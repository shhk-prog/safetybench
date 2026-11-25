#!/bin/bash
# SafeMergeBench Standard Track Merge Runner (Section 3.3).
# Runs priority 1..4 combinations in order; skips any model whose
# merge_metadata.json already exists.
#
# Usage:
#   bash scripts/run_standard_track.sh [--priority 1|2|3|4|all] [--gpu 1] [--seed 42] [--dry-run]

set -euo pipefail

PRIORITY="1"
GPU_ID="1"
SEED="42"
DRY_RUN="0"
SAVE_ROOT="./merged_models/"
MERGEALIGN_DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --priority) PRIORITY="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --mergealign-data) MERGEALIGN_DATA="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
ALPHAS=(0.2 0.4 0.6 0.8 1.0)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_or_skip() {
  local out_dir="$1"
  shift
  if [[ -f "$out_dir/merge_metadata.json" ]]; then
    echo "[SKIP] $out_dir (merge_metadata.json exists)"
    return 0
  fi
  echo "[RUN] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

task_to_domain() {
  case "$1" in
    DartMath) echo "math" ;;
    WildguardMix) echo "safety" ;;
    MagiCoder) echo "coding" ;;
    Aya) echo "multilingual" ;;
    Tulu3IF) echo "instruction" ;;
    *) echo "unknown" ;;
  esac
}

run_combo() {
  local base_model="$1"
  local task_names="$2"   # e.g. WildguardMix-DartMath
  local base_short="${base_model##*/}"
  local merged_root="${SAVE_ROOT%/}/${base_short}_merged"

  IFS='-' read -r -a tasks_arr <<< "$task_names"
  local n_models="${#tasks_arr[@]}"
  local soup_coef
  soup_coef=$(python3 -c "print(round(1.0 / ${n_models}, 6))")

  local inputs=()
  local orders=()
  for t in "${tasks_arr[@]}"; do
    local dom
    dom=$(task_to_domain "$t")
    inputs+=("MergeBench/${base_short}_${dom}")
    orders+=("$dom")
  done

  echo "============================================================"
  echo "Base: $base_model | Tasks: $task_names (N=$n_models)"
  echo "============================================================"

  # 1. Model Soup (TaskArithmetic with 1/N, recorded as ModelSoup)
  run_or_skip "${merged_root}/ModelSoup_task_names_${task_names}_scaling_coef_${soup_coef}" \
    python3 ./merging/main.py --algo TaskArithmetic --record-as ModelSoup \
      --base-model "$base_model" --task_names "$task_names" \
      --scaling-coef "$soup_coef" --seed "$SEED" --save-path "$SAVE_ROOT"

  # 2. Coefficient-sweep methods: TA, TIES (K=0.3), DARE (p=0.9), Consensus TA
  for a in "${ALPHAS[@]}"; do
    run_or_skip "${merged_root}/TaskArithmetic_task_names_${task_names}_scaling_coef_${a}" \
      python3 ./merging/main.py --algo TaskArithmetic \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/TIES_task_names_${task_names}_scaling_coef_${a}_merge_func_sum_K_0.3" \
      python3 ./merging/main.py --algo TIES --K 0.3 \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/DARE_task_names_${task_names}_scaling_coef_${a}_p_0.9" \
      python3 ./merging/main.py --algo DARE --p 0.9 \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/Consensus_task_names_${task_names}_scaling_coef_${a}_k_2_lamda_0.5" \
      python3 ./merging/main.py --algo Consensus \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    # External: DELLA (mergekit)
    run_or_skip "${merged_root}/DELLA_task_names_${task_names}_alpha_${a}" \
      python3 ./safemerge/external/run_della.py \
        --base "$base_model" --inputs "${inputs[@]}" \
        --task-names "$task_names" --alpha "$a" --density 0.5 --epsilon 0.05 --lambda-coef 1.0 \
        --seed "$SEED" --out "${merged_root}/DELLA_task_names_${task_names}_alpha_${a}"

    # External: MergeAlign (temperature sweep mapped to alpha grid if dataset provided)
    if [[ -n "$MERGEALIGN_DATA" ]]; then
      run_or_skip "${merged_root}/MergeAlign_task_names_${task_names}_temp_${a}" \
        python3 ./safemerge/external/run_mergealign.py \
          --base "$base_model" --inputs "${inputs[@]}" \
          --task-names "$task_names" --dataset-path "$MERGEALIGN_DATA" --temp "$a" \
          --seed "$SEED" --out "${merged_root}/MergeAlign_task_names_${task_names}_temp_${a}"
    fi
  done

  # 3. Single-point MergeBench methods: Localize-and-Stitch, RegMean, RegMean++, Fisher
  run_or_skip "${merged_root}/LocalizeAndStitch_task_names_${task_names}_sparsity_0.1_dataless_True_sigmoid_bias_3" \
    python3 ./merging/main.py --algo LocalizeAndStitch --sparsity 0.1 --dataless \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  run_or_skip "${merged_root}/RegMean_task_names_${task_names}_reduction_0.5" \
    python3 ./merging/main.py --algo RegMean --reduction 0.5 \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  run_or_skip "${merged_root}/RegMeanPlusPlus_task_names_${task_names}_reduction_0.1" \
    python3 ./merging/main.py --algo RegMeanPlusPlus --reduction 0.1 \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  # Fisher (stage 1 per task + stage 2 merge)
  local fisher_out="${merged_root}/Fisher_task_names_${task_names}_model_coeff_value_1.0"
  if [[ ! -f "$fisher_out/merge_metadata.json" ]]; then
    for t in "${tasks_arr[@]}"; do
      run_or_skip "${fisher_out}" \
        python3 ./merging/main.py --algo Fisher --base-model "$base_model" \
          --task_names "$t" --save_group "$task_names" --fisher_only --model_coeff_value 1.0 --seed "$SEED" --save-path "$SAVE_ROOT"
    done
    run_or_skip "${fisher_out}" \
      python3 ./merging/main.py --algo Fisher --base-model "$base_model" \
        --task_names "$task_names" --save_group "$task_names" --merge_only --keep_checkpoints \
        --model_coeff_value 1.0 --seed "$SEED" --save-path "$SAVE_ROOT"
  else
    echo "[SKIP] $fisher_out (merge_metadata.json exists)"
  fi

  # External: LED-Merging (single default point)
  local model_fam="llama3"
  [[ "$base_short" == *gemma* ]] && model_fam="gemma2"
  run_or_skip "${merged_root}/LEDMerging_task_names_${task_names}" \
    python3 ./safemerge/external/run_led_merging.py \
      --base "$base_model" --inputs "${inputs[@]}" --orders "${orders[@]}" \
      --task-names "$task_names" --model-family "$model_fam" --seed "$SEED" \
      --out "${merged_root}/LEDMerging_task_names_${task_names}"
}

should_run() {
  [[ "$PRIORITY" == "all" || "$PRIORITY" == "$1" ]]
}

if should_run "1"; then
  run_combo "meta-llama/Llama-3.1-8B-Instruct" "WildguardMix-DartMath"
  run_combo "meta-llama/Llama-3.1-8B-Instruct" "WildguardMix-MagiCoder"
fi

if should_run "2"; then
  run_combo "meta-llama/Llama-3.1-8B-Instruct" "DartMath-WildguardMix-MagiCoder-Aya-Tulu3IF"
fi

if should_run "3"; then
  run_combo "google/gemma-2-9b-it" "WildguardMix-DartMath"
  run_combo "google/gemma-2-9b-it" "WildguardMix-MagiCoder"
fi

if should_run "4"; then
  run_combo "meta-llama/Llama-3.1-8B-Instruct" "WildguardMix-Aya"
  run_combo "meta-llama/Llama-3.1-8B-Instruct" "WildguardMix-Tulu3IF"
fi
