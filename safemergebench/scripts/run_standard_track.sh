#!/bin/bash
# SafeMergeBench Standard Track Merge Runner (Section 3.3).
# Runs priority 1..4 combinations in order; skips any model whose
# merge_metadata.json already exists.
#
# Notes:
#   - Hammoud et al. (EMNLP 2024, LM-Cocktail via hammoudhasan/MergeAlign) is run
#     as a single default point (temp=1.0) using MergeBench/{domain}_val calibration
#     data — NOT swept on the alpha grid (Item 5 / P3).
#   - Optional `--include-asymmetric-ta` also runs Asymmetric Task Arithmetic
#     (domain_coef=1.0 fixed, safety_coef in {0.2..1.0}).
#
# Usage:
#   bash scripts/run_standard_track.sh --run-dir RUN_DIR [--priority 1|2|3|4|all] [--gpu 1] [--seed 42] [--include-asymmetric-ta] [--dry-run]

set -euo pipefail

ORIG_ARGS=("$@")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_run_dir_lib.sh"
extract_run_dir_arg "$@"
set -- "${REMAINING_ARGS[@]}"
record_run_command "$RUN_DIR" bash "$0" "${ORIG_ARGS[@]}"

PRIORITY="1"
GPU_ID="1"
SEED="42"
DRY_RUN="0"
INCLUDE_ASYM_TA="0"
SAVE_ROOT="./merged_models/"
PYTHON_BIN="${MERGING_PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --priority) PRIORITY="$2"; shift 2 ;;
    --gpu) GPU_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --save-root) SAVE_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --include-asymmetric-ta) INCLUDE_ASYM_TA="1"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
ALPHAS=(0.2 0.4 0.6 0.8 1.0)
cd "$ROOT"

if [[ "$PYTHON_BIN" == "python3" && -x "$ROOT/.venvs/merging/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venvs/merging/bin/python"
fi

record_merge_artifact() {
  local out_dir="$1"
  if [[ -f "$out_dir/merge_metadata.json" ]]; then
    mkdir -p "$RUN_DIR/outputs/merge_metadata"
    local base_tag
    base_tag="$(basename "$(dirname "$out_dir")")__$(basename "$out_dir")"
    cp "$out_dir/merge_metadata.json" "$RUN_DIR/outputs/merge_metadata/${base_tag}.json"
    printf '%s\n' "$(cd "$out_dir" && pwd)" >> "$RUN_DIR/outputs/merged_model_paths.txt"
  fi
}

run_or_skip() {
  local out_dir="$1"
  shift
  if [[ -f "$out_dir/merge_metadata.json" ]]; then
    echo "[SKIP] $out_dir (merge_metadata.json exists)"
    record_merge_artifact "$out_dir"
    return 0
  fi
  echo "[RUN] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
  record_merge_artifact "$out_dir"
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
  local domains=()
  for t in "${tasks_arr[@]}"; do
    local dom
    dom=$(task_to_domain "$t")
    inputs+=("MergeBench/${base_short}_${dom}")
    domains+=("$dom")
    if [[ "$dom" == "coding" ]]; then
      orders+=("code")
    else
      orders+=("$dom")
    fi
  done

  echo "============================================================"
  echo "Base: $base_model | Tasks: $task_names (N=$n_models)"
  echo "============================================================"

  # 1. Model Soup (TaskArithmetic with 1/N, recorded as ModelSoup)
  run_or_skip "${merged_root}/ModelSoup_task_names_${task_names}_scaling_coef_${soup_coef}" \
    "$PYTHON_BIN" ./merging/main.py --algo TaskArithmetic --record-as ModelSoup \
      --base-model "$base_model" --task_names "$task_names" \
      --scaling-coef "$soup_coef" --seed "$SEED" --save-path "$SAVE_ROOT"

  # 2. Coefficient-sweep methods: TA, TIES (K=0.3), DARE (p=0.9), Consensus TA (effective lamda=0.2), DELLA
  for a in "${ALPHAS[@]}"; do
    run_or_skip "${merged_root}/TaskArithmetic_task_names_${task_names}_scaling_coef_${a}" \
      "$PYTHON_BIN" ./merging/main.py --algo TaskArithmetic \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/TIES_task_names_${task_names}_scaling_coef_${a}_merge_func_sum_K_0.3" \
      "$PYTHON_BIN" ./merging/main.py --algo TIES --K 0.3 \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/DARE_task_names_${task_names}_scaling_coef_${a}_p_0.9" \
      "$PYTHON_BIN" ./merging/main.py --algo DARE --p 0.9 \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/Consensus_task_names_${task_names}_scaling_coef_${a}_k_2_lamda_0.2" \
      "$PYTHON_BIN" ./merging/main.py --algo Consensus \
        --base-model "$base_model" --task_names "$task_names" \
        --scaling-coef "$a" --seed "$SEED" --save-path "$SAVE_ROOT"

    run_or_skip "${merged_root}/DELLA_task_names_${task_names}_alpha_${a}" \
      "$PYTHON_BIN" ./safemerge/external/run_della.py \
        --base "$base_model" --inputs "${inputs[@]}" \
        --task-names "$task_names" --alpha "$a" --density 0.5 --epsilon 0.05 --lambda-coef 1.0 \
        --seed "$SEED" --out "${merged_root}/DELLA_task_names_${task_names}_alpha_${a}"

    # Optional Asymmetric Task Arithmetic (2-model safety + domain only: domain_coef=1.0, safety_coef=a)
    if [[ "$INCLUDE_ASYM_TA" == "1" && "$n_models" == "2" ]]; then
      run_or_skip "${merged_root}/AsymTaskArithmetic_task_names_${task_names}_safety_coef_${a}_domain_coef_1.0" \
        "$PYTHON_BIN" ./safemerge/external/run_asymmetric_ta.py \
          --base "$base_model" --safety "${inputs[0]}" --domain "${inputs[1]}" \
          --task-names "$task_names" --safety-coef "$a" --domain-coef 1.0 \
          --seed "$SEED" --out "${merged_root}/AsymTaskArithmetic_task_names_${task_names}_safety_coef_${a}_domain_coef_1.0"
    fi
  done

  # 3. Single-point methods: Localize-and-Stitch, RegMean, RegMean++, Fisher, LED-Merging, Hammoud et al. (LM-Cocktail)
  run_or_skip "${merged_root}/LocalizeAndStitch_task_names_${task_names}_sparsity_0.1_dataless_True_sigmoid_bias_3" \
    "$PYTHON_BIN" ./merging/main.py --algo LocalizeAndStitch --sparsity 0.1 --dataless \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  # RegMean: float32 primary comparison version + upstream bfloat16 reference (Section 3.6)
  run_or_skip "${merged_root}/RegMeanFP32_task_names_${task_names}_reduction_0.5_regmean_fp32_True" \
    "$PYTHON_BIN" ./merging/main.py --algo RegMean --reduction 0.5 --regmean-fp32 \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  run_or_skip "${merged_root}/RegMean_task_names_${task_names}_reduction_0.5" \
    "$PYTHON_BIN" ./merging/main.py --algo RegMean --reduction 0.5 \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  run_or_skip "${merged_root}/RegMeanPlusPlus_task_names_${task_names}_reduction_0.1" \
    "$PYTHON_BIN" ./merging/main.py --algo RegMeanPlusPlus --reduction 0.1 \
      --base-model "$base_model" --task_names "$task_names" --seed "$SEED" --save-path "$SAVE_ROOT"

  # Fisher (stage 1 per task + stage 2 merge; Bug 1 fixed in merging/main.py)
  local fisher_out="${merged_root}/Fisher_task_names_${task_names}_model_coeff_value_1.0"
  if [[ ! -f "$fisher_out/merge_metadata.json" ]]; then
    for t in "${tasks_arr[@]}"; do
      run_or_skip "${fisher_out}" \
        "$PYTHON_BIN" ./merging/main.py --algo Fisher --base-model "$base_model" \
          --task_names "$t" --save_group "$task_names" --fisher_only --model_coeff_value 1.0 --seed "$SEED" --save-path "$SAVE_ROOT"
    done
    run_or_skip "${fisher_out}" \
      "$PYTHON_BIN" ./merging/main.py --algo Fisher --base-model "$base_model" \
        --task_names "$task_names" --save_group "$task_names" --merge_only --keep_checkpoints \
        --model_coeff_value 1.0 --seed "$SEED" --save-path "$SAVE_ROOT"
  else
    echo "[SKIP] $fisher_out (merge_metadata.json exists)"
  fi

  # External: Hammoud et al. (EMNLP 2024, LM-Cocktail) — single official default point (temp=1.0)
  run_or_skip "${merged_root}/LMCocktail_task_names_${task_names}_temp_1.0" \
    "$PYTHON_BIN" ./safemerge/external/run_mergealign.py \
      --base "$base_model" --inputs "${inputs[@]}" \
      --mergebench-domains "${domains[@]}" \
      --task-names "$task_names" --temp 1.0 --max-per-task 20 --seed "$SEED" \
      --out "${merged_root}/LMCocktail_task_names_${task_names}_temp_1.0"

  # External: LED-Merging (single default point)
  local model_fam="llama3"
  [[ "$base_short" == *gemma* ]] && model_fam="gemma2"
  run_or_skip "${merged_root}/LEDMerging_task_names_${task_names}" \
    "$PYTHON_BIN" ./safemerge/external/run_led_merging.py \
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
