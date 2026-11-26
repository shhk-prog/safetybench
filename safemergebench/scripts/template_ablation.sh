#!/bin/bash
# Separates "collapse caused by merging" from "collapse caused by the chat template".
# Runs the MergeBench safety tasks on the base, every single expert, and the merged
# model, once with MergeBench's fixed "llama3" template and once with the model's own
# template, then audits response validity for each run.
#
# Usage: bash scripts/template_ablation.sh --run-dir RUN_DIR BASE GPU_ID OUT_ROOT MERGED OWN_TEMPLATE
#   e.g. bash scripts/template_ablation.sh --run-dir results/runs/... meta-llama/Llama-3.2-3B 1 results/runs/.../outputs/ablation \
#        merged_models/Llama-3.2-3B_merged/RegMeanPlusPlus_... plain
# For pretrained bases with no chat template, pass OWN_TEMPLATE=plain
# (resolved to safemerge/data/plain_template.txt).
# For Gemma instruct models, pass OWN_TEMPLATE=gemma (resolved to safety-eval's `hf`).

set -euo pipefail

ORIG_ARGS=("$@")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_run_dir_lib.sh"
extract_run_dir_arg "$@"
set -- "${REMAINING_ARGS[@]}"
record_run_command "$RUN_DIR" bash "$0" "${ORIG_ARGS[@]}"

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/template_ablation.sh --run-dir RUN_DIR BASE GPU_ID OUT_ROOT MERGED OWN_TEMPLATE" >&2
  exit 2
fi

BASE=$1; GPU_ID=$2; OUT_ROOT=$3; MERGED=$4; OWN_TEMPLATE=$5
NAME=$(basename "$BASE")
MODELS=("$BASE")
for c in instruction math coding safety multilingual; do MODELS+=("MergeBench/${NAME}_${c}"); done
MODELS+=("$MERGED")

export CUDA_VISIBLE_DEVICES=$GPU_ID
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate safety-eval
elif [ -f "$ROOT/.venvs/safety-eval/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$ROOT/.venvs/safety-eval/bin/activate"
fi

resolve_template() {
  case "$1" in
    llama3_mb) echo "llama3" ;;
    llama3)    echo "llama3" ;;
    hf)        echo "hf" ;;
    gemma)     echo "hf" ;;
    plain)     echo "$ROOT/safemerge/data/plain_template.txt" ;;
    *)         echo "$1" ;;
  esac
}

if [ -f "/mnt/nas/home/hiromi/src/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "/mnt/nas/home/hiromi/src/.env"
  set +a
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
for M in "${MODELS[@]}"; do
  for T in llama3 "$OWN_TEMPLATE"; do
    OUT="$OUT_ROOT/$(basename "$M")/template_$T"
    [ -f "$OUT/safety_generation.json" ] && continue
    mkdir -p "$OUT"
    T_ARG="$(resolve_template "$T")"
    (cd "$ROOT/safety-eval-fork" && python evaluation/eval.py generators \
      --model_name_or_path "$M" --use_vllm \
      --model_input_template_path_or_name "$T_ARG" \
      --tasks wildguardtest,harmbench,xstest,do_anything_now \
      --report_output_path "$ROOT/$OUT/safety_eval.json" \
      --save_individual_results_path "$ROOT/$OUT/safety_generation.json" \
      --batch_size 8)
    python3 "$ROOT/safemerge/validity_audit.py" "$OUT" --out "$OUT/validity" --sample 0
  done
done
