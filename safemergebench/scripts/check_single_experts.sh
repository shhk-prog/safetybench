#!/bin/bash
# Pre-merge sanity check: evaluate single bases & experts (6-8 models) BEFORE
# running the 72-model standard-track merge sweep, verifying that the math/coding
# experts actually degrade safety relative to the Instruct base model and that
# single-expert utility passes the P6 validity check.
#
# Usage:
#   bash scripts/check_single_experts.sh --run-dir RUN_DIR [--gpu 1] [--out-root DIR] [--dry-run]

set -euo pipefail

ORIG_ARGS=("$@")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_run_dir_lib.sh"
extract_run_dir_arg "$@"
set -- "${REMAINING_ARGS[@]}"
record_run_command "$RUN_DIR" bash "$0" "${ORIG_ARGS[@]}"

GPU_ID="1"
OUT_ROOT="$RUN_DIR/outputs/experts"
DRY_RUN="0"
FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --filter) FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$ROOT"

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct|meta-llama/Llama-3.1-8B-Instruct|llama3"
  "MergeBench/Llama-3.1-8B-Instruct_safety|meta-llama/Llama-3.1-8B-Instruct|llama3"
  "MergeBench/Llama-3.1-8B-Instruct_math|meta-llama/Llama-3.1-8B-Instruct|llama3"
  "MergeBench/Llama-3.1-8B-Instruct_coding|meta-llama/Llama-3.1-8B-Instruct|llama3"
  "google/gemma-2-9b-it|google/gemma-2-9b-it|gemma"
  "MergeBench/gemma-2-9b-it_safety|google/gemma-2-9b-it|gemma"
  "MergeBench/gemma-2-9b-it_math|google/gemma-2-9b-it|gemma"
  "MergeBench/gemma-2-9b-it_coding|google/gemma-2-9b-it|gemma"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL BASE TPL <<< "$entry"
  if [[ -n "$FILTER" && "$MODEL" != *"$FILTER"* ]]; then
    continue
  fi
  TAG="${MODEL##*/}"
  OUT="$OUT_ROOT/$TAG"
  if [[ -f "$OUT/safety_generation_extra.json" ]]; then
    echo "[SKIP] $MODEL ($OUT already evaluated)"
    continue
  fi
  CMD=(bash scripts/evaluate_safemerge.sh --run-dir "$RUN_DIR" "$MODEL" "$GPU_ID" "$OUT" "$BASE" "$MODEL" "$TPL")
  echo "[RUN] SKIP_INLINE_JUDGE=1 ${CMD[*]}"
  if [[ "$DRY_RUN" == "0" ]]; then
    SKIP_INLINE_JUDGE=1 "${CMD[@]}"
  fi
done

echo "[NEXT] After generation finishes, run multi-judge scoring once across all experts:"
echo "  .venvs/safety-eval/bin/python safemerge/judge.py ${OUT_ROOT}/*/safety_generation*.json --judges harmbench,wildguard,strongreject"
echo "  python3 safemerge/metrics.py ${OUT_ROOT}/* --track standard --out-dir ${RUN_DIR}/outputs/metrics"
echo "  python3 safemerge/utility_check.py --measured ${OUT_ROOT} --out ${RUN_DIR}/outputs/metrics/utility_check.json"
