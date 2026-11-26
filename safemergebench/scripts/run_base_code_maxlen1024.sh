#!/usr/bin/env bash
# Sensitivity check (Item 3): Base Llama-3.1-8B-Instruct on MBPP+ & HumanEval+
# with max_length_generation=1024 (1-off check; main protocol remains 512).
#
# Usage:
#   bash scripts/run_base_code_maxlen1024.sh --gpu <CUDA_ID>
# Estimated runtime on H100: ~65-90 minutes (MBPP+ 378x10 + HumanEval+ 164x10 at max_length=1024).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPY="$ROOT/.venvs/merging/bin/python"
BPY="$ROOT/.venvs/bigcode/bin/python"
BACC="$ROOT/.venvs/bigcode/bin/accelerate"

GPU_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU_ID" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_ID="$CUDA_VISIBLE_DEVICES"
  else
    GPU_ID="0"
  fi
fi

if [[ -f "/mnt/nas/home/hiromi/src/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "/mnt/nas/home/hiromi/src/.env"
  set +a
fi

export CUDA_MPS_PIPE_DIRECTORY=/nonexistent
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

RUN_DIR=$("$MPY" "$ROOT/safemerge/new_run.py" \
  --name std-llama31-base-code-maxlen1024-sensitivity \
  --why "点3：Base(Llama-3.1-8B-Instruct)のMBPP+(512切断率51.5%)・HumanEval+(8.1%)についてmax_length_generation=1024で1回だけ感度確認する（本設定は512維持）" \
  --what "meta-llama/Llama-3.1-8B-Instruct に対し bigcode-evaluation-harness (--max_length_generation 1024 --tasks humanevalplus,mbppplus --temperature 0.2 --n_samples 10 --batch_size 10 --save_generations) と問題単位ブートストラップCI・切断率ゲートを実行する" \
  --expect "512切断率が大きく下がり、pass@1の変動幅から512上限による影響度を定量化できる" \
  --paper-use "Section 2.2 / Appendix sensitivity check (max_length_generation 512 vs 1024)" \
  --related "20261126-1525_std-llama31-code-utility-gate,20261126-1718_std-llama31-code-gate-prob-boot-trunc")

echo "RUN_DIR=$RUN_DIR | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
OUT="$RUN_DIR/outputs/base_maxlen1024"
mkdir -p "$OUT" "$RUN_DIR/logs"
printf 'bash %q --gpu %q\n' "$0" "$GPU_ID" >> "$RUN_DIR/commands.sh"

echo "=== [$(date '+%H:%M:%S')] Starting bigcode max_length_generation=1024 (Base) ===" | tee -a "$RUN_DIR/logs/run.log"
(
  cd "$ROOT/bigcode-evaluation-harness"
  "$BACC" launch --num_processes 1 main.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" --max_length_generation 1024 --precision bf16 \
    --tasks humanevalplus,mbppplus --temperature 0.2 --n_samples 10 --batch_size 10 \
    --allow_code_execution --save_generations \
    --save_generations_path "$OUT/code_generations.json" \
    --metric_output_path "$OUT/code_eval.json" --use_auth_token
) 2>&1 | tee -a "$RUN_DIR/logs/base_maxlen1024_bigcode.log"

for TASK in humanevalplus mbppplus; do
  echo "=== [$(date '+%H:%M:%S')] Scoring per-sample details: $TASK ===" | tee -a "$RUN_DIR/logs/run.log"
  "$BPY" "$ROOT/safemerge/eval_code_samples.py" \
    --task "$TASK" \
    --generations "$OUT/code_generations_${TASK}.json" \
    --out-jsonl "$OUT/${TASK}_details.jsonl" \
    --summary-out "$OUT/${TASK}_details_summary.json" \
    2>&1 | tee -a "$RUN_DIR/logs/${TASK}_details.log"

  "$MPY" "$ROOT/safemerge/utility_gate.py" check-code \
    --model "meta-llama/Llama-3.1-8B-Instruct" --task "$TASK" \
    --generations "$OUT/${TASK}_details.jsonl" \
    --metric-json "$OUT/code_eval.json" \
    --max-length-generation 1024 \
    --review-out "$OUT/samples_review_${TASK}.md" \
    --report "$OUT/code_gate_${TASK}.json" || true
done

echo "=== [$(date '+%H:%M:%S')] DONE (RUN_DIR=$RUN_DIR) ===" | tee -a "$RUN_DIR/logs/run.log"
