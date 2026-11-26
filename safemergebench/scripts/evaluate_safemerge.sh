#!/bin/bash
# SafeMergeBench evaluation = MergeBench evaluation (scripts/evaluate.sh) with the
# fixes described in README_SAFEMERGE.md and SafeMergeBench 実装指示書:
#   1. integrity gate before generation (config / chat template / merge recipe)
#   2. the safety chat template is chosen per model family instead of "llama3" for all
#      (Gemma-it uses safety-eval's `hf` template -> tokenizer.apply_chat_template;
#       pretrained bases use safemerge/data/plain_template.txt)
#   3. response-validity audit of every safety generation
#   4. extra safety benchmarks (held out from coefficient tuning) + 3 judges + metrics
#
# Usage:
#   bash scripts/evaluate_safemerge.sh --run-dir RUN_DIR MODEL GPU_ID OUTPUT_PATH BASE_MODEL "INPUT1 INPUT2 ..." [TEMPLATE] [EXPECT_ALGO]
# TEMPLATE: llama3 | gemma | plain (default: inferred from the model name; pretrained
#           bases without a chat template must be given explicitly).

set -euo pipefail

ORIG_ARGS=("$@")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/_run_dir_lib.sh"
extract_run_dir_arg "$@"
set -- "${REMAINING_ARGS[@]}"
record_run_command "$RUN_DIR" bash "$0" "${ORIG_ARGS[@]}"

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/evaluate_safemerge.sh --run-dir RUN_DIR MODEL GPU_ID OUTPUT_PATH BASE_MODEL \"INPUT1 INPUT2 ...\" [TEMPLATE] [EXPECT_ALGO]" >&2
  exit 2
fi

MODEL=$1
GPU_ID=$2
OUTPUT_PATH=$3
BASE_MODEL=$4
INPUTS=$5
TEMPLATE=${6:-}
EXPECT_ALGO=${7:-}

if [ -z "$TEMPLATE" ]; then
  case "$(basename "$MODEL") $(basename "$BASE_MODEL")" in
    *Instruct*) TEMPLATE=llama3 ;;
    *gemma*-it*) TEMPLATE=gemma ;;
    *) echo "ERROR: cannot infer chat template for $MODEL (pretrained base?). Pass TEMPLATE explicitly (e.g. plain)." >&2; exit 2 ;;
  esac
fi

# Map SafeMergeBench template family -> safety-eval-fork --model_input_template_path_or_name
# Verified in safety-eval-fork (commit 2920bb85) src/templates/single_turn.py:
#   - SafeMergeBench main evaluation uses `hf` (tokenizer.apply_chat_template) for ALL
#     instruct models (Llama-3-Instruct, Gemma-2-it, etc.) for F3 consistency.
#   - `llama3_mb` (or `MERGEBENCH_REPRO=1` with `llama3`) selects safety-eval's built-in
#     `llama3` template strictly for MergeBench reproduction / ablation comparisons.
#   - `plain` maps to safemerge/data/plain_template.txt ({instruction}) for pretrained bases.
GATE_TEMPLATE="$TEMPLATE"
case "$TEMPLATE" in
  llama3)
    if [ "${MERGEBENCH_REPRO:-0}" = "1" ]; then
      SAFETY_EVAL_TEMPLATE="llama3"
    else
      SAFETY_EVAL_TEMPLATE="hf"
    fi
    ;;
  llama3_mb)
    GATE_TEMPLATE="llama3"
    SAFETY_EVAL_TEMPLATE="llama3"
    ;;
  gemma)  SAFETY_EVAL_TEMPLATE="hf" ;;
  hf)     GATE_TEMPLATE=""; SAFETY_EVAL_TEMPLATE="hf" ;;
  plain)  GATE_TEMPLATE=""; SAFETY_EVAL_TEMPLATE="$ROOT/safemerge/data/plain_template.txt" ;;
  *)      SAFETY_EVAL_TEMPLATE="$TEMPLATE" ;;
esac

echo "model=$MODEL template=$TEMPLATE safety_eval_template=$SAFETY_EVAL_TEMPLATE gpu=$GPU_ID out=$OUTPUT_PATH"

export CUDA_VISIBLE_DEVICES=$GPU_ID
mkdir -p "$OUTPUT_PATH"
ABS_OUT="$(cd "$OUTPUT_PATH" && pwd)"

# Support both conda environments and uv/venv environments under $ROOT/.venvs/<name>
activate_env() {
  local env_name="$1"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$env_name"
  elif [ -f "$ROOT/.venvs/$env_name/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$ROOT/.venvs/$env_name/bin/activate"
  else
    echo "WARNING: neither conda env '$env_name' nor '$ROOT/.venvs/$env_name' found; using current PATH." >&2
  fi
}

deactivate_env() {
  if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_PREFIX:-}" ]; then
    conda deactivate || true
  elif type deactivate >/dev/null 2>&1; then
    deactivate || true
  fi
}

# ---- 1. integrity gate (fails fast; the model is reported as gate-failed) ----
GATE_PY="python3"
if [ -x "$ROOT/.venvs/merging/bin/python" ]; then
  GATE_PY="$ROOT/.venvs/merging/bin/python"
fi

# ---- 1b. Section 2.3 eval_lock verification for merged models ----
# Single experts/bases pass SINGLE_EXPERT_EVAL=1 (or MODEL==BASE_MODEL / MODEL==INPUTS);
# merged models MUST match configs/eval_lock.yaml hash before evaluation starts.
LOCK_FILE="${EVAL_LOCK_FILE:-$ROOT/configs/eval_lock.yaml}"
IS_SINGLE_EXPERT="0"
if [ "${SINGLE_EXPERT_EVAL:-0}" = "1" ] || [ "$MODEL" = "$BASE_MODEL" ] || [ "$MODEL" = "$INPUTS" ]; then
  IS_SINGLE_EXPERT="1"
fi
if [ "$IS_SINGLE_EXPERT" != "1" ]; then
  if [ ! -f "$LOCK_FILE" ]; then
    echo "EVAL LOCK ERROR: $LOCK_FILE not found. Verify single experts via utility_gate.py and lock settings before evaluating merged models (Section 2.3)." >&2
    exit 4
  fi
  if [ -n "${EVAL_RUNTIME_CONFIG:-}" ]; then
    if ! "$GATE_PY" "$ROOT/safemerge/utility_gate.py" verify-lock --lock "$LOCK_FILE" --runtime-config "$EVAL_RUNTIME_CONFIG"; then
      echo "EVAL LOCK MISMATCH: runtime config does not match $LOCK_FILE (Section 2.3)." >&2
      exit 4
    fi
  fi
  cp "$LOCK_FILE" "$RUN_DIR/config/eval_lock.yaml"
fi

GATE_ARGS=(--base "$BASE_MODEL" --inputs $INPUTS --merged "$MODEL" --report "$ABS_OUT/gate.json")
[ -n "$GATE_TEMPLATE" ] && GATE_ARGS+=(--eval-template "$GATE_TEMPLATE")
[ -n "$EXPECT_ALGO" ] && GATE_ARGS+=(--expect-algo "$EXPECT_ALGO")
if ! "$GATE_PY" "$ROOT/safemerge/integrity_gate.py" "${GATE_ARGS[@]}"; then
  echo "GATE FAILED: see $ABS_OUT/gate.json" >&2
  exit 3
fi

cat > "$ABS_OUT/eval_config.json" <<EOF
{
  "model": "$MODEL",
  "base": "$BASE_MODEL",
  "inputs": "$INPUTS",
  "template": "$TEMPLATE",
  "safety_eval_template": "$SAFETY_EVAL_TEMPLATE",
  "expect_algo": "$EXPECT_ALGO",
  "safety_eval_default_judges": {
    "wildguardtest": "WildGuard",
    "harmbench": "WildGuard",
    "xstest": "WildGuard",
    "do_anything_now": "WildGuard"
  }
}
EOF

# ---- 2. utility / generality: identical to MergeBench ----
has_lmeval_task() {
  local dir="$1" task="$2"
  grep -rq --include="results*.json" "\"$task\"" "$dir" 2>/dev/null
}

if [ "${SKIP_UTILITY:-0}" != "1" ]; then
  activate_env lmeval
  if ! has_lmeval_task "$ABS_OUT" "gsm8k_cot"; then
    lm_eval --model hf --model_args pretrained="$MODEL" --tasks gsm8k_cot \
      --device cuda:0 --batch_size 16 --log_samples --output_path "$ABS_OUT"
  else
    echo "[SKIP] gsm8k_cot already present under $ABS_OUT"
  fi
  if [ "${SKIP_MULTILINGUAL:-0}" != "1" ]; then
    if ! has_lmeval_task "$ABS_OUT" "m_mmlu_fr"; then
      lm_eval --model hf --model_args pretrained="$MODEL" \
        --tasks m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru \
        --device cuda:0 --batch_size 8 --output_path "$ABS_OUT"
    else
      echo "[SKIP] multilingual already present under $ABS_OUT"
    fi
  fi
  if [ "${SKIP_IFEVAL:-0}" != "1" ]; then
    if ! has_lmeval_task "$ABS_OUT" "ifeval"; then
      lm_eval --model hf --model_args pretrained="$MODEL" --tasks ifeval \
        --device cuda:0 --batch_size 8 --output_path "$ABS_OUT"
    else
      echo "[SKIP] ifeval already present under $ABS_OUT"
    fi
  fi
  deactivate_env

  if [ "${SKIP_BIGCODE:-0}" != "1" ]; then
    if [ ! -f "$ABS_OUT/code_eval.json" ]; then
      activate_env bigcode
      pushd "$ROOT/bigcode-evaluation-harness" >/dev/null
      accelerate launch main.py --model "$MODEL" --max_length_generation 512 --precision bf16 \
        --tasks humanevalplus,mbppplus --temperature 0.2 --n_samples 10 --batch_size 10 \
        --allow_code_execution --save_generations --save_generations_path "$ABS_OUT/code_generations.json" \
        --metric_output_path "$ABS_OUT/code_eval.json" --use_auth_token
      popd >/dev/null
      deactivate_env
    else
      echo "[SKIP] code_eval.json already present under $ABS_OUT"
    fi
  fi
fi

# ---- 3. safety: MergeBench tasks with the model's own template ----
if [ -f "/mnt/nas/home/hiromi/src/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "/mnt/nas/home/hiromi/src/.env"
  set +a
fi
SAFETY_EVAL_TASKS="${SAFETY_EVAL_TASKS:-wildguardtest,harmbench,xstest,do_anything_now}"
activate_env safety-eval
if [ ! -f "$ABS_OUT/safety_generation.json" ]; then
  pushd "$ROOT/safety-eval-fork" >/dev/null
  export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
  python evaluation/eval.py generators \
    --model_name_or_path "$MODEL" --use_vllm \
    --model_input_template_path_or_name "$SAFETY_EVAL_TEMPLATE" \
    --tasks "$SAFETY_EVAL_TASKS" \
    --report_output_path "$ABS_OUT/safety_eval.json" \
    --save_individual_results_path "$ABS_OUT/safety_generation.json" \
    --batch_size 8
  popd >/dev/null
else
  echo "[SKIP] safety_generation.json already present under $ABS_OUT"
fi

# ---- 4. extra safety benchmarks (held out from coefficient tuning) ----
EXTRA_ARGS=(
  --model "$MODEL"
  --out "$ABS_OUT/safety_generation_extra.json"
  --tasks "jailbreakbench,strongreject,wildjailbreak,trustllm_jailbreaktrigger"
  --wildjailbreak-tune-file "$ROOT/safety-eval-fork/evaluation/tasks/generation/wildjailbreak/harmful.jsonl"
  --seed 42
)
[ "$TEMPLATE" = "plain" ] && EXTRA_ARGS+=(--allow-plain-fallback)
if [ -n "${TRUSTLLM_TRAIN_EXCLUDE:-}" ]; then
  EXTRA_ARGS+=(--trustllm-train-exclude "$TRUSTLLM_TRAIN_EXCLUDE")
fi
python "$ROOT/safemerge/extra_safety.py" "${EXTRA_ARGS[@]}"

# Optional inline judging (for bulk runs across many models, skip here with SKIP_INLINE_JUDGE=1
# and run safemerge/judge.py once across all models to load each 13B judge only once)
if [ "${SKIP_INLINE_JUDGE:-0}" != "1" ]; then
  python "$ROOT/safemerge/judge.py" \
    "$ABS_OUT/safety_generation.json" "$ABS_OUT/safety_generation_extra.json" \
    --judges harmbench,wildguard,strongreject \
    --config-out "$ABS_OUT/judge_config.json"
fi
deactivate_env

# ---- 5. response-validity audit & metrics ----
python3 "$ROOT/safemerge/validity_audit.py" "$ABS_OUT" --out "$ABS_OUT/validity" --sample 50
python3 "$ROOT/safemerge/metrics.py" "$ABS_OUT" --track "${TRACK:-standard}" --out-dir "$RUN_DIR/outputs/metrics"
