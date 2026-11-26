#!/usr/bin/env bash
# SafeMergeBench Standard Track: Llama-3.1-8B-Instruct 4 single-expert verification batch
#   Stage A: Patched WildGuardTest regeneration (Base, _safety, _math, _coding)
#   Stage B: Official 3-judge scoring (HarmBench, WildGuard, StrongREJECT) + v2 OR-VRR/XSTest/CI metrics
#   Stage C: GSM8K (gsm8k_cot --log_samples, 1,319 items) for Base, _safety, _coding (+ reuse logged _math)
#   Stage D: HumanEval+ & MBPP+ (--save_generations, n_samples=10) for _safety (Table 11)
#   Stage E: Section 2.2 utility gates (math_unified_v1 + problem-level bootstrap code gate + 20-sample review sheets)
#
# Usage:
#   bash scripts/run_llama31_experts_v2.sh --gpu <CUDA_ID> [--stages A,B,C,D,E] [--vllm-mem 0.50] [--run-dir PATH]
# Example (all stages A-E on CUDA device 1 = physical GPU 2 on iag-02):
#   bash scripts/run_llama31_experts_v2.sh --gpu 1 --vllm-mem 0.50 --stages A,B,C,D,E
# Example (split: Stages A-C first):
#   bash scripts/run_llama31_experts_v2.sh --gpu 1 --vllm-mem 0.50 --stages A,B,C

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPY="$ROOT/.venvs/merging/bin/python"
SPY="$ROOT/.venvs/safety-eval/bin/python"
LPY="$ROOT/.venvs/lmeval/bin/lm_eval"
BPY="$ROOT/.venvs/bigcode/bin/python"
BACC="$ROOT/.venvs/bigcode/bin/accelerate"

GPU_ID=""
STAGES="A,B,C,D,E"
VLLM_MEM="0.50"
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --vllm-mem) VLLM_MEM="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$GPU_ID" ]]; then
  echo "ERROR: --gpu <CUDA_VISIBLE_DEVICES_ID> is required (e.g. --gpu 1)." >&2
  exit 2
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
export VLLM_GPU_MEMORY_UTILIZATION="$VLLM_MEM"
export VLLM_MAX_MODEL_LEN=8192
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venvs/safety-eval/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

has_stage() {
  [[ ",$STAGES," == *",$1,"* ]]
}

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR=$("$MPY" "$ROOT/safemerge/new_run.py" \
    --name std-llama31-experts-v2-wgt-gsm8k-safetycode \
    --why "第3節の2（WildGuardTest修正後再生成＋3判定器採点）・第2.2/2.3節（4モデルGSM8K/Codeゲート通過＆eval_lock.yaml固定）・表11【要測定】項目（_safetyのGSM8K/HE+/MBPP+、_codingのGSM8K）を完了する" \
    --what "Llama-3.1-8B-Instruct単体4モデル(Base,_safety,_math,_coding)に対し、指定ステージ($STAGES)を実行する" \
    --expect "WildGuardTest参照回答一致率≈0%、全4モデルVRR>=0.95、_math GSM8K 82.94%(>Base)、表11【要測定】値が揃い、2.2ゲート通過後にeval_lock.yamlを固定できる" \
    --paper-use "Section 2.2-2.3 (eval_lock), Section 3.2 (WildGuardTest fix), Table 11 single experts" \
    --related "20261126-1525_std-llama31-code-utility-gate,20261126-1718_std-llama31-code-gate-prob-boot-trunc,safemergebench/results/experts")
fi

if [[ ! -f "$RUN_DIR/RUN.md" ]]; then
  echo "ERROR: $RUN_DIR/RUN.md does not exist. Create a run dir via safemerge/new_run.py first." >&2
  exit 2
fi

echo "RUN_DIR=$RUN_DIR | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES | STAGES=$STAGES | VLLM_MEM=$VLLM_MEM"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/outputs"
printf 'bash %q --gpu %q --stages %q --vllm-mem %q --run-dir %q\n' "$0" "$GPU_ID" "$STAGES" "$VLLM_MEM" "$RUN_DIR" >> "$RUN_DIR/commands.sh"

declare -A MODELS=(
  ["base"]="meta-llama/Llama-3.1-8B-Instruct"
  ["safety"]="MergeBench/Llama-3.1-8B-Instruct_safety"
  ["math"]="MergeBench/Llama-3.1-8B-Instruct_math"
  ["coding"]="MergeBench/Llama-3.1-8B-Instruct_coding"
)
TAGS=(base safety math coding)

# ---- Stage A: Patched WildGuardTest regeneration ----
if has_stage "A"; then
  for TAG in "${TAGS[@]}"; do
    MODEL="${MODELS[$TAG]}"
    OUT="$RUN_DIR/outputs/$TAG"
    mkdir -p "$OUT"
    echo "=== [$(date '+%H:%M:%S')] Stage A: WildGuardTest regen ($TAG: $MODEL) ===" | tee -a "$RUN_DIR/logs/run.log"
    (
      cd "$ROOT/safety-eval-fork"
      "$SPY" evaluation/eval.py generators \
        --model_name_or_path "$MODEL" --use_vllm \
        --model_input_template_path_or_name hf \
        --tasks wildguardtest \
        --report_output_path "$OUT/wgt_eval.json" \
        --save_individual_results_path "$OUT/wgt_generation.json" \
        --batch_size 8
    ) 2>&1 | tee -a "$RUN_DIR/logs/${TAG}_wgt_gen.log"
  done
fi

# ---- Stage B: Official 3-judge scoring + v2 metrics.py ----
if has_stage "B"; then
  echo "=== [$(date '+%H:%M:%S')] Stage B: 3-judge scoring WildGuardTest (4 models) ===" | tee -a "$RUN_DIR/logs/run.log"
  "$SPY" "$ROOT/safemerge/judge.py" \
    "$RUN_DIR/outputs/base/wgt_generation.json" \
    "$RUN_DIR/outputs/safety/wgt_generation.json" \
    "$RUN_DIR/outputs/math/wgt_generation.json" \
    "$RUN_DIR/outputs/coding/wgt_generation.json" \
    --judges harmbench,wildguard,strongreject \
    --config-out "$RUN_DIR/outputs/wgt_judge_config.json" \
    2>&1 | tee -a "$RUN_DIR/logs/wgt_judge.log"

  echo "=== [$(date '+%H:%M:%S')] Stage B2: Verify WGT non-overwrite & run v2 metrics.py ===" | tee -a "$RUN_DIR/logs/run.log"
  "$MPY" - "$RUN_DIR" "$ROOT" << 'PY'
import json, os, shutil, sys
from datasets import load_dataset

run_dir, root = sys.argv[1], sys.argv[2]
ds = load_dataset("allenai/wildguardmix", "wildguardtest")["test"]
ref_by_id, ref_by_prompt = {}, {}
for idx, row in enumerate(ds):
    resp = (row.get("response") or "").strip()
    if resp:
        ref_by_id[str(row.get("id", idx))] = resp
        ref_by_prompt[(row.get("prompt") or "").strip()] = resp

old_map = {
    "base": os.path.join(root, "results/experts/Llama-3.1-8B-Instruct"),
    "safety": os.path.join(root, "results/experts/Llama-3.1-8B-Instruct_safety"),
    "math": os.path.join(root, "results/experts/Llama-3.1-8B-Instruct_math"),
    "coding": os.path.join(root, "results/experts/Llama-3.1-8B-Instruct_coding"),
}
verif = {}
for tag, old_dir in old_map.items():
    out_sub = os.path.join(run_dir, "outputs", tag)
    wgt_j = json.load(open(os.path.join(out_sub, "wgt_generation_judged.json"), encoding="utf-8"))
    wgt_recs = wgt_j.get("wildguardtest", [])
    n_match = sum(
        1 for r in wgt_recs
        if (r.get("response") or "").strip() and (
            (r.get("response") or "").strip()
            == (ref_by_id.get(str(r.get("id"))) or ref_by_prompt.get((r.get("prompt") or "").strip(), ""))
        )
    )
    verif[tag] = {"n": len(wgt_recs), "exact_ref_match": n_match, "match_rate": round(n_match / max(1, len(wgt_recs)), 6)}

    old_main = json.load(open(os.path.join(old_dir, "safety_generation_judged.json"), encoding="utf-8"))
    merged_main = {k: v for k, v in old_main.items() if k != "wildguardtest"}
    merged_main["wildguardtest"] = wgt_recs
    with open(os.path.join(out_sub, "safety_generation_judged.json"), "w", encoding="utf-8") as f:
        json.dump(merged_main, f, indent=2, ensure_ascii=False)
    shutil.copy2(os.path.join(old_dir, "safety_generation_extra_judged.json"),
                 os.path.join(out_sub, "safety_generation_extra_judged.json"))

with open(os.path.join(run_dir, "outputs", "wgt_ref_match_check.json"), "w", encoding="utf-8") as f:
    json.dump(verif, f, indent=2, ensure_ascii=False)
print("WGT ref match check:", json.dumps(verif, indent=2))
PY

  "$MPY" "$ROOT/safemerge/metrics.py" \
    "$RUN_DIR/outputs/base" \
    "$RUN_DIR/outputs/safety" \
    "$RUN_DIR/outputs/math" \
    "$RUN_DIR/outputs/coding" \
    --track standard \
    --out-dir "$RUN_DIR/outputs/metrics_v2" \
    2>&1 | tee -a "$RUN_DIR/logs/metrics_v2.log"
fi

# ---- Stage C: GSM8K (--log_samples) for base, safety, coding (+ copy logged math) ----
if has_stage "C"; then
  mkdir -p "$RUN_DIR/outputs/math/gsm8k_logged"
  if [[ ! -d "$RUN_DIR/outputs/math/gsm8k_logged/MergeBench__Llama-3.1-8B-Instruct_math" ]]; then
    cp -a "$ROOT/results/experts/Llama-3.1-8B-Instruct_math_gsm8k_logged/MergeBench__Llama-3.1-8B-Instruct_math" \
          "$RUN_DIR/outputs/math/gsm8k_logged/"
  fi
  for TAG in base safety coding; do
    MODEL="${MODELS[$TAG]}"
    OUT="$RUN_DIR/outputs/$TAG/gsm8k_logged"
    mkdir -p "$OUT"
    echo "=== [$(date '+%H:%M:%S')] Stage C: gsm8k_cot --log_samples ($TAG: $MODEL) ===" | tee -a "$RUN_DIR/logs/run.log"
    "$LPY" --model hf --model_args pretrained="$MODEL" --tasks gsm8k_cot \
      --device cuda:0 --batch_size 16 --log_samples --output_path "$OUT" \
      2>&1 | tee -a "$RUN_DIR/logs/${TAG}_gsm8k.log"
  done
fi

# ---- Stage D: _safety HumanEval+ & MBPP+ (Table 11) ----
if has_stage "D"; then
  SAFE_CODE_OUT="$RUN_DIR/outputs/safety/code"
  mkdir -p "$SAFE_CODE_OUT"
  echo "=== [$(date '+%H:%M:%S')] Stage D: bigcode humanevalplus,mbppplus (safety expert) ===" | tee -a "$RUN_DIR/logs/run.log"
  (
    cd "$ROOT/bigcode-evaluation-harness"
    "$BACC" launch --num_processes 1 main.py \
      --model "MergeBench/Llama-3.1-8B-Instruct_safety" --max_length_generation 512 --precision bf16 \
      --tasks humanevalplus,mbppplus --temperature 0.2 --n_samples 10 --batch_size 10 \
      --allow_code_execution --save_generations \
      --save_generations_path "$SAFE_CODE_OUT/code_generations.json" \
      --metric_output_path "$SAFE_CODE_OUT/code_eval.json" --use_auth_token
  ) 2>&1 | tee -a "$RUN_DIR/logs/safety_bigcode.log"

  for TASK in humanevalplus mbppplus; do
    echo "=== [$(date '+%H:%M:%S')] Stage D2: per-sample details safety / $TASK ===" | tee -a "$RUN_DIR/logs/run.log"
    "$BPY" "$ROOT/safemerge/eval_code_samples.py" \
      --task "$TASK" \
      --generations "$SAFE_CODE_OUT/code_generations_${TASK}.json" \
      --out-jsonl "$SAFE_CODE_OUT/${TASK}_details.jsonl" \
      --summary-out "$SAFE_CODE_OUT/${TASK}_details_summary.json" \
      2>&1 | tee -a "$RUN_DIR/logs/safety_${TASK}_details.log"
  done
fi

# ---- Stage E: Section 2.2 Utility Gates ----
if has_stage "E"; then
  echo "=== [$(date '+%H:%M:%S')] Stage E: Running Section 2.2 utility gates ===" | tee -a "$RUN_DIR/logs/run.log"
  "$MPY" - "$RUN_DIR" "$ROOT" << 'PY'
import csv, json, os, subprocess, sys

run_dir, root = sys.argv[1], sys.argv[2]
py = sys.executable

vrr_by_tag = {}
avg_csv = os.path.join(run_dir, "outputs", "metrics_v2", "attack_averages.csv")
if os.path.isfile(avg_csv):
    for r in csv.DictReader(open(avg_csv, encoding="utf-8")):
        if r["split"] == "attack_all" and r["judge"] in ("harmbench", "wildguard", "strongreject"):
            m = r["model"]
            v = float(r["micro_VRR"])
            vrr_by_tag[m] = min(vrr_by_tag.get(m, 1.0), v)
print("Min OR-rule safety VRR by tag:", vrr_by_tag)

models = {
    "base": "meta-llama/Llama-3.1-8B-Instruct",
    "safety": "MergeBench/Llama-3.1-8B-Instruct_safety",
    "math": "MergeBench/Llama-3.1-8B-Instruct_math",
    "coding": "MergeBench/Llama-3.1-8B-Instruct_coding",
}

base_gsm8k = None
base_eval_dir = os.path.join(run_dir, "outputs", "base", "gsm8k_logged")
if os.path.isdir(base_eval_dir):
    base_rep = os.path.join(run_dir, "outputs", "base", "gsm8k_gate.json")
    cmd = [
        py, os.path.join(root, "safemerge/utility_gate.py"), "check",
        "--model", models["base"], "--task", "gsm8k_cot",
        "--eval-dir", base_eval_dir,
        "--review-out", os.path.join(run_dir, "outputs", "base", "samples_review_gsm8k.md"),
        "--report", base_rep,
    ]
    if "base" in vrr_by_tag:
        cmd += ["--safety-vrr", str(vrr_by_tag["base"])]
    subprocess.run(cmd, check=False)
    base_gsm8k = json.load(open(base_rep)).get("score")

for tag in ["math", "safety", "coding"]:
    ev = os.path.join(run_dir, "outputs", tag, "gsm8k_logged")
    if not os.path.isdir(ev):
        continue
    rep = os.path.join(run_dir, "outputs", tag, "gsm8k_gate.json")
    cmd = [
        py, os.path.join(root, "safemerge/utility_gate.py"), "check",
        "--model", models[tag], "--task", "gsm8k_cot",
        "--eval-dir", ev,
        "--review-out", os.path.join(run_dir, "outputs", tag, "samples_review_gsm8k.md"),
        "--report", rep,
    ]
    if tag in vrr_by_tag:
        cmd += ["--safety-vrr", str(vrr_by_tag[tag])]
    if tag == "math" and base_gsm8k is not None:
        cmd += ["--base-score", str(base_gsm8k)]
    subprocess.run(cmd, check=False)

math_ev = os.path.join(run_dir, "outputs", "math", "gsm8k_logged")
if os.path.isdir(math_ev):
    cmd = [
        py, os.path.join(root, "safemerge/utility_gate.py"), "check",
        "--model", models["math"], "--task", "gsm8k_cot",
        "--eval-dir", math_ev, "--use-lmeval-filtered",
        "--report", os.path.join(run_dir, "outputs", "math", "gsm8k_gate_lmeval_raw_p6.json"),
    ]
    if base_gsm8k is not None:
        cmd += ["--base-score", str(base_gsm8k)]
    subprocess.run(cmd, check=False)

safe_code = os.path.join(run_dir, "outputs", "safety", "code")
for task in ["humanevalplus", "mbppplus"]:
    det = os.path.join(safe_code, f"{task}_details.jsonl")
    if not os.path.isfile(det):
        continue
    cmd = [
        py, os.path.join(root, "safemerge/utility_gate.py"), "check-code",
        "--model", models["safety"], "--task", task,
        "--generations", det,
        "--metric-json", os.path.join(safe_code, "code_eval.json"),
        "--max-length-generation", "512",
        "--review-out", os.path.join(safe_code, f"samples_review_{task}.md"),
        "--report", os.path.join(safe_code, f"code_gate_{task}.json"),
    ]
    if "safety" in vrr_by_tag:
        cmd += ["--safety-vrr", str(vrr_by_tag["safety"])]
    subprocess.run(cmd, check=False)
PY
fi

echo "=== [$(date '+%H:%M:%S')] DONE (STAGES=$STAGES, RUN_DIR=$RUN_DIR) ===" | tee -a "$RUN_DIR/logs/run.log"
