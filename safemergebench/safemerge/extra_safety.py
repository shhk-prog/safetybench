"""Extra safety benchmarks generation via vLLM (Section 4.2).

Uses the model's OWN tokenizer chat template (`tokenizer.apply_chat_template`),
greedy decoding (`temperature=0.0`), and `max_tokens=512`.
Saves output in the exact same schema as safety-eval's `safety_generation.json`
(`{task_name: [{"id": ..., "prompt": ..., "response": ...}, ...]}`).

Usage:
  python safemerge/extra_safety.py \
    --model merged_models/... \
    --out results/.../safety_generation_extra.json \
    --tasks jailbreakbench,strongreject,wildjailbreak,trustllm_jailbreaktrigger \
    --wildjailbreak-tune-file ... \
    --trustllm-train-exclude /mnt/nas/home/hiromi/src/sst_v2/v3/data/response_dataframe.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.datasets import DEFAULT_TASKS, load_task  # noqa: E402
from integrity_gate import template_family  # noqa: E402


def format_prompts_with_tokenizer(tokenizer: Any, prompts: List[str], allow_plain_fallback: bool = False) -> Tuple[List[str], str]:
    """Apply the model's own chat template to each prompt."""
    has_tpl = bool(getattr(tokenizer, "chat_template", None))
    if not has_tpl:
        if not allow_plain_fallback:
            raise ValueError(
                "Tokenizer has no chat_template (pretrained base?). "
                "Pass --allow-plain-fallback explicitly to run without a chat template."
            )
        return list(prompts), "plain"

    formatted: List[str] = []
    for p in prompts:
        formatted.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    tpl_str = getattr(tokenizer, "chat_template", "")
    fam = template_family({"chat_template": tpl_str}) or "custom"
    return formatted, fam


def build_generation_records(
    task_records: List[Dict[str, Any]],
    responses: List[str],
) -> List[Dict[str, Any]]:
    if len(task_records) != len(responses):
        raise ValueError(f"Mismatched lengths: {len(task_records)} records vs {len(responses)} responses")
    out: List[Dict[str, Any]] = []
    for rec, resp in zip(task_records, responses):
        item = dict(rec)
        item["response"] = resp
        out.append(item)
    return out


def save_json_atomic(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run extra safety benchmarks with vLLM")
    ap.add_argument("--model", required=True, help="merged or base model path/repo")
    ap.add_argument("--out", required=True, help="output path (e.g. .../safety_generation_extra.json)")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated list from: jailbreakbench,strongreject,wildjailbreak,trustllm_jailbreaktrigger")
    ap.add_argument("--wildjailbreak-tune-file", default=None,
                    help="tuning prompts file (e.g. wildjailbreak:harmful) to exclude from WildJailbreak eval")
    ap.add_argument("--trustllm-train-exclude", default=None,
                    help="safety SFT prompts file to exclude from TrustLLM eval")
    ap.add_argument("--overlap-thr", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.70")))
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192")))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None, help="optional cap per task for smoke testing")
    ap.add_argument("--allow-plain-fallback", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing completed tasks")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    existing: Dict[str, List[Dict[str, Any]]] = {}
    if os.path.exists(args.out) and not args.force:
        with open(args.out, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # Load datasets first so failures happen before GPU model loading
    loaded_tasks: Dict[str, List[Dict[str, Any]]] = {}
    manifests: Dict[str, Dict[str, Any]] = {}
    for t in tasks:
        recs, man = load_task(
            t,
            wildjailbreak_tune_file=args.wildjailbreak_tune_file,
            trustllm_train_exclude=args.trustllm_train_exclude,
            overlap_thr=args.overlap_thr,
            limit=args.limit,
        )
        manifests[t] = man
        if t in existing and len(existing[t]) == len(recs) and not args.force:
            print(f"[SKIP] {t} already present with {len(recs)} records in {args.out}")
            continue
        loaded_tasks[t] = recs

    if not loaded_tasks:
        print("All extra safety tasks already generated.")
        return 0

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    started = datetime.now(timezone.utc).isoformat()
    tpl_family = None
    for t, recs in loaded_tasks.items():
        formatted, tpl_family = format_prompts_with_tokenizer(
            tokenizer, [r["prompt"] for r in recs], allow_plain_fallback=args.allow_plain_fallback
        )
        outputs = llm.generate(formatted, sampling_params)
        responses = [o.outputs[0].text for o in outputs]
        existing[t] = build_generation_records(recs, responses)
        save_json_atomic(args.out, existing)
        print(f"[DONE] {t}: generated {len(responses)} responses -> {args.out}")

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "extra_safety_manifest.json")
    save_json_atomic(manifest_path, {
        "model": args.model,
        "template_family": tpl_family,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "tasks": manifests,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
