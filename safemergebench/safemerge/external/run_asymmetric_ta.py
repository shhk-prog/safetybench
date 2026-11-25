"""Asymmetric Task Arithmetic (domain weight fixed at 1.0, sweep safety weight alpha).

Implements:
    theta_merged = theta_base + domain_coef * (theta_domain - theta_base)
                              + safety_coef * (theta_safety - theta_base)
in bfloat16 while excluding `embed_tokens` and `lm_head` to match MergeBench's
exact task-vector convention (without modifying MergeBench's upstream
`merging/merging_methods/task_arithmetic.py`).

Usage:
  python safemerge/external/run_asymmetric_ta.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --safety MergeBench/Llama-3.1-8B-Instruct_safety \
    --domain MergeBench/Llama-3.1-8B-Instruct_math \
    --safety-coef 0.4 --domain-coef 1.0 \
    --out merged_models/.../AsymTaskArithmetic_... --seed 42
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import set_seed, write_merge_metadata  # noqa: E402


def is_excluded_param(name: str) -> bool:
    """Match MergeBench's `get_task_vector` exclusion of embeddings and lm_head."""
    return "embed_tokens" in name or "lm_head" in name


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Asymmetric Task Arithmetic (fixed domain weight, swept safety weight)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--safety", required=True, help="safety expert checkpoint")
    ap.add_argument("--domain", required=True, help="domain expert checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-names", default=None)
    ap.add_argument("--safety-coef", type=float, required=True)
    ap.add_argument("--domain-coef", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    print(
        f"AsymTaskArithmetic: base={args.base}, safety={args.safety} (w={args.safety_coef}), "
        f"domain={args.domain} (w={args.domain_coef}) -> {args.out}"
    )
    if args.dry_run:
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc).isoformat()
    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    safe_sd = AutoModelForCausalLM.from_pretrained(args.safety, torch_dtype=torch.bfloat16).state_dict()
    dom_sd = AutoModelForCausalLM.from_pretrained(args.domain, torch_dtype=torch.bfloat16).state_dict()

    merged_sd = base_model.state_dict()
    changed = 0
    skipped_keys: List[str] = []
    with torch.no_grad():
        for k, v0 in merged_sd.items():
            if is_excluded_param(k) or k not in safe_sd or k not in dom_sd:
                skipped_keys.append(k)
                continue
            tau_s = safe_sd[k] - v0
            tau_d = dom_sd[k] - v0
            merged_sd[k] = v0 + args.domain_coef * tau_d + args.safety_coef * tau_s
            changed += 1

    base_model.load_state_dict(merged_sd)
    os.makedirs(args.out, exist_ok=True)
    base_model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.safety).save_pretrained(args.out)
    finished = datetime.now(timezone.utc).isoformat()

    write_merge_metadata(
        out_dir=args.out,
        algo="AsymTaskArithmetic",
        kwargs={
            "safety_coef": args.safety_coef,
            "domain_coef": args.domain_coef,
        },
        base_model=args.base,
        inputs=[args.safety, args.domain],
        task_names=args.task_names,
        seed=args.seed,
        dtype="bfloat16",
        started=started,
        finished=finished,
        extra={
            "changed_tensors": changed,
            "excluded_tensors": skipped_keys,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
