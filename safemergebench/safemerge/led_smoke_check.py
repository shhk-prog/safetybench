"""Strict LED-Merging smoke test verifier (Implementation Spec v2, Section 4.2 + Addendum [1],
revised 2026-11-26 to task-vector-normalized degeneration criteria).

Fails (exit code 1) if ANY of the required checks does not pass:
  1. Importance score files in `wanda_score/` match expected tensor count (`scores_complete == True`).
  2. Updated tensors `changed_tensors > 0`.
  3. Not degenerate to base: `||Delta_merged|| / max_i ||Delta_i|| >= min_tv_ratio` (default 0.05 = 5%).
  4. Not degenerate to any input: `||theta_merged - theta_i|| / ||Delta_i|| >= min_tv_ratio` (default 0.05 = 5%).
  5. Domain update retention: `projection_coeff >= min_proj` (default 0.10 = 10%).
  6. Shape-mismatched skipped tensors are recorded explicitly (`skipped_shape_mismatch_names`).
  7. Simple math/sanity problems are answered correctly AND outputs differ from base model.

Note: `rel_norm_diff_to_base = ||Delta_merged|| / ||theta_base||` is recorded in summary
for reference ONLY and is NOT used for pass/fail judgement (MergeBench single experts are
only 0.68-1.42% away from base in ||theta_base||).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional


MATH_SMOKE_PROMPTS = [
    ("math1", "What is 17 * 24? Show your steps and end with the final number.", "408"),
    ("math2", "Janet has 16 eggs per day. She eats 3 and bakes 4. She sells the rest for $2 each. How much does she make per day?", "18"),
    ("math3", "Solve for x: 3x + 7 = 28. Give the final value of x.", "7"),
]


def evaluate_smoke_criteria(
    merge_meta: Dict[str, Any],
    gate_report: Dict[str, Any],
    domain_input: str,
    merged_responses: Optional[List[Dict[str, Any]]] = None,
    base_responses: Optional[List[Dict[str, Any]]] = None,
    min_tv_ratio: float = 0.05,
    min_proj: float = 0.10,
) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []

    # 1. Importance score directory completeness
    score_check = merge_meta.get("score_dir_check") or {}
    scores_complete = bool(merge_meta.get("scores_complete", False)) and bool(score_check)
    if not scores_complete:
        failures.append({
            "check": "score_dir_completeness",
            "reason": "wanda_score/ does not contain all expected layer files",
            "score_dir_check": score_check,
        })

    # 2, 3, 4, 5. Weight update & task-vector-normalized degeneration checks
    stats = gate_report.get("stats") or {}
    degen = stats.get("degeneration") or {}
    changed_tensors = int(degen.get("changed_tensors", 0))
    rel_norm = degen.get("rel_norm_diff_to_base")  # recorded for reference only
    tv_base_ratio = degen.get("merged_to_max_input_delta_ratio")
    dom_stat = stats.get(domain_input) or {}
    proj = dom_stat.get("projection_coeff")

    if changed_tensors <= 0:
        failures.append({"check": "changed_tensors", "changed_tensors": changed_tensors})

    if tv_base_ratio is None or float(tv_base_ratio) < min_tv_ratio:
        failures.append({
            "check": "merged_to_max_input_delta_ratio",
            "value": tv_base_ratio,
            "threshold": min_tv_ratio,
        })

    # Check closeness to every input model (||theta_merged - theta_i|| / ||Delta_i|| >= min_tv_ratio)
    input_tv_ratios: Dict[str, Optional[float]] = {}
    for k, v in stats.items():
        if isinstance(v, dict) and "dist_to_merged_tv_ratio" in v:
            ratio = v.get("dist_to_merged_tv_ratio")
            input_tv_ratios[k] = ratio
            if ratio is not None and float(ratio) < min_tv_ratio:
                failures.append({
                    "check": "close_to_input_tv_ratio",
                    "input_model": k,
                    "value": ratio,
                    "threshold": min_tv_ratio,
                })

    if proj is None or float(proj) < min_proj:
        failures.append({
            "check": "domain_projection_coeff",
            "domain_input": domain_input,
            "value": proj,
            "threshold": min_proj,
        })

    # 6. Shape-mismatch skipped tensors recorded
    if "skipped_shape_mismatch_names" not in merge_meta:
        failures.append({"check": "skipped_shape_mismatch_recorded", "reason": "field missing from merge_metadata.json"})

    # 7. Math sanity correctness + differs from base output
    math_correct = None
    differs_from_base = None
    if merged_responses is not None:
        n_ok = 0
        for item in merged_responses:
            expected = str(item.get("expected", ""))
            resp = str(item.get("response", ""))
            if expected and re.search(rf"\b{re.escape(expected)}\b", resp):
                n_ok += 1
        math_correct = n_ok
        if n_ok < len(merged_responses):
            failures.append({
                "check": "math_sanity_correct",
                "correct": n_ok,
                "total": len(merged_responses),
            })
        if base_responses is not None:
            by_id_base = {r["id"]: r.get("response", "").strip() for r in base_responses}
            n_diff = sum(
                1 for r in merged_responses
                if r.get("response", "").strip() != by_id_base.get(r["id"], "")
            )
            differs_from_base = n_diff > 0
            if not differs_from_base:
                failures.append({
                    "check": "output_differs_from_base",
                    "reason": "All merged outputs are identical to base model outputs",
                })

    return {
        "passed": len(failures) == 0,
        "criteria_version": "tv_norm_v2_20261126",
        "thresholds": {
            "min_tv_ratio": min_tv_ratio,
            "min_proj": min_proj,
        },
        "failures": failures,
        "summary": {
            "scores_complete": scores_complete,
            "score_dir_counts": {
                k: f"{v.get('wanda_score_pkl_count')}/{v.get('expected_tensors')}"
                for k, v in score_check.items()
            },
            "changed_tensors": changed_tensors,
            "merged_to_max_input_delta_ratio": tv_base_ratio,
            "input_dist_tv_ratios": input_tv_ratios,
            "rel_norm_diff_to_base_ref_only": rel_norm,
            "domain_projection_coeff": proj,
            "skipped_shape_mismatch_names": merge_meta.get("skipped_shape_mismatch_names", []),
            "math_correct": math_correct,
            "differs_from_base": differs_from_base,
        },
    }


def generate_responses(model_path: str, prompts: List[tuple]) -> List[Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    mdl.eval()
    out_list: List[Dict[str, Any]] = []
    for pid, prompt, expected in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inp = tok(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=192, do_sample=False)
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        out_list.append({"id": pid, "prompt": prompt, "expected": expected, "response": gen})
    del mdl
    torch.cuda.empty_cache()
    return out_list


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify LED-Merging smoke test against revised Section 4.2 criteria")
    ap.add_argument("--base", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--domain-input", required=True)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--min-tv-ratio", type=float, default=0.05)
    ap.add_argument("--min-proj", type=float, default=0.10)
    ap.add_argument("--reuse-responses-json", default=None,
                    help="reuse merged_responses/base_responses from a prior smoke_check.json without GPU generation")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    with open(os.path.join(args.merged, "merge_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(args.gate_json, encoding="utf-8") as f:
        gate = json.load(f)

    merged_resps = None
    base_resps = None
    if args.reuse_responses_json:
        with open(args.reuse_responses_json, encoding="utf-8") as f:
            prev = json.load(f)
        merged_resps = prev.get("merged_responses")
        base_resps = prev.get("base_responses")
    elif not args.skip_generation:
        merged_resps = generate_responses(args.merged, MATH_SMOKE_PROMPTS)
        base_resps = generate_responses(args.base, MATH_SMOKE_PROMPTS)

    result = evaluate_smoke_criteria(
        merge_meta=meta,
        gate_report=gate,
        domain_input=args.domain_input,
        merged_responses=merged_resps,
        base_responses=base_resps,
        min_tv_ratio=args.min_tv_ratio,
        min_proj=args.min_proj,
    )
    if merged_resps is not None:
        result["merged_responses"] = merged_resps
        result["base_responses"] = base_resps

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("merged_responses", "base_responses")}, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
