"""Evaluate bigcode-evaluation-harness saved generations (`code_generations_<task>.json`)
using bigcode's own task references and `compute_code_eval` to produce per-sample JSONL
records (`task_id`, `completion_id`, `code`, `correct`, `error`) for Addendum [3]
(`utility_gate.py check-code`).

Does NOT modify `bigcode-evaluation-harness/`.
Run inside `.venvs/bigcode/bin/python`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace


def main() -> int:
    ap = argparse.ArgumentParser(description="Score bigcode saved generations per sample for utility_gate check-code")
    ap.add_argument("--task", required=True, choices=["humanevalplus", "mbppplus"])
    ap.add_argument("--generations", required=True, help="Path to code_generations_<task>.json")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL path with per-sample pass/error")
    ap.add_argument("--summary-out", default=None, help="Optional JSON summary path with pass@k")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bigcode_dir = os.path.join(root, "bigcode-evaluation-harness")
    if bigcode_dir not in sys.path:
        sys.path.insert(0, bigcode_dir)

    from bigcode_eval import tasks
    from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

    with open(args.generations, "r", encoding="utf-8") as f:
        generations = json.load(f)

    dummy_args = SimpleNamespace(
        allow_code_execution=True,
        prompt="prompt",
        load_data_path=None,
    )
    task = tasks.get_task(args.task, dummy_args)
    dataset = task.get_dataset()
    n_tasks = len(generations)
    references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
    timeout = args.timeout if args.timeout is not None else (20.0 if args.task == "humanevalplus" else 10.0)

    pass_at_k, results = compute_code_eval(
        references=references,
        predictions=generations,
        k=[1, 10],
        num_workers=args.num_workers,
        timeout=timeout,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    n_total = 0
    n_passed = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for t_idx in range(n_tasks):
            doc = dataset[t_idx]
            tid = str(doc.get("task_id", f"{args.task}/{t_idx}"))
            per_comp = sorted(results[t_idx], key=lambda x: x[0])
            cands = generations[t_idx]
            for c_idx, res in per_comp:
                passed = bool(res.get("passed", False))
                result_str = str(res.get("result", ""))
                n_total += 1
                if passed:
                    n_passed += 1
                rec = {
                    "task_id": f"{tid}#s{c_idx}",
                    "problem_id": tid,
                    "sample_index": c_idx,
                    "code": cands[c_idx] if c_idx < len(cands) else "",
                    "correct": passed,
                    "passed": passed,
                    "error": "" if passed else result_str,
                    "result": result_str,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "task": args.task,
        "generations_path": os.path.abspath(args.generations),
        "out_jsonl": os.path.abspath(args.out_jsonl),
        "n_tasks": n_tasks,
        "n_samples_total": n_total,
        "n_samples_passed": n_passed,
        "pass_at_k": pass_at_k,
    }
    if args.summary_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_out)), exist_ok=True)
        with open(args.summary_out, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
