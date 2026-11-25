"""MergeAlign wrapper via official hammoudhasan/MergeAlign (EMNLP 2024 Findings).

Paper verified from README:
  Hammoud et al., "Model Merging and Safety Alignment: One Bad Model Spoils
  the Bunch", Findings of EMNLP 2024 (arXiv:2406.14563).

Official implementation uses data-aware merging via `LM_Cocktail`
(`mix_models_with_data`) with alignment + task examples (or evolutionary merge
via `mergekit-evolve`). This wrapper invokes `LM_Cocktail.mix_models_with_data`
directly with the requested models and dataset, preserving the exact official
algorithm while exposing SafeMergeBench's standard CLI & metadata contract.

Usage:
  python safemerge/external/run_mergealign.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --inputs MergeBench/Llama-3.1-8B-Instruct_safety MergeBench/Llama-3.1-8B-Instruct_math \
    --out merged_models/.../MergeAlign_... \
    --dataset-path data/mergealign_examples.jsonl --temp 1.0 --max-per-task 20 --seed 42
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ensure_tokenizer_saved,
    git_commit,
    resolve_external_repo,
    set_seed,
    write_merge_metadata,
)


def load_example_data(
    dataset_path: Optional[str],
    max_per_task: int = 20,
    include_all_types: bool = True,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load alignment + task examples for LM_Cocktail `mix_models_with_data`.

    Supports:
      - Hugging Face `datasets.load_from_disk` directory (official MergeAlign format)
      - JSON / JSONL with `prompt`/`input` and `response`/`output` (+ optional `type`)
      - CSV with `prompt` and `response` columns (e.g. sst_v2 response_dataframe.csv)
    """
    if not dataset_path:
        raise ValueError("--dataset-path is required for MergeAlign (LM_Cocktail data-aware merging).")

    records: List[Dict[str, Any]] = []
    if os.path.isdir(dataset_path):
        import datasets
        ds = datasets.load_from_disk(dataset_path)
        ds = ds.shuffle(seed=seed)
        records = [dict(x) for x in ds]
    elif dataset_path.endswith(".csv"):
        with open(dataset_path, "r", encoding="utf-8", newline="") as f:
            records = list(csv.DictReader(f))
        random.Random(seed).shuffle(records)
    elif dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        random.Random(seed).shuffle(records)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data if isinstance(data, list) else data.get("examples", [])
        random.Random(seed).shuffle(records)

    examples: List[Dict[str, str]] = []
    counts_by_type: Dict[str, int] = {}
    for rec in records:
        prompt = rec.get("prompt") or rec.get("input") or rec.get("instruction") or ""
        response = rec.get("response") or rec.get("output") or ""
        if not prompt or not response:
            continue
        rec_type = str(rec.get("type", "alignment"))
        if not include_all_types and rec_type != "task":
            continue
        if counts_by_type.get(rec_type, 0) >= max_per_task:
            continue
        counts_by_type[rec_type] = counts_by_type.get(rec_type, 0) + 1
        examples.append({"input": str(prompt), "output": str(response)})
    return examples


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run MergeAlign (Hammoud et al., EMNLP 2024) via LM_Cocktail")
    ap.add_argument("--base", required=True, help="base model repo id or path")
    ap.add_argument("--inputs", nargs="+", required=True, help="models to merge (e.g. safety + domain)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--task-names", default=None)
    ap.add_argument("--dataset-path", default=None, help="path to HF dataset dir, jsonl, json, or csv")
    ap.add_argument("--temp", type=float, default=1.0, help="LM_Cocktail temperature (controls weight sharpness)")
    ap.add_argument("--max-per-task", type=int, default=20, help="max examples per type (task / alignment)")
    ap.add_argument("--task-only", action="store_true", help="omit alignment examples (ablation)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo-dir", default=None, help="path to hammoudhasan/MergeAlign checkout")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    repo_dir = resolve_external_repo("MergeAlign", args.repo_dir)
    ext_commit = git_commit(repo_dir) if os.path.isdir(repo_dir) else "unknown"

    examples = load_example_data(
        args.dataset_path,
        max_per_task=args.max_per_task,
        include_all_types=not args.task_only,
        seed=args.seed,
    )
    print(f"MergeAlign: loaded {len(examples)} examples from {args.dataset_path}, temp={args.temp}")
    if args.dry_run:
        return 0

    from LM_Cocktail import mix_models_with_data

    os.makedirs(args.out, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    mix_models_with_data(
        model_names_or_paths=list(args.inputs),
        model_type="decoder",
        example_data=examples,
        temperature=args.temp,
        output_path=args.out,
    )
    finished = datetime.now(timezone.utc).isoformat()

    ensure_tokenizer_saved(args.inputs[0], args.out)
    write_merge_metadata(
        out_dir=args.out,
        algo="MergeAlign",
        kwargs={
            "temp": args.temp,
            "max_per_task": args.max_per_task,
            "all_types": not args.task_only,
            "dataset_path": args.dataset_path,
            "num_examples": len(examples),
        },
        base_model=args.base,
        inputs=args.inputs,
        task_names=args.task_names,
        seed=args.seed,
        dtype="bfloat16",
        started=started,
        finished=finished,
        external_repo="hammoudhasan/MergeAlign",
        external_commit=ext_commit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
