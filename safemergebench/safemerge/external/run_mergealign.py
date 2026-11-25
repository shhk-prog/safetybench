"""Hammoud et al. (EMNLP 2024 Findings) LM-Cocktail wrapper via hammoudhasan/MergeAlign.

Note on method identity (Item 5):
  - `hammoudhasan/MergeAlign` implements Hammoud et al., "Model Merging and Safety
    Alignment: One Bad Model Spoils the Bunch" (Findings of EMNLP 2024,
    arXiv:2406.14563), which is data-aware merging via `LM_Cocktail`
    (`mix_models_with_data`). It is distinct from Thakkar et al. (ACL 2025).
  - Run at the official default single point (`--temp 1.0`), NOT swept on the
    alpha grid {0.2..1.0} (temperature is not comparable to task-vector scaling
    coef alpha; sweeping them on the same axis would commit P3).
  - Calibration data for the standard track is extracted from
    `MergeBench/{safety,math,coding,multilingual,instruction}_val` (disjoint from
    evaluation benchmarks).

Usage:
  python safemerge/external/run_mergealign.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --inputs MergeBench/Llama-3.1-8B-Instruct_safety MergeBench/Llama-3.1-8B-Instruct_math \
    --mergebench-domains safety math \
    --out merged_models/.../LMCocktail_... --temp 1.0 --max-per-task 20 --seed 42
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

# Field mappings matching MergeBench/merging/taskloader.py
MERGEBENCH_VAL_FIELDS: Dict[str, tuple[str, str]] = {
    "safety": ("prompt", "response"),
    "math": ("query", "response"),
    "coding": ("instruction", "response"),
    "multilingual": ("inputs", "targets"),
    "instruction": ("instruction", "output"),
}


def load_mergebench_val_examples(
    domains: List[str],
    max_per_task: int = 20,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Extract calibration examples from `MergeBench/{domain}_val` (disjoint from eval)."""
    from datasets import load_dataset

    examples: List[Dict[str, str]] = []
    for dom in domains:
        in_key, out_key = MERGEBENCH_VAL_FIELDS.get(dom, ("prompt", "response"))
        ds = load_dataset(f"MergeBench/{dom}_val", split="train").shuffle(seed=seed)
        taken = 0
        for row in ds:
            inp = row.get(in_key) or row.get("prompt") or row.get("instruction") or row.get("query") or ""
            out = row.get(out_key) or row.get("response") or row.get("output") or row.get("targets") or ""
            if not inp or not out:
                continue
            examples.append({"input": str(inp), "output": str(out)})
            taken += 1
            if taken >= max_per_task:
                break
    return examples


def load_example_data(
    dataset_path: Optional[str],
    max_per_task: int = 20,
    include_all_types: bool = True,
    seed: int = 42,
    mergebench_domains: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Load alignment + task examples for LM_Cocktail `mix_models_with_data`."""
    if mergebench_domains and not dataset_path:
        return load_mergebench_val_examples(mergebench_domains, max_per_task=max_per_task, seed=seed)
    if not dataset_path:
        raise ValueError("Provide either --mergebench-domains or --dataset-path for Hammoud et al. (LM-Cocktail).")

    records: List[Dict[str, Any]] = []
    if os.path.isdir(dataset_path):
        import datasets
        ds = datasets.load_from_disk(dataset_path).shuffle(seed=seed)
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
    ap = argparse.ArgumentParser(description="Run Hammoud et al. (EMNLP 2024) LM-Cocktail via MergeAlign repo")
    ap.add_argument("--base", required=True, help="base model repo id or path")
    ap.add_argument("--inputs", nargs="+", required=True, help="models to merge (e.g. safety + domain)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--task-names", default=None)
    ap.add_argument("--dataset-path", default=None, help="path to HF dataset dir, jsonl, json, or csv")
    ap.add_argument("--mergebench-domains", nargs="+", default=None,
                    help="extract calibration examples from MergeBench/{domain}_val (e.g. safety math)")
    ap.add_argument("--temp", type=float, default=1.0, help="LM_Cocktail default temperature (1.0)")
    ap.add_argument("--max-per-task", type=int, default=20, help="max examples per domain/type")
    ap.add_argument("--task-only", action="store_true", help="omit alignment examples (ablation)")
    ap.add_argument("--record-as", default="LMCocktail", help="algorithm name written to merge_metadata.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo-dir", default=None, help="path to hammoudhasan/MergeAlign checkout")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    repo_dir = resolve_external_repo("MergeAlign", args.repo_dir)
    ext_commit = git_commit(repo_dir) if os.path.isdir(repo_dir) else "unknown"

    if args.dry_run:
        src_desc = f"MergeBench/{args.mergebench_domains}_val" if args.mergebench_domains else args.dataset_path
        print(f"LMCocktail (Hammoud et al. 2024): inputs={args.inputs}, cal={src_desc}, temp={args.temp}")
        return 0

    examples = load_example_data(
        args.dataset_path,
        max_per_task=args.max_per_task,
        include_all_types=not args.task_only,
        seed=args.seed,
        mergebench_domains=args.mergebench_domains,
    )
    print(f"LMCocktail: loaded {len(examples)} examples, temp={args.temp}")

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
        algo=args.record_as,
        kwargs={
            "temp": args.temp,
            "max_per_task": args.max_per_task,
            "all_types": not args.task_only,
            "dataset_path": args.dataset_path,
            "mergebench_domains": args.mergebench_domains,
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
        extra={"paper": "Hammoud et al., Findings of EMNLP 2024 (arXiv:2406.14563)"},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
