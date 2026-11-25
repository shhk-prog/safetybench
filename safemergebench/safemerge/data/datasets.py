"""Dataset loaders for SafeMergeBench extra safety benchmarks (Section 4.2).

Supported tasks:
  - jailbreakbench            : JailbreakBench/JBB-Behaviors (100 harmful goals)
  - strongreject              : StrongREJECT full forbidden prompts
  - wildjailbreak             : allenai/wildjailbreak eval split (harmful),
                                de-duplicated against tuning prompts via split_overlap
  - trustllm_jailbreaktrigger : allenai/tulu-3-trustllm-jailbreaktrigger-eval,
                                optionally de-duplicated against real-world safety SFT prompts

Every loader returns `(records, manifest)` where each record has at least
`{"id": str, "prompt": str}` and `manifest` records counts, removed overlap
counts, and SHA-256 hash of the prompt list.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import split_overlap as so  # noqa: E402

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TASKS = [
    "jailbreakbench",
    "strongreject",
    "wildjailbreak",
    "trustllm_jailbreaktrigger",
]


def prompts_sha256(prompts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def deduplicate_records(
    records: List[Dict[str, Any]],
    exclude_prompts: List[str],
    thr: float = 0.8,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Remove records matching `exclude_prompts` (exact normalized or 5-gram Jaccard >= thr)."""
    if not exclude_prompts:
        return list(records), {"raw": len(records), "exact_removed": 0, "near_removed": 0, "kept": len(records)}

    ex_exact = {so.norm(t) for t in exclude_prompts if t}
    ex_shingles = [so.shingles(t) for t in exclude_prompts if t]
    inv_index: Dict[str, set] = {}
    for idx, sh in enumerate(ex_shingles):
        for g in sh:
            inv_index.setdefault(g, set()).add(idx)

    kept: List[Dict[str, Any]] = []
    exact_rm = 0
    near_rm = 0
    for rec in records:
        p = rec.get("prompt", "")
        if so.norm(p) in ex_exact:
            exact_rm += 1
            continue
        sh = so.shingles(p)
        cands = set().union(*(inv_index.get(g, set()) for g in sh)) if sh else set()
        if any(len(sh & ex_shingles[c]) / max(1, len(sh | ex_shingles[c])) >= thr for c in cands):
            near_rm += 1
            continue
        kept.append(rec)

    return kept, {
        "raw": len(records),
        "exact_removed": exact_rm,
        "near_removed": near_rm,
        "kept": len(kept),
    }


def load_prompts_from_file(path: str, field: Optional[str] = None) -> List[str]:
    """Read raw prompt strings from json, jsonl, or csv."""
    if path.endswith(".csv"):
        prompts: List[str] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                keys = [field] if field else so.FIELDS
                val = next((row[k] for k in keys if k in row and row[k]), None)
                if val:
                    prompts.append(val)
        return prompts
    return [t for t, _ in so.load(path, field=field)]


def load_jailbreakbench(limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    recs: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        recs.append({
            "id": str(row.get("Index", i)),
            "prompt": str(row["Goal"]),
            "category": row.get("Category"),
            "behavior": row.get("Behavior"),
        })
    if limit:
        recs = recs[:limit]
    manifest = {
        "task": "jailbreakbench",
        "source": "JailbreakBench/JBB-Behaviors (behaviors/harmful)",
        "count": len(recs),
        "sha256": prompts_sha256(r["prompt"] for r in recs),
    }
    return recs, manifest


def load_strongreject(limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset
    source = None
    ds = None
    for repo_id, split in [
        ("AlignmentResearch/StrongREJECT", "validation"),
        ("walledai/StrongREJECT", "train"),
        ("AlignmentResearch/StrongREJECT", "train"),
    ]:
        try:
            ds = load_dataset(repo_id, split=split)
            if len(ds) > 0:
                source = f"{repo_id} ({split})"
                break
            ds = None
        except Exception:
            continue
    if ds is None:
        local_fallback = "/mnt/nas/home/hiromi/src/sst_v2/v3/data/eval/eval_harmful_strongreject.json"
        if os.path.exists(local_fallback):
            with open(local_fallback, "r", encoding="utf-8") as f:
                ds = json.load(f)
            source = f"local:{local_fallback}"
        else:
            from strong_reject.load_datasets import load_strongreject as _sr_load
            ds = _sr_load()
            source = "strong_reject.load_datasets.load_strongreject()"

    recs: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt") or row.get("forbidden_prompt") or ""
        if not prompt and row.get("content"):
            c = row["content"]
            prompt = c[0] if isinstance(c, list) and c else str(c)
        if not prompt:
            continue
        recs.append({
            "id": str(row.get("id", i)),
            "prompt": str(prompt),
            "category": row.get("category"),
        })
    if limit:
        recs = recs[:limit]
    manifest = {
        "task": "strongreject",
        "source": source,
        "count": len(recs),
        "sha256": prompts_sha256(r["prompt"] for r in recs),
    }
    return recs, manifest


def load_wildjailbreak(
    tune_file: Optional[str] = None,
    overlap_thr: float = 0.8,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset
    # Prefer the official eval split ("eval" config); fall back to local eval JSON if gated
    ds = None
    source = None
    for cfg, split in [("eval", "eval"), ("eval", "train"), (None, "eval"), (None, "test")]:
        try:
            ds = load_dataset("allenai/wildjailbreak", cfg, split=split) if cfg else load_dataset("allenai/wildjailbreak", split=split)
            source = f"allenai/wildjailbreak ({cfg or 'default'}/{split})"
            break
        except Exception:
            continue
    if ds is None:
        local_fallback = "/mnt/nas/home/hiromi/src/sst_v2/v3/data/eval/eval_harmful_wildjailbreak.json"
        if os.path.exists(local_fallback):
            with open(local_fallback, "r", encoding="utf-8") as f:
                ds = json.load(f)
            source = f"local:{local_fallback}"
        else:
            raise RuntimeError("Could not load allenai/wildjailbreak eval split. Check HF access approval.")

    raw_recs: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        dtype = str(row.get("data_type", ""))
        if dtype and "harmful" not in dtype.lower():
            continue
        adv = row.get("adversarial") or ""
        van = row.get("vanilla") or ""
        prompt = adv or van or row.get("prompt") or ""
        if not prompt:
            continue
        raw_recs.append({
            "id": str(row.get("id", i)),
            "prompt": str(prompt),
            "vanilla": str(van) if van else str(prompt),
            "adversarial": str(adv) if adv else "",
            "data_type": dtype,
        })

    exclude_prompts = load_prompts_from_file(tune_file) if tune_file and os.path.exists(tune_file) else []
    recs, dedup_stats = deduplicate_records(raw_recs, exclude_prompts, thr=overlap_thr)
    if limit:
        recs = recs[:limit]

    manifest = {
        "task": "wildjailbreak",
        "source": source,
        "tune_exclude_file": tune_file,
        "overlap_thr": overlap_thr,
        "dedup": dedup_stats,
        "count": len(recs),
        "sha256": prompts_sha256(r["prompt"] for r in recs),
    }
    return recs, manifest


def load_trustllm_jailbreaktrigger(
    train_exclude_file: Optional[str] = None,
    overlap_thr: float = 0.8,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset
    ds = None
    source = None
    for split in ("test", "train", "eval"):
        try:
            ds = load_dataset("allenai/tulu-3-trustllm-jailbreaktrigger-eval", split=split)
            source = f"allenai/tulu-3-trustllm-jailbreaktrigger-eval ({split})"
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load allenai/tulu-3-trustllm-jailbreaktrigger-eval.")

    raw_recs: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        prompt = row.get("prompt") or row.get("query") or ""
        if not prompt and isinstance(row.get("messages"), list) and row["messages"]:
            prompt = row["messages"][0].get("content", "")
        if not prompt:
            continue
        raw_recs.append({
            "id": str(row.get("id", i)),
            "prompt": str(prompt),
            "label": row.get("label"),
        })

    exclude_prompts = (
        load_prompts_from_file(train_exclude_file)
        if train_exclude_file and os.path.exists(train_exclude_file)
        else []
    )
    recs, dedup_stats = deduplicate_records(raw_recs, exclude_prompts, thr=overlap_thr)
    if limit:
        recs = recs[:limit]

    manifest = {
        "task": "trustllm_jailbreaktrigger",
        "source": source,
        "train_exclude_file": train_exclude_file,
        "overlap_thr": overlap_thr,
        "dedup": dedup_stats,
        "count": len(recs),
        "sha256": prompts_sha256(r["prompt"] for r in recs),
    }
    return recs, manifest


def load_distribution_map(yaml_path: Optional[str] = None) -> Dict[str, Dict[str, Optional[bool]]]:
    path = yaml_path or os.path.join(DATA_DIR, "distribution.yaml")
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback parser for simple 2-level YAML
        out: Dict[str, Dict[str, Optional[bool]]] = {}
        cur_section: Optional[str] = None
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.split("#", 1)[0].rstrip()
                if not line.strip():
                    continue
                if not line.startswith(" ") and line.endswith(":"):
                    cur_section = line[:-1].strip()
                    out[cur_section] = {}
                elif cur_section and ":" in line:
                    k, v = line.strip().split(":", 1)
                    vs = v.strip().lower()
                    val: Optional[bool] = True if vs == "true" else (False if vs == "false" else None)
                    out[cur_section][k.strip()] = val
        return out


def load_task(
    task: str,
    wildjailbreak_tune_file: Optional[str] = None,
    trustllm_train_exclude: Optional[str] = None,
    overlap_thr: float = 0.8,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if task == "jailbreakbench":
        return load_jailbreakbench(limit=limit)
    if task == "strongreject":
        return load_strongreject(limit=limit)
    if task == "wildjailbreak":
        return load_wildjailbreak(tune_file=wildjailbreak_tune_file, overlap_thr=overlap_thr, limit=limit)
    if task == "trustllm_jailbreaktrigger":
        return load_trustllm_jailbreaktrigger(train_exclude_file=trustllm_train_exclude, overlap_thr=overlap_thr, limit=limit)
    raise ValueError(f"Unsupported extra safety task: {task}")
