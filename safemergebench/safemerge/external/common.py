"""Shared helpers for SafeMergeBench external merge method wrappers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import random
import subprocess
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def git_commit(repo_dir: str = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def write_merge_metadata(
    out_dir: str,
    algo: str,
    kwargs: Dict[str, Any],
    base_model: str,
    inputs: List[str],
    task_names: Optional[str] = None,
    seed: int = 42,
    dtype: str = "bfloat16",
    started: Optional[str] = None,
    finished: Optional[str] = None,
    external_repo: Optional[str] = None,
    external_commit: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    meta: Dict[str, Any] = {
        "algo": algo,
        "kwargs": kwargs,
        "base_model": base_model,
        "inputs": list(inputs),
        "task_names": task_names,
        "dtype": dtype,
        "seed": seed,
        "git_commit": git_commit(REPO_ROOT),
        "started": started or now,
        "finished": finished or now,
    }
    if external_repo:
        meta["external_repo"] = external_repo
    if external_commit:
        meta["external_commit"] = external_commit
    if extra:
        meta.update(extra)

    path = os.path.join(out_dir, "merge_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def resolve_external_repo(name: str, override: Optional[str] = None) -> str:
    """Resolve external official repository path (submodule or sibling checkout)."""
    candidates = []
    if override:
        candidates.append(override)
    candidates.extend([
        os.path.join(REPO_ROOT, "external", name),
        os.path.join(REPO_ROOT, "safemerge", "external", "repos", name),
        os.path.join(WORKSPACE_ROOT, name),
    ])
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[-1])


def ensure_tokenizer_saved(source_model: str, out_dir: str) -> bool:
    """Ensure tokenizer files exist in out_dir (copies from source_model if absent)."""
    tok_cfg = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg):
        return True
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(source_model)
        tok.save_pretrained(out_dir)
        return True
    except Exception:
        return False
