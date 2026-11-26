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


def write_infeasible_metadata(
    out_dir: str,
    algo: str,
    base_model: str,
    inputs: List[str],
    reason: str,
    kwargs: Optional[Dict[str, Any]] = None,
    task_names: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Record an infeasible merge attempt (e.g., vocab_size mismatch in LM-Cocktail, Section 4.4)."""
    return write_merge_metadata(
        out_dir=out_dir,
        algo=algo,
        kwargs=kwargs or {},
        base_model=base_model,
        inputs=inputs,
        task_names=task_names,
        seed=seed,
        extra={
            "status": "infeasible",
            "infeasible_reason": reason,
        },
    )


def record_safety_injection_norm(
    merged_dir: str,
    no_safety_ref: str,
    ta_same_alpha_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute float32 L2 norm ||theta_merged - theta_no_safety|| and ratio vs same-alpha TA (Section 4.1),
    and persist into `merged_dir/merge_metadata.json`.
    """
    import math
    import sys
    import torch

    safemerge_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if safemerge_dir not in sys.path:
        sys.path.insert(0, safemerge_dir)
    from integrity_gate import _TensorStore  # noqa: E402

    m_store = _TensorStore(merged_dir)
    r_store = _TensorStore(no_safety_ref)
    try:
        diff_sq = 0.0
        ref_sq = 0.0
        n_tensors = 0
        for k in m_store.keys():
            if not r_store.has(k):
                continue
            mv = m_store.get_f32(k)
            rv = r_store.get_f32(k)
            if mv.shape != rv.shape:
                continue
            d = mv - rv
            diff_sq += float(torch.dot(d.flatten(), d.flatten()).item())
            ref_sq += float(torch.dot(rv.flatten(), rv.flatten()).item())
            n_tensors += 1
        norm = math.sqrt(diff_sq)
        rel_norm = (norm / math.sqrt(ref_sq)) if ref_sq > 0 else None
        ratio = (norm / ta_same_alpha_norm) if (ta_same_alpha_norm and ta_same_alpha_norm > 0) else None
    finally:
        m_store.close()
        r_store.close()

    info = {
        "no_safety_reference": no_safety_ref,
        "safety_update_l2_norm": round(norm, 6),
        "safety_update_rel_norm": round(rel_norm, 6) if rel_norm is not None else None,
        "ta_same_alpha_l2_norm": round(ta_same_alpha_norm, 6) if ta_same_alpha_norm is not None else None,
        "ratio_to_same_alpha_ta": round(ratio, 6) if ratio is not None else None,
        "n_tensors_compared": n_tensors,
    }
    meta_path = os.path.join(merged_dir, "merge_metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["safety_injection"] = info
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return info
