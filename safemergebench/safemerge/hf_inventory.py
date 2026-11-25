"""Hugging Face model inventory for SafeMergeBench (Task A, P4/P6).

Fetches ONLY lightweight metadata files (`config.json`, `tokenizer_config.json`,
`generation_config.json`, `README.md`) via `huggingface_hub.hf_hub_download`
without downloading model weights, for the 8 base models and 40 MergeBench
domain experts (8 bases x 5 domains).

Usage:
  python safemerge/hf_inventory.py --out results/inventory/hf_inventory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrity_gate import CONFIG_KEYS, template_family  # noqa: E402

BASE_MODELS: List[Tuple[str, str]] = [
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
    ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ("gemma-2-2b", "google/gemma-2-2b"),
    ("gemma-2-2b-it", "google/gemma-2-2b-it"),
    ("gemma-2-9b", "google/gemma-2-9b"),
    ("gemma-2-9b-it", "google/gemma-2-9b-it"),
]

DOMAINS: List[str] = ["instruction", "math", "coding", "safety", "multilingual"]

CARD_KEYWORDS: List[str] = [
    "gemma",
    "llama3",
    "GRPO",
    "WildGuardMix",
    "WildJailbreak",
    "axolotl",
]

DIFF_KEYS: List[str] = [
    "model_type",
    "rope_theta",
    "rope_scaling",
    "vocab_size",
    "tie_word_embeddings",
    "torch_dtype",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "eos_token_id",
    "chat_template_family",
]

CSV_FIELDS: List[str] = [
    "repo",
    "base",
    "domain",
    "model_type",
    "rope_theta",
    "rope_scaling",
    "vocab_size",
    "tie_word_embeddings",
    "torch_dtype",
    "chat_template_family",
    "eos_token_id",
    "differs_from_base",
    "card_mentions",
    "fetch_status",
]


def build_repo_list(
    bases: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    include_bases: bool = True,
) -> List[Tuple[str, str, str]]:
    """Return list of (repo_id, base_short_name, domain) tuples."""
    selected_bases = BASE_MODELS
    if bases:
        base_set = set(bases)
        selected_bases = [(short, repo) for short, repo in BASE_MODELS if short in base_set or repo in base_set]
    selected_domains = domains or DOMAINS

    items: List[Tuple[str, str, str]] = []
    for short, base_repo in selected_bases:
        if include_bases:
            items.append((base_repo, short, "base"))
        for dom in selected_domains:
            items.append((f"MergeBench/{short}_{dom}", short, dom))
    return items


def default_fetcher(repo_id: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a single metadata file from local dir or Hugging Face Hub.

    Returns (text_content, error_message).
    """
    if os.path.isdir(repo_id):
        path = os.path.join(repo_id, filename)
        if not os.path.exists(path):
            return None, None
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), None

    from huggingface_hub import hf_hub_download

    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:
        msg = str(exc)
        # Missing optional files (e.g. README.md or generation_config.json) are distinct from gated/401/403
        if "404" in msg or "EntryNotFoundError" in type(exc).__name__ or "LocalEntryNotFoundError" in type(exc).__name__:
            return None, None
        return None, f"{type(exc).__name__}: {msg.splitlines()[0][:180]}"

    with open(local_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), None


def _parse_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_card_mentions(readme_text: Optional[str], keywords: Iterable[str] = CARD_KEYWORDS) -> List[str]:
    if not readme_text:
        return []
    found: List[str] = []
    for kw in keywords:
        if kw.lower() == "llama3":
            if re.search(r"llama[-_\s]?3", readme_text, flags=re.IGNORECASE):
                found.append(kw)
        else:
            if re.search(re.escape(kw), readme_text, flags=re.IGNORECASE):
                found.append(kw)
    return found


def _canonical(val: Any) -> Any:
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True)
    return val


def _format_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val, sort_keys=True)
    return str(val)


def inspect_repo(
    repo_id: str,
    base_short: str,
    domain: str,
    fetcher: Callable[[str, str], Tuple[Optional[str], Optional[str]]] = default_fetcher,
) -> Dict[str, Any]:
    errors: List[str] = []
    raw: Dict[str, Optional[str]] = {}
    for fname in ("config.json", "tokenizer_config.json", "generation_config.json", "README.md"):
        content, err = fetcher(repo_id, fname)
        raw[fname] = content
        if err:
            errors.append(f"{fname}:{err}")

    cfg = _parse_json(raw["config.json"]) or {}
    tok_cfg = _parse_json(raw["tokenizer_config.json"]) or {}
    gen_cfg = _parse_json(raw["generation_config.json"]) or {}

    tpl_fam = template_family(tok_cfg) if tok_cfg else None
    eos = gen_cfg.get("eos_token_id")
    if eos is None:
        eos = cfg.get("eos_token_id")

    mentions = extract_card_mentions(raw["README.md"])

    status = "ok"
    if errors:
        status = "; ".join(errors)
    elif not cfg:
        status = "missing_config"

    row: Dict[str, Any] = {
        "repo": repo_id,
        "base": base_short,
        "domain": domain,
        "model_type": cfg.get("model_type"),
        "rope_theta": cfg.get("rope_theta"),
        "rope_scaling": cfg.get("rope_scaling"),
        "vocab_size": cfg.get("vocab_size"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
        "torch_dtype": cfg.get("torch_dtype"),
        "chat_template_family": tpl_fam,
        "eos_token_id": eos,
        "differs_from_base": "",
        "card_mentions": ";".join(mentions),
        "fetch_status": status,
        "_extra_cfg": {k: cfg.get(k) for k in CONFIG_KEYS},
    }
    return row


def compute_diffs_against_base(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_by_short: Dict[str, Dict[str, Any]] = {
        r["base"]: r for r in rows if r["domain"] == "base"
    }
    for r in rows:
        if r["domain"] == "base":
            r["differs_from_base"] = ""
            continue
        ref = base_by_short.get(r["base"])
        if not ref or ref["fetch_status"] != "ok" or r["fetch_status"] != "ok":
            r["differs_from_base"] = "BASE_OR_MODEL_UNAVAILABLE" if (not ref or ref["fetch_status"] != "ok" or r["fetch_status"] != "ok") else ""
            continue

        diffs: List[str] = []
        for k in DIFF_KEYS:
            if k in r:
                v_cur, v_ref = r.get(k), ref.get(k)
            else:
                v_cur = r.get("_extra_cfg", {}).get(k)
                v_ref = ref.get("_extra_cfg", {}).get(k)
            if _canonical(v_cur) != _canonical(v_ref):
                diffs.append(k)
        r["differs_from_base"] = ";".join(diffs)
    return rows


def run_inventory(
    out_csv: str,
    bases: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    include_bases: bool = True,
    fetcher: Callable[[str, str], Tuple[Optional[str], Optional[str]]] = default_fetcher,
) -> Dict[str, Any]:
    targets = build_repo_list(bases=bases, domains=domains, include_bases=include_bases)
    rows: List[Dict[str, Any]] = []
    for repo_id, base_short, domain in targets:
        rows.append(inspect_repo(repo_id, base_short, domain, fetcher=fetcher))
    compute_diffs_against_base(rows)

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _format_cell(r.get(k)) for k in CSV_FIELDS})

    expert_rows = [r for r in rows if r["domain"] != "base"]
    diff_rows = [
        {
            "repo": r["repo"],
            "base": r["base"],
            "domain": r["domain"],
            "differs_from_base": r["differs_from_base"],
            "fetch_status": r["fetch_status"],
        }
        for r in expert_rows
        if r["differs_from_base"]
    ]
    non_llama3_rows = [
        {
            "repo": r["repo"],
            "base": r["base"],
            "domain": r["domain"],
            "chat_template_family": r["chat_template_family"],
            "fetch_status": r["fetch_status"],
        }
        for r in rows
        if r["chat_template_family"] != "llama3"
    ]
    failed_rows = [
        {"repo": r["repo"], "fetch_status": r["fetch_status"]}
        for r in rows
        if r["fetch_status"] != "ok"
    ]

    summary = {
        "total_rows": len(rows),
        "base_rows": len(rows) - len(expert_rows),
        "expert_rows": len(expert_rows),
        "failed_count": len(failed_rows),
        "failed_repos": failed_rows,
        "differs_from_base_count": len(diff_rows),
        "differs_from_base_rows": diff_rows,
        "non_llama3_template_count": len(non_llama3_rows),
        "non_llama3_template_rows": non_llama3_rows,
    }
    summary_path = os.path.splitext(out_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="SafeMergeBench HF metadata inventory (Task A)")
    ap.add_argument("--out", default="results/inventory/hf_inventory.csv")
    ap.add_argument("--bases", nargs="*", default=None, help="subset of base short names or repo ids")
    ap.add_argument("--domains", nargs="*", default=None, help="subset of domains")
    ap.add_argument("--no-bases", action="store_true", help="omit base model rows (not recommended)")
    args = ap.parse_args(argv)

    summary = run_inventory(
        out_csv=args.out,
        bases=args.bases,
        domains=args.domains,
        include_bases=not args.no_bases,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
