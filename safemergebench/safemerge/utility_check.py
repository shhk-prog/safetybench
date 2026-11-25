"""Utility evaluation validity check (Section 4.5, P6).

Compares measured single-expert utility scores (from lm-eval and bigcode-eval
outputs) against official/reported scores in `safemerge/data/mergebench_reported.yaml`.
Flags any discrepancy exceeding `--threshold` (default 5.0 percentage points)
as a warning so tables can annotate merged models built from affected experts.

Usage:
  python safemerge/utility_check.py \
    --measured results/experts/ \
    --reported safemerge/data/mergebench_reported.yaml \
    --out results/metrics/utility_check.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_reported_yaml(path: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    p = path or os.path.join(DATA_DIR, "mergebench_reported.yaml")
    try:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("models", data)
    except ImportError:
        # Minimal parser when PyYAML is absent (JSON file also accepted)
        with open(p, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text.startswith("{"):
            data = json.loads(text)
            return data.get("models", data)
        raise RuntimeError("PyYAML is required to parse mergebench_reported.yaml (or pass a .json file)")


def _to_percent(val: float) -> float:
    """Normalise ratios in [0, 1] to percentage points [0, 100]."""
    return val * 100.0 if 0.0 <= val <= 1.0 else val


def extract_utility_scores(eval_dir: str) -> Dict[str, float]:
    """Parse lm-eval and bigcode-eval outputs inside `eval_dir` into percentage scores."""
    scores: Dict[str, float] = {}

    # 1. lm-eval outputs (lm_eval.json or results*.json recursively)
    lm_files = sorted(glob.glob(os.path.join(eval_dir, "**", "results*.json"), recursive=True))
    lm_single = os.path.join(eval_dir, "lm_eval.json")
    if os.path.exists(lm_single) and lm_single not in lm_files:
        lm_files.append(lm_single)

    multilingual_vals: List[float] = []
    for fp in lm_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        results = data.get("results", {})
        if not isinstance(results, dict):
            continue
        for task_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            val = None
            for k in (
                "exact_match,flexible-extract",
                "exact_match,strict-match",
                "prompt_level_strict_acc,none",
                "inst_level_strict_acc,none",
                "acc_norm,none",
                "acc,none",
                "acc",
                "exact_match",
            ):
                if k in metrics and isinstance(metrics[k], (int, float)):
                    val = _to_percent(float(metrics[k]))
                    break
            if val is not None:
                scores[task_name] = round(val, 4)
                if any(task_name.endswith(s) for s in ("_fr", "_es", "_de", "_ru")):
                    multilingual_vals.append(val)

    if multilingual_vals:
        scores["multilingual_avg"] = round(sum(multilingual_vals) / len(multilingual_vals), 4)

    # 2. bigcode-eval output (code_eval.json)
    code_file = os.path.join(eval_dir, "code_eval.json")
    if os.path.exists(code_file):
        try:
            with open(code_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            for task_name, metrics in cdata.items():
                if isinstance(metrics, dict) and "pass@1" in metrics:
                    scores[task_name] = round(_to_percent(float(metrics["pass@1"])), 4)
        except Exception:
            pass

    return scores


def check_experts(
    measured_by_model: Dict[str, Dict[str, float]],
    reported_by_model: Dict[str, Dict[str, Dict[str, Any]]],
    threshold: float = 5.0,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    flagged_models: List[str] = []

    for model_name, measured_tasks in sorted(measured_by_model.items()):
        rep_tasks = reported_by_model.get(model_name, {})
        # Also allow matching by basename if full repo id wasn't used
        if not rep_tasks:
            for rk, rv in reported_by_model.items():
                if rk.split("/")[-1] == model_name.split("/")[-1] or rk.replace("/", "_") == model_name:
                    rep_tasks = rv
                    break

        for task, m_val in sorted(measured_tasks.items()):
            rep_entry = rep_tasks.get(task)
            rep_val = rep_entry.get("score") if isinstance(rep_entry, dict) else rep_entry
            source = rep_entry.get("source", "") if isinstance(rep_entry, dict) else ""
            if rep_val is None:
                diff = None
                warning = False
                status = "no_reported_reference"
            else:
                diff = round(float(m_val) - float(rep_val), 4)
                warning = abs(diff) > threshold
                status = "WARNING_DIFF_EXCEEDS_THRESHOLD" if warning else "ok"
                if warning and model_name not in flagged_models:
                    flagged_models.append(model_name)

            rows.append({
                "model": model_name,
                "task": task,
                "measured": round(float(m_val), 4),
                "reported": rep_val,
                "diff": diff,
                "threshold": threshold,
                "warning": warning,
                "status": status,
                "source": source,
            })

    return {
        "threshold_points": threshold,
        "total_checks": len(rows),
        "warnings_count": sum(1 for r in rows if r["warning"]),
        "flagged_models": flagged_models,
        "checks": rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check single-expert utility scores against reported values (P6)")
    ap.add_argument("--measured", required=True,
                    help="JSON mapping {model: {task: score}}, or root directory with per-model eval subdirs")
    ap.add_argument("--reported", default=os.path.join(DATA_DIR, "mergebench_reported.yaml"))
    ap.add_argument("--threshold", type=float, default=5.0, help="warning threshold in percentage points")
    ap.add_argument("--out", default="results/metrics/utility_check.json")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    reported = load_reported_yaml(args.reported)

    if os.path.isfile(args.measured):
        with open(args.measured, "r", encoding="utf-8") as f:
            measured = json.load(f)
    else:
        measured = {}
        for entry in sorted(os.listdir(args.measured)):
            sub = os.path.join(args.measured, entry)
            if os.path.isdir(sub):
                sc = extract_utility_scores(sub)
                if sc:
                    measured[entry] = sc

    report = check_experts(measured, reported, threshold=args.threshold)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    csv_path = os.path.splitext(args.out)[0] + ".csv"
    if report["checks"]:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(report["checks"][0].keys()))
            w.writeheader()
            w.writerows(report["checks"])

    print(json.dumps({
        "total_checks": report["total_checks"],
        "warnings_count": report["warnings_count"],
        "flagged_models": report["flagged_models"],
        "out": args.out,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
