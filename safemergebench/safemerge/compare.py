"""Safety-utility curve comparison and rank-sensitivity analysis (Task F, P3).

Reads `results/metrics/all.csv` and utility scores, builds safety (`cond_ASR`)
vs utility curves per method across merge coefficients, excludes points with
`VRR < --vrr-floor` (default 0.9, plotted in grey), and computes:
  (a) min_cond_asr_at_90pct_utility: minimum cond_ASR among points with
      VRR >= 0.9 and utility >= 0.9 * expert_utility
  (b) normalized_auc: area under the (utility, 1 - cond_ASR) curve normalized
      by expert_utility
  (c) Judge rank stability (pairwise Kendall's tau across judges)
  (d) Single-alpha ranking vs full-curve ranking side-by-side (P3 evidence)
  (e) Publication-ready PDF figures via matplotlib.

Usage:
  python safemerge/compare.py \
    --metrics results/metrics/all.csv \
    --utility results/metrics/utility.csv \
    --expert-utility 75.0 \
    --out-dir results/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def parse_method_alpha(model_name: str) -> Tuple[str, Optional[float]]:
    """Infer (method, alpha) from a standard/realworld merged model directory name."""
    method = model_name.split("_")[0]
    for pat in (
        r"scaling_coef_([0-9]+(?:\.[0-9]+)?)",
        r"alpha_([0-9]+(?:\.[0-9]+)?)",
        r"temp_([0-9]+(?:\.[0-9]+)?)",
    ):
        m = re.search(pat, model_name)
        if m:
            return method, float(m.group(1))
    return method, None


def kendall_tau(rank_a: Dict[str, float], rank_b: Dict[str, float]) -> Optional[float]:
    """Compute Kendall's tau-b between two method->score dicts (lower or higher consistent)."""
    keys = sorted(set(rank_a.keys()) & set(rank_b.keys()))
    n = len(keys)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = keys[i], keys[j]
            da = rank_a[ki] - rank_a[kj]
            db = rank_b[ki] - rank_b[kj]
            if da == 0 and db == 0:
                continue
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    denom = ((concordant + discordant + ties_a) * (concordant + discordant + ties_b)) ** 0.5
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def normalized_curve_auc(points: List[Tuple[float, float]], max_utility: float) -> Optional[float]:
    """Normalized area under (utility, safety = 1 - cond_ASR) curve in [0, 1].

    Each point is (utility, cond_ASR) with cond_ASR in [0, 1].
    """
    if not points or max_utility <= 0:
        return None
    # Group by utility: take best (lowest) cond_ASR at each utility level
    by_u: Dict[float, float] = {}
    for u, asr in points:
        u_clamped = max(0.0, min(max_utility, float(u)))
        safety = max(0.0, min(1.0, 1.0 - float(asr)))
        if u_clamped not in by_u or safety > by_u[u_clamped]:
            by_u[u_clamped] = safety

    xs = sorted(by_u.keys())
    if len(xs) == 1:
        # Single-point method: rectangle area from 0 to x
        return round((xs[0] * by_u[xs[0]]) / max_utility, 6)

    # Include origin (0, safety_at_first_x) so curves are compared on common [0, max_u] support
    area = xs[0] * by_u[xs[0]]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += 0.5 * dx * (by_u[xs[i - 1]] + by_u[xs[i]])
    return round(area / max_utility, 6)


def compute_method_summaries(
    rows: List[Dict[str, Any]],
    utility_by_model: Dict[str, float],
    expert_utility: float,
    vrr_floor: float = 0.9,
    utility_retention: float = 0.9,
    fixed_alpha: float = 0.6,
) -> Dict[str, Any]:
    """Aggregate per (judge, method) metrics and rankings."""
    # Group by judge -> method -> list of points (averaged across datasets per model)
    by_judge_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        judge = r["judge"]
        model = r["model"]
        vrr = r.get("VRR")
        cond_asr = r.get("cond_ASR")
        if vrr is None or cond_asr is None:
            continue
        rec = by_judge_model.setdefault(judge, {}).setdefault(model, {
            "model": model,
            "method": r.get("method") or parse_method_alpha(model)[0],
            "alpha": r.get("alpha") if r.get("alpha") is not None else parse_method_alpha(model)[1],
            "vrrs": [],
            "cond_asrs": [],
        })
        rec["vrrs"].append(float(vrr))
        rec["cond_asrs"].append(float(cond_asr))

    util_thr = utility_retention * expert_utility
    results_by_judge: Dict[str, Any] = {}

    for judge, model_map in sorted(by_judge_model.items()):
        by_method: Dict[str, List[Dict[str, Any]]] = {}
        for model, info in model_map.items():
            if model not in utility_by_model:
                continue
            pt = {
                "model": model,
                "method": info["method"],
                "alpha": info["alpha"],
                "VRR": sum(info["vrrs"]) / len(info["vrrs"]),
                "cond_ASR": sum(info["cond_asrs"]) / len(info["cond_asrs"]),
                "utility": float(utility_by_model[model]),
            }
            pt["valid_point"] = pt["VRR"] >= vrr_floor
            by_method.setdefault(info["method"], []).append(pt)

        method_stats: Dict[str, Dict[str, Any]] = {}
        for method, pts in sorted(by_method.items()):
            valid_pts = [p for p in pts if p["valid_point"]]
            qualifying = [p for p in valid_pts if p["utility"] >= util_thr]
            min_asr = min((p["cond_ASR"] for p in qualifying), default=None)
            auc = normalized_curve_auc([(p["utility"], p["cond_ASR"]) for p in valid_pts], max_utility=expert_utility)
            fixed_pts = [p for p in valid_pts if p["alpha"] is not None and abs(p["alpha"] - fixed_alpha) < 1e-6]
            fixed_asr = fixed_pts[0]["cond_ASR"] if fixed_pts else None

            method_stats[method] = {
                "n_points": len(pts),
                "n_valid_points": len(valid_pts),
                "min_cond_asr_at_90pct_utility": round(min_asr, 6) if min_asr is not None else None,
                "normalized_auc": auc,
                "fixed_alpha_cond_asr": round(fixed_asr, 6) if fixed_asr is not None else None,
                "points": sorted(pts, key=lambda x: (-1.0 if x["alpha"] is None else x["alpha"])),
            }

        # Rankings (1 = best)
        def _rank_dict(key: str, lower_is_better: bool) -> Dict[str, int]:
            avail = [(m, s[key]) for m, s in method_stats.items() if s[key] is not None]
            avail.sort(key=lambda kv: kv[1] if lower_is_better else -kv[1])
            return {m: idx + 1 for idx, (m, _) in enumerate(avail)}

        rank_min_asr = _rank_dict("min_cond_asr_at_90pct_utility", lower_is_better=True)
        rank_auc = _rank_dict("normalized_auc", lower_is_better=False)
        rank_fixed = _rank_dict("fixed_alpha_cond_asr", lower_is_better=True)

        p3_tau = kendall_tau(
            {m: float(r) for m, r in rank_fixed.items()},
            {m: float(r) for m, r in rank_auc.items()},
        )

        results_by_judge[judge] = {
            "methods": method_stats,
            "rankings": {
                "by_min_cond_asr_90pct_util": rank_min_asr,
                "by_normalized_auc": rank_auc,
                f"by_fixed_alpha_{fixed_alpha}": rank_fixed,
                "kendall_tau_fixed_vs_auc": round(p3_tau, 6) if p3_tau is not None else None,
            },
        }

    # Cross-judge Kendall's tau
    judges = sorted(results_by_judge.keys())
    cross_judge_tau: Dict[str, Optional[float]] = {}
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            ja, jb = judges[i], judges[j]
            ra = {m: float(r) for m, r in results_by_judge[ja]["rankings"]["by_normalized_auc"].items()}
            rb = {m: float(r) for m, r in results_by_judge[jb]["rankings"]["by_normalized_auc"].items()}
            tau = kendall_tau(ra, rb)
            cross_judge_tau[f"{ja}_vs_{jb}"] = round(tau, 6) if tau is not None else None

    return {
        "expert_utility": expert_utility,
        "utility_threshold": util_thr,
        "vrr_floor": vrr_floor,
        "fixed_alpha": fixed_alpha,
        "by_judge": results_by_judge,
        "cross_judge_kendall_tau_auc": cross_judge_tau,
    }


def plot_curves_pdf(summary: Dict[str, Any], out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    generated: List[str] = []
    util_thr = summary["utility_threshold"]

    for judge, jdata in summary["by_judge"].items():
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        ax.axvline(util_thr, color="#666666", linestyle="--", linewidth=1.0, label="90% expert utility")

        grey_labeled = False
        for method, mdata in sorted(jdata["methods"].items()):
            pts = mdata["points"]
            valid = sorted([p for p in pts if p["valid_point"]], key=lambda p: p["utility"])
            invalid = [p for p in pts if not p["valid_point"]]

            if valid:
                ax.plot(
                    [p["utility"] for p in valid],
                    [p["cond_ASR"] * 100.0 for p in valid],
                    marker="o",
                    linewidth=1.8,
                    label=method,
                )
            if invalid:
                ax.scatter(
                    [p["utility"] for p in invalid],
                    [p["cond_ASR"] * 100.0 for p in invalid],
                    color="#999999",
                    marker="x",
                    alpha=0.65,
                    label=(f"Excluded (VRR < {summary['vrr_floor']})" if not grey_labeled else None),
                )
                grey_labeled = True

        ax.set_xlabel("Utility Score (%)")
        ax.set_ylabel(f"Conditional ASR (%, {judge})")
        ax.set_title(f"Safety-Utility Curves ({judge})")
        ax.set_ylim(-2, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        pdf_path = os.path.join(out_dir, f"safety_utility_curve_{judge}.pdf")
        fig.savefig(pdf_path, format="pdf")
        plt.close(fig)
        generated.append(pdf_path)
    return generated


def load_metrics_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = dict(r)
            for k in ("VRR", "cond_ASR", "ASR", "VSR"):
                row[k] = float(r[k]) if r.get(k) not in (None, "") else None
            rows.append(row)
    return rows


def load_utility_table(path: str, task_col: Optional[str] = None) -> Dict[str, float]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, float] = {}
        for m, v in data.items():
            if isinstance(v, (int, float)):
                out[m] = float(v)
            elif isinstance(v, dict):
                if task_col and task_col in v:
                    out[m] = float(v[task_col])
                else:
                    vals = [float(x) for x in v.values() if isinstance(x, (int, float))]
                    if vals:
                        out[m] = sum(vals) / len(vals)
        return out
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            m = r["model"]
            val_str = r.get(task_col) if task_col else r.get("utility", r.get("score"))
            if val_str not in (None, ""):
                out[m] = float(val_str)
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare methods via safety-utility curves (Task F, P3)")
    ap.add_argument("--metrics", required=True, help="results/metrics/all.csv")
    ap.add_argument("--utility", required=True, help="utility CSV or JSON mapping model -> score")
    ap.add_argument("--utility-col", default=None, help="optional task/column name in utility file")
    ap.add_argument("--expert-utility", type=float, required=True, help="single-expert reference utility score (%)")
    ap.add_argument("--vrr-floor", type=float, default=0.9)
    ap.add_argument("--fixed-alpha", type=float, default=0.6)
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--no-plot", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_metrics_csv(args.metrics)
    util = load_utility_table(args.utility, task_col=args.utility_col)
    summary = compute_method_summaries(
        rows, util, expert_utility=args.expert_utility,
        vrr_floor=args.vrr_floor, fixed_alpha=args.fixed_alpha,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "comparison_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pdfs = [] if args.no_plot else plot_curves_pdf(summary, args.out_dir)
    print(json.dumps({"summary_json": out_json, "figures": pdfs, "cross_judge_tau": summary["cross_judge_kendall_tau_auc"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
