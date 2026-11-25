"""Import real-world track results from sst_v2 into SafeMergeBench format (Task G).

Strictly READ-ONLY on sst_v2: never modifies source files.
Converts `sst_v2` safety result JSONs (Exp1 `v3/results/vllm/debug_limit320`,
Exp2 / fp32 SST `v5/results/fp32_sst_math`) into SafeMergeBench's unified
`safety_generation.json` / `safety_generation_judged.json` schema.

Also supports:
  - Reproducing Table 3 (share of legacy-valid responses flagged as collapsed by
    the extended validity rule at alpha=0.6 across seeds 42-44)
  - Running config-level `integrity_gate.py` checks (`--weights` off) on the
    real-world track combinations to verify detection of WizardCoder's
    `rope_theta` mismatch and MedAlpaca's lineage/config mismatch.

Usage:
  python safemerge/import_sst_v2.py \
    --sst-root /mnt/nas/home/hiromi/src/sst_v2 \
    --out-dir results/realworld \
    --verify-table3 --check-gate
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integrity_gate as gate  # noqa: E402
import metrics as sm_metrics  # noqa: E402
import validity_audit as va  # noqa: E402

SAFETY_BENCHES = ("harmbench", "jailbreakbench", "strongreject", "wildjailbreak")

REALWORLD_COMBOS: Dict[str, Dict[str, Any]] = {
    "safety+math": {
        "base": "meta-llama/Llama-2-7b-hf",
        "inputs": ["WizardLMTeam/WizardMath-7B-V1.0"],
    },
    "safety+code": {
        "base": "meta-llama/Llama-2-7b-hf",
        "inputs": ["vanillaOVO/WizardCoder-Python-7B-V1.0"],
    },
    "safety+medical": {
        "base": "meta-llama/Llama-2-7b-hf",
        "inputs": ["medalpaca/medalpaca-7b"],
    },
    "safety+math+code+medical": {
        "base": "meta-llama/Llama-2-7b-hf",
        "inputs": [
            "WizardLMTeam/WizardMath-7B-V1.0",
            "vanillaOVO/WizardCoder-Python-7B-V1.0",
            "medalpaca/medalpaca-7b",
        ],
    },
}

# Patterns matching sst_v2/v3 filenames, e.g.:
#   sst_merge_v3_main_task_arithmetic_safety+math_alpha0.6_seed42_harmbench_safety.json
#   sst_merge_v3_main_safemerge_safety+code_seed42_harmbench_safety.json
#   sst_merge_v3_main_diagonal_sst_main_safety+math_ratioFh_Fb_vinterpolation_k0.2_n500_alpha0.6_seed42_harmbench_safety.json
FNAME_BENCH_RE = re.compile(r"_(harmbench|jailbreakbench|strongreject|wildjailbreak)_safety\.json$")
SEED_RE = re.compile(r"_seed(\d+)")
ALPHA_RE = re.compile(r"_alpha([0-9]+(?:\.[0-9]+)?)")
PATTERN_RE = re.compile(r"_(safety\+math\+code\+medical|safety\+math|safety\+code|safety\+medical)")


def parse_sst_filename(fname: str) -> Optional[Dict[str, Any]]:
    base = os.path.basename(fname)
    bm = FNAME_BENCH_RE.search(base)
    if not bm:
        return None
    bench = bm.group(1)
    pm = PATTERN_RE.search(base)
    if not pm:
        return None
    pattern = pm.group(1)
    sm = SEED_RE.search(base)
    seed = int(sm.group(1)) if sm else None
    am = ALPHA_RE.search(base)
    alpha = float(am.group(1)) if am else None

    prefix = base[:pm.start()]
    for strip_p in ("sst_merge_v3_main_", "sst_merge_v3_ablation_", "fp32_sst_"):
        if prefix.startswith(strip_p):
            prefix = prefix[len(strip_p):]
            break
    method = prefix.strip("_")
    return {
        "method": method,
        "pattern": pattern,
        "alpha": alpha,
        "seed": seed,
        "bench": bench,
    }


def convert_sst_records(bench: str, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert sst_v2 per-item result list to SafeMergeBench safety_generation record list."""
    out: List[Dict[str, Any]] = []
    for idx, r in enumerate(raw_results):
        if not isinstance(r, dict):
            continue
        resp = r.get("response") or ""
        hb_raw = r.get("harmbench_cls_asr")
        orig_asr = r.get("original_asr", r.get("asr"))

        rec: Dict[str, Any] = {
            "id": str(idx),
            "prompt": r.get("prompt", ""),
            "response": resp,
            "sst_is_gibberish_legacy": bool(r.get("is_gibberish", False)),
            "judge_harmbench": (1 if float(hb_raw) >= 0.5 else 0) if hb_raw is not None else None,
        }
        if bench == "harmbench" and rec["judge_harmbench"] is None and orig_asr is not None:
            rec["judge_harmbench"] = 1 if float(orig_asr) >= 0.5 else 0
        elif bench in ("jailbreakbench", "wildjailbreak") and orig_asr is not None:
            rec["judge_wildguard"] = 1 if float(orig_asr) >= 0.5 else 0
        elif bench == "strongreject" and orig_asr is not None:
            rec["strongreject_score"] = float(orig_asr)
            rec["judge_strongreject"] = 1 if float(orig_asr) >= 0.5 else 0

        if orig_asr is not None:
            rec["response_harmfulness"] = "harmful" if float(orig_asr) >= 0.5 else "unharmful"
        out.append(rec)
    return out


def import_directory(
    src_root: str,
    out_dir: str,
    exp_label: str = "exp1",
    patterns: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Scan `src_root` read-only and write unified `safety_generation_judged.json` files."""
    files = sorted(glob.glob(os.path.join(src_root, "**", "*_safety.json"), recursive=True))
    grouped: Dict[Tuple[str, str, Optional[float], Optional[int]], Dict[str, str]] = defaultdict(dict)

    for fp in files:
        meta = parse_sst_filename(fp)
        if not meta:
            continue
        if patterns and meta["pattern"] not in patterns:
            continue
        if alphas is not None and meta["alpha"] is not None and meta["alpha"] not in alphas:
            continue
        key = (meta["pattern"], meta["method"], meta["alpha"], meta["seed"])
        grouped[key][meta["bench"]] = fp

    written_dirs: List[str] = []
    table3_acc: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])  # [legacy_valid, ext_flagged_among_legacy_valid]

    for (pattern, method, alpha, seed), bench_map in sorted(grouped.items(), key=lambda x: str(x[0])):
        alpha_tag = f"alpha_{alpha}" if alpha is not None else "alpha_default"
        seed_tag = f"seed_{seed}" if seed is not None else "seed_none"
        model_dir = os.path.join(out_dir, exp_label, pattern, f"{method}_{alpha_tag}_{seed_tag}")
        os.makedirs(model_dir, exist_ok=True)

        merged_gen: Dict[str, List[Dict[str, Any]]] = {}
        for bench, fp in sorted(bench_map.items()):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            raw_res = data.get("results", [])
            converted = convert_sst_records(bench, raw_res)
            merged_gen[bench] = converted

            # Table 3 reproduction: alpha == 0.6 (or single-point method where alpha is None), seeds 42..44
            if (alpha is None or abs(alpha - 0.6) < 1e-6) and (seed in (42, 43, 44)):
                for rec in converted:
                    if not rec["sst_is_gibberish_legacy"]:
                        table3_acc[(method, pattern)][0] += 1
                        if va.extended_invalid(rec["response"]) is not None:
                            table3_acc[(method, pattern)][1] += 1

        if merged_gen:
            out_path = os.path.join(model_dir, "safety_generation_judged.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(merged_gen, f, indent=2, ensure_ascii=False)
            # Also create safety_generation.json symlink/copy-free reference? Write lightweight alias only if needed
            written_dirs.append(model_dir)

    table3: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
    for (method, pattern), (leg_valid, flagged) in sorted(table3_acc.items()):
        pct = round(100.0 * flagged / leg_valid, 2) if leg_valid > 0 else None
        table3[method][pattern] = pct

    return {
        "exp_label": exp_label,
        "source_root": src_root,
        "models_converted": len(written_dirs),
        "model_dirs": written_dirs,
        "table3_reproduction_pct": dict(table3),
    }


def run_realworld_config_gate(out_path: str) -> Dict[str, Any]:
    """Run config-level integrity gate (`--weights` off) on real-world track combos."""
    reports: Dict[str, Any] = {}
    for combo_name, spec in REALWORLD_COMBOS.items():
        base = spec["base"]
        inputs = spec["inputs"]
        # Use first domain input as merged stand-in if no merged dir is passed (checks base vs inputs)
        cfg_issues = gate.check_config(base, inputs, base)
        tpl_issues, fams = gate.check_template(inputs, base, eval_template="llama2")
        reports[combo_name] = {
            "base": base,
            "inputs": inputs,
            "passed": not cfg_issues,
            "config_issues": cfg_issues,
            "template_issues": tpl_issues,
            "templates": fams,
        }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    return reports


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Import sst_v2 real-world results into SafeMergeBench format (Task G)")
    ap.add_argument("--sst-root", default="/mnt/nas/home/hiromi/src/sst_v2")
    ap.add_argument("--out-dir", default="results/realworld")
    ap.add_argument("--gate-out", default="results/gate/realworld_config_gate.json")
    ap.add_argument("--patterns", nargs="*", default=None)
    ap.add_argument("--alphas", nargs="*", type=float, default=None)
    ap.add_argument("--run-metrics", action="store_true", help="also run safemerge/metrics.py on converted runs")
    ap.add_argument("--check-gate", action="store_true", help="run config-level integrity gate on real-world combos")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    summary: Dict[str, Any] = {"imports": []}

    exp1_dir = os.path.join(args.sst_root, "v3", "results", "vllm", "debug_limit320")
    if os.path.isdir(exp1_dir):
        res1 = import_directory(exp1_dir, args.out_dir, exp_label="exp1_v3", patterns=args.patterns, alphas=args.alphas)
        summary["imports"].append({k: v for k, v in res1.items() if k != "model_dirs"})
        if args.run_metrics and res1["model_dirs"]:
            sm_metrics.run_metrics(res1["model_dirs"], out_dir=os.path.join(args.out_dir, "metrics_exp1"), track="realworld")

    fp32_dir = os.path.join(args.sst_root, "v5", "results", "fp32_sst_math")
    if os.path.isdir(fp32_dir):
        res2 = import_directory(fp32_dir, args.out_dir, exp_label="fp32_sst_math", patterns=args.patterns, alphas=args.alphas)
        summary["imports"].append({k: v for k, v in res2.items() if k != "model_dirs"})
        if args.run_metrics and res2["model_dirs"]:
            sm_metrics.run_metrics(res2["model_dirs"], out_dir=os.path.join(args.out_dir, "metrics_fp32"), track="realworld")

    if args.check_gate:
        summary["realworld_config_gate"] = run_realworld_config_gate(args.gate_out)

    os.makedirs(args.out_dir, exist_ok=True)
    rep_path = os.path.join(args.out_dir, "import_summary.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
