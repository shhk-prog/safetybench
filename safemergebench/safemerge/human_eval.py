"""Human evaluation agreement & detector precision/recall calculator (Task E).

Reads blind human-label CSVs (1 or 2 independent annotators) plus the matching
key file (from `validity_audit.py`'s `human_label_key.csv` or `sst_v2`'s
`human_eval_200_key.json`) and computes:
  - Inter-annotator Cohen's kappa for `is_valid` and `is_harmful`
  - Precision, recall, F1 for:
      * Extended validity rule (`extended_invalid`)
      * Legacy validity rule (`legacy_invalid`)
      * Each safety judge (`harmbench`, `wildguard`, `strongreject`, `safety_eval_default`)
  - Confirmed-collapse rate among responses flagged as collapsed (Table 10 validation)

Usage:
  python safemerge/human_eval.py \
    --annotator1 labels_a.csv [--annotator2 labels_b.csv] \
    --key human_label_key.csv --out results/metrics/human_eval_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validity_audit as va  # noqa: E402


def _parse_binary(val: Any) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "na", "n/a", "none", "null"):
        return None
    if s in ("1", "true", "yes", "y", "valid", "harmful"):
        return 1
    if s in ("0", "false", "no", "n", "invalid", "collapsed", "safe", "unharmful"):
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except ValueError:
        return None


def cohen_kappa(y1: List[int], y2: List[int]) -> Optional[float]:
    """Compute binary Cohen's kappa for paired 0/1 annotations."""
    pairs = [(a, b) for a, b in zip(y1, y2) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    tp = sum(1 for a, b in pairs if a == 1 and b == 1)
    tn = sum(1 for a, b in pairs if a == 0 and b == 0)
    fp = sum(1 for a, b in pairs if a == 1 and b == 0)
    fn = sum(1 for a, b in pairs if a == 0 and b == 1)
    po = (tp + tn) / n
    p1_yes = (tp + fp) / n
    p2_yes = (tp + fn) / n
    p1_no = (tn + fn) / n
    p2_no = (tn + fp) / n
    pe = p1_yes * p2_yes + p1_no * p2_no
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    return (po - pe) / (1.0 - pe)


def binary_prf(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    n = len(pairs)
    tp = sum(1 for t, p in pairs if t == 1 and p == 1)
    fp = sum(1 for t, p in pairs if t == 0 and p == 1)
    fn = sum(1 for t, p in pairs if t == 1 and p == 0)
    tn = sum(1 for t, p in pairs if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    rec = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
    acc = (tp + tn) / n if n > 0 else None
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 6) if prec is not None else None,
        "recall": round(rec, 6) if rec is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
        "accuracy": round(acc, 6) if acc is not None else None,
    }


def load_label_csv(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            item_id = str(row.get("item", row.get("sample_id", row.get("id", i)))).strip()
            valid = _parse_binary(row.get("is_valid(1/0)", row.get("is_valid", row.get("human_valid"))))
            harm = _parse_binary(row.get("is_harmful(1/0)", row.get("is_harmful", row.get("human_harmful"))))
            rows[item_id] = {
                "item": item_id,
                "response": row.get("response", ""),
                "is_valid": valid,
                "is_harmful": harm,
            }
    return rows


def load_key_file(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("items", data.get("samples", []))
        out: Dict[str, Dict[str, Any]] = {}
        for i, rec in enumerate(items):
            item_id = str(rec.get("item", rec.get("sample_id", rec.get("blind_id", i)))).strip()
            out[item_id] = rec
        return out
    out = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            item_id = str(row.get("item", row.get("sample_id", i))).strip()
            out[item_id] = row
    return out


def _extract_judge_from_key(krec: Dict[str, Any], name: str) -> Optional[int]:
    if not krec:
        return None
    if name == "harmbench":
        for k in ("judge_harmbench", "harmbench_cls_asr", "hb_harmful"):
            if k in krec and krec[k] not in (None, ""):
                return _parse_binary(krec[k])
    elif name == "wildguard":
        for k in ("judge_wildguard", "wildguard_harmful"):
            if k in krec and krec[k] not in (None, ""):
                return _parse_binary(krec[k])
    elif name == "strongreject":
        for k in ("judge_strongreject", "strongreject_harmful"):
            if k in krec and krec[k] not in (None, ""):
                return _parse_binary(krec[k])
    elif name == "safety_eval_default":
        for k in ("judge_harm", "native_harmful", "response_harmfulness"):
            if k in krec and krec[k] not in (None, ""):
                return _parse_binary(krec[k])
    return None


def evaluate_human_labels(
    ann1: Dict[str, Dict[str, Any]],
    ann2: Optional[Dict[str, Dict[str, Any]]] = None,
    key_map: Optional[Dict[str, Dict[str, Any]]] = None,
    disagreement_policy: str = "agreed_only",
) -> Dict[str, Any]:
    key_map = key_map or {}
    item_ids = sorted(ann1.keys(), key=lambda x: (len(x), x))

    agreement: Dict[str, Any] = {}
    gold_valid: Dict[str, Optional[int]] = {}
    gold_harm: Dict[str, Optional[int]] = {}

    if ann2:
        common = [i for i in item_ids if i in ann2]
        v1 = [ann1[i]["is_valid"] for i in common]
        v2 = [ann2[i]["is_valid"] for i in common]
        h1 = [ann1[i]["is_harmful"] for i in common]
        h2 = [ann2[i]["is_harmful"] for i in common]
        v_pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
        h_pairs = [(a, b) for a, b in zip(h1, h2) if a is not None and b is not None]
        agreement = {
            "n_common": len(common),
            "valid_kappa": round(cohen_kappa(v1, v2), 6) if v_pairs else None,
            "valid_raw_agreement": round(sum(a == b for a, b in v_pairs) / len(v_pairs), 6) if v_pairs else None,
            "harmful_kappa": round(cohen_kappa(h1, h2), 6) if h_pairs else None,
            "harmful_raw_agreement": round(sum(a == b for a, b in h_pairs) / len(h_pairs), 6) if h_pairs else None,
        }
        for i in common:
            a1v, a2v = ann1[i]["is_valid"], ann2[i]["is_valid"]
            a1h, a2h = ann1[i]["is_harmful"], ann2[i]["is_harmful"]
            if disagreement_policy == "annotator1":
                gold_valid[i] = a1v
                gold_harm[i] = a1h
            else:
                gold_valid[i] = a1v if a1v == a2v else None
                gold_harm[i] = a1h if a1h == a2h else None
    else:
        for i in item_ids:
            gold_valid[i] = ann1[i]["is_valid"]
            gold_harm[i] = ann1[i]["is_harmful"]

    # Positive class for collapse detection = 1 (collapsed / invalid)
    y_true_collapsed: List[Optional[int]] = []
    y_pred_ext_collapsed: List[int] = []
    y_pred_leg_collapsed: List[int] = []

    for i in item_ids:
        gv = gold_valid.get(i)
        resp = ann1[i].get("response") or key_map.get(i, {}).get("response", "")
        ext_flag = 1 if va.extended_invalid(resp) is not None else 0
        leg_flag = 1 if va.legacy_invalid(resp) is not None else 0
        y_true_collapsed.append(None if gv is None else (1 - gv))
        y_pred_ext_collapsed.append(ext_flag)
        y_pred_leg_collapsed.append(leg_flag)

    ext_metrics = binary_prf(y_true_collapsed, y_pred_ext_collapsed)
    leg_metrics = binary_prf(y_true_collapsed, y_pred_leg_collapsed)

    # Proportion of extended-flagged responses confirmed collapsed by human
    flagged_pairs = [(t, p) for t, p in zip(y_true_collapsed, y_pred_ext_collapsed) if t is not None and p == 1]
    confirmed_collapsed_rate = (
        round(sum(1 for t, _ in flagged_pairs if t == 1) / len(flagged_pairs), 6)
        if flagged_pairs else None
    )

    judge_metrics: Dict[str, Any] = {}
    for jname in ("harmbench", "wildguard", "strongreject", "safety_eval_default"):
        yt = [gold_harm.get(i) for i in item_ids]
        yp = [_extract_judge_from_key(key_map.get(i, {}), jname) for i in item_ids]
        prf = binary_prf(yt, yp)
        if prf["n"] > 0:
            judge_metrics[jname] = prf

    return {
        "n_items": len(item_ids),
        "disagreement_policy": disagreement_policy,
        "inter_annotator": agreement,
        "collapse_detection": {
            "extended_rule": ext_metrics,
            "legacy_rule": leg_metrics,
            "flagged_confirmed_collapsed_n": len(flagged_pairs),
            "flagged_confirmed_collapsed_rate": confirmed_collapsed_rate,
        },
        "harm_judges": judge_metrics,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute human-label agreement and detector precision/recall (Task E)")
    ap.add_argument("--annotator1", required=True, help="CSV with annotator 1 labels")
    ap.add_argument("--annotator2", default=None, help="optional CSV with annotator 2 labels")
    ap.add_argument("--key", default=None, help="human_label_key.csv or human_eval_200_key.json")
    ap.add_argument("--disagreement-policy", default="agreed_only", choices=["agreed_only", "annotator1"])
    ap.add_argument("--out", default="results/metrics/human_eval_report.json")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ann1 = load_label_csv(args.annotator1)
    ann2 = load_label_csv(args.annotator2) if args.annotator2 else None
    key_map = load_key_file(args.key)

    report = evaluate_human_labels(ann1, ann2, key_map, disagreement_policy=args.disagreement_policy)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
