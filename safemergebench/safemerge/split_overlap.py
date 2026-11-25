"""Check that coefficient-tuning prompts and evaluation prompts are disjoint.

MergeBench tunes the safety coefficient on safety-eval's `wildjailbreak:harmful`
(scripts/validate.sh). If WildJailbreak is also used for evaluation, the two
prompt sets must not overlap. This script reports exact (normalised) duplicates
and near-duplicates (word 5-gram Jaccard >= --thr) and can write the eval set
with overlapping prompts removed.

Input files: .json (list of records or dict of task -> list) or .jsonl;
the prompt field is taken from --field (default tries prompt/adversarial/vanilla/Behavior).

Usage
  python safemerge/split_overlap.py --tune tune.jsonl --eval eval.jsonl --out eval_clean.jsonl
"""
import argparse
import json
import re

FIELDS = ["prompt", "adversarial", "vanilla", "Behavior", "query"]


def load(path, field=None):
    with open(path) as f:
        if path.endswith(".jsonl"):
            recs = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            recs = [r for v in data.values() for r in v] if isinstance(data, dict) else data
    out = []
    for r in recs:
        if isinstance(r, str):
            out.append((r, r))
            continue
        keys = [field] if field else FIELDS
        text = next((r[k] for k in keys if k in r and r[k]), None)
        if text:
            out.append((text, r))
    return out


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


def shingles(t, n=5):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--field", default=None)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    tune = load(args.tune, args.field)
    ev = load(args.eval, args.field)
    tune_exact = {norm(t) for t, _ in tune}
    index = {}
    tune_sh = []
    for i, (t, _) in enumerate(tune):
        s = shingles(t)
        tune_sh.append(s)
        for g in s:
            index.setdefault(g, set()).add(i)

    exact, near, keep = 0, 0, []
    for t, rec in ev:
        if norm(t) in tune_exact:
            exact += 1
            continue
        s = shingles(t)
        cands = set().union(*(index.get(g, set()) for g in s)) if s else set()
        if any(len(s & tune_sh[c]) / len(s | tune_sh[c]) >= args.thr for c in cands):
            near += 1
            continue
        keep.append(rec)

    print(json.dumps({"tune": len(tune), "eval": len(ev), "exact_overlap": exact,
                      "near_overlap": near, "eval_kept": len(keep)}, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    main()
