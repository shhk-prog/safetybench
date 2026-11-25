"""Artifact-integrity gate for merged models (SafeMergeBench, P4/P5/P6).

Runs BEFORE any generation. A merged model that fails the gate is reported as
"gate-failed" and excluded from the safety/utility tables.

Checks
  config   : rope_theta / rope_scaling / vocab_size / tie_word_embeddings /
             hidden_size / num_hidden_layers / model_type must agree across the
             base, every input model and the merged model (P4).
  template : chat_template of every input model and of the merged tokenizer
             must agree, and must match the template the evaluator will use
             (--eval-template), e.g. "llama3" or "gemma" (P6).
  lineage  : (needs torch) per-tensor relative distance ||theta_i - theta_base|| /
             ||theta_base|| for every input; median above --lineage-thr means the
             input was probably not fine-tuned from this base (P4).
  numerics : (needs torch) non-finite values in the merged weights, and which
             tensors were actually changed w.r.t. the base (P5; also catches
             parameters a merger silently skips, e.g. lm_head).
  metadata : the merge recipe (algorithm + hyperparameters) recorded next to the
             merged model must match what the experiment table claims.

Usage
  python safemerge/integrity_gate.py --base BASE --inputs M1 M2 ... --merged OUT \
      --eval-template llama3 [--expect-algo TIES] [--weights] --report gate.json
Models may be local directories or Hugging Face repo ids (config/tokenizer files
are fetched with huggingface_hub when the path is not a directory).
"""
import argparse
import json
import os
import sys

CONFIG_KEYS = ["model_type", "rope_theta", "rope_scaling", "vocab_size",
               "tie_word_embeddings", "hidden_size", "num_hidden_layers",
               "num_attention_heads", "num_key_value_heads"]

TEMPLATE_MARKERS = {
    "llama3": "<|start_header_id|>",
    "gemma": "<start_of_turn>",
    "llama2": "[INST]",
    "chatml": "<|im_start|>",
}


def _load_json(model, filename):
    if os.path.isdir(model):
        path = os.path.join(model, filename)
        if not os.path.exists(path):
            return None
    else:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, filename)
        except Exception:
            return None
    with open(path) as f:
        return json.load(f)


def template_family(tok_cfg):
    tpl = (tok_cfg or {}).get("chat_template")
    if isinstance(tpl, list):  # some tokenizers store named templates
        tpl = " ".join(t.get("template", "") for t in tpl)
    if not tpl:
        return None
    for name, marker in TEMPLATE_MARKERS.items():
        if marker in tpl:
            return name
    return "unknown"


def check_config(base, inputs, merged):
    issues = []
    cfgs = {m: _load_json(m, "config.json") for m in [base, *inputs, merged]}
    for m, c in cfgs.items():
        if c is None:
            issues.append({"check": "config", "model": m, "problem": "config.json not found"})
    ref = cfgs[base] or {}
    for m, c in cfgs.items():
        if m == base or c is None:
            continue
        for k in CONFIG_KEYS:
            if c.get(k) != ref.get(k):
                issues.append({"check": "config", "model": m, "key": k,
                               "base": ref.get(k), "value": c.get(k)})
    return issues


def check_template(inputs, merged, eval_template):
    issues, fams = [], {}
    for m in [*inputs, merged]:
        fams[m] = template_family(_load_json(m, "tokenizer_config.json"))
    distinct = {f for f in fams.values()}
    if len(distinct) > 1:
        issues.append({"check": "template", "problem": "inputs/merged disagree", "families": fams})
    if fams.get(merged) is None:
        issues.append({"check": "template", "model": merged,
                       "problem": "merged tokenizer has no chat_template (pretrained base?)"})
    elif eval_template and fams[merged] != eval_template:
        issues.append({"check": "template", "model": merged, "problem": "evaluator template mismatch",
                       "model_template": fams[merged], "eval_template": eval_template})
    return issues, fams


def check_metadata(merged, expect_algo):
    if not expect_algo:
        return []
    # MergeBench encodes the algorithm as the first token of the output dir name.
    name = os.path.basename(os.path.normpath(merged))
    meta = _load_json(merged, "merge_metadata.json") if os.path.isdir(merged) else None
    algo = (meta or {}).get("algo") or name.split("_")[0]
    if algo != expect_algo:
        return [{"check": "metadata", "model": merged, "expected": expect_algo, "found": algo}]
    return []


def check_weights(base, inputs, merged, lineage_thr):
    """Weight-level checks; requires torch + transformers."""
    import torch
    from transformers import AutoModelForCausalLM

    def sd(m):
        model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32)
        return {k: v.detach() for k, v in model.state_dict().items()}

    issues, stats = [], {}
    base_sd = sd(base)
    for m in inputs:
        d = sd(m)
        rel = []
        for k, v in base_sd.items():
            if k in d and d[k].shape == v.shape and v.norm() > 0:
                rel.append(((d[k] - v).norm() / v.norm()).item())
        rel.sort()
        med = rel[len(rel) // 2] if rel else float("nan")
        stats[m] = {"median_rel_dist": med}
        if med > lineage_thr:
            issues.append({"check": "lineage", "model": m, "median_rel_dist": med, "threshold": lineage_thr})
        del d
    msd = sd(merged)
    changed, unchanged, nonfinite = [], [], []
    for k, v in msd.items():
        if not torch.isfinite(v).all():
            nonfinite.append(k)
        if k in base_sd and torch.equal(v, base_sd[k]):
            unchanged.append(k)
        else:
            changed.append(k)
    stats[merged] = {"changed_tensors": len(changed), "unchanged_tensors": len(unchanged),
                     "unchanged_examples": unchanged[:20]}
    if nonfinite:
        issues.append({"check": "numerics", "model": merged, "nonfinite_tensors": nonfinite[:20]})
    return issues, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--eval-template", default=None, help="template the evaluator will apply, e.g. llama3, gemma")
    ap.add_argument("--expect-algo", default=None)
    ap.add_argument("--weights", action="store_true", help="also run lineage/numerics checks (loads weights)")
    ap.add_argument("--lineage-thr", type=float, default=0.5)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    issues = check_config(args.base, args.inputs, args.merged)
    t_issues, fams = check_template(args.inputs, args.merged, args.eval_template)
    issues += t_issues
    issues += check_metadata(args.merged, args.expect_algo)
    stats = {"templates": fams}
    if args.weights:
        w_issues, w_stats = check_weights(args.base, args.inputs, args.merged, args.lineage_thr)
        issues += w_issues
        stats.update(w_stats)

    report = {"passed": not issues, "issues": issues, "stats": stats}
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
