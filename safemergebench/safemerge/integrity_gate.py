"""Artifact-integrity gate for merged models (SafeMergeBench v2, P4/P5/P6 & Sections 4.2, 4.4).

Runs BEFORE any generation. Policy decided on 2026-11-26 based on the `rope_theta`
isolation experiment (`sst_v2/v5/results/rope_theta_ablation`):
  - Fixing `rope_theta` alone (10,000 -> 1,000,000) on `safety+code` (alpha=0.6, seed 42)
    did NOT stop collapse (DARE: 97.5% -> 99.4-100%, SST: 89.1% -> 72.0-93.8%), showing
    that config mismatch alone does not explain collapse when pretraining lineage differs.
  - Therefore (Section 4.4 defaults):
      * HARD FAIL (exit code 1, `issues`):
          - lineage mismatch: median per-tensor relative distance > --lineage-thr (default 0.50,
            calibrated on real-world measurements: WizardMath 2.74%, WizardCoder 113.5%, MedAlpaca 154.7%)
          - numerics: non-finite weights in merged model
          - metadata: recorded `algo` disagrees with --expect-algo
      * WARNING (exit code 0, `warnings`, annotated in tables):
          - config mismatch (`rope_theta`, `vocab_size`, `eos_token_id`, etc.)
          - chat template mismatch across inputs/merged/evaluator
          - degeneration (Section 4.2, revised 2026-11-26 to task-vector denominator because
            MergeBench single experts are only 0.68-1.42% from base in ||theta_base||):
              * changed_tensors == 0
              * ||Delta_merged|| / max_i ||Delta_i|| < 5% (close to base)
              * ||theta_merged - theta_i|| / ||Delta_i|| < 5% for any input i (close to input)
              * domain projection_coeff < 10%
            (rel_norm_diff_to_base = ||Delta_merged|| / ||theta_base|| is still recorded
             for reference but NOT used for pass/fail judgement.)
      * INFEASIBLE (recorded by merge wrappers when vocab mismatch prevents merging altogether,
        e.g., LM-Cocktail).

Checks
  config       : rope_theta / rope_scaling / vocab_size / eos_token_id / tie_word_embeddings /
                 hidden_size / num_hidden_layers / model_type agreement -> WARNING (Section 4.4).
  template     : chat_template agreement across inputs, merged model, and
                 evaluator (--eval-template) -> WARNING (Section 4.4).
  lineage      : (needs --weights) per-tensor relative distance
                 ||theta_i - theta_base|| / ||theta_base|| median <= --lineage-thr (0.50) -> HARD FAIL.
  numerics     : (needs --weights) non-finite values in merged weights -> HARD FAIL.
  degeneration : (needs --weights, float32, Section 4.2) -> WARNING / degenerate flag.
  metadata     : merge recipe recorded in merge_metadata.json matches --expect-algo -> HARD FAIL.

Usage
  python safemerge/integrity_gate.py --base BASE --inputs M1 M2 ... --merged OUT \
      --eval-template llama3 [--expect-algo TIES] [--weights] --report gate.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

CONFIG_KEYS = [
    "model_type",
    "rope_theta",
    "rope_scaling",
    "vocab_size",
    "tie_word_embeddings",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
]

TEMPLATE_MARKERS = {
    "llama3": "<|start_header_id|>",
    "gemma": "<start_of_turn>",
    "llama2": "[INST]",
    "chatml": "<|im_start|>",
}

LAYER_RE = re.compile(r"(?:^|\.)(layers\.\d+)(?:\.|$)")


def _load_json(model: str, filename: str) -> Optional[Dict[str, Any]]:
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_text(model: str, filename: str) -> Optional[str]:
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
    with open(path, encoding="utf-8") as f:
        return f.read()


def _normalize_config(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize transformers v5 `rope_parameters` back into `rope_theta` and `rope_scaling`
    so models saved with transformers v4 and v5 compare cleanly on CONFIG_KEYS.
    """
    if not isinstance(cfg, dict):
        return cfg
    out = dict(cfg)
    rp = out.get("rope_parameters")
    if isinstance(rp, dict):
        if out.get("rope_theta") is None and "rope_theta" in rp:
            out["rope_theta"] = rp["rope_theta"]
        if out.get("rope_scaling") is None:
            scaling = {k: v for k, v in rp.items() if k != "rope_theta"}
            if scaling and not (set(scaling.keys()) == {"rope_type"} and scaling["rope_type"] == "default"):
                out["rope_scaling"] = scaling
    return out


def template_family(tok_cfg: Optional[Dict[str, Any]], jinja_text: Optional[str] = None) -> Optional[str]:
    tpl = (tok_cfg or {}).get("chat_template")
    if isinstance(tpl, list):
        tpl = " ".join(t.get("template", "") for t in tpl)
    if not tpl and jinja_text:
        tpl = jinja_text
    if not tpl:
        return None
    for name, marker in TEMPLATE_MARKERS.items():
        if marker in tpl:
            return name
    return "unknown"


def _normalize_eos(val: Any) -> Any:
    if isinstance(val, list):
        return sorted(set(val))
    if isinstance(val, int):
        return [val]
    return val


def check_config(base: str, inputs: List[str], merged: str) -> List[Dict[str, Any]]:
    warnings_list = []
    cfgs = {m: _normalize_config(_load_json(m, "config.json")) for m in [base, *inputs, merged]}
    gen_cfgs = {m: _load_json(m, "generation_config.json") for m in [base, *inputs, merged]}
    for m, c in cfgs.items():
        if c is None:
            warnings_list.append({"check": "config", "model": m, "problem": "config.json not found"})
    ref = cfgs[base] or {}
    ref_eos = _normalize_eos((gen_cfgs.get(base) or {}).get("eos_token_id", ref.get("eos_token_id")))
    for m, c in cfgs.items():
        if m == base or c is None:
            continue
        for k in CONFIG_KEYS:
            if c.get(k) != ref.get(k):
                warnings_list.append({
                    "check": "config",
                    "model": m,
                    "key": k,
                    "base": ref.get(k),
                    "value": c.get(k),
                })
        m_eos = _normalize_eos((gen_cfgs.get(m) or {}).get("eos_token_id", c.get("eos_token_id")))
        if ref_eos is not None and m_eos is not None and m_eos != ref_eos:
            warnings_list.append({
                "check": "config",
                "model": m,
                "key": "eos_token_id",
                "base": ref_eos,
                "value": m_eos,
            })
    return warnings_list


def check_template(inputs: List[str], merged: str, eval_template: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    issues, fams = [], {}
    for m in [*inputs, merged]:
        tok_cfg = _load_json(m, "tokenizer_config.json")
        jinja = None if (tok_cfg or {}).get("chat_template") else _load_text(m, "chat_template.jinja")
        fams[m] = template_family(tok_cfg, jinja_text=jinja)
    distinct = {f for f in fams.values()}
    if len(distinct) > 1:
        issues.append({"check": "template", "problem": "inputs/merged disagree", "families": fams})
    if fams.get(merged) is None:
        issues.append({
            "check": "template",
            "model": merged,
            "problem": "merged tokenizer has no chat_template (pretrained base?)",
        })
    elif eval_template and fams[merged] != eval_template:
        issues.append({
            "check": "template",
            "model": merged,
            "problem": "evaluator template mismatch",
            "model_template": fams[merged],
            "eval_template": eval_template,
        })
    return issues, fams


def check_metadata(merged: str, expect_algo: Optional[str]) -> List[Dict[str, Any]]:
    if not expect_algo:
        return []
    name = os.path.basename(os.path.normpath(merged))
    meta = _load_json(merged, "merge_metadata.json") if os.path.isdir(merged) else None
    algo = (meta or {}).get("algo") or name.split("_")[0]
    if algo != expect_algo:
        return [{"check": "metadata", "model": merged, "expected": expect_algo, "found": algo}]
    return []


def _layer_group(param_name: str) -> str:
    m = LAYER_RE.search(param_name)
    if m:
        return m.group(1)
    if "embed" in param_name:
        return "embed"
    if "lm_head" in param_name:
        return "lm_head"
    if "norm" in param_name:
        return "final_norm"
    return "other"


def _is_safety_model_name(name: str) -> bool:
    low = os.path.basename(name.rstrip("/")).lower()
    return any(tok in low for tok in ("_safety", "safety_", "safety-", "temp_safety", "wildguard"))


class _TensorStore:
    """Lazy, low-memory tensor reader supporting local safetensors, HF hubs, and dicts."""

    def __init__(self, source: Any):
        self.source = source
        self._dict: Optional[Dict[str, Any]] = None
        self._st_map: Optional[Dict[str, str]] = None
        self._open_handles: Dict[str, Any] = {}
        self._model_dir: Optional[str] = None
        self._init_source()

    def _init_source(self) -> None:
        if isinstance(self.source, dict):
            self._dict = self.source
            return

        model_dir = self.source
        if not os.path.isdir(model_dir):
            try:
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(
                    self.source,
                    allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
                )
            except Exception:
                model_dir = None

        if model_dir and os.path.isdir(model_dir):
            self._model_dir = model_dir
            idx_path = os.path.join(model_dir, "model.safetensors.index.json")
            single_st = os.path.join(model_dir, "model.safetensors")
            if os.path.isfile(idx_path):
                with open(idx_path, encoding="utf-8") as f:
                    wmap = json.load(f).get("weight_map", {})
                self._st_map = {k: os.path.join(model_dir, v) for k, v in wmap.items()}
                return
            if os.path.isfile(single_st):
                from safetensors import safe_open
                with safe_open(single_st, framework="pt", device="cpu") as fh:
                    keys = list(fh.keys())
                self._st_map = {k: single_st for k in keys}
                return

        # Fallback: load full state_dict via AutoModelForCausalLM (e.g. pytorch_model.bin)
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(self.source, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        self._dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        del model
        gc.collect()

    def keys(self) -> List[str]:
        if self._dict is not None:
            return list(self._dict.keys())
        assert self._st_map is not None
        return list(self._st_map.keys())

    def get_f32(self, key: str):
        import torch
        if self._dict is not None:
            return self._dict[key].detach().to(dtype=torch.float32, device="cpu")
        assert self._st_map is not None
        fpath = self._st_map[key]
        fh = self._open_handles.get(fpath)
        if fh is None:
            from safetensors import safe_open
            fh = safe_open(fpath, framework="pt", device="cpu")
            self._open_handles[fpath] = fh
        return fh.get_tensor(key).to(dtype=torch.float32)

    def has(self, key: str) -> bool:
        if self._dict is not None:
            return key in self._dict
        assert self._st_map is not None
        return key in self._st_map

    def close(self) -> None:
        self._open_handles.clear()
        self._dict = None


def analyze_weights_from_stores(
    base_name: str,
    base_store: _TensorStore,
    input_stores: List[Tuple[str, _TensorStore]],
    merged_name: str,
    merged_store: _TensorStore,
    lineage_thr: float = 0.5,
    degenerate_tv_ratio_thr: float = 0.05,
    degenerate_proj_thr: float = 0.10,
    domain_inputs: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute float32 lineage, numerics, and Section 4.2 degeneration statistics."""
    import torch

    issues: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}

    merged_keys = merged_store.keys()
    base_keys = set(base_store.keys())

    # 1. Numerics & merged-vs-base statistics
    changed: List[str] = []
    unchanged: List[str] = []
    nonfinite: List[str] = []
    diff_base_sq = 0.0
    base_norm_sq = 0.0
    merged_delta_sq = 0.0
    merged_layer_delta_sq: Dict[str, float] = {}
    max_abs_diff = 0.0
    max_abs_diff_tensor: Optional[str] = None

    for k in merged_keys:
        mv = merged_store.get_f32(k)
        if not torch.isfinite(mv).all():
            nonfinite.append(k)
        if k in base_keys:
            bv = base_store.get_f32(k)
            if mv.shape == bv.shape:
                dm = mv - bv
                d_sq = float(torch.dot(dm.flatten(), dm.flatten()).item())
                b_sq = float(torch.dot(bv.flatten(), bv.flatten()).item())
                diff_base_sq += d_sq
                base_norm_sq += b_sq
                merged_delta_sq += d_sq
                lg = _layer_group(k)
                merged_layer_delta_sq[lg] = merged_layer_delta_sq.get(lg, 0.0) + d_sq
                m_abs = float(dm.abs().max().item()) if dm.numel() > 0 else 0.0
                if m_abs > max_abs_diff:
                    max_abs_diff = m_abs
                    max_abs_diff_tensor = k
                if d_sq == 0.0:
                    unchanged.append(k)
                else:
                    changed.append(k)
            else:
                changed.append(k)
        else:
            changed.append(k)

    rel_norm_to_base = (math.sqrt(diff_base_sq) / math.sqrt(base_norm_sq)) if base_norm_sq > 0 else None
    merged_delta_norm = math.sqrt(merged_delta_sq)

    if nonfinite:
        issues.append({"check": "numerics", "model": merged_name, "nonfinite_tensors": nonfinite[:20]})

    # 2. Per-input lineage, cosine similarity (global + per-layer), projection coeff, rel_norm_to_input
    input_stats_map: Dict[str, Dict[str, Any]] = {}
    for inp_name, inp_store in input_stores:
        rel_dists: List[float] = []
        dot_dm_di = 0.0
        di_sq_total = 0.0
        diff_inp_sq = 0.0
        inp_norm_sq = 0.0
        layer_dot: Dict[str, float] = {}
        layer_di_sq: Dict[str, float] = {}

        for k in merged_keys:
            if k not in base_keys or not inp_store.has(k):
                continue
            bv = base_store.get_f32(k)
            iv = inp_store.get_f32(k)
            mv = merged_store.get_f32(k)
            if not (bv.shape == iv.shape == mv.shape):
                continue

            b_norm = float(bv.norm().item())
            di = iv - bv
            dm = mv - bv
            di_sq = float(torch.dot(di.flatten(), di.flatten()).item())
            if b_norm > 0:
                rel_dists.append(math.sqrt(di_sq) / b_norm)

            dot_val = float(torch.dot(dm.flatten(), di.flatten()).item())
            dot_dm_di += dot_val
            di_sq_total += di_sq

            lg = _layer_group(k)
            layer_dot[lg] = layer_dot.get(lg, 0.0) + dot_val
            layer_di_sq[lg] = layer_di_sq.get(lg, 0.0) + di_sq

            mi = mv - iv
            diff_inp_sq += float(torch.dot(mi.flatten(), mi.flatten()).item())
            inp_norm_sq += float(torch.dot(iv.flatten(), iv.flatten()).item())

        rel_dists.sort()
        med = rel_dists[len(rel_dists) // 2] if rel_dists else float("nan")
        if not math.isnan(med) and med > lineage_thr:
            issues.append({"check": "lineage", "model": inp_name, "median_rel_dist": med, "threshold": lineage_thr})

        di_norm = math.sqrt(di_sq_total)
        diff_inp_norm = math.sqrt(diff_inp_sq)
        global_cos = (dot_dm_di / (merged_delta_norm * di_norm)) if (merged_delta_norm > 0 and di_norm > 0) else 0.0
        proj_coeff = (dot_dm_di / di_sq_total) if di_sq_total > 0 else 0.0
        rel_norm_to_input = (diff_inp_norm / math.sqrt(inp_norm_sq)) if inp_norm_sq > 0 else None
        dist_to_merged_tv_ratio = (diff_inp_norm / di_norm) if di_norm > 0 else None

        per_layer_cos: Dict[str, Optional[float]] = {}
        for lg in sorted(layer_di_sq.keys(), key=lambda x: (0, int(x.split(".")[1])) if x.startswith("layers.") else (1, x)):
            dm_l = math.sqrt(merged_layer_delta_sq.get(lg, 0.0))
            di_l = math.sqrt(layer_di_sq[lg])
            if dm_l > 0 and di_l > 0:
                per_layer_cos[lg] = round(layer_dot.get(lg, 0.0) / (dm_l * di_l), 6)
            else:
                per_layer_cos[lg] = 0.0

        inp_stat = {
            "median_rel_dist": round(med, 6) if not math.isnan(med) else None,
            "task_vector_norm": round(di_norm, 6),
            "input_rel_norm_to_base": round(di_norm / math.sqrt(base_norm_sq), 6) if base_norm_sq > 0 else None,
            "cos_sim_to_merged_delta": round(global_cos, 6),
            "projection_coeff": round(proj_coeff, 6),
            "rel_norm_diff_to_merged": round(rel_norm_to_input, 6) if rel_norm_to_input is not None else None,
            "dist_to_merged_tv_ratio": round(dist_to_merged_tv_ratio, 6) if dist_to_merged_tv_ratio is not None else None,
            "per_layer_cos_sim": per_layer_cos,
        }
        stats[inp_name] = inp_stat
        input_stats_map[inp_name] = inp_stat

    # 3. Degeneration detection (Section 4.2, revised 2026-11-26 to task-vector denominator):
    #   (a) changed_tensors == 0
    #   (b) ||Delta_merged|| / max_i ||Delta_i|| < degenerate_tv_ratio_thr (default 5%)
    #   (c) ||theta_merged - theta_i|| / ||Delta_i|| < degenerate_tv_ratio_thr (default 5%)
    #   (d) domain input projection_coeff < degenerate_proj_thr (default 10%)
    # Note: rel_norm_diff_to_base (||Delta_merged|| / ||theta_base||) is recorded for reference ONLY.
    max_input_delta_norm = max((s["task_vector_norm"] for s in input_stats_map.values()), default=0.0)
    rel_to_max_input_delta = (merged_delta_norm / max_input_delta_norm) if max_input_delta_norm > 0 else None

    degenerate_reasons: List[Dict[str, Any]] = []
    if len(changed) == 0:
        degenerate_reasons.append({
            "reason": "zero_changed_tensors",
            "changed_tensors": 0,
        })
    if rel_to_max_input_delta is not None and rel_to_max_input_delta < degenerate_tv_ratio_thr:
        degenerate_reasons.append({
            "reason": "close_to_base",
            "merged_to_max_input_delta_ratio": round(rel_to_max_input_delta, 6),
            "threshold": degenerate_tv_ratio_thr,
        })

    # Determine which inputs represent domain experts for projection check
    if domain_inputs:
        target_domain_inputs = [m for m in domain_inputs if m in input_stats_map]
    else:
        non_safety = [m for m in input_stats_map if not _is_safety_model_name(m)]
        target_domain_inputs = non_safety if non_safety else list(input_stats_map.keys())

    for inp_name, istat in input_stats_map.items():
        r_tv = istat["dist_to_merged_tv_ratio"]
        if r_tv is not None and r_tv < degenerate_tv_ratio_thr:
            degenerate_reasons.append({
                "reason": "close_to_input",
                "model": inp_name,
                "dist_to_merged_tv_ratio": r_tv,
                "threshold": degenerate_tv_ratio_thr,
            })
        if inp_name in target_domain_inputs and istat["task_vector_norm"] > 0:
            if istat["projection_coeff"] < degenerate_proj_thr:
                degenerate_reasons.append({
                    "reason": "low_domain_update_retention",
                    "model": inp_name,
                    "projection_coeff": istat["projection_coeff"],
                    "threshold": degenerate_proj_thr,
                })

    degeneration_summary = {
        "degenerate": bool(degenerate_reasons),
        "reasons": degenerate_reasons,
        "thresholds": {
            "tv_ratio_thr": degenerate_tv_ratio_thr,
            "domain_projection_thr": degenerate_proj_thr,
        },
        "domain_inputs_checked": target_domain_inputs,
        "rel_norm_diff_to_base": round(rel_norm_to_base, 6) if rel_norm_to_base is not None else None,
        "merged_task_vector_norm": round(merged_delta_norm, 6),
        "max_input_task_vector_norm": round(max_input_delta_norm, 6),
        "merged_to_max_input_delta_ratio": round(rel_to_max_input_delta, 6) if rel_to_max_input_delta is not None else None,
        "max_abs_diff_to_base": round(max_abs_diff, 6),
        "max_abs_diff_tensor": max_abs_diff_tensor,
        "changed_tensors": len(changed),
        "unchanged_tensors": len(unchanged),
        "unchanged_examples": unchanged[:20],
    }

    stats[merged_name] = {
        "changed_tensors": len(changed),
        "unchanged_tensors": len(unchanged),
        "unchanged_examples": unchanged[:20],
        "rel_norm_diff_to_base": degeneration_summary["rel_norm_diff_to_base"],
        "merged_task_vector_norm": degeneration_summary["merged_task_vector_norm"],
        "max_abs_diff_to_base": degeneration_summary["max_abs_diff_to_base"],
        "max_abs_diff_tensor": max_abs_diff_tensor,
    }
    stats["degeneration"] = degeneration_summary
    return issues, stats


def check_weights(
    base: str,
    inputs: List[str],
    merged: str,
    lineage_thr: float = 0.5,
    degenerate_tv_ratio_thr: float = 0.05,
    degenerate_proj_thr: float = 0.10,
    domain_inputs: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Weight-level checks (lineage, numerics, Section 4.2 degeneration); float32."""
    base_store = _TensorStore(base)
    merged_store = _TensorStore(merged)
    input_stores: List[Tuple[str, _TensorStore]] = []
    try:
        for m in inputs:
            input_stores.append((m, _TensorStore(m)))
        return analyze_weights_from_stores(
            base_name=base,
            base_store=base_store,
            input_stores=input_stores,
            merged_name=merged,
            merged_store=merged_store,
            lineage_thr=lineage_thr,
            degenerate_tv_ratio_thr=degenerate_tv_ratio_thr,
            degenerate_proj_thr=degenerate_proj_thr,
            domain_inputs=domain_inputs,
        )
    finally:
        base_store.close()
        merged_store.close()
        for _, st in input_stores:
            st.close()
        gc.collect()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--eval-template", default=None, help="template the evaluator will apply, e.g. llama3, gemma")
    ap.add_argument("--expect-algo", default=None)
    ap.add_argument("--weights", action="store_true", help="also run lineage/numerics/degeneration checks (float32)")
    ap.add_argument("--lineage-thr", type=float, default=0.5)
    ap.add_argument("--degenerate-tv-ratio-thr", "--degenerate-rel-norm-thr", dest="degenerate_tv_ratio_thr",
                    type=float, default=0.05,
                    help="task-vector ratio threshold for close_to_base (||Delta_m||/max||Delta_i||) and close_to_input (||theta_m-theta_i||/||Delta_i||) (default 0.05)")
    ap.add_argument("--degenerate-proj-thr", type=float, default=0.10,
                    help="domain task-vector projection threshold below which a merge is flagged degenerate (default 0.10)")
    ap.add_argument("--domain-inputs", nargs="+", default=None,
                    help="explicit list of domain-expert input paths/ids for projection retention check")
    ap.add_argument("--skip-config-template", action="store_true",
                    help="skip config/template checks (diagnostic weight-only mode)")
    ap.add_argument("--strict-config", action="store_true",
                    help="treat config/template warnings as hard failures (default: warnings only per Section 4.4)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    issues: List[Dict[str, Any]] = []
    warnings_list: List[Dict[str, Any]] = []
    fams: Dict[str, Optional[str]] = {}
    if not args.skip_config_template:
        cfg_findings = check_config(args.base, args.inputs, args.merged)
        tpl_findings, fams = check_template(args.inputs, args.merged, args.eval_template)
        if args.strict_config:
            issues += cfg_findings + tpl_findings
        else:
            warnings_list += cfg_findings + tpl_findings
    issues += check_metadata(args.merged, args.expect_algo)
    stats: Dict[str, Any] = {"templates": fams}
    degenerate = False
    if args.weights:
        w_issues, w_stats = check_weights(
            args.base,
            args.inputs,
            args.merged,
            lineage_thr=args.lineage_thr,
            degenerate_tv_ratio_thr=args.degenerate_tv_ratio_thr,
            degenerate_proj_thr=args.degenerate_proj_thr,
            domain_inputs=args.domain_inputs,
        )
        issues += w_issues
        stats.update(w_stats)
        degen = w_stats.get("degeneration", {})
        degenerate = bool(degen.get("degenerate", False))
        if degenerate:
            warnings_list.append({
                "check": "degeneration",
                "model": args.merged,
                "reasons": degen.get("reasons", []),
            })

    report = {
        "passed": not issues,
        "degenerate": degenerate,
        "policy_basis": "sst_v2/v5/results/rope_theta_ablation (Section 4.4)",
        "issues": issues,
        "warnings": warnings_list,
        "stats": stats,
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
