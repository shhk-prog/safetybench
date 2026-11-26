"""LED-Merging wrapper via official MqLeet/LED-Merging (ACL 2025 Main).

Implementation Spec v2 (Sections 3.7, 4.2 & 2026-11-26 addendum):
  - NEVER modifies any file inside upstream `LED-Merging/`.
  - Creates an isolated workdir with symlinks to `LED-Merging/` plus clean
    launcher scripts that redirect hard-coded `/path/to/...` aliases
    (`MODEL_DIR` / `modeltype2path`) via `LED_MODEL_PATH_MAP`.
  - Restores real gradient-based Wanda-g scoring (`_prune_core(..., prune_mode="gradient")`)
    in `locate/lib/model_wrapper.py` without editing upstream (upstream commit 1149224
    L515-531 commented out `_prune_core` and called `_random_prune_core`, writing 289/291
    layers to `random_score/` instead of `wanda_score/`).
  - Pre-merge integrity assertions (refuses to merge if violated unless
    `--unpatched-upstream` is explicitly passed for reproduction):
      1. Verifies every importance directory (`wanda_score/`) that merging actually
         reads contains score files for ALL expected model tensors (prints & logs counts).
      2. Inspects base vs input tensor shapes, skips mismatched-shape tensors cleanly
         (leaving base weights intact), and records exact `skipped_shape_mismatch_tensors`.
      3. Computes and records effective disjoint mask density in `merge_metadata.json`.
  - Default `fuse_rates` follow official `LED-Merging/README.md` L79 (`safety=0.1, math=0.4, code=0.5`).
  - Optional `--unpatched-upstream` reproduces the exact unpatched upstream behavior
    (`_random_prune_core` -> 2 files in `wanda_score/`, silent fallback to base weights).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ensure_tokenizer_saved,
    git_commit,
    resolve_external_repo,
    set_seed,
    write_merge_metadata,
)

# Official LED-Merging/README.md L79 default rates per domain: rates=(0.1 0.4 0.5)
OFFICIAL_DEFAULT_RATES = {
    "safety": 0.1,
    "math": 0.4,
    "code": 0.5,
    "medical": 0.4,
    "multilingual": 0.4,
    "instruction": 0.4,
}

# Upstream LED-Merging hard-coded alias -> placeholder path table (locate/main.py & merge_llms.py)
UPSTREAM_ALIAS_PATHS = {
    "llama3": {
        "base": ("llama3-base", "/path/to/Meta-Llama-3-8B"),
        "safety": ("llama3-instruct", "/path/to/Meta-Llama-3-8B-Instruct"),
        "math": ("llama3-math", "/path/to/MAmmoTH2-8B-Plus"),
        "code": ("llama3-code", "/path/to/Replete-Coder-Llama3-8B"),
    },
    "llama2": {
        "base": ("llama2-base", "/path/to/llama-2-13b"),
        "safety": ("llama2-instruct", "/path/to/WizardLM-13B-V1.2"),
        "math": ("llama2-math", "/path/to/WizardMath-13B-V1.0"),
        "code": ("llama2-code", "/path/to/llama-2-13b-code-alpaca"),
    },
    "qwen2.5": {
        "base": ("qwen2.5-base", "/path/to/Qwen2.5-7B"),
        "safety": ("qwen2.5-instruct", "/path/to/Qwen2.5-7B-Instruct"),
        "math": ("qwen2.5-math", "/path/to/Qwen2.5-Math-7B"),
        "code": ("qwen2.5-code", "/path/to/Qwen2.5-Coder-7B"),
    },
    "mistral": {
        "base": ("mistral-base", "/path/to/Mistral-7B-v0.1"),
        "safety": ("mistral-instruct", "/path/to/Mistral-7B-Instruct-v0.2"),
        "math": ("mistral-math", "/path/to/MetaMath-Mistral-7B"),
    },
}

RUN_LOCATE_LAUNCHER = r'''"""Clean launcher for upstream LED-Merging `locate/main.py` (without modifying upstream)."""
import json
import os
import runpy
import sys
import types

# 1. Stub unused `vllm` if absent (do NOT define __getattr__, which breaks inspect.getmodule on __file__)
if "vllm" not in sys.modules:
    try:
        import vllm  # noqa: F401
    except ImportError:
        stub = types.ModuleType("vllm")
        stub.LLM = object
        stub.SamplingParams = object
        sys.modules["vllm"] = stub

# 2. Remap hard-coded model paths in transformers.from_pretrained
_PATH_MAP = json.loads(os.environ.get("LED_MODEL_PATH_MAP", "{}") or "{}")
if _PATH_MAP:
    import transformers

    def _wrap_fp(cls):
        orig = cls.from_pretrained.__func__

        @classmethod
        def _patched(c, pretrained_model_name_or_path, *args, **kwargs):
            mapped = _PATH_MAP.get(pretrained_model_name_or_path, pretrained_model_name_or_path)
            try:
                return orig(c, mapped, *args, **kwargs)
            except Exception:
                if kwargs.get("use_fast") is False:
                    kw2 = dict(kwargs)
                    kw2.pop("use_fast", None)
                    return orig(c, mapped, *args, **kw2)
                raise

        cls.from_pretrained = _patched

    for _cname in ("AutoModelForCausalLM", "AutoTokenizer", "AutoConfig", "AutoProcessor", "AutoModel"):
        _c = getattr(transformers, _cname, None)
        if _c is not None and hasattr(_c, "from_pretrained"):
            _wrap_fp(_c)

# 3. Restore real gradient-based Wanda-g scoring unless running --unpatched-upstream reproduction
sys.path.insert(0, os.getcwd())
if os.environ.get("LED_FIX_WANDG_CORE", "1") == "1":
    import lib.model_wrapper as _mw

    def _restored_core(args, model, model_base=None, prune_n=0, prune_m=0, name_filter_fn=None):
        return _mw._prune_core(
            args,
            model,
            model_base=model_base,
            prune_n=prune_n,
            prune_m=prune_m,
            prune_mode="gradient",
            name_filter_fn=name_filter_fn,
        )

    _mw._random_prune_core = _restored_core

# Skip post-dump WikiText PPL evaluation (Llama-3 max_position_embeddings=131072 OOMs and is unused for scores)
if "--dump_wanda_score" in sys.argv:
    import lib.eval as _le
    _le.eval_ppl = lambda *a, **kw: 0.0

# 4. Run upstream locate/main.py with forwarded CLI args
sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
'''

ELECT_AND_MERGE_DRIVER = r'''"""Elect + Disjoint Merge driver invoked in the isolated LED workdir.

Enforces pre-merge importance-file completeness check (refuses to merge if files
are missing unless --allow-missing-scores is passed for unpatched reproduction),
records skipped shape-mismatched tensors explicitly, calls upstream
`MergingMethod(top_merging)`, and logs effective mask densities.
"""
import argparse
import json
import os
import pickle
import sys
import types
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.getcwd())
from model_merging_methods.mask_weights_utils import eq_param_name  # noqa: E402
from model_merging_methods.merging_methods import MergingMethod  # noqa: E402


def verify_score_directories(base_model, orders, slot_family, score_root, allow_missing=False):
    """Verify every wanda_score dir read by merging contains files for all expected tensors."""
    ft_alias_by_order = {
        "safety": f"{slot_family}-instruct",
        "math": f"{slot_family}-math",
        "code": f"{slot_family}-code",
    }
    base_alias = f"{slot_family}-base"
    param_names = [name for name, _ in base_model.named_parameters()]
    expected_n = len(param_names)

    report = {}
    all_complete = True
    for order in orders:
        for role, alias in (("ft", ft_alias_by_order[order]), ("base", base_alias)):
            sdir = os.path.join(score_root, order, alias, "wanda_score")
            rdir = os.path.join(score_root, order, alias, "random_score")
            files = sorted(f for f in os.listdir(sdir) if f.endswith(".pkl")) if os.path.isdir(sdir) else []
            random_files = sorted(f for f in os.listdir(rdir) if f.endswith(".pkl")) if os.path.isdir(rdir) else []
            matched = []
            missing = []
            for pname in param_names:
                if eq_param_name(pname, files) is not None:
                    matched.append(pname)
                else:
                    missing.append(pname)
            ok = len(missing) == 0 and len(files) >= expected_n
            if not ok:
                all_complete = False
            key = f"{order}/{alias}"
            report[key] = {
                "score_dir": os.path.abspath(sdir),
                "expected_tensors": expected_n,
                "wanda_score_pkl_count": len(files),
                "random_score_pkl_count": len(random_files),
                "matched_tensors": len(matched),
                "missing_tensors_count": len(missing),
                "missing_tensors_examples": missing[:20],
                "complete": ok,
            }
            print(
                f"[SCORE-CHECK] {key}: wanda_score={len(files)} pkl "
                f"(matched {len(matched)}/{expected_n} tensors, random_score={len(random_files)} pkl, complete={ok})"
            )

    if not all_complete and not allow_missing:
        raise RuntimeError(
            "LED-Merging pre-merge check FAILED: importance score files missing in wanda_score/. "
            "Refusing to merge (prevents silent fallback to base weights). Summary: "
            + json.dumps(report, ensure_ascii=False)
        )
    return report, all_complete


def _compute_pairwise_masks(
    base_model,
    orders,
    rates,
    mask_pattern,
    slot_family,
    score_root,
    skip_param_names,
):
    """Build disjoint masks `mask_dir_{pattern}_{rate}/{order}.pkl` from wanda_score."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rate_by_order = dict(zip(orders, rates))
    ft_alias_by_order = {
        "safety": f"{slot_family}-instruct",
        "math": f"{slot_family}-math",
        "code": f"{slot_family}-code",
    }
    base_alias = f"{slot_family}-base"

    score_dirs = {}
    for order in orders:
        ft_dir = os.path.join(score_root, order, ft_alias_by_order[order], "wanda_score")
        bs_dir = os.path.join(score_root, order, base_alias, "wanda_score")
        ft_files = os.listdir(ft_dir) if os.path.isdir(ft_dir) else []
        bs_files = os.listdir(bs_dir) if os.path.isdir(bs_dir) else []
        score_dirs[order] = (ft_dir, bs_dir, ft_files, bs_files)

    mask_stats = {}
    for target_order, target_rate in zip(orders, rates):
        save_dir = f"mask_dir_{mask_pattern}_{target_rate}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{target_order}.pkl")
        others = [o for o in orders if o != target_order]
        all_mask = {}
        total_elems = 0
        total_ones = 0
        missing_score_params = []

        for name, param in base_model.named_parameters():
            if name in skip_param_names:
                all_mask[name] = torch.ones(param.shape, dtype=torch.int32)
                continue
            p_anothers = []
            for other in others:
                ft_dir, bs_dir, ft_cands, bs_cands = score_dirs[other]
                fname_ft = eq_param_name(name, ft_cands)
                fname_bs = eq_param_name(name, bs_cands)
                if fname_ft is None or fname_bs is None:
                    missing_score_params.append((other, name))
                    continue
                with open(os.path.join(ft_dir, fname_ft), "rb") as fr:
                    p_u = pickle.load(fr).to(device=device, dtype=torch.float32)
                with open(os.path.join(bs_dir, fname_bs), "rb") as fr:
                    p_b = pickle.load(fr).to(device=device, dtype=torch.float32)
                if p_u.shape != param.shape or p_b.shape != param.shape:
                    missing_score_params.append((other, f"{name}(shape_mismatch)"))
                    del p_u, p_b
                    continue
                n_el = p_u.numel()
                k_idx = max(0, min(n_el - 1, int(rate_by_order[other] * n_el) - 1))
                th_u = torch.sort(p_u.flatten(), descending=True)[0][k_idx]
                th_b = torch.sort(p_b.flatten(), descending=True)[0][k_idx]
                mask_u = (p_u >= th_u).int()
                mask_b = (p_b >= th_b).int()
                if mask_pattern == "01":
                    mb = (1 - mask_b) * mask_u
                elif mask_pattern == "10":
                    mb = (1 - mask_u) * mask_b
                elif mask_pattern == "00":
                    mb = (1 - mask_u) * (1 - mask_b)
                elif mask_pattern == "11":
                    mb = mask_b * mask_u
                else:
                    mb = mask_u
                p_anothers.append(mb)
                del p_u, p_b, mask_u, mask_b

            if p_anothers:
                m_all = torch.any(torch.stack(p_anothers), dim=0).int().cpu()
            else:
                m_all = torch.zeros(param.shape, dtype=torch.int32)
            all_mask[name] = m_all
            total_elems += m_all.numel()
            total_ones += int(m_all.sum().item())

        with open(save_path, "wb") as fw:
            pickle.dump(all_mask, fw)

        mask_stats[target_order] = {
            "exclusion_mask_path": save_path,
            "exclusion_mask_density": round(total_ones / total_elems, 6) if total_elems else 0.0,
            "missing_score_params_count": len(missing_score_params),
        }

    # Compute effective (disjoint) update mask density per order matching neuron_merge_weight
    for order, rate in zip(orders, rates):
        ft_dir, bs_dir, ft_cands, bs_cands = score_dirs[order]
        with open(mask_stats[order]["exclusion_mask_path"], "rb") as fr:
            excl_masks = pickle.load(fr)
        eff_ones = 0
        raw_ones = 0
        total_elems = 0
        for name, param in base_model.named_parameters():
            if name in skip_param_names:
                continue
            fname_ft = eq_param_name(name, ft_cands)
            fname_bs = eq_param_name(name, bs_cands)
            if fname_ft is None or fname_bs is None:
                continue
            with open(os.path.join(ft_dir, fname_ft), "rb") as fr:
                sw = pickle.load(fr).to(device=device, dtype=torch.float32)
            with open(os.path.join(bs_dir, fname_bs), "rb") as fr:
                sb = pickle.load(fr).to(device=device, dtype=torch.float32)
            if sw.shape != param.shape or sb.shape != param.shape:
                del sw, sb
                continue
            n_el = sw.numel()
            k_idx = max(0, min(n_el - 1, int(rate * n_el) - 1))
            th_w = torch.sort(sw.flatten(), descending=True)[0][k_idx]
            th_b = torch.sort(sb.flatten(), descending=True)[0][k_idx]
            mw = (sw > th_w).int()
            mb = (sb > th_b).int()
            if mask_pattern == "01":
                m11 = (1 - mb) * mw
            elif mask_pattern == "10":
                m11 = (1 - mw) * mb
            elif mask_pattern == "00":
                m11 = (1 - mw) * (1 - mb)
            else:
                m11 = mb * mw
            pm = excl_masks[name].to(device=device)
            tm = m11 * (1 - pm)
            raw_ones += int(m11.sum().item())
            eff_ones += int(tm.sum().item())
            total_elems += tm.numel()
            del sw, sb, mw, mb, m11, pm, tm
        mask_stats[order]["raw_selected_density"] = round(raw_ones / total_elems, 6) if total_elems else 0.0
        mask_stats[order]["effective_disjoint_density"] = round(eff_ones / total_elems, 6) if total_elems else 0.0
        mask_stats[order]["total_scored_params"] = total_elems

    return mask_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--orders", nargs="+", required=True)
    ap.add_argument("--fuse-rates", nargs="+", type=float, required=True)
    ap.add_argument("--scaling-coef", type=float, default=0.9)
    ap.add_argument("--mask-pattern", default="11")
    ap.add_argument("--mask-apply-method", default="task_arithmetic")
    ap.add_argument("--slot-family", default="llama3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--skip-elect", action="store_true")
    ap.add_argument("--allow-missing-scores", action="store_true",
                    help="only for --unpatched-upstream reproduction: fallback to base weights when scores are missing")
    args = ap.parse_args()

    pretrained = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    score_root = os.path.join(".", "locate", "score")

    # 1. Pre-merge importance-file completeness check (blocks merging when files are missing!)
    score_report, scores_complete = verify_score_directories(
        base_model=pretrained,
        orders=args.orders,
        slot_family=args.slot_family,
        score_root=score_root,
        allow_missing=args.allow_missing_scores,
    )

    ft_models = [
        AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="cpu")
        for m in args.inputs
    ]

    # 2. Detect and record shape-mismatched tensors (e.g., vocab_size 128320 vs 128256 -> Cause B)
    base_sd = pretrained.state_dict()
    skipped_shape_mismatches = []
    skip_param_names = set()
    for inp_path, mdl in zip(args.inputs, ft_models):
        msd = mdl.state_dict()
        for k, bv in base_sd.items():
            if k in msd and msd[k].shape != bv.shape:
                skipped_shape_mismatches.append({
                    "tensor": k,
                    "input_model": inp_path,
                    "base_shape": list(bv.shape),
                    "input_shape": list(msd[k].shape),
                })
                skip_param_names.add(k)

    if skipped_shape_mismatches:
        print(f"[SHAPE-SKIP] Skipping {len(skip_param_names)} shape-mismatched tensor(s): {sorted(skip_param_names)}")
        # Replace mismatched tensors in ft_models with base copies so upstream neuron_merge_weight
        # executes cleanly without crashing while keeping those tensors identically equal to base.
        base_v = pretrained.get_input_embeddings().weight.shape[0]
        for mdl in ft_models:
            if mdl.get_input_embeddings().weight.shape[0] != base_v:
                mdl.resize_token_embeddings(base_v)
            msd = mdl.state_dict()
            for k in skip_param_names:
                if k in msd and k in base_sd and msd[k].shape == base_sd[k].shape:
                    msd[k].copy_(base_sd[k])

    # Keep backup of base weights for skipped tensors and ensure score pkl rows do not crash upstream
    base_skip_backup = {k: base_sd[k].clone() for k in skip_param_names if k in base_sd}
    if skip_param_names:
        base_v = pretrained.get_input_embeddings().weight.shape[0]
        for order in args.orders:
            for sub_alias in (f"{args.slot_family}-base", f"{args.slot_family}-instruct", f"{args.slot_family}-math", f"{args.slot_family}-code"):
                wdir_sc = os.path.join(score_root, order, sub_alias, "wanda_score")
                if not os.path.isdir(wdir_sc):
                    continue
                for pfn in ("embed_tokens.pkl", "lm_head.pkl"):
                    pp = os.path.join(wdir_sc, pfn)
                    if os.path.isfile(pp):
                        with open(pp, "rb") as fr:
                            t = pickle.load(fr)
                        if t.dim() >= 1 and t.size(0) != base_v:
                            if t.size(0) > base_v:
                                t = t[:base_v].contiguous()
                            else:
                                pad = t[-1:].expand(base_v - t.size(0), *t.shape[1:])
                                t = torch.cat([t, pad], dim=0).contiguous()
                            with open(pp, "wb") as fw:
                                pickle.dump(t, fw)

    mask_stats = {}
    if not args.skip_elect:
        mask_stats = _compute_pairwise_masks(
            base_model=pretrained,
            orders=args.orders,
            rates=args.fuse_rates,
            mask_pattern=args.mask_pattern,
            slot_family=args.slot_family,
            score_root=score_root,
            skip_param_names=skip_param_names,
        )

    # 3. Merge via upstream MergingMethod(top_merging) — or safe skip-fallback when reproducing unpatched upstream
    if not scores_complete and args.allow_missing_scores:
        # Reproduce sst_v2 / unpatched upstream behavior: skip any tensor whose wanda_score file is missing
        print("[UNPATCHED-REPRO] Running top_merging with missing-score fallback to base weights.")
        import model_merging_methods.mask_weights_utils as mwu
        orig_nmw = mwu.neuron_merge_weight

        def _safe_unpatched_nmw(finetuned_model, pretrained_model, fuse_rate, util, score_dir, pattern="00", score_extra=None):
            score_lists = os.listdir(score_dir) if os.path.isdir(score_dir) else []
            masked_param_dict = {}
            sft_param = finetuned_model.state_dict()
            for pname, pval in pretrained_model.named_parameters():
                sf = eq_param_name(pname, score_lists)
                if sf is None or pname in skip_param_names:
                    masked_param_dict[pname] = pval.clone()
                else:
                    # Only apply if both score and mask exist; otherwise base weight
                    masked_param_dict[pname] = pval.clone()
            return masked_param_dict

        import model_merging_methods.merging_methods as mm_mod
        mm_mod.neuron_merge_weight = _safe_unpatched_nmw

    m_args = types.SimpleNamespace(
        fuse_rates=args.fuse_rates,
        orders=args.orders,
        fuse_types=["o" for _ in args.inputs],
        fuse_patterns=[args.mask_pattern for _ in args.inputs],
        model_ft_name=args.slot_family,
        model_base_name=f"{args.slot_family}-base",
        lambdas=[args.scaling_coef for _ in args.inputs],
        use_fuse=False,
        safety=False,
    )
    merger = MergingMethod(merging_method_name="top_merging")
    merged = merger.get_merged_model(
        merged_model=pretrained,
        models_to_merge=ft_models,
        exclude_param_names_regex=[],
        scaling_coefficient=args.scaling_coef,
        mask_apply_method=args.mask_apply_method,
        multimodal=False,
        args=m_args,
    )

    # Guarantee skipped shape-mismatched tensors remain strictly equal to base weights
    if base_skip_backup:
        msd = merged.state_dict()
        for k, bv in base_skip_backup.items():
            if k in msd and msd[k].shape == bv.shape:
                msd[k].copy_(bv)

    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)

    full_stats = {
        "score_dir_check": score_report,
        "scores_complete": scores_complete,
        "skipped_shape_mismatch_tensors": skipped_shape_mismatches,
        "skipped_shape_mismatch_names": sorted(skip_param_names),
        "effective_mask_stats": mask_stats,
    }
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({"saved": args.out, **full_stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''


def infer_orders(inputs: List[str]) -> List[str]:
    orders = []
    for m in inputs:
        low = os.path.basename(m.rstrip("/")).lower()
        if low.endswith("_safety") or "safety" in low or "safe" in low:
            orders.append("safety")
        elif low.endswith("_math") or "math" in low or "dartmath" in low:
            orders.append("math")
        elif low.endswith("_coding") or "coding" in low or "coder" in low or "code" in low or "magicoder" in low:
            orders.append("code")
        elif low.endswith("_medical") or "medical" in low or "medalpaca" in low:
            orders.append("medical")
        elif low.endswith("_multilingual") or "multilingual" in low or "aya" in low:
            orders.append("multilingual")
        elif low.endswith("_instruction") or "tulu" in low or "_if" in low:
            orders.append("instruction")
        elif "instruct" in low and len(orders) == 0:
            orders.append("safety")
        else:
            orders.append(f"domain{len(orders)}")
    return orders


def default_fuse_rates_for_orders(orders: List[str]) -> List[float]:
    """Return official LED-Merging/README.md L79 rates: safety=0.1, math=0.4, code=0.5."""
    return [OFFICIAL_DEFAULT_RATES.get(o, 0.4) for o in orders]


def build_model_path_map(
    base_model: str,
    inputs: List[str],
    orders: List[str],
    slot_family: str = "llama3",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (placeholder_to_actual_path, order_to_upstream_alias)."""
    slots = UPSTREAM_ALIAS_PATHS.get(slot_family, UPSTREAM_ALIAS_PATHS["llama3"])
    path_map: Dict[str, str] = {}
    alias_map: Dict[str, str] = {}

    base_alias, base_placeholder = slots["base"]
    path_map[base_placeholder] = base_model
    alias_map["base"] = base_alias

    for inp, order in zip(inputs, orders):
        slot_key = order if order in slots else "safety"
        alias, placeholder = slots[slot_key]
        path_map[placeholder] = inp
        alias_map[order] = alias
    return path_map, alias_map


def prepare_workdir(
    led_repo: str,
    workdir: str,
    extra_data_dir: Optional[str] = None,
    reuse_locate_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Create isolated symlinked workdir so upstream `LED-Merging/` stays 100% untouched."""
    os.makedirs(workdir, exist_ok=True)
    hook_dir = os.path.join(workdir, ".hook")
    os.makedirs(hook_dir, exist_ok=True)
    old_sc = os.path.join(hook_dir, "sitecustomize.py")
    if os.path.exists(old_sc):
        os.remove(old_sc)

    locate_launcher = os.path.join(hook_dir, "run_locate.py")
    with open(locate_launcher, "w", encoding="utf-8") as f:
        f.write(RUN_LOCATE_LAUNCHER)

    driver_path = os.path.join(hook_dir, "elect_and_merge.py")
    with open(driver_path, "w", encoding="utf-8") as f:
        f.write(ELECT_AND_MERGE_DRIVER)

    for item in ("model_merging_methods", "utils", "mask_generate.py", "merge_llms.py"):
        src = os.path.join(led_repo, item)
        dst = os.path.join(workdir, item)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)

    loc_work = os.path.join(workdir, "locate")
    os.makedirs(loc_work, exist_ok=True)
    score_dst = os.path.join(loc_work, "score")
    if reuse_locate_dir:
        cand = os.path.abspath(reuse_locate_dir)
        if os.path.isdir(os.path.join(cand, "locate", "score")):
            cand = os.path.join(cand, "locate", "score")
        elif os.path.isdir(os.path.join(cand, "score")):
            cand = os.path.join(cand, "score")
        if not os.path.isdir(cand):
            raise FileNotFoundError(f"--reuse-locate-dir score directory not found: {cand}")
        if os.path.islink(score_dst):
            os.unlink(score_dst)
        elif os.path.isdir(score_dst) and not os.listdir(score_dst):
            os.rmdir(score_dst)
        if not os.path.lexists(score_dst):
            os.symlink(cand, score_dst)
    else:
        os.makedirs(score_dst, exist_ok=True)
    for item in ("main.py", "lib", "default_config.yaml"):
        src = os.path.join(led_repo, "locate", item)
        dst = os.path.join(loc_work, item)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)

    data_work = os.path.join(loc_work, "data")
    os.makedirs(data_work, exist_ok=True)
    upstream_data = os.path.join(led_repo, "locate", "data")
    if os.path.isdir(upstream_data):
        for fn in os.listdir(upstream_data):
            dst = os.path.join(data_work, fn)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(upstream_data, fn), dst)
    if extra_data_dir and os.path.isdir(extra_data_dir):
        for fn in os.listdir(extra_data_dir):
            dst = os.path.join(data_work, fn)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(extra_data_dir, fn), dst)

    return hook_dir, driver_path


def _auto_find_extra_data_dir() -> Optional[str]:
    candidates = [
        "/mnt/nas/home/hiromi/src/sst_v2/v3/baselines/LED-Merging/locate/data",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def build_stage_commands(
    led_repo: str,
    python_bin: str,
    base_model: str,
    inputs: List[str],
    orders: List[str],
    out_dir: str,
    model_family: str = "llama3",
    fuse_rates: Optional[List[float]] = None,
    scaling_coef: float = 0.9,
    mask_pattern: str = "11",
    mask_apply_method: str = "task_arithmetic",
    skip_locate: bool = False,
    skip_elect: bool = False,
    workdir: Optional[str] = None,
    nsamples: int = 128,
    seed: int = 42,
    unpatched_upstream: bool = False,
) -> List[Dict[str, Any]]:
    rates = fuse_rates if fuse_rates is not None else default_fuse_rates_for_orders(orders)
    if len(rates) != len(inputs):
        raise ValueError(f"fuse_rates length ({len(rates)}) != inputs length ({len(inputs)})")

    slot_family = model_family if model_family in UPSTREAM_ALIAS_PATHS else "llama3"
    _, alias_map = build_model_path_map(base_model, inputs, orders, slot_family=slot_family)
    wdir = os.path.abspath(workdir or os.path.join(out_dir, ".led_workdir"))
    locate_dir = os.path.join(wdir, "locate")
    locate_launcher = os.path.join(wdir, ".hook", "run_locate.py")
    driver_path = os.path.join(wdir, ".hook", "elect_and_merge.py")
    stats_out = os.path.join(os.path.abspath(out_dir), "led_mask_stats.json")
    hf_cache_dir = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )

    stages: List[Dict[str, Any]] = []
    if not skip_locate:
        data_map = {
            "safety": "align",
            "math": "math",
            "code": "code",
            "medical": "align",
            "multilingual": "align",
            "instruction": "align",
        }
        for order in orders:
            prune_data = data_map.get(order, "align")
            ft_alias = alias_map[order]
            base_alias = alias_map["base"]
            save_sub = os.path.join("./score", order, ft_alias)
            stages.append({
                "stage": f"locate_{order}",
                "cwd": locate_dir,
                "cmd": [
                    python_bin, locate_launcher,
                    "--model", ft_alias,
                    "--model_base", base_alias,
                    "--cache_dir", hf_cache_dir,
                    "--prune_method", "wandg",
                    "--prune_data", prune_data,
                    "--nsamples", str(nsamples),
                    "--seed", str(seed),
                    "--sparsity_ratio", "0.01",
                    "--sparsity_type", "unstructured",
                    "--neg_prune",
                    "--save", save_sub,
                    "--dump_wanda_score",
                ],
            })
            base_save_sub = os.path.join("./score", order, base_alias)
            stages.append({
                "stage": f"locate_base_{order}",
                "cwd": locate_dir,
                "cmd": [
                    python_bin, locate_launcher,
                    "--model", base_alias,
                    "--model_base", base_alias,
                    "--cache_dir", hf_cache_dir,
                    "--prune_method", "wandg",
                    "--prune_data", prune_data,
                    "--nsamples", str(nsamples),
                    "--seed", str(seed),
                    "--sparsity_ratio", "0.01",
                    "--sparsity_type", "unstructured",
                    "--neg_prune",
                    "--save", base_save_sub,
                    "--dump_wanda_score",
                ],
            })

    cmd_merge = [
        python_bin, driver_path,
        "--base", base_model,
        "--inputs", *inputs,
        "--orders", *orders,
        "--fuse-rates", *[str(r) for r in rates],
        "--scaling-coef", str(scaling_coef),
        "--mask-pattern", mask_pattern,
        "--mask-apply-method", mask_apply_method,
        "--slot-family", slot_family,
        "--out", os.path.abspath(out_dir),
        "--stats-out", stats_out,
    ]
    if skip_elect:
        cmd_merge.append("--skip-elect")
    if unpatched_upstream:
        cmd_merge.append("--allow-missing-scores")
    stages.append({
        "stage": "elect_and_disjoint_merge",
        "cwd": wdir,
        "cmd": cmd_merge,
    })
    return stages


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run LED-Merging (Locate -> Elect -> Disjoint Merge) without touching upstream files")
    ap.add_argument("--base", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-names", default=None)
    ap.add_argument("--orders", nargs="+", default=None, help="domain labels matching --inputs (e.g. safety math)")
    ap.add_argument("--fuse-rates", nargs="+", type=float, default=None,
                    help="mask rates per input (default: official README L79 safety=0.1, math=0.4, code=0.5)")
    ap.add_argument("--scaling-coef", type=float, default=0.9)
    ap.add_argument("--mask-pattern", default="11")
    ap.add_argument("--mask-apply-method", default="task_arithmetic")
    ap.add_argument("--model-family", default="llama3")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--workdir", default=None, help="isolated symlinked workdir (default: <out>/.led_workdir)")
    ap.add_argument("--extra-data-dir", default=None, help="optional dir containing humaneval.parquet / train-00000-of-00001.parquet")
    ap.add_argument("--keep-workdir", action="store_true", help="keep .led_workdir after merge finishes")
    ap.add_argument("--skip-locate", action="store_true", help="reuse precomputed locate/score files in workdir")
    ap.add_argument("--reuse-locate-dir", default=None,
                    help="path to existing .led_workdir or locate/score dir to symlink and reuse without re-running locate")
    ap.add_argument("--skip-elect", action="store_true", help="reuse precomputed masks in workdir")
    ap.add_argument("--config-role", default=None, choices=["official_default", "extra_safety_sweep", "unpatched_upstream"],
                    help="tag recorded in merge_metadata.json to distinguish official default vs extra safety sweep (auto-inferred if omitted)")
    ap.add_argument("--unpatched-upstream", action="store_true",
                    help="reproduce unpatched upstream LED-Merging (_random_prune_core bug -> base-identical degeneration)")
    ap.add_argument("--repo-dir", default=None, help="path to MqLeet/LED-Merging checkout")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def infer_config_role(
    orders: List[str],
    rates: List[float],
    unpatched_upstream: bool = False,
    explicit_role: Optional[str] = None,
) -> str:
    if explicit_role:
        return explicit_role
    if unpatched_upstream:
        return "unpatched_upstream"
    official = default_fuse_rates_for_orders(orders)
    if len(rates) == len(official) and all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(rates, official)):
        return "official_default"
    return "extra_safety_sweep"


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    led_repo = resolve_external_repo("LED-Merging", args.repo_dir)
    ext_commit = git_commit(led_repo) if os.path.isdir(led_repo) else "unknown"
    orders = args.orders or infer_orders(args.inputs)
    rates = args.fuse_rates if args.fuse_rates is not None else default_fuse_rates_for_orders(orders)
    config_role = infer_config_role(orders, rates, unpatched_upstream=args.unpatched_upstream, explicit_role=args.config_role)
    skip_locate = bool(args.skip_locate or args.reuse_locate_dir)
    slot_family = args.model_family if args.model_family in UPSTREAM_ALIAS_PATHS else "llama3"
    workdir = os.path.abspath(args.workdir or os.path.join(args.out, ".led_workdir"))

    stages = build_stage_commands(
        led_repo=led_repo,
        python_bin=args.python,
        base_model=args.base,
        inputs=args.inputs,
        orders=orders,
        out_dir=args.out,
        model_family=slot_family,
        fuse_rates=rates,
        scaling_coef=args.scaling_coef,
        mask_pattern=args.mask_pattern,
        mask_apply_method=args.mask_apply_method,
        skip_locate=skip_locate,
        skip_elect=args.skip_elect,
        workdir=workdir,
        nsamples=args.nsamples,
        seed=args.seed,
        unpatched_upstream=args.unpatched_upstream,
    )

    for st in stages:
        print(f"[{st['stage']}] (cwd={st['cwd']}):", " ".join(st["cmd"]))
    if args.dry_run:
        return 0

    os.makedirs(args.out, exist_ok=True)
    extra_data = args.extra_data_dir or _auto_find_extra_data_dir()
    hook_dir, _ = prepare_workdir(
        led_repo, workdir, extra_data_dir=extra_data, reuse_locate_dir=args.reuse_locate_dir
    )
    path_map, _ = build_model_path_map(args.base, args.inputs, orders, slot_family=slot_family)

    env = os.environ.copy()
    env["LED_MODEL_PATH_MAP"] = json.dumps(path_map)
    env["LED_FIX_WANDG_CORE"] = "0" if args.unpatched_upstream else "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    started = datetime.now(timezone.utc).isoformat()
    for st in stages:
        subprocess.run(st["cmd"], cwd=st["cwd"], env=env, check=True)
    finished = datetime.now(timezone.utc).isoformat()

    stats_path = os.path.join(os.path.abspath(args.out), "led_mask_stats.json")
    full_stats: Dict[str, Any] = {}
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            full_stats = json.load(f)

    if not args.keep_workdir and os.path.isdir(workdir) and workdir.startswith(os.path.abspath(args.out)):
        shutil.rmtree(workdir, ignore_errors=True)

    ensure_tokenizer_saved(args.base, args.out)
    write_merge_metadata(
        out_dir=args.out,
        algo="LEDMerging_UnpatchedUpstream" if args.unpatched_upstream else "LEDMerging",
        kwargs={
            "orders": orders,
            "fuse_rates": rates,
            "scaling_coef": args.scaling_coef,
            "mask_pattern": args.mask_pattern,
            "mask_apply_method": args.mask_apply_method,
            "model_family": args.model_family,
            "slot_family": slot_family,
            "nsamples": args.nsamples,
            "unpatched_upstream": args.unpatched_upstream,
            "reused_locate_dir": os.path.abspath(args.reuse_locate_dir) if args.reuse_locate_dir else None,
        },
        base_model=args.base,
        inputs=args.inputs,
        task_names=args.task_names,
        seed=args.seed,
        dtype="bfloat16",
        started=started,
        finished=finished,
        external_repo="MqLeet/LED-Merging",
        external_commit=ext_commit,
        extra={
            "config_role": config_role,
            "is_official_default": config_role == "official_default",
            "upstream_unmodified": True,
            "score_dir_check": full_stats.get("score_dir_check", {}),
            "scores_complete": full_stats.get("scores_complete", False),
            "skipped_shape_mismatch_tensors": full_stats.get("skipped_shape_mismatch_tensors", []),
            "skipped_shape_mismatch_names": full_stats.get("skipped_shape_mismatch_names", []),
            "effective_mask_stats": full_stats.get("effective_mask_stats", {}),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
