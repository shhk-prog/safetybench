"""LED-Merging wrapper via official MqLeet/LED-Merging (ACL 2025 Main).

Runs the 3 official stages in order:
  1. Locate: compute neuron importance scores (`locate/main.py`) in float32-safe mode
  2. Elect : generate pairwise disjoint masks (`mask_generate.py`)
  3. Disjoint Merge: apply masks and merge (`merge_llms.py`)

Also records effective mask density statistics into `merge_metadata.json` to
satisfy SafeMergeBench's P5 integrity transparency requirement.

Usage:
  python safemerge/external/run_led_merging.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --inputs MergeBench/Llama-3.1-8B-Instruct_safety MergeBench/Llama-3.1-8B-Instruct_math \
    --orders safety math \
    --out merged_models/.../LEDMerging_... \
    --seed 42
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ensure_tokenizer_saved,
    git_commit,
    resolve_external_repo,
    set_seed,
    write_merge_metadata,
)


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
) -> List[Dict[str, Any]]:
    rates = fuse_rates or [0.5 for _ in inputs]
    if len(rates) != len(inputs):
        raise ValueError(f"fuse_rates length ({len(rates)}) != inputs length ({len(inputs)})")

    stages: List[Dict[str, Any]] = []
    locate_dir = os.path.join(led_repo, "locate")

    if not skip_locate:
        data_map = {
            "safety": "align",
            "math": "math",
            "code": "code",
            "medical": "medical",
            "multilingual": "align",
            "instruction": "align",
        }
        for model_path, order in zip(inputs, orders):
            prune_data = data_map.get(order, "align")
            save_sub = os.path.join("./score", order, f"{model_family}-{order}")
            stages.append({
                "stage": f"locate_{order}",
                "cwd": locate_dir,
                "cmd": [
                    python_bin, "main.py",
                    "--model", model_path,
                    "--model_base", base_model,
                    "--prune_method", "wandg",
                    "--prune_data", prune_data,
                    "--sparsity_ratio", "0.01",
                    "--sparsity_type", "unstructured",
                    "--neg_prune",
                    "--save", save_sub,
                    "--dump_wanda_score",
                ],
            })
            base_save_sub = os.path.join("./score", order, f"{model_family}-base")
            stages.append({
                "stage": f"locate_base_{order}",
                "cwd": locate_dir,
                "cmd": [
                    python_bin, "main.py",
                    "--model", base_model,
                    "--model_base", base_model,
                    "--prune_method", "wandg",
                    "--prune_data", prune_data,
                    "--sparsity_ratio", "0.01",
                    "--sparsity_type", "unstructured",
                    "--neg_prune",
                    "--save", base_save_sub,
                    "--dump_wanda_score",
                ],
            })

    if not skip_elect:
        rate_args = [str(r) for r in rates]
        while len(rate_args) < 3:
            rate_args.append(str(rates[-1]))
        stages.append({
            "stage": "elect",
            "cwd": led_repo,
            "cmd": [python_bin, "mask_generate.py", *rate_args[:3], mask_pattern, model_family],
        })

    stages.append({
        "stage": "disjoint_merge",
        "cwd": led_repo,
        "cmd": [
            python_bin, "merge_llms.py",
            "--models_to_merge", *inputs,
            "--pretrained_model_name", base_model,
            "--merging_method_name", "top_merging",
            "--scaling_coefficient", str(scaling_coef),
            "--mask_apply_method", mask_apply_method,
            "--fuse_rates", *[str(r) for r in rates],
            "--orders", *orders,
            "--fuse_types", *["o" for _ in inputs],
            "--fuse_patterns", *[mask_pattern for _ in inputs],
            "--model_ft_name", model_family,
            "--model_base_name", f"{model_family}-base",
            "--save", os.path.abspath(out_dir),
        ],
    })
    return stages


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run LED-Merging (Locate -> Elect -> Disjoint Merge)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-names", default=None)
    ap.add_argument("--orders", nargs="+", default=None, help="domain labels matching --inputs (e.g. safety math)")
    ap.add_argument("--fuse-rates", nargs="+", type=float, default=None, help="mask rates per input")
    ap.add_argument("--scaling-coef", type=float, default=0.9)
    ap.add_argument("--mask-pattern", default="11")
    ap.add_argument("--mask-apply-method", default="task_arithmetic")
    ap.add_argument("--model-family", default="llama3")
    ap.add_argument("--skip-locate", action="store_true", help="reuse precomputed locate/score files")
    ap.add_argument("--skip-elect", action="store_true", help="reuse precomputed masks")
    ap.add_argument("--repo-dir", default=None, help="path to MqLeet/LED-Merging checkout")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    led_repo = resolve_external_repo("LED-Merging", args.repo_dir)
    ext_commit = git_commit(led_repo) if os.path.isdir(led_repo) else "unknown"
    orders = args.orders or infer_orders(args.inputs)

    stages = build_stage_commands(
        led_repo=led_repo,
        python_bin=args.python,
        base_model=args.base,
        inputs=args.inputs,
        orders=orders,
        out_dir=args.out,
        model_family=args.model_family,
        fuse_rates=args.fuse_rates,
        scaling_coef=args.scaling_coef,
        mask_pattern=args.mask_pattern,
        mask_apply_method=args.mask_apply_method,
        skip_locate=args.skip_locate,
        skip_elect=args.skip_elect,
    )

    for st in stages:
        print(f"[{st['stage']}] (cwd={st['cwd']}):", " ".join(st["cmd"]))
    if args.dry_run:
        return 0

    os.makedirs(args.out, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    for st in stages:
        subprocess.run(st["cmd"], cwd=st["cwd"], check=True)
    finished = datetime.now(timezone.utc).isoformat()

    ensure_tokenizer_saved(args.inputs[0], args.out)
    write_merge_metadata(
        out_dir=args.out,
        algo="LEDMerging",
        kwargs={
            "orders": orders,
            "fuse_rates": args.fuse_rates or [0.5 for _ in args.inputs],
            "scaling_coef": args.scaling_coef,
            "mask_pattern": args.mask_pattern,
            "mask_apply_method": args.mask_apply_method,
            "model_family": args.model_family,
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
