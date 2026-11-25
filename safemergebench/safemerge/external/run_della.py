"""DELLA merging wrapper via official arcee-ai/mergekit (`merge_method: della`).

Usage:
  python safemerge/external/run_della.py \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --inputs MergeBench/Llama-3.1-8B-Instruct_safety MergeBench/Llama-3.1-8B-Instruct_math \
    --out merged_models/Llama-3.1-8B-Instruct_merged/DELLA_... \
    --alpha 0.6 --density 0.5 --epsilon 0.05 --lambda-coef 1.0 --seed 42
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ensure_tokenizer_saved, set_seed, write_merge_metadata  # noqa: E402


def build_della_config(
    base_model: str,
    inputs: List[str],
    weights: Optional[List[float]] = None,
    alpha: Optional[float] = None,
    density: float = 0.5,
    epsilon: float = 0.05,
    lambda_coef: float = 1.0,
    dtype: str = "bfloat16",
    normalize: bool = False,
) -> Dict[str, Any]:
    if weights is not None:
        if len(weights) != len(inputs):
            raise ValueError(f"weights length ({len(weights)}) != inputs length ({len(inputs)})")
        model_weights = [float(w) for w in weights]
    elif alpha is not None:
        # Match MergeBench scaling-coef convention: uniform scaling `alpha` on task vectors
        model_weights = [float(alpha) for _ in inputs]
    else:
        model_weights = [1.0 / len(inputs) for _ in inputs]

    models_entries = []
    for m, w in zip(inputs, model_weights):
        models_entries.append({
            "model": m,
            "parameters": {
                "weight": w,
                "density": float(density),
                "epsilon": float(epsilon),
            },
        })

    return {
        "models": models_entries,
        "merge_method": "della",
        "base_model": base_model,
        "parameters": {
            "lambda": float(lambda_coef),
            "normalize": bool(normalize),
            "int8_mask": False,
        },
        "dtype": dtype,
        "tokenizer_source": inputs[0] if inputs else base_model,
    }


def dump_yaml(cfg: Dict[str, Any], path: str) -> None:
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    except ImportError:
        # Minimal YAML serializer fallback when PyYAML is not installed
        lines = ["models:"]
        for entry in cfg["models"]:
            lines.append(f"  - model: {entry['model']}")
            lines.append("    parameters:")
            for pk, pv in entry["parameters"].items():
                lines.append(f"      {pk}: {pv}")
        lines.append(f"merge_method: {cfg['merge_method']}")
        lines.append(f"base_model: {cfg['base_model']}")
        lines.append("parameters:")
        for pk, pv in cfg["parameters"].items():
            val = str(pv).lower() if isinstance(pv, bool) else pv
            lines.append(f"  {pk}: {val}")
        lines.append(f"dtype: {cfg['dtype']}")
        if "tokenizer_source" in cfg:
            lines.append(f"tokenizer_source: {cfg['tokenizer_source']}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def build_command(
    yaml_path: str,
    out_dir: str,
    seed: int = 42,
    cuda: bool = True,
    copy_tokenizer: bool = True,
    mergekit_bin: str = "mergekit-yaml",
) -> List[str]:
    cmd = [mergekit_bin, yaml_path, out_dir, "--random-seed", str(seed)]
    if cuda:
        cmd.append("--cuda")
    if copy_tokenizer:
        cmd.append("--copy-tokenizer")
    return cmd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run DELLA merge via mergekit-yaml")
    ap.add_argument("--base", required=True, help="base model repo id or path")
    ap.add_argument("--inputs", nargs="+", required=True, help="expert models (e.g. safety + domain)")
    ap.add_argument("--out", required=True, help="output directory for merged model")
    ap.add_argument("--task-names", default=None, help="task names tag for metadata")
    ap.add_argument("--alpha", type=float, default=None, help="scaling coefficient applied to each input")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="explicit per-input weights")
    ap.add_argument("--density", type=float, default=0.5, help="DELLA density parameter")
    ap.add_argument("--epsilon", type=float, default=0.05, help="DELLA epsilon parameter")
    ap.add_argument("--lambda-coef", dest="lambda_coef", type=float, default=1.0, help="DELLA lambda parameter")
    ap.add_argument("--normalize", action="store_true", help="normalize weights in mergekit")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--mergekit-bin", default="mergekit-yaml")
    ap.add_argument("--dry-run", action="store_true", help="write YAML and print command without executing")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = build_della_config(
        base_model=args.base,
        inputs=args.inputs,
        weights=args.weights,
        alpha=args.alpha,
        density=args.density,
        epsilon=args.epsilon,
        lambda_coef=args.lambda_coef,
        dtype=args.dtype,
        normalize=args.normalize,
    )
    yaml_path = os.path.join(args.out, "della_mergekit.yml")
    dump_yaml(cfg, yaml_path)

    cmd = build_command(
        yaml_path=yaml_path,
        out_dir=args.out,
        seed=args.seed,
        cuda=not args.no_cuda,
        mergekit_bin=args.mergekit_bin,
    )
    print("Command:", " ".join(cmd))
    if args.dry_run:
        return 0

    if shutil.which(args.mergekit_bin) is None:
        raise FileNotFoundError(
            f"'{args.mergekit_bin}' not found in PATH. Install arcee-ai/mergekit in the merging environment."
        )

    started = datetime.now(timezone.utc).isoformat()
    subprocess.run(cmd, check=True)
    finished = datetime.now(timezone.utc).isoformat()

    ensure_tokenizer_saved(args.inputs[0], args.out)
    write_merge_metadata(
        out_dir=args.out,
        algo="DELLA",
        kwargs={
            "alpha": args.alpha,
            "weights": args.weights,
            "density": args.density,
            "epsilon": args.epsilon,
            "lambda": args.lambda_coef,
            "normalize": args.normalize,
        },
        base_model=args.base,
        inputs=args.inputs,
        task_names=args.task_names,
        seed=args.seed,
        dtype=args.dtype,
        started=started,
        finished=finished,
        external_repo="arcee-ai/mergekit",
        extra={"mergekit_config": yaml_path},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
