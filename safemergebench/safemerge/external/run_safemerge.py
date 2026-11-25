"""SafeMERGE wrapper via official aladinD/SafeMERGE (ACL 2026 Findings).

Real-world track ONLY (requires LoRA adapters; cannot merge full-weight SFT
MergeBench checkpoints). Invokes SafeMERGE's `SafeLoRAMerger`, exports the
merged model + tokenizer, and records `layers_merged` and `layers_skipped` in
`merge_metadata.json` (addresses H3 / P4 / P5 transparency).

Usage:
  python safemerge/external/run_safemerge.py \
    --base meta-llama/Llama-2-7b-chat-hf \
    --inputs path/to/safety_lora path/to/domain_lora \
    --safelora-unaligned meta-llama/Llama-2-7b-hf \
    --safelora-aligned meta-llama/Llama-2-7b-chat-hf \
    --out merged_models/realworld/SafeMERGE_... --seed 42
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import re
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


def parse_safemerge_layer_logs(stdout_text: str) -> Tuple[List[str], List[str]]:
    """Extract merged vs skipped layer names from SafeMERGE stdout/logs."""
    merged: List[str] = []
    skipped: List[str] = []
    for line in stdout_text.splitlines():
        low = line.lower()
        # SafeMERGE utils.py prints layer cosine similarities and merge/skip decisions
        m_layer = re.search(r"(model\.layers\.\d+\.[\w\.]+|layers\.\d+\.[\w\.]+)", line)
        layer_name = m_layer.group(1) if m_layer else None
        if not layer_name:
            continue
        if "skip" in low or "keeping finetuned" in low or "no merge" in low or "non-2d" in low:
            if layer_name not in skipped:
                skipped.append(layer_name)
        elif "merg" in low or "partial" in low:
            if layer_name not in merged:
                merged.append(layer_name)
    return merged, skipped


def build_command(
    repo_dir: str,
    python_bin: str,
    base_model: str,
    safety_lora: str,
    domain_lora: str,
    unaligned_model: str,
    aligned_model: str,
    out_dir: str,
    cos_threshold: float = 0.35,
    default_merge_ratio: Optional[float] = 0.2,
    merge_type: str = "linear",
    density: Optional[float] = None,
) -> List[str]:
    script = os.path.join(repo_dir, "get_safemerge_model.py")
    cmd = [
        python_bin, script,
        "--base_model_id", base_model,
        "--finetuned_model_id", domain_lora,
        "--safety_model_id", safety_lora,
        "--safelora_unaligned_model_id", unaligned_model,
        "--safelora_aligned_model_id", aligned_model,
        "--cos_threshold", str(cos_threshold),
        "--merge_type", merge_type,
        "--output_path", os.path.abspath(out_dir),
    ]
    if default_merge_ratio is not None:
        cmd.extend(["--default_merge_ratio", str(default_merge_ratio)])
    if density is not None:
        cmd.extend(["--density", str(density)])
    return cmd


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SafeMERGE (LoRA real-world track only)")
    ap.add_argument("--base", required=True, help="base model for PEFT loading")
    ap.add_argument("--inputs", nargs=2, required=True, metavar=("SAFETY_LORA", "DOMAIN_LORA"),
                    help="two LoRA adapters: safety adapter first, domain adapter second")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--task-names", default=None)
    ap.add_argument("--safelora-unaligned", default=None, help="unaligned reference model (defaults to --base)")
    ap.add_argument("--safelora-aligned", required=True, help="aligned reference model for safety subspace")
    ap.add_argument("--cos-threshold", type=float, default=0.35)
    ap.add_argument("--default-merge-ratio", type=float, default=0.2)
    ap.add_argument("--merge-type", default="linear", choices=["linear", "dare_linear"])
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--repo-dir", default=None, help="path to aladinD/SafeMERGE checkout")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    repo_dir = resolve_external_repo("SafeMERGE", args.repo_dir)
    ext_commit = git_commit(repo_dir) if os.path.isdir(repo_dir) else "unknown"
    safety_lora, domain_lora = args.inputs
    unaligned = args.safelora_unaligned or args.base

    cmd = build_command(
        repo_dir=repo_dir,
        python_bin=args.python,
        base_model=args.base,
        safety_lora=safety_lora,
        domain_lora=domain_lora,
        unaligned_model=unaligned,
        aligned_model=args.safelora_aligned,
        out_dir=args.out,
        cos_threshold=args.cos_threshold,
        default_merge_ratio=args.default_merge_ratio,
        merge_type=args.merge_type,
        density=args.density,
    )
    print("Command:", " ".join(cmd))
    if args.dry_run:
        return 0

    os.makedirs(args.out, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, check=True)
    finished = datetime.now(timezone.utc).isoformat()

    combined_log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    layers_merged, layers_skipped = parse_safemerge_layer_logs(combined_log)

    with open(os.path.join(args.out, "safemerge_run.log"), "w", encoding="utf-8") as f:
        f.write(combined_log)

    ensure_tokenizer_saved(args.base, args.out)
    write_merge_metadata(
        out_dir=args.out,
        algo="SafeMERGE",
        kwargs={
            "cos_threshold": args.cos_threshold,
            "default_merge_ratio": args.default_merge_ratio,
            "merge_type": args.merge_type,
            "density": args.density,
            "safelora_unaligned": unaligned,
            "safelora_aligned": args.safelora_aligned,
        },
        base_model=args.base,
        inputs=list(args.inputs),
        task_names=args.task_names,
        seed=args.seed,
        dtype="float32_proj",
        started=started,
        finished=finished,
        external_repo="aladinD/SafeMERGE",
        external_commit=ext_commit,
        extra={
            "layers_merged_count": len(layers_merged),
            "layers_merged": layers_merged,
            "layers_skipped_count": len(layers_skipped),
            "layers_skipped": layers_skipped,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
