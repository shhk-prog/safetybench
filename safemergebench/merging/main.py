from datetime import datetime, timezone
import importlib
import json
import os
import random
import subprocess

from prepare_args import prepare_args, create_parser
from taskloader import get_ft_ckpts


def _set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _git_commit():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def write_merge_metadata(
    save_path,
    algo,
    kwargs,
    base_model,
    inputs,
    task_names,
    seed=42,
    dtype="bfloat16",
    started=None,
    finished=None,
    extra=None,
):
    os.makedirs(save_path, exist_ok=True)
    meta = {
        "algo": algo,
        "kwargs": kwargs,
        "base_model": base_model,
        "inputs": list(inputs),
        "task_names": task_names,
        "dtype": dtype,
        "seed": seed,
        "git_commit": _git_commit(),
        "started": started or datetime.now(timezone.utc).isoformat(),
        "finished": finished or datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    out_path = os.path.join(save_path, "merge_metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_path


def parse_args():
    parser = create_parser()

    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-3B', type=str)
    parser.add_argument('--algo', default='TaskArithmetic', type=str, choices=['TaskArithmetic', 'TIES', 'DARE', 'LocalizeAndStitch', 'Consensus', 'RegMean', 'RegMeanPlusPlus', 'Fisher'])
    parser.add_argument('--save-path', default='./merged_models/', type=str)

    return parser.parse_args()


def main(args):
    _set_seed(args.seed)
    kwargs = prepare_args(args)
    merger_module = importlib.import_module("merging_methods")
    task_names = args.task_names.split('-') if args.task_names else None
    ft_ckpts = get_ft_ckpts(args.base_model, task_names)

    kwargs_str = "_".join(f"{key}_{value}" for key, value in kwargs.items() if key not in ['fisher_only','merge_only','save_group','task_names','keep_checkpoints'])
    if args.save_group:
        task_group = args.save_group
    elif args.task_names:
        task_group = args.task_names
    else:
        task_group = None

    recorded_algo = args.record_as or args.algo
    save_path = args.save_path + args.base_model.split('/')[1] + '_merged/' + recorded_algo
    if task_group:
        save_path += '_task_names_' + task_group
    if kwargs_str != '':
        save_path += '_' + kwargs_str

    print('merged model save to:', save_path)
    merger = getattr(merger_module, args.algo)(args.base_model, ft_ckpts, save_path)
    print(args)
    print(kwargs)
    started = datetime.now(timezone.utc).isoformat()
    merger.merge(**kwargs)
    finished = datetime.now(timezone.utc).isoformat()

    if not kwargs.get("fisher_only", False):
        write_merge_metadata(
            save_path=save_path,
            algo=recorded_algo,
            kwargs=kwargs,
            base_model=args.base_model,
            inputs=ft_ckpts,
            task_names=task_group,
            seed=args.seed,
            dtype="bfloat16",
            started=started,
            finished=finished,
            extra={"impl_algo": args.algo} if args.record_as else None,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
