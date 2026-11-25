from datetime import datetime, timezone
import importlib
import json
import os
import random
import subprocess

from prepare_args import prepare_args, create_parser


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


def resolve_ckpt_task_names(algo, task_names_arg, save_group_arg):
    """Return the ordered list of task names whose checkpoints must be loaded.

    Bug 1 fix: Fisher's merge() zips `save_group.split('-')` with `self.ft_ckpts`.
    When stage-1 (--fisher_only) is invoked with a single `--task_names` and a
    multi-task `--save_group`, loading only `--task_names` causes mis-mapping and
    a KeyError. Therefore Fisher with `save_group` must load checkpoints for all
    tasks in `save_group`.
    """
    if algo == "Fisher" and save_group_arg:
        return save_group_arg.split("-")
    return task_names_arg.split("-") if task_names_arg else None


def normalize_recorded_kwargs(algo, kwargs):
    """Return (recorded_kwargs, extra_metadata) reflecting effective runtime values.

    Item 6 fix: upstream Consensus.merge() hard-codes lamda=0.2 and ignores
    kwargs['lamda'] (CLI default 0.5). Record the effective value (0.2) in
    merge_metadata.json and note the ignored CLI value.
    """
    rec_kwargs = dict(kwargs)
    extra = {}
    if algo == "Consensus":
        cli_lamda = rec_kwargs.get("lamda")
        rec_kwargs["lamda"] = 0.2
        extra["effective_lamda"] = 0.2
        if cli_lamda is not None and cli_lamda != 0.2:
            extra["cli_lamda_ignored"] = cli_lamda
            extra["upstream_note"] = "Consensus.merge() hard-codes lamdas=[0.2,...] and ignores CLI --lamda."
    return rec_kwargs, extra


def parse_args():
    parser = create_parser()

    parser.add_argument('--base-model', default='meta-llama/Llama-3.2-3B', type=str)
    parser.add_argument('--algo', default='TaskArithmetic', type=str, choices=['TaskArithmetic', 'TIES', 'DARE', 'LocalizeAndStitch', 'Consensus', 'RegMean', 'RegMeanPlusPlus', 'Fisher'])
    parser.add_argument('--save-path', default='./merged_models/', type=str)

    return parser.parse_args()


def main(args):
    from taskloader import get_ft_ckpts
    _set_seed(args.seed)
    kwargs = prepare_args(args)
    merger_module = importlib.import_module("merging_methods")
    ckpt_tasks = resolve_ckpt_task_names(args.algo, args.task_names, args.save_group)
    ft_ckpts = get_ft_ckpts(args.base_model, ckpt_tasks)

    rec_kwargs, extra_meta = normalize_recorded_kwargs(args.algo, kwargs)
    kwargs_str = "_".join(f"{key}_{value}" for key, value in rec_kwargs.items() if key not in ['fisher_only','merge_only','save_group','task_names','keep_checkpoints'])
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
        if args.record_as:
            extra_meta["impl_algo"] = args.algo
        write_merge_metadata(
            save_path=save_path,
            algo=recorded_algo,
            kwargs=rec_kwargs,
            base_model=args.base_model,
            inputs=ft_ckpts,
            task_names=task_group,
            seed=args.seed,
            dtype="bfloat16",
            started=started,
            finished=finished,
            extra=extra_meta or None,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
