"""Create a timestamped, non-overwriting experiment run directory (SafeMergeBench v2, Section 1).

Usage:
  python safemerge/new_run.py \
    --name std-llama31-safety-math \
    --why "Verify single experts before main merge" \
    --what "Llama-3.1-8B-Instruct base/safety/math/coding"

Prints the created run directory path to stdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from typing import Any, Dict, List, Optional

JST = timezone(timedelta(hours=9))
NAME_RE = re.compile(r"^[A-Za-z0-9-]+$")

INDEX_HEADER = """# SafeMergeBench 実験一覧 (INDEX)

ルール（実装指示書 v2 第1節）:
- 既存の実験フォルダには上書きしない。やり直す場合も新しいフォルダを作る。
- 使わなくなった結果は削除せず、状態に「置き換え済み」と書き、理由と置き換え先を記載する。
- 論文に載せる数値は、本表で状態が「完了」かつ置き換えられていない実験からのみ取る。

## 新形式の実験 (`results/runs/`)

| フォルダ名 | 目的 | 状態 | 論文での使い道 | 備考・置き換え記録 |
|---|---|---|---|---|
"""


def validate_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or not NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid --name {name!r}: must contain only ASCII alphanumerics and hyphens [A-Za-z0-9-]."
        )
    return cleaned


def allocate_run_dir(runs_root: str, name: str, now: Optional[datetime] = None) -> str:
    """Create a unique `YYYYMMDD-HHMM_<name>` directory (appending `-2`, `-3`, ... on collision)."""
    validate_name(name)
    ts = (now or datetime.now(JST)).astimezone(JST).strftime("%Y%m%d-%H%M")
    os.makedirs(runs_root, exist_ok=True)
    base_dir_name = f"{ts}_{name}"
    candidate = os.path.join(runs_root, base_dir_name)
    suffix = 2
    while True:
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = os.path.join(runs_root, f"{base_dir_name}-{suffix}")
            suffix += 1


def _run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 30) -> TupleResult:
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


TupleResult = Dict[str, Any]


def _find_git_root(start_dir: str) -> Optional[str]:
    res = _run_cmd(["git", "rev-parse", "--show-toplevel"], cwd=start_dir)
    if res["returncode"] == 0 and res["stdout"]:
        return res["stdout"]
    return None


def collect_env_info(
    repo_root: str,
    config_dir: str,
    venv_root: Optional[str] = None,
    collect_pip: bool = True,
    collect_gpu: bool = True,
) -> Dict[str, Any]:
    git_root = _find_git_root(repo_root) or repo_root
    head = _run_cmd(["git", "rev-parse", "HEAD"], cwd=git_root)
    branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
    status = _run_cmd(["git", "status", "--porcelain"], cwd=git_root)
    dirty = bool(status["stdout"])
    uncommitted_diff_path: Optional[str] = None

    if dirty:
        diff_res = _run_cmd(["git", "diff", "HEAD"], cwd=git_root, timeout=60)
        untracked = [
            line[3:] for line in status["stdout"].splitlines() if line.startswith("?? ")
        ]
        diff_file = os.path.join(config_dir, "uncommitted.diff")
        with open(diff_file, "w", encoding="utf-8") as f:
            f.write(diff_res["stdout"])
            if diff_res["stdout"] and not diff_res["stdout"].endswith("\n"):
                f.write("\n")
            if untracked:
                f.write("\n# Untracked files:\n")
                for u in untracked:
                    f.write(f"#   {u}\n")
        uncommitted_diff_path = "config/uncommitted.diff"

    submodules: Dict[str, Any] = {}
    sm_res = _run_cmd(["git", "submodule", "status", "--recursive"], cwd=git_root)
    if sm_res["returncode"] == 0 and sm_res["stdout"]:
        for line in sm_res["stdout"].splitlines():
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            commit_token = parts[0]
            path_token = parts[1] if len(parts) > 1 else "unknown"
            submodules[path_token] = {
                "commit": commit_token.lstrip("+-U "),
                "status_prefix": commit_token[0] if commit_token[0] in "+-U" else "",
                "describe": " ".join(parts[2:]) if len(parts) > 2 else "",
            }

    # Also capture nested git repos under safemergebench (safety-eval-fork, lm-evaluation-harness, bigcode)
    for rel in ("safemergebench/safety-eval-fork", "safemergebench/lm-evaluation-harness", "safemergebench/bigcode-evaluation-harness"):
        p = os.path.join(git_root, rel)
        if os.path.isdir(os.path.join(p, ".git")):
            c = _run_cmd(["git", "rev-parse", "HEAD"], cwd=p)
            st = _run_cmd(["git", "status", "--porcelain"], cwd=p)
            submodules[rel] = {
                "commit": c["stdout"] if c["returncode"] == 0 else "unknown",
                "dirty": bool(st["stdout"]),
            }

    gpu_summary: Any = "skipped"
    if collect_gpu:
        nv = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version",
                "--format=csv,noheader",
            ],
            timeout=15,
        )
        if nv["returncode"] == 0:
            gpu_summary = [line.strip() for line in nv["stdout"].splitlines() if line.strip()]
        else:
            gpu_summary = {"error": nv["stderr"] or "nvidia-smi unavailable"}

    pip_freezes: Dict[str, Any] = {}
    if collect_pip:
        py_targets: Dict[str, str] = {"current_python": sys.executable}
        vroot = venv_root or os.path.join(repo_root, ".venvs")
        if os.path.isdir(vroot):
            for entry in sorted(os.listdir(vroot)):
                py_bin = os.path.join(vroot, entry, "bin", "python")
                if os.path.isfile(py_bin) and os.access(py_bin, os.X_OK):
                    py_targets[f"venv:{entry}"] = py_bin
        for label, py_bin in py_targets.items():
            fr = _run_cmd([py_bin, "-m", "pip", "freeze"], timeout=30)
            if fr["returncode"] == 0:
                pkgs = [ln.strip() for ln in fr["stdout"].splitlines() if ln.strip()]
                pip_freezes[label] = {"python": py_bin, "count": len(pkgs), "packages": pkgs}
            else:
                pip_freezes[label] = {"python": py_bin, "error": fr["stderr"]}

    return {
        "created_at_jst": datetime.now(JST).isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "git": {
            "root": git_root,
            "commit": head["stdout"] if head["returncode"] == 0 else "unknown",
            "branch": branch["stdout"] if branch["returncode"] == 0 else "unknown",
            "has_uncommitted_changes": dirty,
            "uncommitted_diff": uncommitted_diff_path,
            "status_porcelain": status["stdout"].splitlines() if status["stdout"] else [],
        },
        "submodules": submodules,
        "gpu": gpu_summary,
        "python_version": sys.version,
        "pip_freeze": pip_freezes,
    }


def render_run_md(
    run_name: str,
    started_jst: str,
    runner: str,
    related: str,
    why: str,
    what: str,
    expect: str,
) -> str:
    return (
        f"# {run_name}\n"
        f"- 開始: {started_jst} JST / 終了: \n"
        f"- 実行者: {runner}\n"
        f"- 関連する実験: {related}\n"
        f"\n"
        f"## なぜやるか\n"
        f"{why}\n"
        f"\n"
        f"## 何をやるか\n"
        f"{what}\n"
        f"\n"
        f"## 予想\n"
        f"{expect}\n"
        f"\n"
        f"## 結果（終了後に記入）\n"
        f"<!-- 主な数値。想定と違った点。失敗や中断があればその理由 -->\n"
        f"\n"
        f"## 判断と次の一手\n"
        f"<!-- この結果を受けて何を決めたか。結果を使わない場合はその理由 -->\n"
    )


def _escape_md_cell(text: str) -> str:
    return " ".join((text or "").replace("|", "\\|").splitlines()).strip()


def append_index_entry(
    runs_root: str,
    folder_name: str,
    purpose: str,
    status: str = "実行中",
    paper_use: str = "",
    note: str = "",
) -> str:
    index_path = os.path.join(runs_root, "INDEX.md")
    if not os.path.exists(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(INDEX_HEADER)

    row = (
        f"| `{folder_name}` | {_escape_md_cell(purpose)} | {_escape_md_cell(status)} "
        f"| {_escape_md_cell(paper_use)} | {_escape_md_cell(note)} |\n"
    )
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Insert right after the new-format table if a legacy section exists, otherwise append at end
    legacy_marker = "\n## 旧形式の結果"
    if legacy_marker in content:
        before, after = content.split(legacy_marker, 1)
        if not before.endswith("\n"):
            before += "\n"
        new_content = before + row + legacy_marker + after
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    else:
        with open(index_path, "a", encoding="utf-8") as f:
            if content and not content.endswith("\n"):
                f.write("\n")
            f.write(row)
    return index_path


def create_run(
    name: str,
    why: str,
    what: str,
    expect: str = "",
    runner: str = "AI",
    related: str = "なし",
    paper_use: str = "",
    runs_root: Optional[str] = None,
    repo_root: Optional[str] = None,
    copy_configs: Optional[List[str]] = None,
    now: Optional[datetime] = None,
    collect_pip: bool = True,
    collect_gpu: bool = True,
) -> str:
    smb_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rroot = runs_root or os.path.join(smb_root, "results", "runs")
    now_dt = (now or datetime.now(JST)).astimezone(JST)

    run_dir = allocate_run_dir(rroot, name, now=now_dt)
    folder_name = os.path.basename(run_dir)
    config_dir = os.path.join(run_dir, "config")
    logs_dir = os.path.join(run_dir, "logs")
    outputs_dir = os.path.join(run_dir, "outputs")
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    run_md = render_run_md(
        run_name=folder_name,
        started_jst=now_dt.strftime("%Y-%m-%d %H:%M"),
        runner=runner,
        related=related,
        why=why,
        what=what,
        expect=expect or "（記入予定）",
    )
    with open(os.path.join(run_dir, "RUN.md"), "w", encoding="utf-8") as f:
        f.write(run_md)

    cmd_sh = os.path.join(run_dir, "commands.sh")
    with open(cmd_sh, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n# Auto-recorded commands for " + folder_name + "\nset -euo pipefail\n\n")
    os.chmod(cmd_sh, 0o755)

    # Copy eval_lock.yaml automatically if present, plus any user-specified config files
    default_lock = os.path.join(smb_root, "configs", "eval_lock.yaml")
    to_copy = list(copy_configs or [])
    if os.path.isfile(default_lock) and default_lock not in to_copy:
        to_copy.append(default_lock)
    for cfg in to_copy:
        if os.path.isfile(cfg):
            shutil.copy2(cfg, os.path.join(config_dir, os.path.basename(cfg)))

    env_info = collect_env_info(
        repo_root=smb_root,
        config_dir=config_dir,
        venv_root=os.path.join(smb_root, ".venvs"),
        collect_pip=collect_pip,
        collect_gpu=collect_gpu,
    )
    with open(os.path.join(run_dir, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2, ensure_ascii=False)

    append_index_entry(
        runs_root=rroot,
        folder_name=folder_name,
        purpose=why,
        status="実行中",
        paper_use=paper_use,
    )
    return run_dir


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create a new timestamped experiment run folder (Section 1)")
    ap.add_argument("--name", required=True, help="short alphanumeric/hyphen run name, e.g. std-llama31-safety-math")
    ap.add_argument("--why", required=True, help="why this experiment is run (1-2 sentences)")
    ap.add_argument("--what", required=True, help="models, methods, alphas, datasets, judges, changes")
    ap.add_argument("--expect", default="", help="expected outcome and decision branches")
    ap.add_argument("--runner", default="AI (Antigravity)", help="person or AI running the experiment")
    ap.add_argument("--related", default="なし", help="prerequisite or superseded run folder(s)")
    ap.add_argument("--paper-use", default="", help="paper table/section reference for INDEX.md")
    ap.add_argument("--runs-root", default=None, help="override results/runs directory")
    ap.add_argument("--copy-config", action="append", default=[], help="config file(s) to copy into config/")
    ap.add_argument("--skip-pip-freeze", action="store_true", help="skip pip freeze (useful for fast unit tests)")
    ap.add_argument("--skip-gpu", action="store_true", help="skip nvidia-smi query")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = create_run(
        name=args.name,
        why=args.why,
        what=args.what,
        expect=args.expect,
        runner=args.runner,
        related=args.related,
        paper_use=args.paper_use,
        runs_root=args.runs_root,
        copy_configs=args.copy_config,
        collect_pip=not args.skip_pip_freeze,
        collect_gpu=not args.skip_gpu,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
