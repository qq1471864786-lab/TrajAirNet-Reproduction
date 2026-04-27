import argparse
import datetime as dt
import json
import os
import re
import shlex
import sys
from typing import Any, List

from remote_common import add_remote_conda_args, add_remote_project_root_arg, add_remote_target_args, run_target_command


REMOTE_LAUNCH_SCRIPT = r"""
import json
import os
import shlex
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(os.environ["PROJECT_ROOT"])
run_name = os.environ["RUN_NAME"]
conda_sh = os.environ["CONDA_SH"]
conda_env = os.environ["CONDA_ENV"]
python_exec = os.environ.get("PYTHON_EXEC", "python")
train_script = os.environ.get("TRAIN_SCRIPT", "train.py")
train_args = json.loads(os.environ["TRAIN_ARGS_JSON"])
resolved_run_dir = os.environ["RESOLVED_RUN_DIR"]

runs_root = project_root / ".remote_runs"
runs_root.mkdir(parents=True, exist_ok=True)
run_root = runs_root / run_name
run_root.mkdir(parents=True, exist_ok=True)

command_list = [python_exec, train_script] + train_args
command_str = " ".join(shlex.quote(item) for item in command_list)
shell_cmd = (
    f"source {shlex.quote(conda_sh)} && "
    f"conda activate {shlex.quote(conda_env)} && "
    f"cd {shlex.quote(str(project_root))} && "
    f"{command_str}"
)

(run_root / "command.sh").write_text(shell_cmd + "\n", encoding="utf-8")

meta = {
    "run_name": run_name,
    "project_root": str(project_root),
    "resolved_run_dir": resolved_run_dir,
    "conda_env": conda_env,
    "python_exec": python_exec,
    "train_script": train_script,
    "train_args": train_args,
    "train_command": command_str,
    "launched_at": datetime.now().isoformat(),
}

stdout_path = run_root / "stdout.log"
stderr_path = run_root / "stderr.log"
with open(stdout_path, "ab") as stdout_handle, open(stderr_path, "ab") as stderr_handle:
    proc = subprocess.Popen(
        ["bash", "-lc", shell_cmd],
        stdout=stdout_handle,
        stderr=stderr_handle,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

meta["pid"] = proc.pid
(run_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
(run_root / "pid").write_text(str(proc.pid), encoding="utf-8")

latest_link = runs_root / "latest"
if latest_link.exists() or latest_link.is_symlink():
    latest_link.unlink()
latest_link.symlink_to(run_root, target_is_directory=True)

print(json.dumps({"run_root": str(run_root), "pid": proc.pid, "resolved_run_dir": resolved_run_dir}, ensure_ascii=False))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a remote ProtoBasis-Net training run with traceable logs.")
    add_remote_target_args(parser)
    add_remote_project_root_arg(parser)
    add_remote_conda_args(parser)
    parser.add_argument("--python-exec", default="python")
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--name", default="", help="Optional readable name prefix for the remote run.")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Arguments passed through to train.py after '--'.")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "train"


def extract_arg(args: List[str], key: str, default: str) -> str:
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            return args[idx + 1]
    key_eq = key + "="
    for item in args:
        if item.startswith(key_eq):
            return item[len(key_eq) :]
    return default


def main() -> int:
    args = parse_args()
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = sanitize_name(args.name) if args.name else "train"
    run_name = f"{timestamp}_{prefix}"

    dataset_name = extract_arg(train_args, "--dataset_name", "111_days")
    seed = extract_arg(train_args, "--seed", "3407")
    save_dir = extract_arg(train_args, "--save_dir", "save_model")
    resolved_run_dir = f"{args.project_root}/{save_dir}/{dataset_name}/seed{seed}"

    env_parts = {
        "PROJECT_ROOT": args.project_root,
        "RUN_NAME": run_name,
        "CONDA_SH": args.conda_sh,
        "CONDA_ENV": args.conda_env,
        "PYTHON_EXEC": args.python_exec,
        "TRAIN_SCRIPT": args.train_script,
        "TRAIN_ARGS_JSON": json.dumps(train_args, ensure_ascii=False),
        "RESOLVED_RUN_DIR": resolved_run_dir,
    }
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items())
    command = f"{env_prefix} python3 - <<'PY'\n{REMOTE_LAUNCH_SCRIPT}\nPY"

    try:
        stdout = run_target_command(
            args.host,
            args.project_root,
            args.user,
            args.port,
            command,
            password_env=args.password_env,
        )
    except Exception as exc:
        print("远端训练启动失败。", file=sys.stderr)
        if getattr(exc, "stdout", ""):
            print(exc.stdout, file=sys.stderr)
        if getattr(exc, "stderr", ""):
            print(exc.stderr, file=sys.stderr)
        if not getattr(exc, "stdout", "") and not getattr(exc, "stderr", ""):
            print(str(exc), file=sys.stderr)
        return 1

    payload: dict[str, Any] = json.loads(stdout)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
