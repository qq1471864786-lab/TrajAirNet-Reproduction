import argparse
import datetime as dt
import json
import re
import shlex
import sys
from typing import Any

from remote_common import add_remote_conda_args, add_remote_project_root_arg, add_remote_target_args, run_target_command


REMOTE_SUITE_SCRIPT = r"""
import json
import os
import shlex
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
suite_name = os.environ["SUITE_NAME"]
conda_sh = os.environ["CONDA_SH"]
conda_env = os.environ["CONDA_ENV"]
python_exec = os.environ.get("PYTHON_EXEC", "python")
train_script = os.environ.get("TRAIN_SCRIPT", "train.py")
datasets = json.loads(os.environ["DATASETS_JSON"])
devices = json.loads(os.environ["DEVICES_JSON"])
train_args = json.loads(os.environ["TRAIN_ARGS_JSON"])
save_dir = os.environ["SAVE_DIR"]
seed = os.environ["SEED"]


def extract_arg(args, key, default):
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            return args[idx + 1]
    key_eq = key + "="
    for item in args:
        if item.startswith(key_eq):
            return item[len(key_eq):]
    return default


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc), "_path": str(path)}


def tail_lines(path, n=40):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=n)]
    except Exception as exc:
        return [f"[load error] {exc}"]


suite_root = project_root / ".remote_runs" / suite_name
suite_root.mkdir(parents=True, exist_ok=True)

jobs = []
for dataset in datasets:
    run_dir = project_root / save_dir / dataset / f"seed{seed}"
    stdout_path = suite_root / f"{dataset}.stdout.log"
    stderr_path = suite_root / f"{dataset}.stderr.log"
    jobs.append(
        {
            "dataset": dataset,
            "run_dir": run_dir,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "status": "pending",
            "device": None,
            "pid": None,
        }
    )


def launch(job, device):
    command_list = [python_exec, train_script, job["dataset"], "--device", device, "--save_dir", save_dir] + train_args
    command_str = " ".join(shlex.quote(item) for item in command_list)
    shell_cmd = (
        f"source {shlex.quote(conda_sh)} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"cd {shlex.quote(str(project_root))} && "
        f"{command_str}"
    )
    with open(job["stdout_path"], "ab") as stdout_handle, open(job["stderr_path"], "ab") as stderr_handle:
        proc = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    job["status"] = "running"
    job["device"] = device
    job["pid"] = proc.pid
    job["launched_at"] = datetime.now().isoformat()
    job["proc"] = proc
    job["command"] = command_list


pending = jobs[:]
running = []
free_devices = devices[:]

while pending or running:
    while pending and free_devices:
        device = free_devices.pop(0)
        job = pending.pop(0)
        launch(job, device)
        running.append(job)

    next_running = []
    for job in running:
        ret = job["proc"].poll()
        if ret is None:
            next_running.append(job)
            continue
        job["returncode"] = ret
        job["finished_at"] = datetime.now().isoformat()
        job["status"] = "completed" if ret == 0 else "failed"
        free_devices.append(job["device"])
    running = next_running
    if pending or running:
        time.sleep(5)


results = []
for job in jobs:
    run_summary = load_json(job["run_dir"] / "run_summary.json")
    latest = run_summary.get("latest_epoch", {}) if isinstance(run_summary, dict) else {}
    best = (run_summary or {}).get("best", {}).get("best20", {}) if isinstance(run_summary, dict) else {}
    results.append(
        {
            "dataset": job["dataset"],
            "device": job["device"],
            "status": job["status"],
            "returncode": job.get("returncode"),
            "pid": job.get("pid"),
            "run_dir": str(job["run_dir"]),
            "best20": {
                "epoch": best.get("epoch"),
                "ADE@20": best.get("value"),
                "FDE@20": best.get("fde_value"),
                "rare_FDE@20": best.get("rare_fde_value"),
            },
            "final": {
                "epoch": latest.get("epoch"),
                "ADE@20": (latest.get("metrics") or {}).get("ADE@20"),
                "FDE@20": (latest.get("metrics") or {}).get("FDE@20"),
                "rare_FDE@20": (latest.get("metrics") or {}).get("rare_FDE@20"),
            },
            "stdout_tail": tail_lines(job["stdout_path"], 20),
            "stderr_tail": tail_lines(job["stderr_path"], 20),
        }
    )

payload = {
    "suite_name": suite_name,
    "project_root": str(project_root),
    "save_dir": save_dir,
    "seed": seed,
    "datasets": datasets,
    "devices": devices,
    "results": results,
}

(suite_root / "suite_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 7days1~4 on the remote server and summarize final results.")
    add_remote_target_args(parser)
    add_remote_project_root_arg(parser)
    add_remote_conda_args(parser)
    parser.add_argument("--python-exec", default="python")
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--datasets", default="7days1,7days2,7days3,7days4")
    parser.add_argument("--devices", default="cuda:1", help="Comma-separated device list, e.g. cuda:0,cuda:1")
    parser.add_argument("--save-dir", default="save_model_7days_suite")
    parser.add_argument("--name", default="7days-suite", help="Readable suite name prefix.")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Extra train.py args after '--'.")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "7days-suite"


def extract_arg(args: list[str], key: str, default: str) -> str:
    if key in args:
        idx = args.index(key)
        if idx + 1 < len(args):
            return args[idx + 1]
    key_eq = key + "="
    for item in args:
        if item.startswith(key_eq):
            return item[len(key_eq) :]
    return default


def print_summary(payload: dict[str, Any]) -> None:
    print(f"套件: {payload['suite_name']}")
    print(f"保存目录: {payload['save_dir']}")
    print("结果:")
    for item in payload.get("results", []):
        best = item.get("best20", {})
        final = item.get("final", {})
        print(
            f"- {item['dataset']} [{item.get('device')}] status={item.get('status')} "
            f"best(ADE20={best.get('ADE@20')}, FDE20={best.get('FDE@20')}, "
            f"rare={best.get('rare_FDE@20')}, epoch={best.get('epoch')}) "
            f"final(ADE20={final.get('ADE@20')}, FDE20={final.get('FDE@20')}, "
            f"rare={final.get('rare_FDE@20')}, epoch={final.get('epoch')})"
        )


def main() -> int:
    args = parse_args()
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not datasets:
        print("没有指定数据集。", file=sys.stderr)
        return 1
    if not devices:
        print("没有指定可用设备。", file=sys.stderr)
        return 1

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"{timestamp}_{sanitize_name(args.name)}"
    seed = extract_arg(train_args, "--seed", "3407")

    env_parts = {
        "PROJECT_ROOT": args.project_root,
        "SUITE_NAME": suite_name,
        "CONDA_SH": args.conda_sh,
        "CONDA_ENV": args.conda_env,
        "PYTHON_EXEC": args.python_exec,
        "TRAIN_SCRIPT": args.train_script,
        "DATASETS_JSON": json.dumps(datasets, ensure_ascii=False),
        "DEVICES_JSON": json.dumps(devices, ensure_ascii=False),
        "TRAIN_ARGS_JSON": json.dumps(train_args, ensure_ascii=False),
        "SAVE_DIR": args.save_dir,
        "SEED": seed,
    }
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items())
    command = f"{env_prefix} python3 - <<'PY'\n{REMOTE_SUITE_SCRIPT}\nPY"

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
        print("远端 7days 套件运行失败。", file=sys.stderr)
        if getattr(exc, "stdout", ""):
            print(exc.stdout, file=sys.stderr)
        if getattr(exc, "stderr", ""):
            print(exc.stderr, file=sys.stderr)
        if not getattr(exc, "stdout", "") and not getattr(exc, "stderr", ""):
            print(str(exc), file=sys.stderr)
        return 1

    payload: dict[str, Any] = json.loads(stdout)
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
