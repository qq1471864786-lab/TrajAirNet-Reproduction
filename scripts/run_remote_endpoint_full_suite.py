import argparse
import datetime as dt
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from remote_common import add_remote_conda_args, add_remote_project_root_arg, add_remote_target_args, run_target_command


REMOTE_SUPERVISOR_SCRIPT = r"""
import json
import os
import shlex
import subprocess
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
suite_name = os.environ["SUITE_NAME"]
conda_sh = os.environ["CONDA_SH"]
conda_env = os.environ["CONDA_ENV"]
python_exec = os.environ.get("PYTHON_EXEC", "python")
train_script = os.environ.get("TRAIN_SCRIPT", "train.py")
save_dir = os.environ["SAVE_DIR"]
seed = os.environ["SEED"]
device_spec = os.environ["DEVICES"]
min_free_mb_111 = int(os.environ["MIN_FREE_MB_111"])
min_free_mb_7days = int(os.environ["MIN_FREE_MB_7DAYS"])
max_7days_parallel = int(os.environ["MAX_7DAYS_PARALLEL"])
poll_seconds = int(os.environ["POLL_SECONDS"])
wait_minutes = int(os.environ["WAIT_MINUTES"])
base_train_args = json.loads(os.environ["BASE_TRAIN_ARGS_JSON"])
datasets_7days = json.loads(os.environ["DATASETS_7DAYS_JSON"])
stop_on_111_failure = os.environ.get("STOP_ON_111_FAILURE", "1") == "1"

suite_root = project_root / ".remote_runs" / suite_name
job_root = suite_root / "jobs"
job_root.mkdir(parents=True, exist_ok=True)
status_path = suite_root / "suite_status.json"
summary_path = suite_root / "suite_summary.json"


def now():
    return datetime.now().isoformat(timespec="seconds")


def parse_devices(spec):
    if not spec or spec.strip().lower() == "auto":
        return None
    devices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("cuda:"):
            devices.append(int(token.split(":", 1)[1]))
        else:
            devices.append(int(token))
    return devices


allowed_device_indices = parse_devices(device_spec)


def run_command(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def gpu_snapshot():
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run_command(query)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "gpus": []}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        index = int(parts[0])
        if allowed_device_indices is not None and index not in allowed_device_indices:
            continue
        used = int(float(parts[2]))
        total = int(float(parts[3]))
        gpus.append(
            {
                "index": index,
                "device": f"cuda:{index}",
                "name": parts[1],
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_free_mb": total - used,
                "utilization_gpu": int(float(parts[4])),
                "temperature_c": int(float(parts[5])),
            }
        )
    return {"ok": True, "gpus": gpus}


def eligible_devices(min_free_mb):
    snap = gpu_snapshot()
    if not snap["ok"]:
        return [], snap
    devices = [gpu for gpu in snap["gpus"] if gpu["memory_free_mb"] >= min_free_mb]
    devices.sort(key=lambda item: item["memory_free_mb"], reverse=True)
    return devices, snap


def wait_for_eligible(min_free_mb, needed=1):
    deadline = time.time() + wait_minutes * 60
    last_snap = None
    while True:
        devices, snap = eligible_devices(min_free_mb)
        last_snap = snap
        if len(devices) >= needed:
            return devices, snap
        if time.time() >= deadline:
            return devices, last_snap
        write_status(
            status="waiting_for_gpu",
            phase="gpu_wait",
            message=f"Need {needed} device(s) with >= {min_free_mb} MB free.",
            gpu_snapshot=last_snap,
        )
        time.sleep(poll_seconds)


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


jobs = []


def run_dir_for(dataset):
    return project_root / save_dir / dataset / f"seed{seed}"


def summarize_job(job):
    run_summary = load_json(job["run_dir"] / "run_summary.json")
    latest = run_summary.get("latest_epoch", {}) if isinstance(run_summary, dict) else {}
    best = (run_summary or {}).get("best", {}).get("best20", {}) if isinstance(run_summary, dict) else {}
    return {
        "dataset": job["dataset"],
        "stage": job["stage"],
        "device": job.get("device"),
        "status": job.get("status"),
        "returncode": job.get("returncode"),
        "pid": job.get("pid"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "run_dir": str(job["run_dir"]),
        "best20": {
            "epoch": best.get("epoch"),
            "ADE@20": best.get("value"),
            "FDE@20": best.get("fde_value"),
            "GLeV@20": best.get("glev_value"),
            "rare_FDE@20": best.get("rare_fde_value"),
        },
        "final": {
            "epoch": latest.get("epoch"),
            "ADE@20": (latest.get("metrics") or {}).get("ADE@20"),
            "FDE@20": (latest.get("metrics") or {}).get("FDE@20"),
            "GLeV@20": (latest.get("metrics") or {}).get("GLeV@20"),
            "rare_FDE@20": (latest.get("metrics") or {}).get("rare_FDE@20"),
        },
        "stdout_tail": tail_lines(job["stdout_path"], 16),
        "stderr_tail": tail_lines(job["stderr_path"], 16),
    }


def write_status(**extra):
    payload = {
        "suite_name": suite_name,
        "project_root": str(project_root),
        "save_dir": save_dir,
        "seed": seed,
        "updated_at": now(),
        "base_train_args": base_train_args,
        "datasets_7days": datasets_7days,
        "jobs": [summarize_job(job) for job in jobs],
    }
    payload.update(extra)
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_job(dataset, device, stage):
    stdout_path = job_root / f"{dataset}.stdout.log"
    stderr_path = job_root / f"{dataset}.stderr.log"
    run_dir = run_dir_for(dataset)
    command_list = [
        python_exec,
        train_script,
        "--dataset_name",
        dataset,
        "--device",
        device,
        "--save_dir",
        save_dir,
    ] + base_train_args
    command_str = " ".join(shlex.quote(item) for item in command_list)
    shell_cmd = (
        f"source {shlex.quote(conda_sh)} && "
        f"conda activate {shlex.quote(conda_env)} && "
        f"cd {shlex.quote(str(project_root))} && "
        f"{command_str}"
    )
    with open(stdout_path, "ab") as stdout_handle, open(stderr_path, "ab") as stderr_handle:
        proc = subprocess.Popen(
            ["bash", "-lc", shell_cmd],
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    job = {
        "dataset": dataset,
        "stage": stage,
        "device": device,
        "pid": proc.pid,
        "proc": proc,
        "status": "running",
        "started_at": now(),
        "run_dir": run_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "command": command_list,
    }
    jobs.append(job)
    return job


def wait_job(job):
    while True:
        ret = job["proc"].poll()
        if ret is not None:
            job["returncode"] = ret
            job["status"] = "completed" if ret == 0 else "failed"
            job["finished_at"] = now()
            return ret
        write_status(status="running", phase=job["dataset"], gpu_snapshot=gpu_snapshot())
        time.sleep(poll_seconds)


def run_7days_suite():
    pending = list(datasets_7days)
    running = []
    free_devices = []
    selected, snap = wait_for_eligible(min_free_mb_7days, needed=1)
    if not selected:
        write_status(
            status="failed",
            phase="7days_preflight",
            message=f"No 7days device reached >= {min_free_mb_7days} MB free.",
            gpu_snapshot=snap,
        )
        return 1
    limit = len(selected) if max_7days_parallel <= 0 else min(max_7days_parallel, len(selected))
    free_devices = [item["device"] for item in selected[:limit]]
    write_status(status="running", phase="7days", message=f"Launching 7days on {free_devices}.", gpu_snapshot=snap)

    while pending or running:
        while pending and free_devices:
            device = free_devices.pop(0)
            dataset = pending.pop(0)
            running.append(launch_job(dataset, device, "7days"))
            write_status(status="running", phase="7days", gpu_snapshot=gpu_snapshot())

        next_running = []
        for job in running:
            ret = job["proc"].poll()
            if ret is None:
                next_running.append(job)
                continue
            job["returncode"] = ret
            job["status"] = "completed" if ret == 0 else "failed"
            job["finished_at"] = now()
            free_devices.append(job["device"])
        running = next_running
        write_status(status="running", phase="7days", gpu_snapshot=gpu_snapshot())
        if pending or running:
            time.sleep(poll_seconds)

    return 0 if all(job["status"] == "completed" for job in jobs if job["stage"] == "7days") else 1


def main():
    write_status(status="starting", phase="preflight", gpu_snapshot=gpu_snapshot())
    selected_111, snap_111 = wait_for_eligible(min_free_mb_111, needed=1)
    if not selected_111:
        write_status(
            status="failed",
            phase="111_days_preflight",
            message=f"No 111_days device reached >= {min_free_mb_111} MB free.",
            gpu_snapshot=snap_111,
        )
        return 1

    device_111 = selected_111[0]["device"]
    write_status(status="running", phase="111_days", message=f"Launching 111_days on {device_111}.", gpu_snapshot=snap_111)
    job_111 = launch_job("111_days", device_111, "111_days")
    ret_111 = wait_job(job_111)
    if ret_111 != 0 and stop_on_111_failure:
        write_status(status="failed", phase="111_days", message="111_days failed; 7days not launched.", gpu_snapshot=gpu_snapshot())
        summary_path.write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
        return ret_111

    ret_7days = run_7days_suite()
    final_status = "completed" if ret_111 == 0 and ret_7days == 0 else "failed"
    write_status(status=final_status, phase="done", gpu_snapshot=gpu_snapshot())
    summary_path.write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
    return 0 if final_status == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error_path = suite_root / "supervisor_exception.txt"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        write_status(status="failed", phase="exception", message=str(error_path), gpu_snapshot=gpu_snapshot())
        raise
"""


REMOTE_GPU_PREFLIGHT_SCRIPT = r"""
import json
import os
import subprocess

devices = os.environ["DEVICES"]
min_free_111 = int(os.environ["MIN_FREE_MB_111"])
min_free_7days = int(os.environ["MIN_FREE_MB_7DAYS"])


def parse_devices(spec):
    if not spec or spec.strip().lower() == "auto":
        return None
    parsed = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(int(token.split(":", 1)[1] if token.startswith("cuda:") else token))
    return parsed


allowed = parse_devices(devices)
result = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
gpus = []
for line in result.stdout.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        continue
    index = int(parts[0])
    if allowed is not None and index not in allowed:
        continue
    used = int(float(parts[2]))
    total = int(float(parts[3]))
    gpu = {
        "index": index,
        "device": f"cuda:{index}",
        "name": parts[1],
        "memory_used_mb": used,
        "memory_total_mb": total,
        "memory_free_mb": total - used,
        "utilization_gpu": int(float(parts[4])),
        "temperature_c": int(float(parts[5])),
    }
    gpu["eligible_111"] = gpu["memory_free_mb"] >= min_free_111
    gpu["eligible_7days"] = gpu["memory_free_mb"] >= min_free_7days
    gpus.append(gpu)
gpus.sort(key=lambda item: item["memory_free_mb"], reverse=True)
print(json.dumps({"devices": devices, "min_free_mb_111": min_free_111, "min_free_mb_7days": min_free_7days, "gpus": gpus}, ensure_ascii=False))
"""


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return cleaned or "endpoint-full-suite"


def extract_arg(args: list[str], key: str, default: str) -> str:
    if key in args:
        index = args.index(key)
        if index + 1 < len(args):
            return args[index + 1]
    prefix = key + "="
    for item in args:
        if item.startswith(prefix):
            return item[len(prefix) :]
    return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a remote full endpoint-calibration suite: 111_days first, then 7days1~4."
    )
    add_remote_target_args(parser)
    add_remote_project_root_arg(parser)
    add_remote_conda_args(parser)
    parser.add_argument("--python-exec", default="python")
    parser.add_argument("--train-script", default="train.py")
    parser.add_argument("--name", default="endpoint-proto-hit-full")
    parser.add_argument("--save-dir", default="save_model_endpoint_proto_hit_full")
    parser.add_argument("--devices", default="auto", help="auto or comma-separated devices, e.g. cuda:2,cuda:3")
    parser.add_argument("--datasets-7days", default="7days1,7days2,7days3,7days4")
    parser.add_argument("--min-free-mb-111", type=int, default=12000)
    parser.add_argument("--min-free-mb-7days", type=int, default=7000)
    parser.add_argument("--max-7days-parallel", type=int, default=0, help="0 uses all eligible devices.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-minutes", type=int, default=720)
    parser.add_argument("--continue-on-111-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect eligible GPUs; do not launch training.")
    parser.add_argument("--status", default="", help="Read status for an existing suite name.")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Extra train.py args after '--'.")
    return parser


def default_train_args() -> list[str]:
    return [
        "--n_proto",
        "96",
        "--topk_proto",
        "5",
        "--micro_per_proto",
        "4",
        "--basis_dim",
        "16",
        "--local_basis_dim",
        "2",
        "--support_aware_local_basis",
        "--two_stage_decoder",
        "--no_two_stage_rescore",
        "--endpoint_conditioning",
        "proto",
        "--endpoint_residual_supervision",
        "hit_only",
    ]


def normalize_train_args(raw_args: list[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    return default_train_args() + args


def print_gpu_preflight(payload: dict[str, Any]) -> None:
    print("GPU preflight:")
    print(f"- devices={payload.get('devices')} min111={payload.get('min_free_mb_111')}MB min7days={payload.get('min_free_mb_7days')}MB")
    for gpu in payload.get("gpus", []):
        print(
            f"- {gpu['device']} free={gpu['memory_free_mb']}MB used={gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB "
            f"util={gpu['utilization_gpu']}% eligible111={gpu['eligible_111']} eligible7days={gpu['eligible_7days']}"
        )


def print_status(payload: dict[str, Any]) -> None:
    print(f"Suite: {payload.get('suite_name')}")
    print(f"Status: {payload.get('status')} phase={payload.get('phase')} updated={payload.get('updated_at')}")
    if payload.get("message"):
        print(f"Message: {payload.get('message')}")
    print(f"Save dir: {payload.get('save_dir')}")
    for job in payload.get("jobs", []):
        best = job.get("best20", {})
        final = job.get("final", {})
        print(
            f"- {job.get('dataset')} [{job.get('device')}] {job.get('status')} "
            f"best ADE20={best.get('ADE@20')} FDE20={best.get('FDE@20')} epoch={best.get('epoch')} "
            f"final ADE20={final.get('ADE@20')} epoch={final.get('epoch')}"
        )


def run_dry_run(args: argparse.Namespace) -> int:
    env_prefix = " ".join(
        [
            f"DEVICES={shlex.quote(args.devices)}",
            f"MIN_FREE_MB_111={args.min_free_mb_111}",
            f"MIN_FREE_MB_7DAYS={args.min_free_mb_7days}",
        ]
    )
    command = f"{env_prefix} python3 - <<'PY'\n{REMOTE_GPU_PREFLIGHT_SCRIPT}\nPY"
    stdout = run_target_command(
        args.host,
        args.project_root,
        args.user,
        args.port,
        command,
        password_env=args.password_env,
    )
    print_gpu_preflight(json.loads(stdout))
    return 0


def run_status(args: argparse.Namespace) -> int:
    suite_name = sanitize_name(args.status)
    status_path = f"{args.project_root}/.remote_runs/{suite_name}/suite_status.json"
    command = f"test -f {shlex.quote(status_path)} && cat {shlex.quote(status_path)}"
    stdout = run_target_command(
        args.host,
        args.project_root,
        args.user,
        args.port,
        command,
        password_env=args.password_env,
    )
    print_status(json.loads(stdout))
    return 0


def launch_suite(args: argparse.Namespace) -> int:
    train_args = normalize_train_args(args.train_args)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_name = f"{timestamp}_{sanitize_name(args.name)}"
    seed = extract_arg(train_args, "--seed", "3407")
    datasets_7days = [item.strip() for item in args.datasets_7days.split(",") if item.strip()]
    if not datasets_7days:
        print("No 7days datasets specified.", file=sys.stderr)
        return 1

    suite_root = f"{args.project_root}/.remote_runs/{suite_name}"
    supervisor_path = f"{suite_root}/supervisor.py"
    stdout_path = f"{suite_root}/supervisor_stdout.log"
    stderr_path = f"{suite_root}/supervisor_stderr.log"
    pid_path = f"{suite_root}/supervisor.pid"
    env_parts = {
        "PROJECT_ROOT": args.project_root,
        "SUITE_NAME": suite_name,
        "CONDA_SH": args.conda_sh,
        "CONDA_ENV": args.conda_env,
        "PYTHON_EXEC": args.python_exec,
        "TRAIN_SCRIPT": args.train_script,
        "SAVE_DIR": args.save_dir,
        "SEED": seed,
        "DEVICES": args.devices,
        "MIN_FREE_MB_111": str(args.min_free_mb_111),
        "MIN_FREE_MB_7DAYS": str(args.min_free_mb_7days),
        "MAX_7DAYS_PARALLEL": str(args.max_7days_parallel),
        "POLL_SECONDS": str(args.poll_seconds),
        "WAIT_MINUTES": str(args.wait_minutes),
        "BASE_TRAIN_ARGS_JSON": json.dumps(train_args, ensure_ascii=False),
        "DATASETS_7DAYS_JSON": json.dumps(datasets_7days, ensure_ascii=False),
        "STOP_ON_111_FAILURE": "0" if args.continue_on_111_failure else "1",
    }
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items())
    command = (
        f"mkdir -p {shlex.quote(suite_root)} && "
        f"cat > {shlex.quote(supervisor_path)} <<'PY'\n{REMOTE_SUPERVISOR_SCRIPT}\nPY\n"
        f"nohup env {env_prefix} python3 {shlex.quote(supervisor_path)} "
        f"> {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)} < /dev/null & "
        f"pid=$!; echo $pid > {shlex.quote(pid_path)}; "
        f"python3 - <<PY\n"
        f"import json\n"
        f"print(json.dumps({{'suite_name': {suite_name!r}, 'pid': int(open({pid_path!r}).read()), "
        f"'suite_root': {suite_root!r}, 'status_path': {suite_root + '/suite_status.json'!r}, "
        f"'save_dir': {args.save_dir!r}, 'train_args': {train_args!r}}}, ensure_ascii=False))\n"
        f"PY"
    )
    stdout = run_target_command(
        args.host,
        args.project_root,
        args.user,
        args.port,
        command,
        password_env=args.password_env,
        timeout=120,
    )
    payload = json.loads(stdout)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Status command: python scripts\\run_remote_endpoint_full_suite.py --status {suite_name}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.status:
        return run_status(args)
    if args.dry_run:
        return run_dry_run(args)
    return launch_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
