import argparse
import json
import shlex
import subprocess
import sys
from typing import Any


REMOTE_DISCOVER_SCRIPT = r"""
import glob
import json
import os
import pwd
import shutil
import socket
import subprocess
from datetime import datetime

project_root = os.environ["PROJECT_ROOT"]
requested_run_dir = os.environ.get("RUN_DIR", "").strip()


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"_load_error": str(exc), "_path": path}


def tail_file(path, n=40):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [line.rstrip("\n") for line in f.readlines()[-n:]]
    except Exception as exc:
        return [f"[load error] {exc}"]


def run_command(command):
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}
    return {
        "ok": True,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_snapshot():
    head = run_command(["git", "-C", project_root, "rev-parse", "HEAD"])
    status = run_command(["git", "-C", project_root, "status", "--short"])
    branch = run_command(["git", "-C", project_root, "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "head": head["stdout"] if head["ok"] else "",
        "branch": branch["stdout"] if branch["ok"] else "",
        "status_lines": status["stdout"].splitlines() if status["ok"] and status["stdout"] else [],
        "status_error": status["error"] if not status["ok"] else "",
    }


def memory_snapshot():
    meminfo = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "MemTotal": meminfo.get("MemTotal", ""),
        "MemAvailable": meminfo.get("MemAvailable", ""),
        "SwapTotal": meminfo.get("SwapTotal", ""),
        "SwapFree": meminfo.get("SwapFree", ""),
    }


def gpu_snapshot():
    gpu_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    gpus = []
    if gpu_query["ok"] and gpu_query["stdout"]:
        for line in gpu_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "utilization_gpu": parts[2],
                    "memory_used_mb": parts[3],
                    "memory_total_mb": parts[4],
                    "temperature_c": parts[5],
                }
            )

    processes = []
    if process_query["ok"] and process_query["stdout"]:
        for line in process_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            processes.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_gpu_memory_mb": parts[3],
                }
            )

    return {
        "gpus": gpus,
        "processes": processes,
        "query_error": gpu_query["error"] if not gpu_query["ok"] else "",
    }


def discover_project_processes():
    results = []
    uid = os.getuid()
    user = pwd.getpwuid(uid).pw_name
    proc_root = "/proc"
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        pid = int(name)
        proc_dir = os.path.join(proc_root, name)
        try:
            stat_info = os.stat(proc_dir)
        except Exception:
            continue
        if stat_info.st_uid != uid:
            continue

        cmdline_path = os.path.join(proc_dir, "cmdline")
        cwd_path = os.path.join(proc_dir, "cwd")
        try:
            raw_cmd = open(cmdline_path, "rb").read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except Exception:
            raw_cmd = ""
        try:
            cwd = os.readlink(cwd_path)
        except Exception:
            cwd = ""

        if not raw_cmd and not cwd:
            continue
        if project_root not in cwd and project_root not in raw_cmd and "train.py" not in raw_cmd and "test.py" not in raw_cmd:
            continue

        results.append(
            {
                "pid": pid,
                "user": user,
                "cwd": cwd,
                "cmdline": raw_cmd,
            }
        )
    results.sort(key=lambda item: item["pid"])
    return results


def discover_launcher_run():
    runs_root = os.path.join(project_root, ".remote_runs")
    if not os.path.isdir(runs_root):
        return None
    candidates = []
    for name in os.listdir(runs_root):
        path = os.path.join(runs_root, name)
        if name == "latest":
            continue
        if os.path.isdir(path):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    run_root = candidates[0]
    meta = load_json(os.path.join(run_root, "meta.json"))
    pid_path = os.path.join(run_root, "pid")
    pid = None
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r", encoding="utf-8") as f:
                pid = int(f.read().strip())
        except Exception:
            pid = None
    proc_alive = bool(pid and os.path.exists(f"/proc/{pid}"))
    return {
        "run_root": run_root,
        "meta": meta,
        "pid": pid,
        "pid_alive": proc_alive,
        "stdout_tail": tail_file(os.path.join(run_root, "stdout.log"), 60),
        "stderr_tail": tail_file(os.path.join(run_root, "stderr.log"), 60),
        "command_tail": tail_file(os.path.join(run_root, "command.sh"), 20),
    }


launcher_run = discover_launcher_run()
project_processes = discover_project_processes()

run_dir = requested_run_dir
if not run_dir and launcher_run and isinstance(launcher_run.get("meta"), dict):
    run_dir = launcher_run["meta"].get("resolved_run_dir", "")

if not run_dir:
    candidates = sorted(
        glob.glob(os.path.join(project_root, "save_model", "*", "seed*")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    run_dir = candidates[0] if candidates else ""
elif not os.path.isabs(run_dir):
    run_dir = os.path.join(project_root, run_dir)

live_status = load_json(os.path.join(run_dir, "live_status.json")) if run_dir else None
run_summary = load_json(os.path.join(run_dir, "run_summary.json")) if run_dir else None
run_config = load_json(os.path.join(run_dir, "run_config.json")) if run_dir else None

epoch_tail = []
epoch_metrics_path = os.path.join(run_dir, "epoch_metrics.jsonl") if run_dir else ""
if epoch_metrics_path and os.path.exists(epoch_metrics_path):
    try:
        with open(epoch_metrics_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-5:]
        for line in lines:
            line = line.strip()
            if line:
                epoch_tail.append(json.loads(line))
    except Exception as exc:
        epoch_tail = [{"_load_error": str(exc), "_path": epoch_metrics_path}]

root_logs = []
for name in sorted(os.listdir(project_root)):
    if name.endswith(".log") or name == "nohup.out":
        path = os.path.join(project_root, name)
        root_logs.append(
            {
                "name": name,
                "path": path,
                "mtime": os.path.getmtime(path),
                "tail": tail_file(path, 40),
            }
        )
root_logs.sort(key=lambda item: item["mtime"], reverse=True)

disk_total, disk_used, disk_free = shutil.disk_usage(project_root)

payload = {
    "server_time": datetime.now().isoformat(),
    "hostname": socket.gethostname(),
    "project_root": project_root,
    "project_processes": project_processes,
    "run_dir": run_dir,
    "launcher_run": launcher_run,
    "live_status": live_status,
    "run_summary": run_summary,
    "run_config": run_config,
    "epoch_tail": epoch_tail,
    "root_logs": root_logs[:3],
    "system_snapshot": {
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else [],
        "disk": {
            "total_bytes": disk_total,
            "used_bytes": disk_used,
            "free_bytes": disk_free,
        },
        "memory": memory_snapshot(),
        "gpu": gpu_snapshot(),
        "git": git_snapshot(),
    },
}
print(json.dumps(payload, ensure_ascii=False))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect remote training status for ProtoBasis-Net.")
    parser.add_argument("--host", default="10.23.66.99")
    parser.add_argument("--user", default="wangzhilin")
    parser.add_argument("--project-root", default="/home/wangzhilin/ProtoBasis-Net")
    parser.add_argument("--run-dir", default="", help="Absolute remote run dir or path relative to project root.")
    parser.add_argument("--show-logs", action="store_true", help="Print recent stdout/stderr tails.")
    parser.add_argument("--show-epochs", action="store_true", help="Print last few epoch JSON rows.")
    parser.add_argument("--show-system", action="store_true", help="Print GPU / memory / disk / git snapshot.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON payload for automation.")
    return parser.parse_args()


def run_ssh(user: str, host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", f"{user}@{host}", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def remote_json(user: str, host: str, project_root: str, run_dir: str) -> dict[str, Any]:
    env_parts = [f"PROJECT_ROOT={shlex.quote(project_root)}"]
    if run_dir:
        env_parts.append(f"RUN_DIR={shlex.quote(run_dir)}")
    env_prefix = " ".join(env_parts)
    command = f"{env_prefix} python3 - <<'PY'\n{REMOTE_DISCOVER_SCRIPT}\nPY"
    stdout = run_ssh(user, host, command)
    return json.loads(stdout)


def pick_best_metrics(status_payload: dict[str, Any]) -> dict[str, Any]:
    source = status_payload.get("live_status") or status_payload.get("run_summary") or {}
    best = source.get("best", {}) if isinstance(source, dict) else {}
    compact = {}
    for key, value in best.items():
        if isinstance(value, dict):
            compact[key] = {
                "value": value.get("value"),
                "epoch": value.get("epoch"),
                "path": value.get("path"),
            }
        else:
            compact[key] = value
    return compact


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    args = parse_args()
    try:
        payload = remote_json(args.user, args.host, args.project_root, args.run_dir)
    except subprocess.CalledProcessError as exc:
        print("远程状态查询失败。", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print_section("Remote Project")
    print(payload.get("project_root", args.project_root))
    print(f"host: {payload.get('hostname', '-')}")
    print(f"server_time: {payload.get('server_time', '-')}")

    print_section("Project Processes")
    processes = payload.get("project_processes") or []
    if processes:
        for item in processes:
            print(f"pid={item.get('pid')} cwd={item.get('cwd')}")
            print(f"cmd={item.get('cmdline')}")
    else:
        print("No active project-related process")

    launcher_run = payload.get("launcher_run") or {}
    if launcher_run:
        print_section("Latest Launcher Run")
        meta = launcher_run.get("meta") or {}
        print(f"run_root: {launcher_run.get('run_root')}")
        print(f"pid: {launcher_run.get('pid')}")
        print(f"pid_alive: {launcher_run.get('pid_alive')}")
        print(f"requested_name: {meta.get('run_name')}")
        print(f"resolved_run_dir: {meta.get('resolved_run_dir')}")
        print(f"launched_at: {meta.get('launched_at')}")
        print(f"command: {meta.get('train_command')}")

    print_section("Resolved Run Dir")
    print(payload.get("run_dir") or "(none)")

    live_status = payload.get("live_status") or {}
    run_summary = payload.get("run_summary") or {}
    status = live_status.get("status") or run_summary.get("status") or "unknown"
    current_epoch = live_status.get("current_epoch", "-")
    error_message = run_summary.get("error_message") or live_status.get("error_message")
    print_section("Run Status")
    print(f"status: {status}")
    print(f"current_epoch: {current_epoch}")
    if error_message:
        print(f"error: {error_message}")

    print_section("Best Metrics")
    best = pick_best_metrics(payload)
    if best:
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        print("(no best metrics yet)")

    if args.show_epochs:
        print_section("Epoch Tail")
        epoch_tail = payload.get("epoch_tail") or []
        if epoch_tail:
            print(json.dumps(epoch_tail, ensure_ascii=False, indent=2))
        else:
            print("(no epoch metrics)")

    if args.show_system:
        system_snapshot = payload.get("system_snapshot") or {}
        print_section("System Snapshot")
        print(f"loadavg: {system_snapshot.get('loadavg')}")

        disk = system_snapshot.get("disk") or {}
        if disk:
            gib = 1024 ** 3
            print(
                "disk_gib: "
                f"used={disk.get('used_bytes', 0) / gib:.2f} "
                f"free={disk.get('free_bytes', 0) / gib:.2f} "
                f"total={disk.get('total_bytes', 0) / gib:.2f}"
            )

        memory = system_snapshot.get("memory") or {}
        if memory:
            print(
                "memory: "
                f"MemAvailable={memory.get('MemAvailable', '-')}, "
                f"MemTotal={memory.get('MemTotal', '-')}, "
                f"SwapFree={memory.get('SwapFree', '-')}"
            )

        gpu = system_snapshot.get("gpu") or {}
        gpus = gpu.get("gpus") or []
        if gpus:
            print("\n--- GPUs ---")
            for item in gpus:
                print(
                    f"gpu{item.get('index')}: {item.get('name')} "
                    f"util={item.get('utilization_gpu')}% "
                    f"mem={item.get('memory_used_mb')}/{item.get('memory_total_mb')} MB "
                    f"temp={item.get('temperature_c')}C"
                )
        else:
            print("gpu: (no GPU snapshot)")
            if gpu.get("query_error"):
                print(f"gpu_error: {gpu.get('query_error')}")

        gpu_processes = gpu.get("processes") or []
        if gpu_processes:
            print("\n--- GPU Processes ---")
            for item in gpu_processes:
                print(
                    f"pid={item.get('pid')} name={item.get('process_name')} "
                    f"gpu_mem={item.get('used_gpu_memory_mb')} MB"
                )

        git_info = system_snapshot.get("git") or {}
        if git_info:
            print("\n--- Git ---")
            print(f"branch: {git_info.get('branch') or '-'}")
            print(f"head: {git_info.get('head') or '-'}")
            status_lines = git_info.get("status_lines") or []
            if status_lines:
                print("status:")
                for line in status_lines[:20]:
                    print(line)
            else:
                print("status: clean")

    if args.show_logs:
        print_section("Recent Launcher Logs")
        if launcher_run:
            print("\n--- command.sh ---")
            for line in launcher_run.get("command_tail") or ["(empty)"]:
                print(line)
            print("\n--- stdout.log ---")
            for line in launcher_run.get("stdout_tail") or ["(empty)"]:
                print(line)
            print("\n--- stderr.log ---")
            for line in launcher_run.get("stderr_tail") or ["(empty)"]:
                print(line)
        else:
            print("(no launcher run found)")

        print_section("Recent Root Logs")
        root_logs = payload.get("root_logs") or []
        if not root_logs:
            print("(no .log / nohup.out found)")
        for item in root_logs:
            print(f"\n--- {item['name']} ({item['path']}) ---")
            tail = item.get("tail") or []
            if tail:
                for line in tail:
                    print(line)
            else:
                print("(empty)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
