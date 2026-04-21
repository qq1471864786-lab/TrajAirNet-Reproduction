import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


REMOTE_DISCOVER_SCRIPT = r"""
import glob
import json
import os
import pwd
import shutil
import socket
import subprocess
from collections import deque
from datetime import datetime

project_root = os.environ["PROJECT_ROOT"]
requested_run_dir = os.environ.get("RUN_DIR", "").strip()
include_logs = os.environ.get("INCLUDE_LOGS") == "1"
include_epochs = os.environ.get("INCLUDE_EPOCHS") == "1"
include_system = os.environ.get("INCLUDE_SYSTEM") == "1"
include_run_config = os.environ.get("INCLUDE_RUN_CONFIG") == "1"


def trim_text(text, limit=220):
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"_load_error": str(exc), "_path": path}


def read_jsonl_tail(path, n):
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = deque(handle, maxlen=n)
        for line in lines:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception as exc:
        rows = [{"_load_error": str(exc), "_path": path}]
    return rows


def tail_file(path, n=40):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[-n:]]
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


def git_snapshot(include_status_lines=False):
    head = run_command(["git", "-C", project_root, "rev-parse", "HEAD"])
    branch = run_command(["git", "-C", project_root, "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_command(["git", "-C", project_root, "status", "--short"])
    status_lines = status["stdout"].splitlines() if status["ok"] and status["stdout"] else []
    return {
        "head": head["stdout"] if head["ok"] else "",
        "branch": branch["stdout"] if branch["ok"] else "",
        "dirty_count": len(status_lines),
        "status_lines": status_lines[:40] if include_status_lines else [],
        "status_error": status["error"] if not status["ok"] else "",
    }


def memory_snapshot():
    meminfo = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
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


def gpu_snapshot(project_pids, include_all_processes=False):
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

    all_processes = []
    if process_query["ok"] and process_query["stdout"]:
        for line in process_query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            all_processes.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": trim_text(parts[2], limit=120),
                    "used_gpu_memory_mb": parts[3],
                }
            )

    project_processes = [item for item in all_processes if item["pid"] in project_pids]
    return {
        "gpus": gpus,
        "project_processes": project_processes,
        "processes": all_processes if include_all_processes else [],
        "query_error": gpu_query["error"] if not gpu_query["ok"] else "",
    }


def discover_project_processes():
    results = []
    uid = os.getuid()
    user = pwd.getpwuid(uid).pw_name
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        proc_dir = os.path.join("/proc", name)
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
        compact_cmd = trim_text(raw_cmd)
        try:
            cwd = os.readlink(cwd_path)
        except Exception:
            cwd = ""

        if not compact_cmd and not cwd:
            continue
        if project_root not in cwd and project_root not in raw_cmd and "train.py" not in raw_cmd and "test.py" not in raw_cmd:
            continue
        if compact_cmd in {"bash", "-bash", "/bin/bash"}:
            continue
        if "python3 - <<'PY'" in compact_cmd or "server_status.py" in compact_cmd:
            continue

        results.append(
            {
                "pid": pid,
                "user": user,
                "cwd": cwd,
                "cmdline": compact_cmd,
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
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    run_root = candidates[0]
    meta = load_json(os.path.join(run_root, "meta.json"))
    pid_path = os.path.join(run_root, "pid")
    pid = None
    if os.path.exists(pid_path):
        try:
            with open(pid_path, "r", encoding="utf-8") as handle:
                pid = int(handle.read().strip())
        except Exception:
            pid = None

    payload = {
        "run_root": run_root,
        "meta": meta,
        "pid": pid,
        "pid_alive": bool(pid and os.path.exists(f"/proc/{pid}")),
    }
    if include_logs:
        payload["stdout_tail"] = tail_file(os.path.join(run_root, "stdout.log"), 60)
        payload["stderr_tail"] = tail_file(os.path.join(run_root, "stderr.log"), 60)
        payload["command_tail"] = tail_file(os.path.join(run_root, "command.sh"), 20)
    return payload


def resolve_run_dir(launcher_run):
    run_dir = requested_run_dir
    if not run_dir and launcher_run and isinstance(launcher_run.get("meta"), dict):
        run_dir = launcher_run["meta"].get("resolved_run_dir", "")

    if not run_dir:
        candidates = sorted(
            glob.glob(os.path.join(project_root, "save_model", "*", "seed*")),
            key=lambda path: os.path.getmtime(path),
            reverse=True,
        )
        return candidates[0] if candidates else ""
    if os.path.isabs(run_dir):
        return run_dir
    return os.path.join(project_root, run_dir)


def discover_root_logs():
    logs = []
    for name in sorted(os.listdir(project_root)):
        if not name.endswith(".log") and name != "nohup.out":
            continue
        path = os.path.join(project_root, name)
        logs.append(
            {
                "name": name,
                "path": path,
                "mtime": os.path.getmtime(path),
                "tail": tail_file(path, 40),
            }
        )
    logs.sort(key=lambda item: item["mtime"], reverse=True)
    return logs[:3]


def latest_epoch_record(live_status, epoch_metrics_path):
    if isinstance(live_status, dict):
        latest = live_status.get("latest_epoch")
        if isinstance(latest, dict) and latest:
            return latest
    tail_rows = read_jsonl_tail(epoch_metrics_path, 1)
    if tail_rows:
        return tail_rows[-1]
    return {}


launcher_run = discover_launcher_run()
project_processes = discover_project_processes()
project_pids = {str(item["pid"]) for item in project_processes}
run_dir = resolve_run_dir(launcher_run)

live_status = load_json(os.path.join(run_dir, "live_status.json")) if run_dir else None
run_summary = load_json(os.path.join(run_dir, "run_summary.json")) if run_dir else None
run_config = load_json(os.path.join(run_dir, "run_config.json")) if run_dir and include_run_config else None
epoch_metrics_path = os.path.join(run_dir, "epoch_metrics.jsonl") if run_dir else ""
latest_epoch = latest_epoch_record(live_status, epoch_metrics_path)

epoch_tail = read_jsonl_tail(epoch_metrics_path, 5) if include_epochs and epoch_metrics_path else []
root_logs = discover_root_logs() if include_logs else []

disk_total, disk_used, disk_free = shutil.disk_usage(project_root)
system_snapshot = {
    "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else [],
    "disk": {
        "total_bytes": disk_total,
        "used_bytes": disk_used,
        "free_bytes": disk_free,
    },
    "memory": memory_snapshot(),
    "gpu": gpu_snapshot(project_pids, include_all_processes=include_system),
    "git": git_snapshot(include_status_lines=include_system),
}

payload = {
    "server_time": datetime.now().isoformat(),
    "hostname": socket.gethostname(),
    "project_root": project_root,
    "project_processes": project_processes,
    "run_dir": run_dir,
    "launcher_run": launcher_run,
    "live_status": live_status,
    "run_summary": run_summary,
    "latest_epoch": latest_epoch,
    "system_snapshot": system_snapshot,
}
if include_epochs:
    payload["epoch_tail"] = epoch_tail
if include_logs:
    payload["root_logs"] = root_logs
if include_run_config:
    payload["run_config"] = run_config

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
    parser.add_argument("--json", action="store_true", help="Print JSON. Default is compact JSON; add --full for raw payload.")
    parser.add_argument("--full", action="store_true", help="Fetch full payload for JSON or deep inspection.")
    return parser.parse_args()


def run_ssh(user: str, host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", f"{user}@{host}", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_local(command: str) -> str:
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def local_addresses() -> set[str]:
    addresses = {"127.0.0.1", "::1", "localhost"}
    for name in {socket.gethostname(), socket.getfqdn()}:
        if name:
            addresses.add(name.lower())
        try:
            for _, _, _, _, sockaddr in socket.getaddrinfo(name, None):
                if sockaddr and sockaddr[0]:
                    addresses.add(sockaddr[0].lower())
        except OSError:
            continue
    return addresses


def should_run_locally(host: str, project_root: str) -> bool:
    if not Path(project_root).exists():
        return False

    host_norm = host.strip().lower()
    local_addrs = local_addresses()
    if host_norm in local_addrs:
        return True

    try:
        for _, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if sockaddr and sockaddr[0].lower() in local_addrs:
                return True
    except OSError:
        return False
    return False


def remote_json(
    user: str,
    host: str,
    project_root: str,
    run_dir: str,
    *,
    include_logs: bool,
    include_epochs: bool,
    include_system: bool,
    include_run_config: bool,
) -> dict[str, Any]:
    env_parts = [f"PROJECT_ROOT={shlex.quote(project_root)}"]
    if run_dir:
        env_parts.append(f"RUN_DIR={shlex.quote(run_dir)}")
    if include_logs:
        env_parts.append("INCLUDE_LOGS=1")
    if include_epochs:
        env_parts.append("INCLUDE_EPOCHS=1")
    if include_system:
        env_parts.append("INCLUDE_SYSTEM=1")
    if include_run_config:
        env_parts.append("INCLUDE_RUN_CONFIG=1")
    env_prefix = " ".join(env_parts)
    command = f"{env_prefix} python3 - <<'PY'\n{REMOTE_DISCOVER_SCRIPT}\nPY"
    if should_run_locally(host, project_root):
        stdout = run_local(command)
    else:
        stdout = run_ssh(user, host, command)
    return json.loads(stdout)


def format_scalar(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def shorten_head(head: str) -> str:
    return head[:7] if head else "-"


def bytes_to_gib(value: Any) -> float:
    try:
        return float(value) / (1024 ** 3)
    except Exception:
        return 0.0


def pick_best_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    source = payload.get("live_status") or payload.get("run_summary") or {}
    best = source.get("best", {}) if isinstance(source, dict) else {}
    compact: dict[str, Any] = {}
    for key, value in best.items():
        if isinstance(value, dict):
            compact[key] = {
                "value": value.get("value"),
                "epoch": value.get("epoch"),
            }
        else:
            compact[key] = value
    return compact


def select_metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    preferred = (
        "ADE@5",
        "FDE@5",
        "ADE@20",
        "FDE@20",
        "rare_FDE@20",
        "GLeV@20",
        "Top1_ADE",
        "Top1_FDE",
        "proto_top1_acc",
        "proto_rare_recall",
        "score_entropy",
        "endpoint_var",
    )
    selected = {name: metrics[name] for name in preferred if name in metrics}
    if selected:
        return selected
    return metrics


def build_compact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    live_status = payload.get("live_status") or {}
    run_summary = payload.get("run_summary") or {}
    latest_epoch = payload.get("latest_epoch") or {}
    system_snapshot = payload.get("system_snapshot") or {}
    launcher_run = payload.get("launcher_run") or {}
    launcher_meta = launcher_run.get("meta") or {}
    git_info = system_snapshot.get("git") or {}
    gpu_info = system_snapshot.get("gpu") or {}
    disk_info = system_snapshot.get("disk") or {}
    memory_info = system_snapshot.get("memory") or {}

    status = live_status.get("status") or run_summary.get("status") or "unknown"
    error_message = run_summary.get("error_message") or live_status.get("error_message")
    metrics = select_metric_subset(latest_epoch.get("metrics") or {})

    best_summary = {}
    for key, value in pick_best_metrics(payload).items():
        if isinstance(value, dict):
            best_summary[key] = {
                "value": value.get("value"),
                "epoch": value.get("epoch"),
            }
        else:
            best_summary[key] = value

    return {
        "host": payload.get("hostname"),
        "server_time": payload.get("server_time"),
        "project_root": payload.get("project_root"),
        "run_dir": payload.get("run_dir"),
        "status": status,
        "current_epoch": live_status.get("current_epoch", run_summary.get("current_epoch")),
        "phase": latest_epoch.get("phase"),
        "train_loss": latest_epoch.get("train_loss"),
        "lr": latest_epoch.get("lr"),
        "peak_memory_mb": latest_epoch.get("peak_memory_mb"),
        "metrics": metrics,
        "best": best_summary,
        "error": error_message,
        "launcher": {
            "run_root": launcher_run.get("run_root"),
            "run_name": launcher_meta.get("run_name"),
            "pid": launcher_run.get("pid"),
            "pid_alive": launcher_run.get("pid_alive"),
            "launched_at": launcher_meta.get("launched_at"),
            "resolved_run_dir": launcher_meta.get("resolved_run_dir"),
            "train_command": launcher_meta.get("train_command"),
        },
        "project_processes": [
            {
                "pid": item.get("pid"),
                "cmdline": item.get("cmdline"),
            }
            for item in (payload.get("project_processes") or [])
        ],
        "gpu": {
            "gpus": gpu_info.get("gpus") or [],
            "project_processes": gpu_info.get("project_processes") or [],
        },
        "git": {
            "branch": git_info.get("branch"),
            "head": shorten_head(git_info.get("head", "")),
            "dirty_count": git_info.get("dirty_count", 0),
        },
        "memory": {
            "MemAvailable": memory_info.get("MemAvailable"),
            "SwapFree": memory_info.get("SwapFree"),
        },
        "disk_gib": {
            "used": round(bytes_to_gib(disk_info.get("used_bytes", 0)), 2),
            "free": round(bytes_to_gib(disk_info.get("free_bytes", 0)), 2),
            "total": round(bytes_to_gib(disk_info.get("total_bytes", 0)), 2),
        },
    }


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_compact_summary(summary: dict[str, Any]) -> None:
    print_section("Remote Snapshot")
    print(f"host: {summary.get('host') or '-'}")
    print(f"server_time: {summary.get('server_time') or '-'}")
    print(f"project_root: {summary.get('project_root') or '-'}")
    print(f"run_dir: {summary.get('run_dir') or '(none)'}")

    status_line = f"status: {summary.get('status') or 'unknown'}"
    current_epoch = summary.get("current_epoch")
    if current_epoch not in (None, ""):
        status_line += f" epoch={current_epoch}"
    phase = summary.get("phase")
    if phase:
        status_line += f" phase={phase}"
    print(status_line)

    latest_parts = []
    if summary.get("train_loss") is not None:
        latest_parts.append(f"loss={format_scalar(summary.get('train_loss'))}")
    if summary.get("lr") is not None:
        latest_parts.append(f"lr={format_scalar(summary.get('lr'), digits=6)}")
    if summary.get("peak_memory_mb") is not None:
        latest_parts.append(f"mem={format_scalar(summary.get('peak_memory_mb'), digits=1)}MB")
    metrics = summary.get("metrics") or {}
    for name in ("ADE@5", "FDE@5", "ADE@20", "FDE@20", "rare_FDE@20", "GLeV@20"):
        if name in metrics:
            latest_parts.append(f"{name}={format_scalar(metrics[name])}")
    if latest_parts:
        print("latest: " + " ".join(latest_parts))

    best = summary.get("best") or {}
    if best:
        best_tokens = []
        for key, value in best.items():
            if isinstance(value, dict):
                best_tokens.append(f"{key}={format_scalar(value.get('value'))}@e{value.get('epoch')}")
            else:
                best_tokens.append(f"{key}={format_scalar(value)}")
        print("best: " + " ".join(best_tokens))

    launcher = summary.get("launcher") or {}
    if launcher.get("run_name") or launcher.get("pid"):
        print(
            "launcher: "
            f"name={launcher.get('run_name') or '-'} "
            f"pid={launcher.get('pid') or '-'} "
            f"alive={launcher.get('pid_alive')} "
            f"launched_at={launcher.get('launched_at') or '-'}"
        )

    processes = summary.get("project_processes") or []
    if processes:
        print("project_processes:")
        for item in processes:
            print(f"  pid={item.get('pid')} cmd={item.get('cmdline')}")
    else:
        print("project_processes: none")

    gpu = summary.get("gpu") or {}
    gpu_lines = []
    for item in gpu.get("gpus") or []:
        gpu_lines.append(
            f"gpu{item.get('index')}:util={item.get('utilization_gpu')}% "
            f"mem={item.get('memory_used_mb')}/{item.get('memory_total_mb')}MB "
            f"temp={item.get('temperature_c')}C"
        )
    if gpu_lines:
        print("gpu: " + " | ".join(gpu_lines))
    else:
        print("gpu: unavailable")

    gpu_processes = gpu.get("project_processes") or []
    if gpu_processes:
        print("gpu_project_processes:")
        for item in gpu_processes:
            print(
                f"  pid={item.get('pid')} name={item.get('process_name')} "
                f"gpu_mem={item.get('used_gpu_memory_mb')}MB"
            )

    git_info = summary.get("git") or {}
    print(
        "git: "
        f"branch={git_info.get('branch') or '-'} "
        f"head={git_info.get('head') or '-'} "
        f"dirty={git_info.get('dirty_count', 0)}"
    )

    memory = summary.get("memory") or {}
    disk = summary.get("disk_gib") or {}
    print(
        "resources: "
        f"MemAvailable={memory.get('MemAvailable') or '-'} "
        f"SwapFree={memory.get('SwapFree') or '-'} "
        f"disk_free={disk.get('free', 0):.2f}GiB"
    )

    if summary.get("error"):
        print(f"error: {summary.get('error')}")


def print_system_snapshot(payload: dict[str, Any]) -> None:
    system_snapshot = payload.get("system_snapshot") or {}
    print_section("System Snapshot")
    print(f"loadavg: {system_snapshot.get('loadavg')}")

    disk = system_snapshot.get("disk") or {}
    if disk:
        print(
            "disk_gib: "
            f"used={bytes_to_gib(disk.get('used_bytes', 0)):.2f} "
            f"free={bytes_to_gib(disk.get('free_bytes', 0)):.2f} "
            f"total={bytes_to_gib(disk.get('total_bytes', 0)):.2f}"
        )

    memory = system_snapshot.get("memory") or {}
    if memory:
        print(
            "memory: "
            f"MemAvailable={memory.get('MemAvailable', '-')}, "
            f"MemTotal={memory.get('MemTotal', '-')}, "
            f"SwapFree={memory.get('SwapFree', '-')}, "
            f"SwapTotal={memory.get('SwapTotal', '-')}"
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
        print(f"dirty_count: {git_info.get('dirty_count', 0)}")
        status_lines = git_info.get("status_lines") or []
        if status_lines:
            print("status:")
            for line in status_lines:
                print(line)
        else:
            print("status: clean")


def print_epoch_tail(payload: dict[str, Any]) -> None:
    print_section("Epoch Tail")
    epoch_tail = payload.get("epoch_tail") or []
    if epoch_tail:
        print(json.dumps(epoch_tail, ensure_ascii=False, indent=2))
    else:
        print("(no epoch metrics)")


def print_logs(payload: dict[str, Any]) -> None:
    launcher_run = payload.get("launcher_run") or {}
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
        return
    for item in root_logs:
        print(f"\n--- {item['name']} ({item['path']}) ---")
        tail = item.get("tail") or []
        if tail:
            for line in tail:
                print(line)
        else:
            print("(empty)")


def main() -> int:
    args = parse_args()
    include_logs = args.show_logs or args.full
    include_epochs = args.show_epochs or args.full
    include_system = args.show_system or args.full
    include_run_config = args.full

    try:
        payload = remote_json(
            args.user,
            args.host,
            args.project_root,
            args.run_dir,
            include_logs=include_logs,
            include_epochs=include_epochs,
            include_system=include_system,
            include_run_config=include_run_config,
        )
    except subprocess.CalledProcessError as exc:
        print("远程状态查询失败。", file=sys.stderr)
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        return 1

    compact = build_compact_summary(payload)
    if args.json:
        if args.full:
            payload["compact"] = compact
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(compact, ensure_ascii=False, indent=2))
        return 0

    print_compact_summary(compact)

    if args.show_epochs:
        print_epoch_tail(payload)
    if args.show_system:
        print_system_snapshot(payload)
    if args.show_logs:
        print_logs(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
