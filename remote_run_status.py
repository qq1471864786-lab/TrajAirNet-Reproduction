import argparse
import json
import shlex
from pathlib import PurePosixPath

from deploy import REMOTE_PATH, SERVER_IP, USERNAME, resolve_ssh_options, run_command


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one remote training run.")
    parser.add_argument("--dataset_variant", required=True, choices=["social", "no_social"])
    parser.add_argument("--dataset_name", default="7days1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="save_model")
    return parser.parse_args()


def remote_cat(remote_file):
    ssh_options = resolve_ssh_options()
    result = run_command(
        [
            "ssh",
            *ssh_options,
            f"{USERNAME}@{SERVER_IP}",
            f"cat {shlex.quote(remote_file)}",
        ],
        capture_output=True,
    )
    return result.stdout


def remote_exists(remote_file):
    ssh_options = resolve_ssh_options()
    result = run_command(
        [
            "ssh",
            *ssh_options,
            f"{USERNAME}@{SERVER_IP}",
            f"test -f {shlex.quote(remote_file)} && echo 1 || echo 0",
        ],
        capture_output=True,
    )
    return result.stdout.strip() == "1"


def remote_pid_alive(pid):
    ssh_options = resolve_ssh_options()
    result = run_command(
        [
            "ssh",
            *ssh_options,
            f"{USERNAME}@{SERVER_IP}",
            f"kill -0 {int(pid)} >/dev/null 2>&1 && echo 1 || echo 0",
        ],
        capture_output=True,
    )
    return result.stdout.strip() == "1"


def main():
    args = parse_args()
    run_dir = PurePosixPath(REMOTE_PATH) / args.save_dir / args.dataset_variant / args.dataset_name / str(args.seed)
    live_status_path = str(run_dir / "live_status.json")
    summary_path = str(run_dir / "run_summary.json")
    checkpoint_events_path = str(run_dir / "checkpoint_events.jsonl")

    print(f"remote_run_dir: {run_dir}")
    if not remote_exists(live_status_path):
        raise SystemExit("live_status.json not found on remote run directory.")

    live_status = json.loads(remote_cat(live_status_path))
    status = live_status["status"]
    pid = live_status.get("pid")
    if status == "running" and pid is not None and not remote_pid_alive(pid):
        status = "stale_running"
    print(f"status: {status}")
    if pid is not None:
        print(f"pid: {pid}")
    print(f"current_epoch: {live_status['current_epoch']}")
    print(f"best_epoch: {live_status['best_epoch']}")
    if live_status.get("best_metrics"):
        print("best_metrics:", " ".join(f"{k}={v:.4f}" for k, v in live_status["best_metrics"].items()))
    if live_status.get("latest_metrics"):
        print("latest_metrics:", " ".join(f"{k}={v:.4f}" for k, v in live_status["latest_metrics"].items()))
    if live_status.get("latest_checkpoint"):
        print(f"latest_checkpoint: {live_status['latest_checkpoint']}")
    if live_status.get("error_message"):
        print(f"error_message: {live_status['error_message']}")

    if remote_exists(checkpoint_events_path):
        raw_lines = remote_cat(checkpoint_events_path).strip().splitlines()
        if raw_lines:
            event = json.loads(raw_lines[-1])
            print(f"last_save_event: epoch={event['epoch']} reason={event['reason']}")

    if status in {"completed", "failed", "interrupted", "stale_running"} and remote_exists(summary_path):
        summary = json.loads(remote_cat(summary_path))
        print("diagnostics:")
        for item in summary.get("diagnostics", []):
            print(f"- {item}")
        print("suggestions:")
        for item in summary.get("suggestions", []):
            print(f"- {item}")


if __name__ == "__main__":
    main()
