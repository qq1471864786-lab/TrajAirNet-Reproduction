import argparse
import json
import os


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Show a concise summary for one training run.")
    parser.add_argument("--run_dir", required=True, help="Example: save_model/social/7days1/42")
    args = parser.parse_args()

    summary_path = os.path.join(args.run_dir, "run_summary.json")
    config_path = os.path.join(args.run_dir, "run_config.json")
    epoch_path = os.path.join(args.run_dir, "epoch_metrics.jsonl")
    checkpoint_path = os.path.join(args.run_dir, "checkpoint_events.jsonl")

    summary = read_json(summary_path)
    config = read_json(config_path)
    epochs = read_jsonl(epoch_path) if os.path.exists(epoch_path) else []
    checkpoints = read_jsonl(checkpoint_path) if os.path.exists(checkpoint_path) else []

    print(f"run_dir: {args.run_dir}")
    print(f"started_at: {config['started_at']}")
    print(f"dataset: {config['args']['dataset_variant']}/{config['args']['dataset_name']}")
    print(f"seed: {config['args']['seed']}")
    print(f"best_epoch: {summary.get('best_epoch')}")
    if summary.get("best_metrics"):
        print("best_metrics:", " ".join(f"{k}={v:.4f}" for k, v in summary["best_metrics"].items()))
    if epochs:
        last = epochs[-1]
        print("last_epoch:", f"{last['epoch']}", " ".join(f"{k}={v:.4f}" for k, v in last["metrics"].items()))
    if checkpoints:
        last_ckpt = checkpoints[-1]
        print(f"last_checkpoint_event: epoch={last_ckpt['epoch']} path={last_ckpt['checkpoint_path']}")
    print("diagnostics:")
    for item in summary.get("diagnostics", []):
        print(f"- {item}")
    print("suggestions:")
    for item in summary.get("suggestions", []):
        print(f"- {item}")


if __name__ == "__main__":
    main()
