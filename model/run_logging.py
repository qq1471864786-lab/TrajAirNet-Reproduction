import json
import os
from datetime import datetime

import numpy as np
import torch



def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")



def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)



def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


class RunRecorder:
    def __init__(self, run_dir, config, extra_metadata=None):
        self.run_dir = run_dir
        self.config = dict(config)
        self.extra_metadata = extra_metadata or {}
        self.started_at = datetime.now()
        self.history = []

        os.makedirs(run_dir, exist_ok=True)
        self.run_config_path = os.path.join(run_dir, "run_config.json")
        self.epoch_metrics_path = os.path.join(run_dir, "epoch_metrics.jsonl")
        self.summary_path = os.path.join(run_dir, "run_summary.json")
        self.live_status_path = os.path.join(run_dir, "live_status.json")

        for path in (self.run_config_path, self.epoch_metrics_path, self.summary_path, self.live_status_path):
            if os.path.exists(path):
                os.remove(path)

        _write_json(
            self.run_config_path,
            {
                "started_at": self.started_at,
                "config": self.config,
                "extra_metadata": self.extra_metadata,
            },
        )
        _write_json(
            self.live_status_path,
            {
                "status": "running",
                "started_at": self.started_at,
                "current_epoch": 0,
                "best": {},
                "extra_metadata": self.extra_metadata,
            },
        )

    def log_epoch(self, epoch, phase_name, train_loss, metrics, lr, peak_memory_mb):
        payload = {
            "timestamp": datetime.now(),
            "epoch": epoch,
            "phase": phase_name,
            "train_loss": train_loss,
            "lr": lr,
            "peak_memory_mb": peak_memory_mb,
            "metrics": metrics,
        }
        self.history.append(payload)
        _append_jsonl(self.epoch_metrics_path, payload)

    def update_live_status(self, epoch, best):
        _write_json(
            self.live_status_path,
            {
                "status": "running",
                "started_at": self.started_at,
                "updated_at": datetime.now(),
                "current_epoch": epoch,
                "best": best,
                "extra_metadata": self.extra_metadata,
            },
        )

    def finalize(self, best):
        payload = {
            "status": "completed",
            "started_at": self.started_at,
            "finished_at": datetime.now(),
            "history_length": len(self.history),
            "best": best,
            "run_dir": self.run_dir,
            "extra_metadata": self.extra_metadata,
        }
        _write_json(self.summary_path, payload)
        _write_json(
            self.live_status_path,
            {
                "status": "completed",
                "started_at": self.started_at,
                "finished_at": datetime.now(),
                "current_epoch": len(self.history),
                "best": best,
                "extra_metadata": self.extra_metadata,
            },
        )

    def finalize_incomplete(self, status, error_message):
        payload = {
            "status": status,
            "started_at": self.started_at,
            "finished_at": datetime.now(),
            "history_length": len(self.history),
            "error_message": error_message,
            "run_dir": self.run_dir,
            "extra_metadata": self.extra_metadata,
        }
        _write_json(self.summary_path, payload)
        _write_json(self.live_status_path, payload)
