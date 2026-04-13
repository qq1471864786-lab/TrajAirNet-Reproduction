import json
import os
from datetime import datetime


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _safe_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > 1e-12 else 0.0


def _select_primary_metric(metrics):
    for candidate in ("ADE_best5", "ADE", "ADE_1"):
        if candidate in metrics:
            return candidate
    return next(iter(metrics.keys()))


def _select_vertical_metric(metrics):
    for candidate in ("z-ADE_best5", "z-ADE", "z-ADE_1"):
        if candidate in metrics:
            return candidate
    return None


def build_run_summary(history, best_epoch, best_metrics, best_checkpoint):
    if not history:
        return {
            "status": "empty",
            "diagnostics": ["No epoch history was recorded."],
            "suggestions": ["Check whether training exited before the first evaluation."],
        }

    first = history[0]
    last = history[-1]
    diagnostics = []
    suggestions = []
    primary_key = _select_primary_metric(best_metrics)
    vertical_key = _select_vertical_metric(best_metrics)

    primary_drop = first["metrics"][primary_key] - best_metrics[primary_key]
    best_epoch_ratio = _safe_ratio(best_epoch, len(history))

    if best_epoch_ratio <= 0.35 and last["metrics"][primary_key] > best_metrics[primary_key] * 1.10:
        diagnostics.append(
            f"Best {primary_key} appeared early and later epochs regressed, indicating instability or overfitting."
        )
        suggestions.append("Try reducing learning rate or stopping earlier.")

    if best_epoch == len(history) and last["metrics"][primary_key] <= first["metrics"][primary_key] * 0.75:
        diagnostics.append(f"{primary_key} was still improving at the last epoch.")
        suggestions.append("Increase epochs before changing architecture.")

    if last["train_loss"] > first["train_loss"] * 0.90 and last["metrics"][primary_key] > first["metrics"][primary_key] * 0.90:
        diagnostics.append(f"Training loss and {primary_key} both improved only slightly.")
        suggestions.append("This run is likely undertrained or under-capacity.")

    if (
        vertical_key is not None
        and best_metrics[primary_key] < last["metrics"][primary_key]
        and best_metrics[vertical_key] > last["metrics"].get(vertical_key, best_metrics[vertical_key])
    ):
        diagnostics.append("Trajectory and vertical objectives peaked at different epochs.")
        suggestions.append("Recheck state-conditioning strength because the main and vertical objectives are peaking at different epochs.")

    if primary_drop <= 0:
        diagnostics.append(f"{primary_key} did not improve over the run.")
        suggestions.append("Check optimizer settings or data batching before changing the model.")

    if vertical_key is not None:
        vertical_drop = first["metrics"][vertical_key] - best_metrics[vertical_key]
        if vertical_drop <= 0:
            diagnostics.append(f"{vertical_key} did not improve over the run.")
            suggestions.append("Check vertical-state features and conditioning strength.")

    if not diagnostics:
        diagnostics.append("Run behavior looked stable under the current training budget.")
        suggestions.append("Promote this setting to a larger-budget run if the result is worth keeping.")

    return {
        "status": "completed",
        "history_length": len(history),
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "primary_metric": primary_key,
        "best_metrics": best_metrics,
        "last_epoch_metrics": last["metrics"],
        "last_train_loss": last["train_loss"],
        "last_model_state": last.get("model_state"),
        "diagnostics": diagnostics,
        "suggestions": suggestions,
    }


class RunRecorder:
    def __init__(self, run_dir, args, extra_metadata=None):
        self.run_dir = run_dir
        self.args = dict(args)
        self.extra_metadata = extra_metadata or {}
        self.history = []
        self.started_at = datetime.now()
        self.pid = os.getpid()
        self.run_config_path = os.path.join(run_dir, "run_config.json")
        self.epoch_metrics_path = os.path.join(run_dir, "epoch_metrics.jsonl")
        self.checkpoint_events_path = os.path.join(run_dir, "checkpoint_events.jsonl")
        self.summary_path = os.path.join(run_dir, "run_summary.json")
        self.live_status_path = os.path.join(run_dir, "live_status.json")

        os.makedirs(run_dir, exist_ok=True)
        for path in (
            self.run_config_path,
            self.epoch_metrics_path,
            self.checkpoint_events_path,
            self.summary_path,
            self.live_status_path,
        ):
            if os.path.exists(path):
                os.remove(path)
        _write_json(
            self.run_config_path,
            {
                "started_at": self.started_at,
                "pid": self.pid,
                "args": self.args,
                "extra_metadata": self.extra_metadata,
            },
        )
        _write_json(
            self.live_status_path,
            {
                "status": "running",
                "started_at": self.started_at,
                "pid": self.pid,
                "current_epoch": 0,
                "best_epoch": 0,
                "best_metrics": None,
                "latest_metrics": None,
                "latest_train_loss": None,
                "latest_model_state": None,
                "latest_checkpoint": None,
            },
        )

    def log_epoch(
        self,
        epoch,
        train_loss,
        metrics,
        train_batches,
        eval_scenes,
        best_epoch,
        best_metrics,
        best_checkpoint,
        model_state=None,
    ):
        payload = {
            "timestamp": datetime.now(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_batches": train_batches,
            "eval_scenes": eval_scenes,
            "metrics": metrics,
        }
        if model_state is not None:
            payload["model_state"] = model_state
        self.history.append(payload)
        _append_jsonl(self.epoch_metrics_path, payload)
        _write_json(
            self.live_status_path,
            {
                "status": "running",
                "started_at": self.started_at,
                "pid": self.pid,
                "updated_at": datetime.now(),
                "current_epoch": epoch,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "latest_metrics": metrics,
                "latest_train_loss": train_loss,
                "latest_model_state": model_state,
                "latest_checkpoint": best_checkpoint,
            },
        )

    def log_checkpoint(self, epoch, checkpoint_path, metrics, reason):
        _append_jsonl(
            self.checkpoint_events_path,
            {
                "timestamp": datetime.now(),
                "epoch": epoch,
                "checkpoint_path": checkpoint_path,
                "reason": reason,
                "metrics": metrics,
            },
        )

    def finalize(self, best_epoch, best_metrics, best_checkpoint):
        summary = build_run_summary(self.history, best_epoch, best_metrics, best_checkpoint)
        summary["finished_at"] = datetime.now()
        summary["run_dir"] = self.run_dir
        _write_json(self.summary_path, summary)
        _write_json(
            self.live_status_path,
            {
                "status": "completed",
                "started_at": self.started_at,
                "pid": self.pid,
                "finished_at": summary["finished_at"],
                "current_epoch": len(self.history),
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "latest_metrics": summary.get("last_epoch_metrics"),
                "latest_train_loss": summary.get("last_train_loss"),
                "latest_model_state": summary.get("last_model_state"),
                "latest_checkpoint": best_checkpoint,
            },
        )
        return summary

    def finalize_incomplete(self, status, error_message=None):
        if status not in {"failed", "interrupted"}:
            raise ValueError(f"Unsupported incomplete status: {status}")

        finished_at = datetime.now()
        payload = {
            "status": status,
            "started_at": self.started_at,
            "pid": self.pid,
            "finished_at": finished_at,
            "current_epoch": len(self.history),
            "best_epoch": 0,
            "best_metrics": None,
            "latest_metrics": None,
            "latest_train_loss": None,
            "latest_model_state": None,
            "latest_checkpoint": None,
        }
        if self.history:
            last = self.history[-1]
            best_entry = min(self.history, key=lambda item: item["metrics"]["ADE"])
            payload.update(
                {
                    "best_epoch": best_entry["epoch"],
                    "best_metrics": best_entry["metrics"],
                    "latest_metrics": last["metrics"],
                    "latest_train_loss": last["train_loss"],
                    "latest_model_state": last.get("model_state"),
                }
            )

        if error_message:
            payload["error_message"] = error_message

        _write_json(self.live_status_path, payload)
        _write_json(
            self.summary_path,
            {
                **payload,
                "run_dir": self.run_dir,
                "history_length": len(self.history),
                "diagnostics": [error_message] if error_message else [],
                "suggestions": [],
            },
        )
        return payload
