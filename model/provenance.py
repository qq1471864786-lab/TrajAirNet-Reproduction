import hashlib
import json
import os
from typing import Any, Dict

import numpy as np


def _normalize(value: Any):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def stable_hash(payload: Any) -> str:
    normalized = _normalize(payload)
    encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_hash(config: Dict[str, Any]) -> str:
    keys = [
        "protocol_name",
        "dataset_variant",
        "dataset_name",
        "obs",
        "preds",
        "obs_stride",
        "pred_stride",
        "obs_horizon_sec",
        "pred_horizon_sec",
        "max_agents",
        "topk_proto",
        "micro_per_proto",
        "basis_dim",
        "micro_coeff_anchors",
        "n_proto",
        "eval_topk_primary",
        "eval_topk_secondary",
        "glev_topn_primary",
        "glev_topn_secondary",
    ]
    subset = {key: config.get(key) for key in keys}
    return stable_hash(subset)


def model_artifact_hash(model_artifact: Dict[str, Any]) -> str:
    subset = {
        "summary_5d": model_artifact["summary_5d"],
        "frequency": model_artifact["frequency"],
        "rare_ids": model_artifact["rare_ids"],
        "basis_bank": model_artifact["basis_bank"],
        "micro_coeff_anchors": model_artifact.get("micro_coeff_anchors"),
        "n_proto": model_artifact["n_proto"],
        "basis_dim": model_artifact["basis_dim"],
        "micro_per_proto": model_artifact.get("micro_per_proto"),
        "rare_threshold": model_artifact["rare_threshold"],
        "obs_len": model_artifact.get("obs_len"),
        "pred_len": model_artifact.get("pred_len"),
        "obs_stride": model_artifact.get("obs_stride"),
        "pred_stride": model_artifact.get("pred_stride"),
    }
    return stable_hash(subset)


def basis_hash(model_artifact: Dict[str, Any]) -> str:
    return stable_hash(model_artifact["basis_bank"])


def git_commit(project_root: str) -> str:
    git_dir = os.path.join(project_root, ".git")
    head_path = os.path.join(git_dir, "HEAD")
    if not os.path.exists(head_path):
        return "unknown"
    with open(head_path, "r", encoding="utf-8") as handle:
        head = handle.read().strip()
    if head.startswith("ref:"):
        ref_path = os.path.join(git_dir, head.split(" ", 1)[1].strip().replace("/", os.sep))
        if os.path.exists(ref_path):
            with open(ref_path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        return "unknown"
    return head or "unknown"
