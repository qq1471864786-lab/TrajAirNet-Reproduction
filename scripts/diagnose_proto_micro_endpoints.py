import argparse
import json
from pathlib import Path

import numpy as np

from model.data import ProtoBasisSceneDataset, _run_kmeans, resolve_split_dir


def _anchor_metrics(future_local: np.ndarray, endpoints: np.ndarray, pred_len: int) -> tuple[float, float]:
    alpha = np.linspace(0.0, 1.0, num=pred_len, dtype=np.float32)[:, None]
    errors = []
    final_errors = []
    for index in range(future_local.shape[0]):
        path = alpha * endpoints[index][None, :]
        l2 = np.linalg.norm(path - future_local[index], axis=-1)
        errors.append(float(l2.mean()))
        final_errors.append(float(l2[-1]))
    return float(np.mean(errors)), float(np.mean(final_errors))


def main():
    parser = argparse.ArgumentParser(description="Diagnose whether each prototype contains useful endpoint sub-modes.")
    parser.add_argument("dataset", help="Dataset name such as 111_days or 7days1.")
    parser.add_argument("--obs", type=int, default=40)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--obs-stride", type=int, default=1)
    parser.add_argument("--pred-stride", type=int, default=1)
    parser.add_argument("--max-agents", type=int, default=7)
    parser.add_argument("--n-proto", type=int, default=64)
    parser.add_argument("--basis-dim", type=int, default=16)
    parser.add_argument("--rare-threshold", type=float, default=0.02)
    parser.add_argument("--n-micro", type=int, default=4)
    args = parser.parse_args()

    project_root = str(Path(__file__).resolve().parents[1])
    train_dir = resolve_split_dir(project_root, "social", args.dataset, "train")
    test_dir = resolve_split_dir(project_root, "social", args.dataset, "test")

    train_ds = ProtoBasisSceneDataset(
        train_dir,
        "train",
        obs_len=args.obs,
        pred_len=args.preds,
        obs_stride=args.obs_stride,
        pred_stride=args.pred_stride,
        max_agents=args.max_agents,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        rare_threshold=args.rare_threshold,
    )
    artifact = train_ds.export_model_artifact()
    test_ds = ProtoBasisSceneDataset(
        test_dir,
        "test",
        obs_len=args.obs,
        pred_len=args.preds,
        obs_stride=args.obs_stride,
        pred_stride=args.pred_stride,
        max_agents=args.max_agents,
        model_artifact=artifact,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        rare_threshold=args.rare_threshold,
    )

    train_future = train_ds.samples["future_local"]
    train_proto = train_ds.samples["gt_proto_id"]
    test_future = test_ds.samples["future_local"]
    test_proto = test_ds.samples["gt_proto_id"]
    proto_endpoints = train_ds.prototype_summary_5d[:, :3].astype(np.float32)

    micro_centers = np.zeros((proto_endpoints.shape[0], args.n_micro, 3), dtype=np.float32)
    proto_counts = []
    for proto_id in range(proto_endpoints.shape[0]):
        proto_mask = train_proto == proto_id
        proto_end_samples = train_future[proto_mask, -1]
        proto_counts.append(int(proto_mask.sum()))
        if proto_end_samples.shape[0] >= args.n_micro:
            centers, _ = _run_kmeans(
                proto_end_samples,
                n_clusters=args.n_micro,
                random_state=3407 + proto_id,
                iters=30,
            )
            micro_centers[proto_id] = centers
        elif proto_end_samples.shape[0] > 0:
            micro_centers[proto_id, : proto_end_samples.shape[0]] = proto_end_samples
            micro_centers[proto_id, proto_end_samples.shape[0] :] = proto_end_samples[-1]
        else:
            micro_centers[proto_id] = proto_endpoints[proto_id][None, :]

    base_endpoints = proto_endpoints[test_proto]
    base_ade, base_fde = _anchor_metrics(test_future, base_endpoints, test_ds.pred_len)

    micro_endpoints = np.zeros_like(base_endpoints)
    for index, proto_id in enumerate(test_proto):
        candidates = micro_centers[proto_id]
        endpoint = test_future[index, -1]
        distances = ((candidates - endpoint[None, :]) ** 2).sum(axis=1)
        micro_endpoints[index] = candidates[np.argmin(distances)]
    micro_ade, micro_fde = _anchor_metrics(test_future, micro_endpoints, test_ds.pred_len)

    result = {
        "dataset": args.dataset,
        "n_proto": int(proto_endpoints.shape[0]),
        "n_micro": int(args.n_micro),
        "train_proto_min_count": int(min(proto_counts)),
        "train_proto_median_count": float(np.median(proto_counts)),
        "train_proto_mean_count": float(np.mean(proto_counts)),
        "base_anchor_ADE_test": float(base_ade),
        "base_anchor_FDE_test": float(base_fde),
        "micro_anchor_ADE_test": float(micro_ade),
        "micro_anchor_FDE_test": float(micro_fde),
        "delta_ADE": float(micro_ade - base_ade),
        "delta_FDE": float(micro_fde - base_fde),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
