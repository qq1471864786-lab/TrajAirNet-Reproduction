import argparse
import json
from pathlib import Path

import numpy as np

from model.data import ProtoBasisSceneDataset, resolve_split_dir


def _fit_basis(residual_bank: np.ndarray, basis_dim: int) -> np.ndarray:
    flat = residual_bank.reshape(residual_bank.shape[0], -1).astype(np.float32, copy=False)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    rank = min(basis_dim, vt.shape[0])
    basis_flat = vt[:rank]
    if rank < basis_dim:
        pad = np.zeros((basis_dim - rank, flat.shape[1]), dtype=np.float32)
        basis_flat = np.concatenate([basis_flat, pad], axis=0)
    return basis_flat.reshape(basis_dim, residual_bank.shape[1], residual_bank.shape[2]).astype(np.float32)


def _solve_reconstruction(target: np.ndarray, basis: np.ndarray) -> np.ndarray:
    target_flat = target.reshape(-1)
    if basis.size == 0:
        return np.zeros_like(target)
    coeff, *_ = np.linalg.lstsq(basis, target_flat, rcond=None)
    recon_flat = basis @ coeff
    return recon_flat.reshape(target.shape)


def _metrics(futures: np.ndarray, bases: np.ndarray) -> tuple[float, float]:
    error = np.linalg.norm(bases - futures, axis=-1)
    return float(error.mean()), float(error[:, -1].mean())


def main():
    parser = argparse.ArgumentParser(description="Diagnose proto/path/local-basis representation upper bounds.")
    parser.add_argument("dataset", help="Dataset name such as 111_days or 7days1.")
    parser.add_argument("--obs", type=int, default=40)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--obs-stride", type=int, default=1)
    parser.add_argument("--pred-stride", type=int, default=1)
    parser.add_argument("--max-agents", type=int, default=7)
    parser.add_argument("--n-proto", type=int, default=64)
    parser.add_argument("--basis-dim", type=int, default=16)
    parser.add_argument("--local-basis-dim", type=int, default=4)
    parser.add_argument("--rare-threshold", type=float, default=0.02)
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
    pred_len = train_ds.pred_len

    alpha = np.linspace(0.0, 1.0, num=pred_len, dtype=np.float32)[:, None]
    proto_end = train_ds.prototype_summary_5d[:, :3].astype(np.float32)
    global_basis = train_ds.basis_bank.reshape(train_ds.basis_bank.shape[0], -1).T

    proto_mean_path = np.zeros((args.n_proto, pred_len, 3), dtype=np.float32)
    local_basis = np.zeros((args.n_proto, args.local_basis_dim, pred_len, 3), dtype=np.float32)
    proto_counts = []
    for proto_id in range(args.n_proto):
        mask = train_proto == proto_id
        proto_samples = train_future[mask]
        proto_counts.append(int(mask.sum()))
        if proto_samples.shape[0] == 0:
            proto_mean_path[proto_id] = alpha * proto_end[proto_id][None, :]
            continue
        proto_mean_path[proto_id] = proto_samples.mean(axis=0)
        local_basis[proto_id] = _fit_basis(proto_samples - proto_mean_path[proto_id][None, :], args.local_basis_dim)

    anchor_path = alpha[None, :, :] * proto_end[test_proto][:, None, :]
    mean_path = proto_mean_path[test_proto]

    current_recon = np.zeros_like(test_future)
    mean_global_recon = np.zeros_like(test_future)
    mean_local_recon = np.zeros_like(test_future)
    mean_gl_recon = np.zeros_like(test_future)

    for idx in range(test_future.shape[0]):
        residual_anchor = test_future[idx] - anchor_path[idx]
        current_recon[idx] = anchor_path[idx] + _solve_reconstruction(residual_anchor, global_basis)

        residual_mean = test_future[idx] - mean_path[idx]
        local = local_basis[test_proto[idx]].reshape(args.local_basis_dim, -1).T
        gl = np.concatenate([global_basis, local], axis=1)
        mean_global_recon[idx] = mean_path[idx] + _solve_reconstruction(residual_mean, global_basis)
        mean_local_recon[idx] = mean_path[idx] + _solve_reconstruction(residual_mean, local)
        mean_gl_recon[idx] = mean_path[idx] + _solve_reconstruction(residual_mean, gl)

    result = {
        "dataset": args.dataset,
        "train_proto_min_count": int(min(proto_counts)),
        "train_proto_median_count": float(np.median(proto_counts)),
        "train_proto_mean_count": float(np.mean(proto_counts)),
        "anchor_only": {
            "ADE": _metrics(test_future, anchor_path)[0],
            "FDE": _metrics(test_future, anchor_path)[1],
        },
        "anchor_plus_global_basis": {
            "ADE": _metrics(test_future, current_recon)[0],
            "FDE": _metrics(test_future, current_recon)[1],
        },
        "proto_mean_path": {
            "ADE": _metrics(test_future, mean_path)[0],
            "FDE": _metrics(test_future, mean_path)[1],
        },
        "proto_mean_path_plus_global_basis": {
            "ADE": _metrics(test_future, mean_global_recon)[0],
            "FDE": _metrics(test_future, mean_global_recon)[1],
        },
        "proto_mean_path_plus_local_basis": {
            "ADE": _metrics(test_future, mean_local_recon)[0],
            "FDE": _metrics(test_future, mean_local_recon)[1],
        },
        "proto_mean_path_plus_global_local_basis": {
            "ADE": _metrics(test_future, mean_gl_recon)[0],
            "FDE": _metrics(test_future, mean_gl_recon)[1],
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
