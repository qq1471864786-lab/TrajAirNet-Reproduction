import argparse
import json
from pathlib import Path

import numpy as np

from model.data import ProtoBasisSceneDataset, resolve_split_dir


def _solve_reconstruction(target, basis):
    target_flat = target.reshape(-1)
    coeff, *_ = np.linalg.lstsq(basis, target_flat, rcond=None)
    recon_flat = basis @ coeff
    return recon_flat.reshape(target.shape), coeff


def _old_anchor_path(summary_5d, pred_len):
    alpha = np.linspace(0.0, 1.0, num=pred_len, dtype=np.float32)[:, None]
    endpoint = summary_5d[:3].astype(np.float32)
    return alpha * endpoint[None, :]


def run_diagnostic(args):
    project_root = str(Path(__file__).resolve().parents[1])
    data_dir = resolve_split_dir(project_root, "social", args.dataset, "train")
    dataset = ProtoBasisSceneDataset(
        data_dir=data_dir,
        split_name="train",
        obs_len=args.obs,
        pred_len=args.preds,
        obs_stride=args.obs_stride,
        pred_stride=args.pred_stride,
        max_agents=args.max_agents,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        rare_threshold=args.rare_threshold,
    )

    samples = dataset.samples
    sample_count = len(dataset)
    limit = min(args.limit, sample_count) if args.limit > 0 else sample_count
    indices = np.arange(sample_count, dtype=np.int64)
    if limit < sample_count:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(indices, size=limit, replace=False)

    global_basis = dataset.basis_bank.reshape(dataset.basis_bank.shape[0], -1).T
    has_proto_paths = "prototype_mean_path" in dataset.model_artifact
    has_local_basis = "local_basis_bank" in dataset.model_artifact
    local_basis_bank = None
    proto_paths = None
    local_basis_dim = 0
    if has_proto_paths:
        proto_paths = np.asarray(dataset.model_artifact["prototype_mean_path"], dtype=np.float32)
    if has_local_basis:
        local_basis_bank = np.asarray(dataset.model_artifact["local_basis_bank"], dtype=np.float32)
        local_basis_dim = int(local_basis_bank.shape[1])

    proto_base_ade = []
    proto_base_fde = []
    recon_ade = []
    recon_fde = []
    global_coeff_norm = []
    local_coeff_norm = []

    for sample_index in indices.tolist():
        proto_id = int(samples["gt_proto_id"][sample_index])
        future_local = samples["future_local"][sample_index]

        if proto_paths is not None:
            base_path = proto_paths[proto_id]
        else:
            base_path = _old_anchor_path(dataset.prototype_summary_5d[proto_id], dataset.pred_len)

        base_error = np.linalg.norm(base_path - future_local, axis=-1)
        proto_base_ade.append(float(base_error.mean()))
        proto_base_fde.append(float(base_error[-1]))

        residual = future_local - base_path
        basis = global_basis
        if local_basis_bank is not None:
            local_basis = local_basis_bank[proto_id].reshape(local_basis_dim, -1).T
            basis = np.concatenate([global_basis, local_basis], axis=1)

        recon_residual, coeff = _solve_reconstruction(residual, basis)
        recon = base_path + recon_residual
        recon_error = np.linalg.norm(recon - future_local, axis=-1)
        recon_ade.append(float(recon_error.mean()))
        recon_fde.append(float(recon_error[-1]))
        global_coeff_norm.append(float(np.linalg.norm(coeff[: global_basis.shape[1]])))
        if local_basis_bank is not None:
            local_coeff_norm.append(float(np.linalg.norm(coeff[global_basis.shape[1] :])))

    result = {
        "dataset": args.dataset,
        "obs": args.obs,
        "preds": args.preds,
        "sample_count": int(sample_count),
        "evaluated_samples": int(len(indices)),
        "summary_dim": int(dataset.prototype_summary_5d.shape[1]),
        "n_proto": int(dataset.prototype_summary_5d.shape[0]),
        "basis_dim_global": int(dataset.basis_bank.shape[0]),
        "basis_dim_local": int(local_basis_dim),
        "rare_proto_count": int(dataset.rare_proto_ids.shape[0]),
        "uses_prototype_mean_path": bool(has_proto_paths),
        "uses_local_basis": bool(has_local_basis),
        "proto_base_ADE_mean": float(np.mean(proto_base_ade)),
        "proto_base_FDE_mean": float(np.mean(proto_base_fde)),
        "recon_ADE_mean": float(np.mean(recon_ade)),
        "recon_FDE_mean": float(np.mean(recon_fde)),
        "global_coeff_norm_mean": float(np.mean(global_coeff_norm)),
        "local_coeff_norm_mean": float(np.mean(local_coeff_norm)) if local_coeff_norm else 0.0,
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Diagnose ProtoBasis artifact representation quality.")
    parser.add_argument("dataset", help="Dataset name such as 7days1 or 111_days.")
    parser.add_argument("--obs", type=int, default=40)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--obs-stride", type=int, default=1)
    parser.add_argument("--pred-stride", type=int, default=1)
    parser.add_argument("--max-agents", type=int, default=7)
    parser.add_argument("--n-proto", type=int, default=64)
    parser.add_argument("--basis-dim", type=int, default=16)
    parser.add_argument("--rare-threshold", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()
