import argparse
import json
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import (  # noqa: E402
    ProtoBasisNet,
    ProtoBasisSceneDataset,
    proto_basis_collate,
    summarize_batch_metrics,
)
from model.data import resolve_split_dir  # noqa: E402
from model.proto_basis_flight_model import build_anchor, build_global_features, build_local_features  # noqa: E402


def _as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_artifact(checkpoint):
    return {
        "summary_5d": _as_numpy(checkpoint["proto_summary_5d"]).astype(np.float32),
        "frequency": _as_numpy(checkpoint["proto_freq"]).astype(np.float32),
        "rare_ids": np.nonzero(_as_numpy(checkpoint["proto_freq"]) < checkpoint["config"]["rare_threshold"])[0].astype(
            np.int64
        ),
        "basis_bank": _as_numpy(checkpoint["basis_bank"]).astype(np.float32),
        "prototype_mean_path": _as_numpy(checkpoint.get("prototype_mean_path")).astype(np.float32),
        "local_basis_bank": _as_numpy(checkpoint.get("local_basis_bank")).astype(np.float32),
        "micro_coeff_anchors": _as_numpy(checkpoint.get("micro_coeff_anchors")).astype(np.float32)
        if checkpoint.get("micro_coeff_anchors") is not None
        else None,
        "micro_endpoint_anchors": _as_numpy(checkpoint.get("micro_endpoint_anchors")).astype(np.float32)
        if checkpoint.get("micro_endpoint_anchors") is not None
        else None,
        "n_proto": checkpoint["config"]["n_proto"],
        "basis_dim": checkpoint["config"]["basis_dim"],
        "local_basis_dim": checkpoint["config"].get("local_basis_dim", 0),
        "micro_per_proto": checkpoint["config"].get("micro_per_proto", 0)
        if checkpoint["config"].get("micro_coeff_anchors", False)
        else 0,
        "rare_threshold": checkpoint["config"]["rare_threshold"],
        "obs_len": checkpoint["config"]["obs"],
        "pred_len": checkpoint["config"]["preds"],
        "obs_stride": checkpoint["config"].get("obs_stride", 1),
        "pred_stride": checkpoint["config"].get("pred_stride", 1),
    }


def _build_model(config, checkpoint):
    micro_endpoint_anchors = checkpoint.get("micro_endpoint_anchors")
    micro_endpoint_scale = config.get("micro_endpoint_scale")
    if micro_endpoint_scale is None:
        micro_endpoint_scale = 1.0 if micro_endpoint_anchors is not None and config.get("micro_coeff_anchors", False) else 0.0
    model = ProtoBasisNet(
        obs_len=config["obs"],
        pred_len=config["preds"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        ff_dim=config["ff_dim"],
        encoder_layers=config["encoder_layers"],
        social_layers=config["social_layers"],
        topk_proto=config["topk_proto"],
        n_micro=config["micro_per_proto"],
        n_proto=config["n_proto"],
        basis_dim=config["basis_dim"],
        local_basis_dim=config.get("local_basis_dim", 0),
        support_aware_local_basis=config.get("support_aware_local_basis", False),
        two_stage_decoder=bool(config.get("two_stage_decoder", False)),
        two_stage_update_endpoint=not config.get("no_two_stage_update_endpoint", False),
        two_stage_update_coeff=not config.get("no_two_stage_update_coeff", False),
        two_stage_rescore=bool(config.get("two_stage_rescore", True)),
        dropout=config["dropout"],
        proto_summary_5d=torch.tensor(checkpoint["proto_summary_5d"], dtype=torch.float32),
        proto_frequency=torch.tensor(checkpoint["proto_freq"], dtype=torch.float32),
        basis_bank=torch.tensor(checkpoint["basis_bank"], dtype=torch.float32),
        prototype_mean_path=torch.tensor(checkpoint.get("prototype_mean_path"), dtype=torch.float32)
        if checkpoint.get("prototype_mean_path") is not None
        else None,
        local_basis_bank=torch.tensor(checkpoint.get("local_basis_bank"), dtype=torch.float32)
        if checkpoint.get("local_basis_bank") is not None
        else None,
        micro_coeff_anchors=torch.tensor(checkpoint.get("micro_coeff_anchors"), dtype=torch.float32)
        if checkpoint.get("micro_coeff_anchors") is not None
        else None,
        micro_endpoint_anchors=torch.tensor(micro_endpoint_anchors, dtype=torch.float32)
        if micro_endpoint_anchors is not None
        else None,
        use_micro_coeff_anchors=bool(config.get("micro_coeff_anchors", False)),
        micro_endpoint_scale=float(micro_endpoint_scale),
        endpoint_conditioning=config.get("endpoint_conditioning", "rank"),
        disable_social=config.get("disable_social", False),
        disable_router=config.get("disable_router", False),
        disable_refiner=config.get("disable_refiner", False),
    )
    state_dict = checkpoint["model"]
    if "prototype_router.endpoint_head.0.weight" in state_dict:
        first_weight = state_dict["prototype_router.endpoint_head.0.weight"]
        last_weight = state_dict["prototype_router.endpoint_head.3.weight"]
        model.prototype_router.endpoint_head = nn.Sequential(
            nn.Linear(first_weight.shape[1], first_weight.shape[0]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(last_weight.shape[1], last_weight.shape[0]),
        )
        load_result = model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in load_result.unexpected_keys if not key.startswith("prototype_router.generic_endpoint_head")]
        missing = [key for key in load_result.missing_keys if not key.startswith("prototype_router.generic_endpoint_head")]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint compatibility load mismatch: missing={missing}, unexpected={unexpected}")
        return model
    model.load_state_dict(state_dict)
    return model


def _metric_from_paths(pred, gt):
    error = np.linalg.norm(pred - gt, axis=-1)
    return {
        "ADE": float(error.mean()),
        "FDE": float(error[:, -1].mean()),
        "ADE_first40": float(error[:, :40].mean()) if error.shape[1] >= 40 else float(error.mean()),
        "ADE_mid40": float(error[:, 40:80].mean()) if error.shape[1] >= 80 else None,
        "ADE_last40": float(error[:, -40:].mean()) if error.shape[1] >= 40 else float(error.mean()),
    }


def _solve_linear_reconstruction(anchor, gt, basis_matrix):
    residual = (gt - anchor).reshape(gt.shape[0], -1)
    if basis_matrix.size == 0:
        return anchor.copy()
    pinv = np.linalg.pinv(basis_matrix).astype(np.float32)
    coeff = residual @ pinv
    recon = coeff @ basis_matrix
    return anchor + recon.reshape(gt.shape)


def _nearest_bank_oracle(test_summary, test_future, train_summary, train_future, chunk_size=512):
    ade_values = []
    fde_values = []
    for start in range(0, test_summary.shape[0], chunk_size):
        end = min(start + chunk_size, test_summary.shape[0])
        dist = ((test_summary[start:end, None] - train_summary[None]) ** 2).sum(axis=-1)
        nearest = dist.argmin(axis=1)
        pred = train_future[nearest]
        error = np.linalg.norm(pred - test_future[start:end], axis=-1)
        ade_values.append(error.mean(axis=1))
        fde_values.append(error[:, -1])
    ade = np.concatenate(ade_values)
    fde = np.concatenate(fde_values)
    return {
        "ADE": float(ade.mean()),
        "FDE": float(fde.mean()),
        "ADE_p50": float(np.percentile(ade, 50)),
        "ADE_p90": float(np.percentile(ade, 90)),
    }


def _collect_motion_baselines(dataset, indices):
    hold_errors = []
    cv_errors = []
    for index in indices:
        item = dataset[int(index)]
        obs = item["obs_xyz"][0].numpy()
        fut = item["fut_xyz"].numpy()
        last = obs[-1]
        velocity = obs[-1] - obs[-2]
        steps = np.arange(1, fut.shape[0] + 1, dtype=np.float32)[:, None]
        hold = np.repeat(last[None], fut.shape[0], axis=0)
        cv = last[None] + steps * velocity[None]
        hold_errors.append(np.linalg.norm(hold - fut, axis=-1))
        cv_errors.append(np.linalg.norm(cv - fut, axis=-1))
    hold_errors = np.stack(hold_errors)
    cv_errors = np.stack(cv_errors)
    return {
        "hold_last": {"ADE": float(hold_errors.mean()), "FDE": float(hold_errors[:, -1].mean())},
        "constant_velocity": {"ADE": float(cv_errors.mean()), "FDE": float(cv_errors[:, -1].mean())},
    }


def _evaluate_model(model, loader, device, config, variant, limit_batches=0):
    model.eval()
    primary_k = config.get("eval_topk_primary", 5)
    secondary_k = config.get("eval_topk_secondary", 20)
    metric_sums = {}
    total = 0
    rare_total = 0
    endpoint_errors = []
    horizon_errors = []
    xy_errors = []
    z_errors = []
    endpoint_oracle_stats = {
        "pred_endpoint_straight": [],
        "pred_endpoint_plus_global_basis_ls": [],
        "gt_endpoint_plus_pred_coeff": [],
    }
    router_hits = 0
    basis_matrix = model.basis_bank.basis_bank.reshape(model.basis_bank.basis_bank.size(0), -1)
    basis_pinv = torch.linalg.pinv(basis_matrix).to(device)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if limit_batches and batch_index >= limit_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            force_gt = variant == "force_gt_proto"
            with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
                state = _forward_with_local_state(
                    model,
                    batch["obs_xyz"],
                    batch["obs_mask"],
                    gt_proto_id=batch["gt_proto_id"] if force_gt else None,
                    force_gt_proto=force_gt,
                    enable_refiner=True,
                )
            outputs = state["outputs"]
            metrics, count, rare_count = summarize_batch_metrics(
                outputs,
                batch,
                primary_k=primary_k,
                secondary_k=secondary_k,
                glev_topn_primary=config.get("glev_topn_primary", 2),
                glev_topn_secondary=config.get("glev_topn_secondary", 5),
            )
            for name, value in metrics.items():
                if np.isnan(value):
                    continue
                metric_sums[name] = metric_sums.get(name, 0.0) + value * count
            total += count
            rare_total += rare_count

            pred = outputs["pred_xyz"]
            gt = batch["fut_xyz"]
            l2 = torch.linalg.norm(pred - gt[:, None], dim=-1)
            ade = l2.mean(dim=-1)
            best_idx = ade.argmin(dim=1)
            gather_idx = best_idx[:, None, None, None].expand(-1, 1, pred.size(2), pred.size(3))
            best = torch.gather(pred, 1, gather_idx).squeeze(1)
            endpoint_errors.append(torch.linalg.norm(best[:, -1] - gt[:, -1], dim=-1).cpu())
            per_t = torch.linalg.norm(best - gt, dim=-1)
            horizon_errors.append(per_t.cpu())
            xy_errors.append(torch.linalg.norm(best[..., :2] - gt[..., :2], dim=-1).cpu())
            z_errors.append(torch.abs(best[..., 2] - gt[..., 2]).cpu())
            router_hits += int(outputs["top_proto_idx"].eq(batch["gt_proto_id"][:, None]).any(dim=1).sum().item())

            gt_local = batch["fut_local"].float()
            endpoint_local = state["endpoint_mode_local"].float()
            coeff = state["coeff"].float()
            alpha = model.anchor_alpha.to(endpoint_local.device, dtype=endpoint_local.dtype)
            pred_endpoint_anchor = build_anchor(endpoint_local, alpha)
            gt_endpoint = gt_local[:, -1]
            gt_endpoint_anchor = build_anchor(gt_endpoint[:, None].expand_as(endpoint_local), alpha)
            gt_rep = gt_local[:, None].expand_as(pred_endpoint_anchor)

            straight_error = torch.linalg.norm(pred_endpoint_anchor - gt_rep, dim=-1).mean(dim=-1)
            endpoint_oracle_stats["pred_endpoint_straight"].append(straight_error.min(dim=1).values.cpu())

            residual = (gt_rep - pred_endpoint_anchor).reshape(-1, gt_local.size(1) * 3)
            ls_coeff = residual @ basis_pinv
            ls_recon = ls_coeff @ basis_matrix
            ls_path = pred_endpoint_anchor.reshape(-1, gt_local.size(1) * 3) + ls_recon
            ls_path = ls_path.reshape_as(pred_endpoint_anchor)
            ls_error = torch.linalg.norm(ls_path - gt_rep, dim=-1).mean(dim=-1)
            endpoint_oracle_stats["pred_endpoint_plus_global_basis_ls"].append(ls_error.min(dim=1).values.cpu())

            gt_endpoint_pred_coeff = model.basis_bank(gt_endpoint_anchor, coeff)
            gt_endpoint_pred_coeff_error = torch.linalg.norm(gt_endpoint_pred_coeff - gt_rep, dim=-1).mean(dim=-1)
            endpoint_oracle_stats["gt_endpoint_plus_pred_coeff"].append(
                gt_endpoint_pred_coeff_error.min(dim=1).values.cpu()
            )

    averaged = {}
    for name, value in metric_sums.items():
        divisor = rare_total if name.startswith("rare_FDE@") else total
        averaged[name] = value / max(divisor, 1)
    horizon = torch.cat(horizon_errors, dim=0).numpy()
    xy = torch.cat(xy_errors, dim=0).numpy()
    z = torch.cat(z_errors, dim=0).numpy()
    endpoints = torch.cat(endpoint_errors, dim=0).numpy()
    averaged.update(
        {
            "count": total,
            "rare_count": rare_total,
            "router_topk_hit": router_hits / max(total, 1),
            "router_topk_miss": 1.0 - router_hits / max(total, 1),
            "best_endpoint_FDE_mean": float(endpoints.mean()),
            "best_endpoint_FDE_p90": float(np.percentile(endpoints, 90)),
            "best_ADE_first40": float(horizon[:, :40].mean()) if horizon.shape[1] >= 40 else float(horizon.mean()),
            "best_ADE_mid40": float(horizon[:, 40:80].mean()) if horizon.shape[1] >= 80 else None,
            "best_ADE_last40": float(horizon[:, -40:].mean()) if horizon.shape[1] >= 40 else float(horizon.mean()),
            "best_xy_ADE": float(xy.mean()),
            "best_z_ADE": float(z.mean()),
        }
    )
    averaged["endpoint_shape_oracles"] = {
        name: {
            "ADE": float(torch.cat(values).mean().item()),
            "ADE_p50": float(torch.quantile(torch.cat(values), 0.5).item()),
            "ADE_p90": float(torch.quantile(torch.cat(values), 0.9).item()),
        }
        for name, values in endpoint_oracle_stats.items()
    }
    return averaged


def _forward_with_local_state(model, obs_xyz, obs_mask, gt_proto_id=None, force_gt_proto=False, enable_refiner=True):
    local_xyz, _, _, origin, rotation = model.pose_normalizer(obs_xyz)
    feats_local = build_local_features(local_xyz)
    feats_global = build_global_features(obs_xyz)

    agent_feat = model.temporal_encoder(feats_local, feats_global)
    if model.disable_social:
        target_ctx = agent_feat[:, 0]
        scene_ctx = (agent_feat * obs_mask.unsqueeze(-1)).sum(dim=1) / obs_mask.sum(dim=1, keepdim=True).clamp_min(1)
    else:
        target_ctx, scene_ctx = model.social_aggregator(agent_feat, obs_mask)

    proto_logits, top_proto_idx, proto_token, endpoint_residual, endpoint_local = model.prototype_router(
        target_ctx,
        scene_ctx,
        model.proto_summary_5d,
        gt_proto_id=gt_proto_id,
        force_gt_proto=force_gt_proto,
        disable_router=model.disable_router,
    )
    micro_coeff_anchor = model.micro_coeff_anchors[top_proto_idx] if model.has_micro_coeff_anchors else None
    micro_endpoint_anchor = model.micro_endpoint_anchors[top_proto_idx] if model.has_micro_endpoint_anchors else None
    query_feat, coeff, pred_score, difficulty_gate, coeff_delta = model.query_decoder(
        proto_token,
        endpoint_local,
        target_ctx,
        agent_feat,
        obs_mask,
        micro_coeff_anchor=micro_coeff_anchor,
    )
    endpoint_mode_local = endpoint_local.repeat_interleave(model.n_micro, dim=1)
    if micro_endpoint_anchor is not None and model.micro_endpoint_scale != 0.0:
        endpoint_mode_local = endpoint_mode_local + model.micro_endpoint_scale * micro_endpoint_anchor.reshape(
            endpoint_mode_local.size(0),
            endpoint_mode_local.size(1),
            3,
        )
    anchor_local = build_anchor(endpoint_mode_local, model.anchor_alpha.to(endpoint_mode_local))
    coarse_local = model.basis_bank(anchor_local, coeff)
    active_query = query_feat
    if model.two_stage_decoder:
        stage2_input = torch.cat([query_feat, coarse_local[:, :, -1], coarse_local.mean(dim=2)], dim=-1)
        stage2_query = query_feat + model.stage2_proj(stage2_input)
        if model.two_stage_update_endpoint:
            endpoint_mode_local = endpoint_mode_local + model.stage2_endpoint_head(stage2_query)
        if model.two_stage_update_coeff:
            stage2_coeff_delta = model.stage2_coeff_head(stage2_query)
            coeff = coeff + stage2_coeff_delta
            coeff_delta = coeff_delta + stage2_coeff_delta
        if model.two_stage_rescore:
            pred_score = model.query_decoder.score_head(stage2_query).squeeze(-1)
            difficulty_gate = torch.sigmoid(model.query_decoder.gate_head(stage2_query)).squeeze(-1)
        anchor_local = build_anchor(endpoint_mode_local, model.anchor_alpha.to(endpoint_mode_local))
        coarse_local = model.basis_bank(anchor_local, coeff)
        active_query = stage2_query
    if model.has_local_basis:
        proto_mean_path = model.prototype_mean_path[top_proto_idx]
        proto_mean_path = proto_mean_path.repeat_interleave(model.n_micro, dim=1)
        proto_mean_endpoint = proto_mean_path[:, :, -1, :]
        aligned_proto_mean = proto_mean_path + (endpoint_mode_local - proto_mean_endpoint)[:, :, None, :]

        local_basis = model.local_basis_bank[top_proto_idx]
        local_basis = local_basis.repeat_interleave(model.n_micro, dim=1)
        local_coeff = model.local_coeff_head(active_query)
        local_residual = torch.einsum("bkm,bkmtd->bktd", local_coeff, local_basis)
        local_path = aligned_proto_mean + local_residual
        local_gate = torch.sigmoid(model.local_mix_gate(active_query)).unsqueeze(-1)
        if model.support_aware_local_basis:
            proto_support = model.proto_frequency[top_proto_idx]
            support_scale = (proto_support / model.proto_frequency.max().clamp_min(1e-6)).clamp_min(1e-6).sqrt()
            support_scale = support_scale.repeat_interleave(model.n_micro, dim=1).unsqueeze(-1).unsqueeze(-1)
            local_gate = local_gate * support_scale
        coarse_local = coarse_local + local_gate * (local_path - coarse_local)
    use_refiner = enable_refiner and (not model.disable_refiner)
    refined_local = model.refiner(coarse_local, active_query, difficulty_gate) if use_refiner else coarse_local
    pred_xyz = model.pose_normalizer.inverse(refined_local, origin, rotation)
    outputs = {
        "pred_xyz": pred_xyz,
        "pred_score": pred_score,
        "proto_logits": proto_logits,
        "top_proto_idx": top_proto_idx,
        "aux": {
            "coeff": coeff,
            "coeff_delta": coeff_delta,
            "endpoint_residual": endpoint_residual,
            "proto_summary_5d": model.proto_summary_5d,
        },
    }
    return {
        "outputs": outputs,
        "endpoint_mode_local": endpoint_mode_local,
        "coeff": coeff,
        "coarse_local": coarse_local,
        "refined_local": refined_local,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose ADE bottlenecks with representation and model oracles.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_limit", type=int, default=8192)
    parser.add_argument("--train_bank_limit", type=int, default=50000)
    parser.add_argument("--limit_model_batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_name = args.dataset_name or config["dataset_name"]
    artifact = _load_artifact(checkpoint)
    project_root = os.getcwd()
    train_dir = resolve_split_dir(project_root, config["dataset_variant"], dataset_name, "train")
    test_dir = resolve_split_dir(project_root, config["dataset_variant"], dataset_name, "test")
    train_ds = ProtoBasisSceneDataset(
        train_dir,
        "train",
        obs_len=config["obs"],
        pred_len=config["preds"],
        obs_stride=config.get("obs_stride", 1),
        pred_stride=config.get("pred_stride", 1),
        max_agents=config["max_agents"],
        model_artifact=artifact,
        n_proto=config["n_proto"],
        basis_dim=config["basis_dim"],
        local_basis_dim=config.get("local_basis_dim", 0),
        micro_per_proto=artifact["micro_per_proto"],
        rare_threshold=config["rare_threshold"],
    )
    test_ds = ProtoBasisSceneDataset(
        test_dir,
        "test",
        obs_len=config["obs"],
        pred_len=config["preds"],
        obs_stride=config.get("obs_stride", 1),
        pred_stride=config.get("pred_stride", 1),
        max_agents=config["max_agents"],
        model_artifact=artifact,
        n_proto=config["n_proto"],
        basis_dim=config["basis_dim"],
        local_basis_dim=config.get("local_basis_dim", 0),
        micro_per_proto=artifact["micro_per_proto"],
        rare_threshold=config["rare_threshold"],
    )

    rng = np.random.default_rng(args.seed)
    test_count = min(args.test_limit, len(test_ds))
    train_count = min(args.train_bank_limit, len(train_ds))
    test_indices = rng.choice(len(test_ds), size=test_count, replace=False)
    train_indices = rng.choice(len(train_ds), size=train_count, replace=False)
    test_indices.sort()
    train_indices.sort()

    test_future = test_ds.samples["future_local"][test_indices].astype(np.float32)
    test_summary = test_ds.samples["proto_summary_5d"][test_indices].astype(np.float32)
    test_proto = test_ds.samples["gt_proto_id"][test_indices].astype(np.int64)
    train_future = train_ds.samples["future_local"][train_indices].astype(np.float32)
    train_summary = train_ds.samples["proto_summary_5d"][train_indices].astype(np.float32)
    pred_len = test_future.shape[1]
    alpha = np.linspace(0.0, 1.0, num=pred_len, dtype=np.float32)[None, :, None]

    proto_endpoints = artifact["summary_5d"][:, :3].astype(np.float32)
    basis_matrix = artifact["basis_bank"].reshape(artifact["basis_bank"].shape[0], -1).astype(np.float32)
    proto_anchor = alpha * proto_endpoints[test_proto][:, None, :]
    gt_endpoint_anchor = alpha * test_future[:, -1:, :]
    proto_mean_path = artifact["prototype_mean_path"][test_proto].astype(np.float32)

    global_basis_recon = _solve_linear_reconstruction(proto_anchor, test_future, basis_matrix)
    gt_endpoint_basis_recon = _solve_linear_reconstruction(gt_endpoint_anchor, test_future, basis_matrix)

    local_recon = np.zeros_like(test_future)
    global_local_recon = np.zeros_like(test_future)
    local_bank = artifact["local_basis_bank"].astype(np.float32)
    for row, proto_id in enumerate(test_proto):
        residual_anchor = proto_mean_path[row : row + 1]
        if local_bank.shape[1] > 0:
            local_matrix = local_bank[proto_id].reshape(local_bank.shape[1], -1).astype(np.float32)
        else:
            local_matrix = np.zeros((0, pred_len * 3), dtype=np.float32)
        local_recon[row : row + 1] = _solve_linear_reconstruction(residual_anchor, test_future[row : row + 1], local_matrix)
        combined = np.concatenate([basis_matrix, local_matrix], axis=0)
        global_local_recon[row : row + 1] = _solve_linear_reconstruction(
            residual_anchor,
            test_future[row : row + 1],
            combined,
        )

    model = _build_model(config, checkpoint).to(device)
    subset = Subset(test_ds, test_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(proto_basis_collate, max_agents=config["max_agents"]),
        pin_memory=device.type == "cuda",
    )

    result = {
        "dataset_name": dataset_name,
        "checkpoint": args.checkpoint,
        "samples": {
            "test_count": int(test_count),
            "train_bank_count": int(train_count),
            "test_total": int(len(test_ds)),
            "train_total": int(len(train_ds)),
        },
        "motion_baselines": _collect_motion_baselines(test_ds, test_indices),
        "dataset_oracles": {
            "nearest_train_future_by_5d_summary": _nearest_bank_oracle(
                test_summary,
                test_future,
                train_summary,
                train_future,
            ),
        },
        "representation_oracles": {
            "gt_proto_straight_anchor": _metric_from_paths(proto_anchor, test_future),
            "gt_proto_anchor_plus_global_basis_ls": _metric_from_paths(global_basis_recon, test_future),
            "gt_endpoint_straight_anchor": _metric_from_paths(gt_endpoint_anchor, test_future),
            "gt_endpoint_plus_global_basis_ls": _metric_from_paths(gt_endpoint_basis_recon, test_future),
            "gt_proto_mean_path": _metric_from_paths(proto_mean_path, test_future),
            "gt_proto_mean_path_plus_local_basis_ls": _metric_from_paths(local_recon, test_future),
            "gt_proto_mean_path_plus_global_local_basis_ls": _metric_from_paths(global_local_recon, test_future),
        },
        "model_variants": {
            "actual": _evaluate_model(model, loader, device, config, "actual", args.limit_model_batches),
            "force_gt_proto": _evaluate_model(model, loader, device, config, "force_gt_proto", args.limit_model_batches),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
