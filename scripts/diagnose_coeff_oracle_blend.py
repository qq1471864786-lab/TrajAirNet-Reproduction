import argparse
import json
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ProtoBasisSceneDataset, proto_basis_collate  # noqa: E402
from model.data import resolve_split_dir  # noqa: E402
from model.proto_basis_flight_model import build_anchor, build_global_features, build_local_features  # noqa: E402
from test import build_model, load_checkpoint_state, resolve_eval_enable_refiner  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Fast oracle blend probe for coefficient-correction headroom.")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=20)
    parser.add_argument("--alphas", default="0,0.02,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output", default="")
    return parser


def autocast_context(device, use_amp):
    if device.type == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_dataset(config, checkpoint, dataset_name, split):
    split_dir = resolve_split_dir(os.getcwd(), config["dataset_variant"], dataset_name, split)
    model_artifact = {
        "summary_5d": checkpoint["proto_summary_5d"],
        "frequency": checkpoint["proto_freq"],
        "rare_ids": torch.nonzero(torch.tensor(checkpoint["proto_freq"]) < config["rare_threshold"]).view(-1).tolist(),
        "basis_bank": checkpoint["basis_bank"],
        "prototype_mean_path": checkpoint.get("prototype_mean_path"),
        "local_basis_bank": checkpoint.get("local_basis_bank"),
        "micro_coeff_anchors": checkpoint.get("micro_coeff_anchors"),
        "n_proto": config["n_proto"],
        "basis_dim": config["basis_dim"],
        "local_basis_dim": config.get("local_basis_dim", 0),
        "micro_per_proto": config.get("micro_per_proto", 0) if config.get("micro_coeff_anchors", False) else 0,
        "rare_threshold": config["rare_threshold"],
        "obs_len": config["obs"],
        "pred_len": config["preds"],
        "obs_stride": config.get("obs_stride", 1),
        "pred_stride": config.get("pred_stride", 1),
    }
    return ProtoBasisSceneDataset(
        data_dir=split_dir,
        split_name=split,
        obs_len=config["obs"],
        pred_len=config["preds"],
        obs_stride=config.get("obs_stride", 1),
        pred_stride=config.get("pred_stride", 1),
        max_agents=config["max_agents"],
        model_artifact=model_artifact,
        n_proto=config["n_proto"],
        basis_dim=config["basis_dim"],
        local_basis_dim=config.get("local_basis_dim", 0),
        micro_per_proto=config.get("micro_per_proto", 0) if config.get("micro_coeff_anchors", False) else 0,
        rare_threshold=config["rare_threshold"],
    )


def forward_pre_refiner_state(model, obs_xyz, obs_mask, enable_refiner=True):
    local_xyz, _, _, origin, rotation = model.pose_normalizer(obs_xyz)
    feats_local = build_local_features(local_xyz)
    feats_global = build_global_features(obs_xyz)

    agent_feat = model.temporal_encoder(feats_local, feats_global)
    if model.disable_social:
        target_ctx = agent_feat[:, 0]
        scene_ctx = (agent_feat * obs_mask.unsqueeze(-1)).sum(dim=1) / obs_mask.sum(dim=1, keepdim=True).clamp_min(1)
    else:
        target_ctx, scene_ctx = model.social_aggregator(agent_feat, obs_mask)

    proto_logits, top_proto_idx, proto_token, _, endpoint_local = model.prototype_router(
        target_ctx,
        scene_ctx,
        model.proto_summary_5d,
        disable_router=model.disable_router,
    )
    micro_coeff_anchor = model.micro_coeff_anchors[top_proto_idx] if model.has_micro_coeff_anchors else None
    query_feat, coeff, pred_score, difficulty_gate, coeff_delta = model.query_decoder(
        proto_token,
        endpoint_local,
        target_ctx,
        agent_feat,
        obs_mask,
        micro_coeff_anchor=micro_coeff_anchor,
    )
    endpoint_mode_local = endpoint_local.repeat_interleave(model.n_micro, dim=1)
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
        anchor_local = build_anchor(endpoint_mode_local, model.anchor_alpha.to(endpoint_mode_local))
        coarse_local = model.basis_bank(anchor_local, coeff)
        active_query = stage2_query

    if getattr(model, "coupled_decoder", None) is not None:
        coeff_before_coupled = coeff
        active_query, endpoint_mode_local, coeff, coarse_local = model.coupled_decoder(
            active_query,
            endpoint_mode_local,
            coeff,
            coarse_local,
            model.basis_bank,
            model.anchor_alpha,
        )
        coeff_delta = coeff_delta + (coeff - coeff_before_coupled)
    candidate_proto_idx = top_proto_idx.repeat_interleave(model.n_micro, dim=1)
    if getattr(model, "candidate_keep_indices", None) is not None and model.candidate_keep_indices.numel() > 0:
        keep = model.candidate_keep_indices.to(device=coeff.device)
        coeff = coeff.index_select(1, keep)
        pred_score = pred_score.index_select(1, keep)
        difficulty_gate = difficulty_gate.index_select(1, keep)
        coeff_delta = coeff_delta.index_select(1, keep)
        endpoint_mode_local = endpoint_mode_local.index_select(1, keep)
        active_query = active_query.index_select(1, keep)
        candidate_proto_idx = candidate_proto_idx.index_select(1, keep)

    return {
        "active_query": active_query,
        "candidate_proto_idx": candidate_proto_idx,
        "coeff": coeff,
        "coeff_delta": coeff_delta,
        "difficulty_gate": difficulty_gate,
        "enable_refiner": enable_refiner and (not model.disable_refiner),
        "endpoint_mode_local": endpoint_mode_local,
        "origin": origin,
        "pred_score": pred_score,
        "proto_logits": proto_logits,
        "rotation": rotation,
        "top_proto_idx": top_proto_idx,
    }


def apply_tail_modules(model, state, coeff):
    endpoint_mode_local = state["endpoint_mode_local"].to(dtype=coeff.dtype)
    anchor_local = build_anchor(endpoint_mode_local, model.anchor_alpha.to(endpoint_mode_local))
    coarse_local = model.basis_bank(anchor_local, coeff)
    active_query = state["active_query"].to(dtype=coeff.dtype)

    if model.has_local_basis:
        candidate_proto_idx = state["candidate_proto_idx"]
        proto_mean_path = model.prototype_mean_path[candidate_proto_idx]
        proto_mean_endpoint = proto_mean_path[:, :, -1, :]
        aligned_proto_mean = proto_mean_path + (endpoint_mode_local - proto_mean_endpoint)[:, :, None, :]
        local_basis = model.local_basis_bank[candidate_proto_idx]
        local_coeff = model.local_coeff_head(active_query)
        local_residual = torch.einsum("bkm,bkmtd->bktd", local_coeff, local_basis)
        local_path = aligned_proto_mean + local_residual
        local_gate = torch.sigmoid(model.local_mix_gate(active_query)).unsqueeze(-1)
        if model.support_aware_local_basis:
            proto_support = model.proto_frequency[candidate_proto_idx]
            support_scale = (proto_support / model.proto_frequency.max().clamp_min(1e-6)).clamp_min(1e-6).sqrt()
            local_gate = local_gate * support_scale.unsqueeze(-1).unsqueeze(-1)
        coarse_local = coarse_local + local_gate * (local_path - coarse_local)

    if state["enable_refiner"]:
        local_pred = model.refiner(coarse_local, active_query, state["difficulty_gate"].to(dtype=coeff.dtype))
    else:
        local_pred = coarse_local
    if getattr(model, "endpoint_shape_refiner", None) is not None:
        local_pred = model.endpoint_shape_refiner(local_pred, active_query)
    if getattr(model, "control_shape_refiner", None) is not None:
        local_pred = model.control_shape_refiner(local_pred, active_query)
    return model.pose_normalizer.inverse(local_pred, state["origin"], state["rotation"])


def minade_minfde_topk(pred_xyz, gt_xyz, pred_score, k):
    safe_k = min(int(k), pred_score.size(1))
    top_idx = pred_score.argsort(dim=-1, descending=True)[:, :safe_k]
    gather_idx = top_idx[:, :, None, None].expand(-1, -1, pred_xyz.size(2), pred_xyz.size(3))
    top_pred = torch.gather(pred_xyz, 1, gather_idx)
    l2 = torch.linalg.norm(top_pred - gt_xyz[:, None], dim=-1)
    ade = l2.mean(dim=-1)
    fde = l2[..., -1]
    return ade.min(dim=1).values, fde.min(dim=1).values


def metric_bucket(ade, fde, mask):
    if not mask.any():
        return {"count": 0, "ADE@20": None, "FDE@20": None}
    return {
        "count": int(mask.sum().item()),
        "ADE@20": float(ade[mask].mean().item()),
        "FDE@20": float(fde[mask].mean().item()),
    }


def main():
    args = build_parser().parse_args()
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_name = args.dataset_name or config["dataset_name"]
    secondary_k = config.get("eval_topk_secondary", 20)
    eval_enable_refiner = resolve_eval_enable_refiner(checkpoint)

    dataset = build_dataset(config, checkpoint, dataset_name, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(proto_basis_collate, max_agents=config["max_agents"]),
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, checkpoint).to(device)
    load_checkpoint_state(model, checkpoint, config["dropout"])
    model.eval()

    totals = {
        alpha: {
            "ade_sum": 0.0,
            "fde_sum": 0.0,
            "count": 0,
            "router_hit_ade_sum": 0.0,
            "router_hit_fde_sum": 0.0,
            "router_hit_count": 0,
            "router_miss_ade_sum": 0.0,
            "router_miss_fde_sum": 0.0,
            "router_miss_count": 0,
            "rare_ade_sum": 0.0,
            "rare_fde_sum": 0.0,
            "rare_count": 0,
        }
        for alpha in alphas
    }
    coeff_l1_sum = 0.0
    coeff_l2_sum = 0.0
    coeff_count = 0
    processed = 0
    total_batches = min(len(loader), args.limit_eval_batches) if args.limit_eval_batches else len(loader)

    basis_matrix = model.basis_bank.basis_bank.reshape(model.basis_bank.basis_bank.size(0), -1)
    basis_pinv = torch.linalg.pinv(basis_matrix).to(device)

    with torch.no_grad():
        progress = tqdm(loader, total=total_batches, desc="coeff oracle blend", leave=False, dynamic_ncols=True)
        for batch_idx, raw_batch in enumerate(progress):
            if args.limit_eval_batches and batch_idx >= args.limit_eval_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            with autocast_context(device, use_amp):
                state = forward_pre_refiner_state(
                    model,
                    batch["obs_xyz"],
                    batch["obs_mask"],
                    enable_refiner=eval_enable_refiner,
                )

            endpoint_mode_local = state["endpoint_mode_local"].float()
            coeff = state["coeff"].float()
            gt_local = batch["fut_local"].float()
            alpha_vec = model.anchor_alpha.to(device=gt_local.device, dtype=gt_local.dtype)
            anchor = build_anchor(endpoint_mode_local, alpha_vec)
            residual = (gt_local[:, None, :, :] - anchor).reshape(-1, gt_local.size(1) * 3)
            ls_coeff = residual @ basis_pinv.to(dtype=gt_local.dtype)
            ls_coeff = ls_coeff.reshape_as(coeff)
            coeff_gap = ls_coeff - coeff
            coeff_l1_sum += float(coeff_gap.abs().mean(dim=-1).sum().item())
            coeff_l2_sum += float(torch.linalg.norm(coeff_gap, dim=-1).sum().item())
            coeff_count += coeff_gap.numel() // coeff_gap.size(-1)

            router_hit = state["top_proto_idx"].eq(batch["gt_proto_id"][:, None]).any(dim=1)
            router_miss = ~router_hit
            rare = batch["is_rare"].bool()
            for alpha in alphas:
                blended_coeff = coeff + alpha * coeff_gap
                pred_xyz = apply_tail_modules(model, state, blended_coeff)
                ade, fde = minade_minfde_topk(pred_xyz, batch["fut_xyz"], state["pred_score"], secondary_k)
                entry = totals[alpha]
                n = ade.numel()
                entry["ade_sum"] += float(ade.sum().item())
                entry["fde_sum"] += float(fde.sum().item())
                entry["count"] += n
                for prefix, mask in (("router_hit", router_hit), ("router_miss", router_miss), ("rare", rare)):
                    if mask.any():
                        entry[f"{prefix}_ade_sum"] += float(ade[mask].sum().item())
                        entry[f"{prefix}_fde_sum"] += float(fde[mask].sum().item())
                        entry[f"{prefix}_count"] += int(mask.sum().item())
            processed += batch["obs_xyz"].size(0)
        progress.close()

    results = {}
    for alpha in alphas:
        entry = totals[alpha]
        count = max(entry["count"], 1)
        result = {
            "ADE@20": entry["ade_sum"] / count,
            "FDE@20": entry["fde_sum"] / count,
            "count": entry["count"],
        }
        for prefix in ("router_hit", "router_miss", "rare"):
            prefix_count = max(entry[f"{prefix}_count"], 1)
            result[prefix] = {
                "ADE@20": entry[f"{prefix}_ade_sum"] / prefix_count,
                "FDE@20": entry[f"{prefix}_fde_sum"] / prefix_count,
                "count": entry[f"{prefix}_count"],
            }
        results[str(alpha)] = result

    output = {
        "alphas": alphas,
        "checkpoint": args.checkpoint,
        "coeff_gap": {
            "mean_l1": coeff_l1_sum / max(coeff_count, 1),
            "mean_l2": coeff_l2_sum / max(coeff_count, 1),
        },
        "dataset_name": dataset_name,
        "eval_enable_refiner": eval_enable_refiner,
        "limit_eval_batches": args.limit_eval_batches,
        "processed": processed,
        "results": results,
        "secondary_k": secondary_k,
        "split": args.split,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
