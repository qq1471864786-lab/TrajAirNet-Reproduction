import argparse
import json
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import (  # noqa: E402
    ProtoBasisNet,
    ProtoBasisSceneDataset,
    average_metric_sums,
    init_metric_sums,
    metric_names_for_protocol,
    proto_basis_collate,
    summarize_batch_metrics,
    update_metric_sums,
)
from model.data import resolve_split_dir  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate router/prototype oracle variants for a checkpoint.")
    parser.add_argument("checkpoint", help="Path to best_best20.pt or last.pt.")
    parser.add_argument("--dataset_name", default="", help="Override checkpoint dataset.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    return parser


def autocast_context(device, use_amp):
    if device.type == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_model(config, checkpoint):
    return ProtoBasisNet(
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
        use_micro_coeff_anchors=bool(config.get("micro_coeff_anchors", False)),
        endpoint_conditioning=config.get("endpoint_conditioning", "rank"),
        candidate_dense_topk=int(config.get("candidate_dense_topk", 0)),
        coupled_decoder=bool(config.get("coupled_decoder", False)),
        coupled_decoder_iters=int(config.get("coupled_decoder_iters", 0)),
        endpoint_shape_refiner=bool(config.get("endpoint_shape_refiner", False)),
        control_shape_refiner=bool(config.get("control_shape_refiner", False)),
        control_shape_points=int(config.get("control_shape_points", 16)),
        trajectory_control_refiner=bool(config.get("trajectory_control_refiner", False)),
        trajectory_control_points=int(config.get("trajectory_control_points", 32)),
        disable_social=config.get("disable_social", False),
        disable_router=config.get("disable_router", False),
        disable_refiner=config.get("disable_refiner", False),
    )


def load_checkpoint_state(model, checkpoint, dropout):
    state_dict = checkpoint["model"]
    if "prototype_router.endpoint_head.0.weight" in state_dict:
        device = next(model.parameters()).device
        first_weight = state_dict["prototype_router.endpoint_head.0.weight"]
        last_weight = state_dict["prototype_router.endpoint_head.3.weight"]
        model.prototype_router.endpoint_head = nn.Sequential(
            nn.Linear(first_weight.shape[1], first_weight.shape[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_weight.shape[1], last_weight.shape[0]),
        ).to(device)
        load_result = model.load_state_dict(state_dict, strict=False)
        unexpected = [
            key for key in load_result.unexpected_keys if not key.startswith("prototype_router.generic_endpoint_head")
        ]
        missing = [
            key for key in load_result.missing_keys if not key.startswith("prototype_router.generic_endpoint_head")
        ]
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint compatibility load mismatch: missing={missing}, unexpected={unexpected}")
        return
    model.load_state_dict(state_dict)


def build_dataset(config, checkpoint, dataset_name, split):
    project_root = os.getcwd()
    split_dir = resolve_split_dir(project_root, config["dataset_variant"], dataset_name, split)
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


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _candidate_l2(pred_xyz, gt_xyz):
    return torch.linalg.norm(pred_xyz - gt_xyz[:, None], dim=-1)


def _batch_rank_correlation(score, quality):
    score_rank = torch.argsort(torch.argsort(score, dim=1), dim=1).float()
    quality_rank = torch.argsort(torch.argsort(quality, dim=1), dim=1).float()
    score_centered = score_rank - score_rank.mean(dim=1, keepdim=True)
    quality_centered = quality_rank - quality_rank.mean(dim=1, keepdim=True)
    numerator = (score_centered * quality_centered).sum(dim=1)
    denominator = torch.sqrt(score_centered.pow(2).sum(dim=1) * quality_centered.pow(2).sum(dim=1)).clamp_min(1e-6)
    return numerator / denominator


def _new_stat_accumulator():
    return {"sum": 0.0, "count": 0}


def _add_stat(accumulator, values):
    if values.numel() == 0:
        return
    accumulator["sum"] += float(values.detach().sum().item())
    accumulator["count"] += int(values.numel())


def _mean_or_none(accumulator):
    if accumulator["count"] <= 0:
        return None
    return accumulator["sum"] / accumulator["count"]


def _safe_rate(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def _metric_accumulators(config):
    primary_k = config.get("eval_topk_primary", 5)
    secondary_k = config.get("eval_topk_secondary", 20)
    metric_names = metric_names_for_protocol(primary_k, secondary_k)
    return init_metric_sums(metric_names), 0, 0


def _update_metric_accumulators(accumulators, metrics, count, rare_count):
    metric_sums, total_count, total_rare = accumulators
    update_metric_sums(metric_sums, metrics, count)
    return metric_sums, total_count + count, total_rare + rare_count


def _finalize_metric_accumulators(accumulators):
    metric_sums, total_count, total_rare = accumulators
    averaged = average_metric_sums(metric_sums, total_count, total_rare)
    averaged["count"] = total_count
    averaged["rare_count"] = total_rare
    return averaged


def evaluate_variant(model, loader, device, config, variant, use_amp, limit_eval_batches, eval_enable_refiner=True):
    primary_k = config.get("eval_topk_primary", 5)
    secondary_k = config.get("eval_topk_secondary", 20)
    metric_names = metric_names_for_protocol(primary_k, secondary_k)
    metric_sums = init_metric_sums(metric_names)
    score_oracle_accumulators = {
        "oracle_by_ADE": _metric_accumulators(config),
        "oracle_by_FDE": _metric_accumulators(config),
        "oracle_by_ADE_FDE": _metric_accumulators(config),
    }
    total_count = 0
    rare_count = 0
    router_topk_hit_count = 0
    router_top1_count = 0
    rare_topk_hit_count = 0
    rare_top1_count = 0
    nonrare_topk_hit_count = 0
    nonrare_top1_count = 0
    nonrare_count = 0
    residual_stats = {
        "current_all": _new_stat_accumulator(),
        "hit_only": _new_stat_accumulator(),
        "miss_slot0": _new_stat_accumulator(),
        "rare_all": _new_stat_accumulator(),
        "rare_hit_only": _new_stat_accumulator(),
        "rare_miss_slot0": _new_stat_accumulator(),
        "gt_residual_norm_all": _new_stat_accumulator(),
        "gt_residual_norm_hit": _new_stat_accumulator(),
        "gt_residual_norm_miss": _new_stat_accumulator(),
        "rare_gt_residual_norm_miss": _new_stat_accumulator(),
        "slot0_to_gt_proto_endpoint_distance_miss": _new_stat_accumulator(),
        "rare_slot0_to_gt_proto_endpoint_distance_miss": _new_stat_accumulator(),
    }
    score_quality_stats = {
        "rank_corr_ADE_FDE": _new_stat_accumulator(),
        "score_top1_matches_best_ADE_FDE": _new_stat_accumulator(),
    }
    model.eval()
    total_batches = min(len(loader), limit_eval_batches) if limit_eval_batches else len(loader)
    progress = tqdm(loader, desc=variant, total=total_batches, leave=False, dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(progress):
            if limit_eval_batches and batch_index >= limit_eval_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            force_gt = variant in {"force_gt_proto", "force_gt_no_refiner", "with_refiner_force_gt"}
            if variant in {"no_refiner", "force_gt_no_refiner"}:
                enable_refiner = False
            elif variant in {"with_refiner", "with_refiner_force_gt"}:
                enable_refiner = True
            else:
                enable_refiner = bool(eval_enable_refiner)
            with autocast_context(device, use_amp):
                outputs = model(
                    batch["obs_xyz"],
                    batch["obs_mask"],
                    gt_proto_id=batch["gt_proto_id"] if force_gt else None,
                    force_gt_proto=force_gt,
                    enable_refiner=enable_refiner,
                )
            metrics, count, rare = summarize_batch_metrics(
                outputs,
                batch,
                primary_k=primary_k,
                secondary_k=secondary_k,
                glev_topn_primary=config.get("glev_topn_primary", 2),
                glev_topn_secondary=config.get("glev_topn_secondary", 5),
            )
            update_metric_sums(metric_sums, metrics, count)
            top_proto_idx = outputs["top_proto_idx"]
            gt_proto_id = batch["gt_proto_id"]
            rare_mask = batch["is_rare"].bool()
            nonrare_mask = ~rare_mask
            match_mask = top_proto_idx.eq(gt_proto_id[:, None])
            topk_hit = match_mask.any(dim=1)
            top1_hit = outputs["proto_logits"].argmax(dim=-1).eq(gt_proto_id)
            router_topk_hit_count += int(topk_hit.sum().item())
            router_top1_count += int(top1_hit.sum().item())
            rare_topk_hit_count += int((topk_hit & rare_mask).sum().item())
            rare_top1_count += int((top1_hit & rare_mask).sum().item())
            nonrare_topk_hit_count += int((topk_hit & nonrare_mask).sum().item())
            nonrare_top1_count += int((top1_hit & nonrare_mask).sum().item())
            total_count += count
            rare_count += rare
            nonrare_count += int(nonrare_mask.sum().item())

            gt_slot = match_mask.float().argmax(dim=1)
            residual = outputs["aux"]["endpoint_residual"]
            pred_residual = residual[torch.arange(gt_slot.size(0), device=gt_slot.device), gt_slot]
            per_sample_res = F.smooth_l1_loss(pred_residual, batch["gt_proto_residual"], reduction="none").mean(dim=1)
            gt_residual_norm = torch.linalg.norm(batch["gt_proto_residual"], dim=1)
            gt_proto_endpoint = model.proto_summary_5d[gt_proto_id, :3]
            slot0_proto_endpoint = model.proto_summary_5d[top_proto_idx[:, 0], :3]
            slot0_endpoint_distance = torch.linalg.norm(slot0_proto_endpoint - gt_proto_endpoint, dim=1)
            _add_stat(residual_stats["current_all"], per_sample_res)
            _add_stat(residual_stats["hit_only"], per_sample_res[topk_hit])
            _add_stat(residual_stats["miss_slot0"], per_sample_res[~topk_hit])
            _add_stat(residual_stats["rare_all"], per_sample_res[rare_mask])
            _add_stat(residual_stats["rare_hit_only"], per_sample_res[topk_hit & rare_mask])
            _add_stat(residual_stats["rare_miss_slot0"], per_sample_res[(~topk_hit) & rare_mask])
            _add_stat(residual_stats["gt_residual_norm_all"], gt_residual_norm)
            _add_stat(residual_stats["gt_residual_norm_hit"], gt_residual_norm[topk_hit])
            _add_stat(residual_stats["gt_residual_norm_miss"], gt_residual_norm[~topk_hit])
            _add_stat(residual_stats["rare_gt_residual_norm_miss"], gt_residual_norm[(~topk_hit) & rare_mask])
            _add_stat(residual_stats["slot0_to_gt_proto_endpoint_distance_miss"], slot0_endpoint_distance[~topk_hit])
            _add_stat(
                residual_stats["rare_slot0_to_gt_proto_endpoint_distance_miss"],
                slot0_endpoint_distance[(~topk_hit) & rare_mask],
            )

            l2 = _candidate_l2(outputs["pred_xyz"], batch["fut_xyz"])
            ade = l2.mean(dim=-1)
            fde = l2[..., -1]
            ade_scale = ade.mean(dim=1, keepdim=True).clamp_min(1e-6)
            fde_scale = fde.mean(dim=1, keepdim=True).clamp_min(1e-6)
            combined_quality = -(ade / ade_scale + 0.75 * fde / fde_scale)
            _add_stat(score_quality_stats["rank_corr_ADE_FDE"], _batch_rank_correlation(outputs["pred_score"], combined_quality))
            score_best = outputs["pred_score"].argmax(dim=1)
            quality_best = combined_quality.argmax(dim=1)
            _add_stat(score_quality_stats["score_top1_matches_best_ADE_FDE"], score_best.eq(quality_best).float())
            oracle_scores = {
                "oracle_by_ADE": -ade,
                "oracle_by_FDE": -fde,
                "oracle_by_ADE_FDE": combined_quality,
            }
            for oracle_name, oracle_score in oracle_scores.items():
                oracle_outputs = dict(outputs)
                oracle_outputs["pred_score"] = oracle_score
                oracle_metrics, oracle_count, oracle_rare = summarize_batch_metrics(
                    oracle_outputs,
                    batch,
                    primary_k=primary_k,
                    secondary_k=secondary_k,
                    glev_topn_primary=config.get("glev_topn_primary", 2),
                    glev_topn_secondary=config.get("glev_topn_secondary", 5),
                )
                score_oracle_accumulators[oracle_name] = _update_metric_accumulators(
                    score_oracle_accumulators[oracle_name],
                    oracle_metrics,
                    oracle_count,
                    oracle_rare,
                )
    progress.close()
    averaged = average_metric_sums(metric_sums, total_count, rare_count)
    averaged["router_topk_hit"] = _safe_rate(router_topk_hit_count, total_count)
    averaged["router_topk_miss"] = _safe_rate(total_count - router_topk_hit_count, total_count)
    averaged["router_top1_acc"] = _safe_rate(router_top1_count, total_count)
    averaged["rare_router_topk_hit"] = _safe_rate(rare_topk_hit_count, rare_count)
    averaged["rare_router_topk_miss"] = _safe_rate(rare_count - rare_topk_hit_count, rare_count)
    averaged["rare_router_top1_acc"] = _safe_rate(rare_top1_count, rare_count)
    averaged["nonrare_router_topk_hit"] = _safe_rate(nonrare_topk_hit_count, nonrare_count)
    averaged["nonrare_router_topk_miss"] = _safe_rate(nonrare_count - nonrare_topk_hit_count, nonrare_count)
    averaged["nonrare_router_top1_acc"] = _safe_rate(nonrare_top1_count, nonrare_count)
    averaged["count"] = total_count
    averaged["rare_count"] = rare_count
    averaged["nonrare_count"] = nonrare_count
    averaged["residual_loss_probe"] = {name: _mean_or_none(stat) for name, stat in residual_stats.items()}
    hit_only = averaged["residual_loss_probe"]["hit_only"]
    polluted = averaged["residual_loss_probe"]["current_all"]
    if hit_only is None or polluted is None:
        averaged["residual_loss_probe"]["pollution_delta_all_minus_hit"] = None
    else:
        averaged["residual_loss_probe"]["pollution_delta_all_minus_hit"] = polluted - hit_only
    averaged["score_oracles"] = {
        name: _finalize_metric_accumulators(accumulators)
        for name, accumulators in score_oracle_accumulators.items()
    }
    averaged["score_quality_probe"] = {name: _mean_or_none(stat) for name, stat in score_quality_stats.items()}
    return averaged


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    eval_enable_refiner = bool(checkpoint.get("meta", {}).get("eval_enable_refiner", True))
    dataset_name = args.dataset_name or config["dataset_name"]
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

    variants = ["actual", "force_gt_proto", "no_refiner", "force_gt_no_refiner"]
    if not eval_enable_refiner:
        variants.extend(["with_refiner", "with_refiner_force_gt"])
    result = {
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "split": args.split,
        "eval_enable_refiner": eval_enable_refiner,
        "diagnostic_notes": {
            "actual": "Uses checkpoint meta.eval_enable_refiner, matching test.py baseline evaluation.",
            "residual_loss_probe.current_all": (
                "Current training behavior: if GT prototype is absent from top-k, slot 0 is used by argmax."
            ),
            "residual_loss_probe.hit_only": "Endpoint residual loss restricted to samples whose GT prototype is in top-k.",
            "score_oracles": "Same candidate trajectories, but scores are replaced by true ADE/FDE-derived ordering.",
            "score_quality_probe.rank_corr_ADE_FDE": "Spearman-style rank correlation; higher means model scores agree with true candidate quality.",
            "GLeV_direction": "Higher is better; model/metrics.py returns local_var/global_var following GooDFlight.",
        },
        "variants": {},
    }
    for variant in variants:
        result["variants"][variant] = evaluate_variant(
            model,
            loader,
            device,
            config,
            variant,
            use_amp,
            args.limit_eval_batches,
            eval_enable_refiner=eval_enable_refiner,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
