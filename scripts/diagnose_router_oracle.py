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
        use_micro_coeff_anchors=bool(config.get("micro_coeff_anchors", False)),
        disable_social=config.get("disable_social", False),
        disable_router=config.get("disable_router", False),
        disable_refiner=config.get("disable_refiner", False),
    )


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


def evaluate_variant(model, loader, device, config, variant, use_amp, limit_eval_batches):
    primary_k = config.get("eval_topk_primary", 5)
    secondary_k = config.get("eval_topk_secondary", 20)
    metric_names = metric_names_for_protocol(primary_k, secondary_k)
    metric_sums = init_metric_sums(metric_names)
    total_count = 0
    rare_count = 0
    route_hit_count = 0
    route_top1_count = 0
    model.eval()
    total_batches = min(len(loader), limit_eval_batches) if limit_eval_batches else len(loader)
    progress = tqdm(loader, desc=variant, total=total_batches, leave=False, dynamic_ncols=True, file=sys.stdout)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(progress):
            if limit_eval_batches and batch_index >= limit_eval_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            force_gt = variant in {"force_gt_proto", "force_gt_no_refiner"}
            enable_refiner = variant not in {"no_refiner", "force_gt_no_refiner"}
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
            route_hit_count += int(top_proto_idx.eq(gt_proto_id[:, None]).any(dim=1).sum().item())
            route_top1_count += int(outputs["proto_logits"].argmax(dim=-1).eq(gt_proto_id).sum().item())
            total_count += count
            rare_count += rare
    progress.close()
    averaged = average_metric_sums(metric_sums, total_count, rare_count)
    averaged["router_topk_hit"] = route_hit_count / max(total_count, 1)
    averaged["router_top1_acc"] = route_top1_count / max(total_count, 1)
    averaged["count"] = total_count
    averaged["rare_count"] = rare_count
    return averaged


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
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
    model.load_state_dict(checkpoint["model"])

    variants = ["actual", "force_gt_proto", "no_refiner", "force_gt_no_refiner"]
    result = {
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "split": args.split,
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
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
