import argparse
import json
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ProtoBasisSceneDataset, proto_basis_collate  # noqa: E402
from model.data import resolve_split_dir  # noqa: E402
from test import build_model, load_checkpoint_state, resolve_eval_enable_refiner  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Break down ProtoBasis errors by failure buckets.")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output", default="")
    return parser


def autocast_context(device, use_amp):
    if device.type == "cuda" and use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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


def topk_best_errors(pred_xyz, gt_xyz, pred_score, k):
    safe_k = min(int(k), pred_score.size(1))
    top_idx = pred_score.argsort(dim=-1, descending=True)[:, :safe_k]
    gather_idx = top_idx[:, :, None, None].expand(-1, -1, pred_xyz.size(2), pred_xyz.size(3))
    top_pred = torch.gather(pred_xyz, 1, gather_idx)
    error = torch.linalg.norm(top_pred - gt_xyz[:, None], dim=-1)
    ade = error.mean(dim=-1)
    best_in_top = ade.argmin(dim=1)
    best_error = error[torch.arange(error.size(0), device=error.device), best_in_top]
    best_pred = top_pred[torch.arange(top_pred.size(0), device=top_pred.device), best_in_top]
    best_global_idx = top_idx[torch.arange(top_idx.size(0), device=top_idx.device), best_in_top]
    xy_error = torch.linalg.norm(best_pred[..., :2] - gt_xyz[..., :2], dim=-1)
    z_error = torch.abs(best_pred[..., 2] - gt_xyz[..., 2])
    return {
        "ade": best_error.mean(dim=1),
        "fde": best_error[:, -1],
        "first40": best_error[:, :40].mean(dim=1) if best_error.size(1) >= 40 else best_error.mean(dim=1),
        "mid40": best_error[:, 40:80].mean(dim=1) if best_error.size(1) >= 80 else best_error.mean(dim=1),
        "last40": best_error[:, -40:].mean(dim=1) if best_error.size(1) >= 40 else best_error.mean(dim=1),
        "xy": xy_error.mean(dim=1),
        "z": z_error.mean(dim=1),
        "best_global_idx": best_global_idx,
    }


def summarize_bucket(values, mask):
    count = int(mask.sum())
    if count <= 0:
        return {"count": 0}
    out = {"count": count}
    for name in ("ade", "fde", "first40", "mid40", "last40", "xy", "z"):
        selected = values[name][mask]
        out[name] = float(selected.mean())
        out[f"{name}_p90"] = float(np.percentile(selected, 90))
    return out


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_name = args.dataset_name or config["dataset_name"]
    eval_enable_refiner = resolve_eval_enable_refiner(checkpoint)
    secondary_k = config.get("eval_topk_secondary", 20)

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

    collected = {name: [] for name in ("ade", "fde", "first40", "mid40", "last40", "xy", "z")}
    hit_values = []
    rare_values = []
    top1_hit_values = []
    best_proto_hit_values = []
    total_batches = min(len(loader), args.limit_eval_batches) if args.limit_eval_batches else len(loader)

    with torch.no_grad():
        progress = tqdm(loader, desc="failure buckets", total=total_batches, leave=False, dynamic_ncols=True)
        for batch_index, raw_batch in enumerate(progress):
            if args.limit_eval_batches and batch_index >= args.limit_eval_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            with autocast_context(device, use_amp):
                outputs = model(batch["obs_xyz"], batch["obs_mask"], enable_refiner=eval_enable_refiner)
            errors = topk_best_errors(outputs["pred_xyz"], batch["fut_xyz"], outputs["pred_score"], secondary_k)
            top_proto_idx = outputs["top_proto_idx"]
            candidate_proto_idx = outputs.get("aux", {}).get("candidate_proto_idx")
            gt_proto_id = batch["gt_proto_id"]
            hit = top_proto_idx.eq(gt_proto_id[:, None]).any(dim=1)
            top1_hit = outputs["proto_logits"].argmax(dim=1).eq(gt_proto_id)
            if candidate_proto_idx is not None and candidate_proto_idx.shape[:2] == outputs["pred_score"].shape[:2]:
                best_proto = candidate_proto_idx.gather(1, errors["best_global_idx"][:, None]).squeeze(1)
            else:
                n_micro = max(outputs["pred_score"].size(1) // max(top_proto_idx.size(1), 1), 1)
                best_proto_slot = (errors["best_global_idx"] // n_micro).clamp(max=top_proto_idx.size(1) - 1)
                best_proto = top_proto_idx.gather(1, best_proto_slot[:, None]).squeeze(1)
            best_proto_hit = best_proto.eq(gt_proto_id)

            for name in collected:
                collected[name].append(errors[name].detach().cpu())
            hit_values.append(hit.detach().cpu())
            rare_values.append(batch["is_rare"].bool().detach().cpu())
            top1_hit_values.append(top1_hit.detach().cpu())
            best_proto_hit_values.append(best_proto_hit.detach().cpu())
        progress.close()

    values = {name: torch.cat(chunks).numpy() for name, chunks in collected.items()}
    hit = torch.cat(hit_values).numpy().astype(bool)
    rare = torch.cat(rare_values).numpy().astype(bool)
    top1_hit = torch.cat(top1_hit_values).numpy().astype(bool)
    best_proto_hit = torch.cat(best_proto_hit_values).numpy().astype(bool)
    endpoint_p90_threshold = float(np.percentile(values["fde"], 90))
    last40_p90_threshold = float(np.percentile(values["last40"], 90))
    high_endpoint = values["fde"] >= endpoint_p90_threshold
    high_last40 = values["last40"] >= last40_p90_threshold

    bucket_masks = {
        "all": np.ones_like(hit, dtype=bool),
        "router_hit": hit,
        "router_miss": ~hit,
        "rare": rare,
        "nonrare": ~rare,
        "rare_hit": rare & hit,
        "rare_miss": rare & ~hit,
        "top1_hit": top1_hit,
        "top1_miss": ~top1_hit,
        "best_candidate_proto_hit": best_proto_hit,
        "best_candidate_proto_miss": ~best_proto_hit,
        "endpoint_p90": high_endpoint,
        "last40_p90": high_last40,
        "router_miss_endpoint_p90": (~hit) & high_endpoint,
        "router_miss_last40_p90": (~hit) & high_last40,
        "rare_miss_endpoint_p90": rare & (~hit) & high_endpoint,
        "rare_miss_last40_p90": rare & (~hit) & high_last40,
    }
    result = {
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "split": args.split,
        "eval_enable_refiner": eval_enable_refiner,
        "secondary_k": secondary_k,
        "count": int(hit.shape[0]),
        "thresholds": {
            "endpoint_fde_p90": endpoint_p90_threshold,
            "last40_ade_p90": last40_p90_threshold,
        },
        "rates": {
            "router_topk_hit": float(hit.mean()),
            "router_topk_miss": float((~hit).mean()),
            "rare": float(rare.mean()),
            "top1_hit": float(top1_hit.mean()),
            "best_candidate_proto_hit": float(best_proto_hit.mean()),
        },
        "buckets": {name: summarize_bucket(values, mask) for name, mask in bucket_masks.items()},
    }

    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
