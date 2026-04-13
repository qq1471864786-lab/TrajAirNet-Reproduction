import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import (
    SceneTrajectoryDataset,
    VerticalRelationTrajectoryModel,
    average_metric_sums,
    init_metric_sums,
    metric_totals,
    resolve_split_dir,
    scene_batch_collate,
    select_best_of_n_prediction,
    update_metric_sums,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate TrajAir rebuild checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_variant", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--best_of_n", type=int, default=0)
    parser.add_argument("--protocol", type=str, default="both", choices=["bestof5", "single", "both"])
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    return parser


def build_model(checkpoint_args):
    return VerticalRelationTrajectoryModel(
        variant=checkpoint_args["variant"],
        obs_len=checkpoint_args["obs"],
        pred_len=checkpoint_args["preds"],
        pred_step=checkpoint_args["preds_step"],
        traj_hidden=checkpoint_args["tcn_channels"],
        context_hidden=checkpoint_args["context_hidden"],
        state_hidden=checkpoint_args["state_hidden"],
        interaction_hidden=checkpoint_args["interaction_hidden"],
        interaction_heads=checkpoint_args["interaction_heads"],
        interaction_topk=checkpoint_args.get("interaction_topk", 3),
        tcn_kernel=checkpoint_args["tcn_kernel"],
        dropout=checkpoint_args["dropout"],
        cvae_latent=checkpoint_args["cvae_latent"],
        cvae_layers=checkpoint_args["cvae_layers"],
        cvae_channel_size=checkpoint_args["cvae_channel_size"],
        condition_dropout=checkpoint_args["condition_dropout"],
        state_scale_init=checkpoint_args.get("state_scale_init"),
        social_scale_init=checkpoint_args.get("social_scale_init"),
        interaction_integration=checkpoint_args.get("interaction_integration", "concat"),
    )


def move_batch_to_device(batch, device):
    return {
        "obs": batch["obs"].to(device, non_blocking=True),
        "target": batch["target"].to(device, non_blocking=True),
        "context": batch["context"].to(device, non_blocking=True),
        "scene_ids": batch["scene_ids"].to(device, non_blocking=True),
        "scene_slices": batch["scene_slices"].to(device, non_blocking=True),
        "scene_count": batch["scene_count"],
        "agent_counts": batch["agent_counts"].to(device, non_blocking=True),
        "future_vertical_ranges": batch["future_vertical_ranges"].to(device, non_blocking=True),
    }


def empty_metric_groups(protocols):
    return {
        subset_name: {
            protocol_name: init_metric_sums()
            for protocol_name in protocols
        }
        for subset_name in ("all", "multi_agent", "strong_vertical")
    }


def empty_count_groups(protocols):
    return {
        subset_name: {
            protocol_name: 0
            for protocol_name in protocols
        }
        for subset_name in ("all", "multi_agent", "strong_vertical")
    }


def subset_masks(batch, vertical_threshold):
    return {
        "all": torch.ones(batch["scene_count"], dtype=torch.bool, device=batch["obs"].device),
        "multi_agent": batch["agent_counts"] >= 2,
        "strong_vertical": batch["future_vertical_ranges"] >= vertical_threshold,
    }


def flatten_metrics(metric_groups):
    flattened = {}
    for subset_name, protocol_metrics in metric_groups.items():
        subset_prefix = "" if subset_name == "all" else f"{subset_name}_"
        for protocol_name, metrics in protocol_metrics.items():
            protocol_suffix = "best5" if protocol_name == "best5" else "1"
            for metric_name, value in metrics.items():
                flattened[f"{subset_prefix}{metric_name}_{protocol_suffix}"] = value
    return flattened


def evaluate_model(model, loader, device, protocols, best_of_n, vertical_threshold, limit_eval_batches=0):
    model.eval()
    metric_groups = empty_metric_groups(protocols)
    count_groups = empty_count_groups(protocols)
    processed_batches = 0
    with torch.no_grad():
        for raw_batch in tqdm(loader, ncols=100):
            if limit_eval_batches and processed_batches >= limit_eval_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            target = batch["target"]
            predictions = {}

            if "single" in protocols:
                predictions["single"] = model(
                    batch["obs"],
                    batch["context"],
                    target=None,
                    scene_ids=batch["scene_ids"],
                    scene_slices=batch["scene_slices"],
                    latent_mode="zero",
                )

            if "best5" in protocols:
                if best_of_n > 1:
                    predictions["best5"], _ = select_best_of_n_prediction(
                        model,
                        batch["obs"],
                        target,
                        batch["context"],
                        batch["scene_ids"],
                        batch["scene_slices"],
                        best_of_n,
                    )
                else:
                    predictions["best5"] = predictions.get("single")
                    if predictions["best5"] is None:
                        predictions["best5"] = model(
                            batch["obs"],
                            batch["context"],
                            target=None,
                            scene_ids=batch["scene_ids"],
                            scene_slices=batch["scene_slices"],
                            latent_mode="zero",
                        )

            masks = subset_masks(batch, vertical_threshold)
            for subset_name, scene_mask in masks.items():
                for protocol_name, prediction in predictions.items():
                    totals, scene_count = metric_totals(
                        prediction,
                        target,
                        scene_slices=batch["scene_slices"],
                        scene_mask=scene_mask,
                    )
                    update_metric_sums(metric_groups[subset_name][protocol_name], totals)
                    count_groups[subset_name][protocol_name] += scene_count
            processed_batches += 1

    averaged = {
        subset_name: {
            protocol_name: average_metric_sums(metrics, count_groups[subset_name][protocol_name])
            for protocol_name, metrics in protocol_metrics.items()
        }
        for subset_name, protocol_metrics in metric_groups.items()
    }
    return flatten_metrics(averaged)


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint["args"]
    vertical_threshold = checkpoint.get("vertical_threshold", 0.0)

    dataset_variant = args.dataset_variant or checkpoint_args["dataset_variant"]
    dataset_name = args.dataset_name or checkpoint_args["dataset_name"]
    best_of_n = args.best_of_n if args.best_of_n > 0 else checkpoint_args.get("best_of_n", 5)

    project_root = os.getcwd()
    data_dir = resolve_split_dir(project_root, dataset_variant, dataset_name, "test")
    dataset = SceneTrajectoryDataset(
        data_dir=data_dir,
        obs_len=checkpoint_args["obs"],
        pred_len=checkpoint_args["preds"],
        pred_step=checkpoint_args["preds_step"],
        skip=checkpoint_args["skip"],
        min_agents=checkpoint_args.get("min_eval_agents", 1),
        delim=" ",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=scene_batch_collate,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    model = build_model(checkpoint_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.protocol == "both":
        protocols = ("best5", "single")
    elif args.protocol == "bestof5":
        protocols = ("best5",)
    else:
        protocols = ("single",)

    metrics = evaluate_model(
        model,
        loader,
        device,
        protocols=protocols,
        best_of_n=best_of_n,
        vertical_threshold=vertical_threshold,
        limit_eval_batches=args.limit_eval_batches,
    )
    print(" ".join(f"{name}={value:.4f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
