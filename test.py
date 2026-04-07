import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import (
    HAINet,
    average_metric_sums,
    init_metric_sums,
    metric_totals,
    resolve_split_dir,
    scene_batch_collate,
    SceneTrajectoryDataset,
    select_best_of_n_prediction,
    update_metric_sums,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate HAINet checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_variant", type=str, default="social", choices=["social", "no_social"])
    parser.add_argument("--dataset_name", type=str, default="7days1")
    parser.add_argument("--obs", type=int, default=11)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--preds_step", type=int, default=10)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--min_eval_agents", type=int, default=1)
    parser.add_argument("--best_of_n", type=int, default=5)
    return parser


def build_model(checkpoint_args):
    return HAINet(
        obs_len=checkpoint_args["obs"],
        pred_len=checkpoint_args["preds"],
        pred_step=checkpoint_args["preds_step"],
        tcn_channel_size=checkpoint_args["tcn_channels"],
        tcn_layers=checkpoint_args["tcn_layers"],
        tcn_kernel=checkpoint_args["tcn_kernel"],
        dropout=checkpoint_args["dropout"],
        cvae_hidden=checkpoint_args["cvae_latent"],
        cvae_layers=checkpoint_args["cvae_layers"],
        cvae_channel_size=checkpoint_args["cvae_channel_size"],
        mlp_layer=checkpoint_args["mlp_layer"],
        gat_hidden=checkpoint_args["gat_hidden"],
        gat_heads=checkpoint_args["gat_heads"],
        gat_dropout=checkpoint_args["gat_dropout"],
        use_interaction=not checkpoint_args.get("disable_interaction", False),
        use_height_conditioning=not checkpoint_args.get("disable_height_conditioning", False),
        use_height_feedback=not checkpoint_args.get("disable_height_feedback", False),
    )


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint["args"]

    project_root = os.getcwd()
    data_dir = resolve_split_dir(project_root, args.dataset_variant, args.dataset_name, "test")
    dataset = SceneTrajectoryDataset(
        data_dir=data_dir,
        obs_len=args.obs,
        pred_len=args.preds,
        pred_step=args.preds_step,
        skip=args.skip,
        min_agents=args.min_eval_agents,
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

    metric_sums = init_metric_sums()
    total_agents = 0

    with torch.no_grad():
        for batch in tqdm(loader, ncols=80):
            obs = batch["obs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            scene_ids = batch["scene_ids"].to(device, non_blocking=True)
            scene_slices = batch["scene_slices"].to(device, non_blocking=True)
            prediction, _ = select_best_of_n_prediction(
                model, obs, target, context, scene_ids, scene_slices, args.best_of_n
            )
            batch_metric_totals, agent_count = metric_totals(prediction, target, scene_slices)
            update_metric_sums(metric_sums, batch_metric_totals)
            total_agents += agent_count

    metrics = average_metric_sums(metric_sums, total_agents)
    print(" ".join(f"{name}={value:.4f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
