import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import (
    SceneTrajectoryDataset,
    TrajectoryForecastLoss,
    VerticalRelationTrajectoryModel,
    average_metric_sums,
    init_metric_sums,
    metric_totals,
    resolve_split_dir,
    scene_batch_collate,
    select_best_of_n_prediction,
    update_metric_sums,
)
from model.run_logging import RunRecorder


def progress_enabled():
    return sys.stderr.isatty()


def supports_color():
    return sys.stdout.isatty()


def green_text(text):
    if not supports_color():
        return text
    return f"\033[92m{text}\033[0m"


def format_epoch_header(epoch, total_epochs, train_loss, train_batches, eval_scenes):
    return (
        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
        f"loss={train_loss:.4f} | train_batches={train_batches} | eval_scenes={eval_scenes}"
    )


def format_metrics(metrics):
    keys = (
        "ADE_best5",
        "FDE_best5",
        "MDE_best5",
        "ADE_1",
        "FDE_1",
        "multi_agent_ADE_best5",
        "strong_vertical_ADE_best5",
        "z-ADE_best5",
    )
    return " | ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


def build_parser():
    parser = argparse.ArgumentParser(description="Train TrajAir rebuild model")

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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_train_agents", type=int, default=1)
    parser.add_argument("--min_eval_agents", type=int, default=1)

    parser.add_argument("--variant", type=str, default="base", choices=["base", "interaction", "state", "full", "naive"])
    parser.add_argument("--protocol", type=str, default="bestof5", choices=["bestof5", "single"])
    parser.add_argument("--tcn_channels", type=int, default=256)
    parser.add_argument("--context_hidden", type=int, default=32)
    parser.add_argument("--state_hidden", type=int, default=128)
    parser.add_argument("--interaction_hidden", type=int, default=256)
    parser.add_argument("--interaction_heads", type=int, default=8)
    parser.add_argument("--interaction_topk", type=int, default=3)
    parser.add_argument("--interaction_integration", type=str, default="concat", choices=["concat", "residualgate"])
    parser.add_argument("--social_scale_init", type=float, default=None)
    parser.add_argument("--state_scale_init", type=float, default=None)
    parser.add_argument("--tcn_kernel", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cvae_latent", type=int, default=128)
    parser.add_argument("--cvae_layers", type=int, default=2)
    parser.add_argument("--cvae_channel_size", type=int, default=128)
    parser.add_argument("--condition_dropout", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--free_bits", type=float, default=0.1)
    parser.add_argument("--kl_anneal_epochs", type=int, default=20)
    parser.add_argument("--vertical_loss_weight", type=float, default=0.0)
    parser.add_argument("--endpoint_loss_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--best_of_n", type=int, default=5)
    parser.add_argument("--subset_vertical_quantile", type=float, default=0.70)

    parser.add_argument("--save_dir", type=str, default="save_model_rebuild")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument("--disable_shuffle", action="store_true")
    parser.add_argument("--multi_agent_weight", type=float, default=None)
    parser.add_argument(
        "--multi_agent_weight_mode",
        type=str,
        default="binary",
        choices=["binary", "linear", "pair_count"],
    )
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloader(project_root, args, split):
    data_dir = resolve_split_dir(project_root, args.dataset_variant, args.dataset_name, split)
    min_agents = args.min_train_agents if split == "train" else args.min_eval_agents
    print(f"[Data] loading {split:<5} from {data_dir}")
    dataset = SceneTrajectoryDataset(
        data_dir=data_dir,
        obs_len=args.obs,
        pred_len=args.preds,
        pred_step=args.preds_step,
        skip=args.skip,
        min_agents=min_agents,
        delim=" ",
        show_progress=True,
        progress_desc=f"Load {split}",
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed + (0 if split == "train" else 10_000))
    sampler = None
    shuffle = split == "train" and not args.disable_shuffle
    has_multi_agent_scene = any(sample["agent_count"] > 1 for sample in dataset.sample_index)

    if (
        split == "train"
        and args.multi_agent_weight is not None
        and args.multi_agent_weight > 1.0
        and has_multi_agent_scene
    ):
        def sample_weight(sample_meta):
            agent_count = sample_meta["agent_count"]
            if agent_count <= 1:
                return 1.0
            if args.multi_agent_weight_mode == "binary":
                return args.multi_agent_weight
            if args.multi_agent_weight_mode == "linear":
                return 1.0 + (agent_count - 1) * (args.multi_agent_weight - 1.0)
            pair_count = agent_count * (agent_count - 1) / 2.0
            return 1.0 + pair_count * (args.multi_agent_weight - 1.0)

        sample_weights = [sample_weight(sample) for sample in dataset.sample_index]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=loader_generator,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=scene_batch_collate,
        generator=loader_generator,
        sampler=sampler,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    print(
        f"[Data] ready   {split:<5} scenes={len(dataset):,} "
        f"batch_size={args.batch_size} batches={len(loader):,}"
    )
    return loader


def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(args):
    return VerticalRelationTrajectoryModel(
        variant=args.variant,
        obs_len=args.obs,
        pred_len=args.preds,
        pred_step=args.preds_step,
        traj_hidden=args.tcn_channels,
        context_hidden=args.context_hidden,
        state_hidden=args.state_hidden,
        interaction_hidden=args.interaction_hidden,
        interaction_heads=args.interaction_heads,
        interaction_topk=args.interaction_topk,
        tcn_kernel=args.tcn_kernel,
        dropout=args.dropout,
        cvae_latent=args.cvae_latent,
        cvae_layers=args.cvae_layers,
        cvae_channel_size=args.cvae_channel_size,
        condition_dropout=args.condition_dropout,
        state_scale_init=args.state_scale_init,
        social_scale_init=args.social_scale_init,
        interaction_integration=args.interaction_integration,
    )


def checkpoint_dir(args):
    return os.path.join(
        args.save_dir,
        args.variant,
        args.protocol,
        args.dataset_variant,
        args.dataset_name,
        str(args.seed),
    )


def checkpoint_path(args):
    save_name = args.save_name or "model.pt"
    return os.path.join(checkpoint_dir(args), save_name)


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
        "source_path": batch["source_path"],
    }


def get_kl_weight(epoch, anneal_epochs, max_weight):
    if anneal_epochs <= 0:
        return max_weight
    return min(1.0, epoch / anneal_epochs) * max_weight


def compute_vertical_threshold(dataset, quantile):
    ranges = [sample["future_vertical_range"] for sample in dataset.sample_index]
    if not ranges:
        return 0.0
    return float(np.quantile(np.asarray(ranges, dtype=np.float32), quantile))


def flatten_metrics(metric_groups):
    flattened = {}
    for subset_name, protocol_metrics in metric_groups.items():
        subset_prefix = "" if subset_name == "all" else f"{subset_name}_"
        for protocol_name, metrics in protocol_metrics.items():
            protocol_suffix = "best5" if protocol_name == "best5" else "1"
            for metric_name, value in metrics.items():
                flattened[f"{subset_prefix}{metric_name}_{protocol_suffix}"] = value
    return flattened


def empty_metric_groups():
    return {
        subset_name: {
            "best5": init_metric_sums(),
            "single": init_metric_sums(),
        }
        for subset_name in ("all", "multi_agent", "strong_vertical")
    }


def empty_count_groups():
    return {
        subset_name: {
            "best5": 0,
            "single": 0,
        }
        for subset_name in ("all", "multi_agent", "strong_vertical")
    }


def subset_masks(batch, vertical_threshold):
    multi_agent_mask = batch["agent_counts"] >= 2
    strong_vertical_mask = batch["future_vertical_ranges"] >= vertical_threshold
    return {
        "all": torch.ones(batch["scene_count"], dtype=torch.bool, device=batch["obs"].device),
        "multi_agent": multi_agent_mask,
        "strong_vertical": strong_vertical_mask,
    }


def evaluate_model(model, loader, device, best_of_n, vertical_threshold, limit_batches=0, desc="Eval"):
    model.eval()
    metric_groups = empty_metric_groups()
    count_groups = empty_count_groups()
    total_scenes = 0
    diversity_sum = 0.0
    diversity_count = 0

    with torch.no_grad():
        for raw_batch in tqdm(
            loader,
            desc=desc,
            unit="batch",
            disable=not progress_enabled(),
            leave=False,
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.5,
            bar_format="{desc:<12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            if limit_batches and total_scenes >= limit_batches * loader.batch_size:
                break
            batch = move_batch_to_device(raw_batch, device)
            target = batch["target"]
            scene_slices = batch["scene_slices"]

            single_prediction = model(
                batch["obs"],
                batch["context"],
                target=None,
                scene_ids=batch["scene_ids"],
                scene_slices=batch["scene_slices"],
                latent_mode="zero",
            )

            if best_of_n > 1:
                best_prediction, sample_diversity = select_best_of_n_prediction(
                    model,
                    batch["obs"],
                    target,
                    batch["context"],
                    batch["scene_ids"],
                    batch["scene_slices"],
                    best_of_n,
                )
                diversity_sum += sample_diversity
                diversity_count += 1
            else:
                best_prediction = single_prediction
                sample_diversity = 0.0

            masks = subset_masks(batch, vertical_threshold)
            predictions = {
                "best5": best_prediction,
                "single": single_prediction,
            }

            for subset_name, scene_mask in masks.items():
                for protocol_name, prediction in predictions.items():
                    totals, scene_count = metric_totals(
                        prediction,
                        target,
                        scene_slices=scene_slices,
                        scene_mask=scene_mask,
                    )
                    update_metric_sums(metric_groups[subset_name][protocol_name], totals)
                    count_groups[subset_name][protocol_name] += scene_count

            total_scenes += batch["scene_count"]

    averaged = {
        subset_name: {
            protocol_name: average_metric_sums(metrics, count_groups[subset_name][protocol_name])
            for protocol_name, metrics in protocol_metrics.items()
        }
        for subset_name, protocol_metrics in metric_groups.items()
    }
    flattened = flatten_metrics(averaged)
    subset_counts = {
        f"{subset_name}_{protocol_name}_scenes": count_groups[subset_name][protocol_name]
        for subset_name in count_groups
        for protocol_name in count_groups[subset_name]
    }
    average_diversity = diversity_sum / max(diversity_count, 1)
    return flattened, subset_counts, total_scenes, average_diversity


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.multi_agent_weight is None:
        args.multi_agent_weight = 4.0 if args.dataset_variant == "social" else 1.0
    set_seed(args.seed)

    recorder = None
    try:
        project_root = os.getcwd()
        os.makedirs(checkpoint_dir(args), exist_ok=True)
        device = select_device(args.device)

        train_loader = build_dataloader(project_root, args, "train")
        test_loader = build_dataloader(project_root, args, "test")
        vertical_threshold = compute_vertical_threshold(train_loader.dataset, args.subset_vertical_quantile)
        print(f"[Eval] strong_vertical threshold={vertical_threshold:.6f} (quantile={args.subset_vertical_quantile:.2f})")

        recorder = RunRecorder(
            checkpoint_dir(args),
            vars(args),
            extra_metadata={
                "device": str(device),
                "train_scenes": len(train_loader.dataset),
                "test_scenes": len(test_loader.dataset),
                "vertical_threshold": vertical_threshold,
            },
        )

        model = build_model(args).to(device)
        param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"[Model] variant={args.variant} | params={param_count:,} | device={device}")

        criterion = TrajectoryForecastLoss(
            kl_weight=args.kl_weight,
            free_bits=args.free_bits,
            vertical_loss_weight=args.vertical_loss_weight,
            endpoint_loss_weight=args.endpoint_loss_weight,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.min_lr,
            )

        primary_key = "ADE_best5" if args.protocol == "bestof5" else "ADE_1"
        best_primary = float("inf")
        best_metrics = None
        best_epoch = 0
        save_path = checkpoint_path(args)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_vertical = 0.0
            epoch_endpoint = 0.0
            batch_count = 0
            diag_delta_abs_sum = 0.0
            diag_delta_abs_max = 0.0
            diag_mu_sum = 0.0
            diag_mu_sq_sum = 0.0
            diag_logvar_sum = 0.0
            diag_logvar_sq_sum = 0.0
            diag_latent_count = 0
            diag_grad_norm_sum = 0.0

            criterion.kl_weight = get_kl_weight(epoch, args.kl_anneal_epochs, args.kl_weight)

            for raw_batch in tqdm(
                train_loader,
                desc=f"Train {epoch:03d}/{args.epochs:03d}",
                unit="batch",
                disable=not progress_enabled(),
                leave=False,
                dynamic_ncols=True,
                ascii=True,
                mininterval=0.5,
                bar_format="{desc:<12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ):
                if args.limit_train_batches and batch_count >= args.limit_train_batches:
                    break
                batch = move_batch_to_device(raw_batch, device)
                optimizer.zero_grad()

                prediction, mu, logvar, decoded_deltas = model(
                    batch["obs"],
                    batch["context"],
                    target=batch["target"],
                    scene_ids=batch["scene_ids"],
                    scene_slices=batch["scene_slices"],
                )
                loss, recon, kl, vertical_aux, endpoint_aux = criterion(prediction, batch["target"], mu, logvar)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                with torch.no_grad():
                    delta_abs = decoded_deltas.abs()
                    diag_delta_abs_sum += delta_abs.mean().item()
                    diag_delta_abs_max = max(diag_delta_abs_max, delta_abs.max().item())
                    diag_mu_sum += mu.mean().item()
                    diag_mu_sq_sum += (mu ** 2).mean().item()
                    diag_logvar_sum += logvar.mean().item()
                    diag_logvar_sq_sum += (logvar ** 2).mean().item()
                    diag_latent_count += 1
                diag_grad_norm_sum += grad_norm.item()

                epoch_loss += loss.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
                epoch_vertical += vertical_aux.item()
                epoch_endpoint += endpoint_aux.item()
                batch_count += 1

            eval_metrics, subset_counts, eval_scene_count, sample_diversity = evaluate_model(
                model,
                test_loader,
                device,
                best_of_n=args.best_of_n,
                vertical_threshold=vertical_threshold,
                limit_batches=args.limit_eval_batches,
                desc=f"Eval  {epoch:03d}/{args.epochs:03d}",
            )

            train_loss = epoch_loss / max(batch_count, 1)
            train_recon = epoch_recon / max(batch_count, 1)
            train_kl = epoch_kl / max(batch_count, 1)
            train_vertical = epoch_vertical / max(batch_count, 1)
            train_endpoint = epoch_endpoint / max(batch_count, 1)

            if diag_latent_count > 0:
                avg_delta_abs = diag_delta_abs_sum / diag_latent_count
                avg_mu = diag_mu_sum / diag_latent_count
                avg_mu_sq = diag_mu_sq_sum / diag_latent_count
                mu_std = max(0.0, avg_mu_sq - avg_mu ** 2) ** 0.5
                avg_logvar = diag_logvar_sum / diag_latent_count
                avg_logvar_sq = diag_logvar_sq_sum / diag_latent_count
                logvar_std = max(0.0, avg_logvar_sq - avg_logvar ** 2) ** 0.5
                avg_grad = diag_grad_norm_sum / max(batch_count, 1)
            else:
                avg_delta_abs = avg_mu = mu_std = avg_logvar = logvar_std = avg_grad = 0.0

            recorder.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                metrics=eval_metrics,
                train_batches=batch_count,
                eval_scenes=eval_scene_count,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                best_checkpoint=save_path if best_metrics is not None else None,
                model_state={
                    "kl_weight": criterion.kl_weight,
                    "recon": train_recon,
                    "kl": train_kl,
                    "vertical_aux": train_vertical,
                    "endpoint_aux": train_endpoint,
                    "delta_mean": avg_delta_abs,
                    "delta_max": diag_delta_abs_max,
                    "mu_mean": avg_mu,
                    "logvar_mean": avg_logvar,
                    "grad_norm": avg_grad,
                    "best_of_n_diversity": sample_diversity,
                    **subset_counts,
                },
            )

            print(format_epoch_header(epoch, args.epochs, train_loss, batch_count, eval_scene_count))
            print(
                f"  loss    | recon={train_recon:.4f} | kl={train_kl:.4f} | "
                f"vert={train_vertical:.4f} | end={train_endpoint:.4f} | "
                f"kl_w={criterion.kl_weight:.3f}"
            )
            print(f"  metrics | {format_metrics(eval_metrics)}")
            print(
                "  diag    | "
                f"delta_mean={avg_delta_abs:.6f} delta_max={diag_delta_abs_max:.6f} "
                f"grad={avg_grad:.4f} bo{args.best_of_n}_div={sample_diversity:.6f}"
            )
            print(
                "  diag    | "
                f"mu={avg_mu:.4f}({mu_std:.4f}) logvar={avg_logvar:.4f}({logvar_std:.4f})"
            )

            if eval_metrics[primary_key] < best_primary:
                best_primary = eval_metrics[primary_key]
                best_metrics = eval_metrics
                best_epoch = epoch
                torch.save(
                    {
                        "args": vars(args),
                        "model_state_dict": model.state_dict(),
                        "metrics": eval_metrics,
                        "vertical_threshold": vertical_threshold,
                    },
                    save_path,
                )
                recorder.log_checkpoint(
                    epoch=epoch,
                    checkpoint_path=save_path,
                    metrics=eval_metrics,
                    reason=f"best_{primary_key}",
                )
                print(green_text(f"  best    | epoch={best_epoch:03d} | {primary_key}={best_metrics[primary_key]:.4f}"))
                print(f"  save    | {save_path}")
            elif best_metrics is not None:
                no_improve = epoch - best_epoch
                patience_str = f" | patience {no_improve}/{args.patience}" if args.patience > 0 else ""
                print(green_text(f"  best    | epoch={best_epoch:03d} | {primary_key}={best_metrics[primary_key]:.4f}{patience_str}"))

            if scheduler is not None:
                scheduler.step()

            if args.patience > 0 and (epoch - best_epoch) >= args.patience:
                print(green_text(f"  early stop | no improvement for {args.patience} epochs since epoch {best_epoch}"))
                break

        if best_metrics is not None:
            summary = recorder.finalize(best_epoch, best_metrics, save_path)
            print(green_text(f"Best checkpoint: {save_path}"))
            print(green_text(f"Best epoch: {best_epoch}"))
            print(green_text(f"Best {primary_key}: {best_metrics[primary_key]:.4f}"))
            print("Run summary:", os.path.join(checkpoint_dir(args), "run_summary.json"))
            if summary.get("diagnostics"):
                print("Diagnostics:", " | ".join(summary["diagnostics"]))
    except KeyboardInterrupt:
        if recorder is not None:
            recorder.finalize_incomplete("interrupted", "Training interrupted before finalize().")
        raise
    except Exception as exc:
        if recorder is not None:
            recorder.finalize_incomplete("failed", f"{type(exc).__name__}: {exc}")
        crash_log = os.path.join(checkpoint_dir(args), "crash.log")
        os.makedirs(os.path.dirname(crash_log), exist_ok=True)
        with open(crash_log, "w", encoding="utf-8") as handle:
            import traceback

            traceback.print_exc(file=handle)
        raise


if __name__ == "__main__":
    main()
