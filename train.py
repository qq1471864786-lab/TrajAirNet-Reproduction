import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import (
    HAINet,
    HAINetLoss,
    SceneTrajectoryDataset,
    average_metric_sums,
    init_metric_sums,
    metric_totals,
    resolve_split_dir,
    scene_batch_collate,
    select_best_of_n_prediction,
    update_metric_sums,
)
from model.run_logging import RunRecorder


def format_metrics(metrics):
    ordered_names = ("ADE", "FDE", "MDE", "AADE", "AFDE", "AMDE")
    return " | ".join(f"{name}={metrics[name]:.4f}" for name in ordered_names)


def format_epoch_header(epoch, total_epochs, train_loss, train_batches, eval_scenes):
    return (
        f"[Epoch {epoch:03d}/{total_epochs:03d}] "
        f"loss={train_loss:.4f} | train_batches={train_batches} | eval_scenes={eval_scenes}"
    )


def progress_enabled():
    return sys.stderr.isatty()


def supports_color():
    return sys.stdout.isatty()


def green_text(text):
    if not supports_color():
        return text
    return f"\033[92m{text}\033[0m"


def build_parser():
    parser = argparse.ArgumentParser(description="Train HAINet")
    # Data
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

    # Model architecture
    parser.add_argument("--tcn_channels", type=int, default=256)
    parser.add_argument("--tcn_layers", type=int, default=2)
    parser.add_argument("--tcn_kernel", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gat_hidden", type=int, default=256)
    parser.add_argument("--gat_heads", type=int, default=8)
    parser.add_argument("--gat_dropout", type=float, default=0.05)
    parser.add_argument("--cvae_latent", type=int, default=128)
    parser.add_argument("--cvae_hidden", type=int, default=128)
    parser.add_argument("--cvae_layers", type=int, default=2)
    parser.add_argument("--cvae_channel_size", type=int, default=128)
    parser.add_argument("--mlp_layer", type=int, default=32)
    parser.add_argument("--condition_dropout", type=float, default=0.1)

    # Ablation switches (only 3, clean)
    parser.add_argument("--disable_interaction", action="store_true",
                        help="Remove AC-GAT (like ACTrajNet)")
    parser.add_argument("--disable_height_conditioning", action="store_true",
                        help="Remove altitude bias in GAT attention")
    parser.add_argument("--disable_height_feedback", action="store_true",
                        help="Remove interaction->height feedback (single direction)")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--kl_weight", type=float, default=1.0)  # ACTrajNet original: implicit 1.0
    parser.add_argument("--free_bits", type=float, default=0.1,
                        help="Free-bits KL floor per latent dimension")
    parser.add_argument("--kl_anneal_epochs", type=int, default=20,
                        help="KL annealing ramp epochs (0=no annealing)")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (0=disabled)")

    # Evaluation
    parser.add_argument("--best_of_n", type=int, default=5,
                        help="Best-of-N sampling at test time")

    # Misc
    parser.add_argument("--save_dir", type=str, default="save_model")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument("--disable_shuffle", action="store_true")
    parser.add_argument("--multi_agent_weight", type=float, default=None)
    parser.add_argument(
        "--multi_agent_weight_mode", type=str, default="binary",
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
    has_multi_agent_scene = any(len(s["agent_ids"]) > 1 for s in dataset.sample_index)
    if split == "train" and args.multi_agent_weight > 1.0 and has_multi_agent_scene:
        def sample_weight(sample_meta):
            agent_count = len(sample_meta["agent_ids"])
            if agent_count <= 1:
                return 1.0
            if args.multi_agent_weight_mode == "binary":
                return args.multi_agent_weight
            if args.multi_agent_weight_mode == "linear":
                return 1.0 + (agent_count - 1) * (args.multi_agent_weight - 1.0)
            pair_count = agent_count * (agent_count - 1) / 2.0
            return 1.0 + pair_count * (args.multi_agent_weight - 1.0)

        sample_weights = [sample_weight(s) for s in dataset.sample_index]
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
    return HAINet(
        obs_len=args.obs,
        pred_len=args.preds,
        pred_step=args.preds_step,
        tcn_channel_size=args.tcn_channels,
        tcn_layers=args.tcn_layers,
        tcn_kernel=args.tcn_kernel,
        dropout=args.dropout,
        cvae_hidden=args.cvae_latent,
        cvae_layers=args.cvae_layers,
        cvae_channel_size=args.cvae_channel_size,
        mlp_layer=args.mlp_layer,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        condition_dropout=args.condition_dropout,
        use_interaction=not args.disable_interaction,
        use_height_conditioning=not args.disable_height_conditioning,
        use_height_feedback=not args.disable_height_feedback,
    )


def checkpoint_dir(args):
    return os.path.join(args.save_dir, args.dataset_variant, args.dataset_name, str(args.seed))


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
        "source_path": batch["source_path"],
    }


def get_kl_weight(epoch, anneal_epochs, max_weight, cyclical=True, n_cycles=4, total_epochs=50):
    """
    KL annealing schedule.
    cyclical=True: cyclical annealing (Fu et al. 2019) — prevents KL collapse
    cyclical=False: linear annealing 0 -> max_weight over anneal_epochs
    """
    if anneal_epochs <= 0:
        return max_weight
    if cyclical:
        cycle_len = total_epochs / n_cycles
        pos_in_cycle = (epoch - 1) % cycle_len
        ramp = cycle_len * 0.5  # first half ramps up, second half stays
        if pos_in_cycle < ramp:
            return (pos_in_cycle / ramp) * max_weight
        return max_weight
    return min(1.0, epoch / anneal_epochs) * max_weight


def evaluate_best_of_n(model, loader, device, n_samples=5, limit_batches=0, desc="Eval"):
    """Best-of-N evaluation with per-agent sample selection."""
    model.eval()
    metric_sums = init_metric_sums()
    processed_batches = 0
    total_agents = 0
    total_scenes = 0
    diversity_sum = 0.0
    diversity_count = 0
    with torch.no_grad():
        for raw_batch in tqdm(
            loader, desc=desc, unit="batch",
            disable=not progress_enabled(), leave=False,
            dynamic_ncols=True, ascii=True, mininterval=0.5,
            bar_format="{desc:<12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            if limit_batches and processed_batches >= limit_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            obs = batch["obs"]
            target = batch["target"]
            context = batch["context"]
            scene_ids = batch["scene_ids"]
            scene_slices = batch["scene_slices"]

            best_prediction, sample_diversity = select_best_of_n_prediction(
                model, obs, target, context, scene_ids, scene_slices, n_samples
            )

            if n_samples > 1:
                diversity_sum += sample_diversity
                diversity_count += 1

            batch_metric_totals, agent_count = metric_totals(best_prediction, target, scene_slices)
            update_metric_sums(metric_sums, batch_metric_totals)
            processed_batches += 1
            total_agents += agent_count
            total_scenes += batch["scene_count"]
    avg_diversity = diversity_sum / max(diversity_count, 1)
    return average_metric_sums(metric_sums, total_agents), total_scenes, avg_diversity


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
        recorder = RunRecorder(
            checkpoint_dir(args),
            vars(args),
            extra_metadata={
                "device": str(device),
                "train_scenes": len(train_loader.dataset),
                "test_scenes": len(test_loader.dataset),
            },
        )

        model = build_model(args).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] HAINet | params={param_count:,} | device={device}")
        print(f"[Model] interaction={not args.disable_interaction} | "
              f"height_cond={not args.disable_height_conditioning} | "
              f"height_feedback={not args.disable_height_feedback}")

        criterion = HAINetLoss(
            kl_weight=args.kl_weight,
            free_bits=args.free_bits,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = None
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.min_lr,
            )

        best_ade = float("inf")
        best_metrics = None
        best_epoch = 0
        save_path = checkpoint_path(args)

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            batch_count = 0
            # Diagnostic accumulators
            diag_acc_abs_sum = 0.0
            diag_acc_abs_max = 0.0
            diag_mu_sum = 0.0
            diag_mu_sq_sum = 0.0
            diag_logvar_sum = 0.0
            diag_logvar_sq_sum = 0.0
            diag_latent_count = 0
            diag_grad_norm_sum = 0.0
            kl_w = get_kl_weight(epoch, args.kl_anneal_epochs, args.kl_weight,
                                 cyclical=False, n_cycles=4, total_epochs=args.epochs)
            criterion.kl_weight = kl_w

            for raw_batch in tqdm(
                train_loader,
                desc=f"Train {epoch:03d}/{args.epochs:03d}",
                unit="batch",
                disable=not progress_enabled(), leave=False,
                dynamic_ncols=True, ascii=True, mininterval=0.5,
                bar_format="{desc:<12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ):
                if args.limit_train_batches and batch_count >= args.limit_train_batches:
                    break
                batch = move_batch_to_device(raw_batch, device)
                optimizer.zero_grad()

                prediction, mu, logvar, acc = model(
                    batch["obs"], batch["context"],
                    target=batch["target"],
                    scene_ids=batch["scene_ids"],
                    scene_slices=batch["scene_slices"],
                )
                loss, recon, kl = criterion(prediction, batch["target"], mu, logvar)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

                # Collect diagnostics
                with torch.no_grad():
                    acc_abs = acc.abs()
                    diag_acc_abs_sum += acc_abs.mean().item()
                    diag_acc_abs_max = max(diag_acc_abs_max, acc_abs.max().item())
                    diag_mu_sum += mu.mean().item()
                    diag_mu_sq_sum += (mu ** 2).mean().item()
                    diag_logvar_sum += logvar.mean().item()
                    diag_logvar_sq_sum += (logvar ** 2).mean().item()
                    diag_latent_count += 1
                diag_grad_norm_sum += grad_norm.item()

                epoch_loss += loss.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
                batch_count += 1

            test_metrics, eval_scene_count, sample_diversity = evaluate_best_of_n(
                model, test_loader, device,
                n_samples=args.best_of_n,
                limit_batches=args.limit_eval_batches,
                desc=f"Eval  {epoch:03d}/{args.epochs:03d}",
            )
            train_loss = epoch_loss / max(batch_count, 1)
            train_recon = epoch_recon / max(batch_count, 1)
            train_kl = epoch_kl / max(batch_count, 1)

            if diag_latent_count > 0:
                avg_acc_abs = diag_acc_abs_sum / diag_latent_count
                avg_mu = diag_mu_sum / diag_latent_count
                avg_mu_sq = diag_mu_sq_sum / diag_latent_count
                mu_std = max(0, avg_mu_sq - avg_mu ** 2) ** 0.5
                avg_logvar = diag_logvar_sum / diag_latent_count
                avg_logvar_sq = diag_logvar_sq_sum / diag_latent_count
                logvar_std = max(0, avg_logvar_sq - avg_logvar ** 2) ** 0.5
                avg_grad = diag_grad_norm_sum / batch_count
            else:
                avg_acc_abs = avg_mu = mu_std = avg_logvar = logvar_std = avg_grad = 0.0

            recorder.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                metrics=test_metrics,
                train_batches=batch_count,
                eval_scenes=eval_scene_count,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                best_checkpoint=save_path if best_metrics is not None else None,
                model_state={
                    "kl_weight": kl_w, "recon": train_recon, "kl": train_kl,
                    "acc_mean": avg_acc_abs, "acc_max": diag_acc_abs_max,
                    "mu_mean": avg_mu, "logvar_mean": avg_logvar,
                    "grad_norm": avg_grad, "bo_n_diversity": sample_diversity,
                },
            )
            print(format_epoch_header(epoch, args.epochs, train_loss, batch_count, eval_scene_count))
            print(f"  loss    | recon={train_recon:.4f} | kl={train_kl:.4f} | kl_w={kl_w:.3f}")
            print(f"  metrics | {format_metrics(test_metrics)}")
            print(f"  diag    | acc_mean={avg_acc_abs:.6f} acc_max={diag_acc_abs_max:.6f} grad={avg_grad:.4f}")
            print(f"  diag    | mu={avg_mu:.4f}({mu_std:.4f}) logvar={avg_logvar:.4f}({logvar_std:.4f}) bo{args.best_of_n}_div={sample_diversity:.6f}")

            if test_metrics["ADE"] < best_ade:
                best_ade = test_metrics["ADE"]
                best_metrics = test_metrics
                best_epoch = epoch
                torch.save(
                    {
                        "args": vars(args),
                        "model_state_dict": model.state_dict(),
                        "metrics": test_metrics,
                    },
                    save_path,
                )
                recorder.log_checkpoint(
                    epoch=epoch, checkpoint_path=save_path,
                    metrics=test_metrics, reason="best_ade",
                )
                print(green_text(f"  best    | epoch={best_epoch:03d} | {format_metrics(best_metrics)}"))
                print(f"  save    | {save_path}")
            elif best_metrics is not None:
                no_improve = epoch - best_epoch
                patience_str = f" | patience {no_improve}/{args.patience}" if args.patience > 0 else ""
                print(f"  best    | epoch={best_epoch:03d} | {format_metrics(best_metrics)}{patience_str}")

            if scheduler is not None:
                scheduler.step()

            # Early stopping
            if args.patience > 0 and (epoch - best_epoch) >= args.patience:
                print(green_text(f"  early stop | no improvement for {args.patience} epochs since epoch {best_epoch}"))
                break

        if best_metrics is not None:
            summary = recorder.finalize(best_epoch, best_metrics, save_path)
            print(green_text(f"Best checkpoint: {save_path}"))
            print(green_text(f"Best epoch: {best_epoch}"))
            print(green_text(f"Best metrics: {format_metrics(best_metrics)}"))
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
        with open(crash_log, "w", encoding="utf-8") as f:
            import traceback
            traceback.print_exc(file=f)
        raise


if __name__ == "__main__":
    main()
