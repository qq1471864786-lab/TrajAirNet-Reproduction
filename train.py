import argparse
import math
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import (
    ProtoBasisFlight,
    ProtoBasisLoss,
    ProtoBasisSceneDataset,
    average_metric_sums,
    basis_hash,
    git_commit,
    init_metric_sums,
    model_artifact_hash,
    proto_basis_collate,
    protocol_hash,
    resolve_protocol,
    resolve_split_dir,
    summarize_batch_metrics,
    update_metric_sums,
)
from model.run_logging import RunRecorder



def build_parser():
    parser = argparse.ArgumentParser(description="Train ProtoBasis-Flight")
    parser.add_argument("--dataset_variant", type=str, default="social", choices=["social"])
    parser.add_argument("--dataset_name", type=str, default="111_days")
    parser.add_argument("--protocol_name", type=str, default="trajair_40to120_best20")
    parser.add_argument("--obs", type=int, default=40)
    parser.add_argument("--preds", type=int, default=120)
    parser.add_argument("--max_agents", type=int, default=7)
    parser.add_argument("--n_proto", type=int, default=64)
    parser.add_argument("--basis_dim", type=int, default=16)
    parser.add_argument("--topk_proto", type=int, default=5)
    parser.add_argument("--micro_per_proto", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=192)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--social_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--phase_a_epochs", type=int, default=10)
    parser.add_argument("--phase_b_epochs", type=int, default=35)
    parser.add_argument("--phase_c_epochs", type=int, default=20)
    parser.add_argument("--stage_a_lr", type=float, default=3e-4)
    parser.add_argument("--stage_b_lr", type=float, default=2e-4)
    parser.add_argument("--stage_c_lr", type=float, default=8e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--rare_threshold", type=float, default=0.02)

    parser.add_argument("--lambda_xyz", type=float, default=1.0)
    parser.add_argument("--lambda_fde", type=float, default=0.8)
    parser.add_argument("--lambda_proto", type=float, default=0.4)
    parser.add_argument("--lambda_res", type=float, default=0.2)
    parser.add_argument("--lambda_score", type=float, default=0.5)
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.05)
    parser.add_argument("--lambda_coeff", type=float, default=0.02)
    parser.add_argument("--lambda_smooth", type=float, default=0.10)

    parser.add_argument("--disable_social", action="store_true")
    parser.add_argument("--disable_router", action="store_true")
    parser.add_argument("--disable_refiner", action="store_true")

    parser.add_argument("--save_dir", type=str, default="save_model")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    return parser



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def select_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def apply_protocol(args):
    spec = resolve_protocol(args.protocol_name)
    args.obs = spec.obs
    args.preds = spec.preds
    return spec



def stage_config(epoch, args):
    if epoch <= args.phase_a_epochs:
        return {
            "name": "basis_warmup",
            "enable_refiner": False,
            "force_gt_proto": True,
            "rank_weight": 0.0,
            "div_weight": 0.0,
            "rare_weight": 1.0,
        }
    if epoch <= args.phase_a_epochs + args.phase_b_epochs:
        return {
            "name": "joint_no_refiner",
            "enable_refiner": False,
            "force_gt_proto": False,
            "rank_weight": 1.0,
            "div_weight": 0.5,
            "rare_weight": 1.0,
        }
    return {
        "name": "joint_refiner",
        "enable_refiner": True,
        "force_gt_proto": False,
        "rank_weight": 1.0,
        "div_weight": 1.0,
        "rare_weight": 1.5,
    }



def set_epoch_lr(optimizer, epoch, args):
    if epoch <= args.phase_a_epochs:
        lr = args.stage_a_lr
    elif epoch <= args.phase_a_epochs + args.phase_b_epochs:
        phase_epoch = epoch - args.phase_a_epochs - 1
        denom = max(args.phase_b_epochs - 1, 1)
        progress = phase_epoch / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.stage_c_lr + (args.stage_b_lr - args.stage_c_lr) * cosine
    else:
        phase_epoch = epoch - args.phase_a_epochs - args.phase_b_epochs - 1
        denom = max(args.phase_c_epochs - 1, 1)
        progress = phase_epoch / denom
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = args.min_lr + (args.stage_c_lr - args.min_lr) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr



def build_train_loader(dataset, args, rare_weight):
    weights = [rare_weight if sample["is_rare"] else 1.0 for sample in dataset.sample_index]
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=partial(proto_basis_collate, max_agents=args.max_agents),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )



def build_eval_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(proto_basis_collate, max_agents=args.max_agents),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )



def build_model(args, model_artifact):
    return ProtoBasisFlight(
        obs_len=args.obs,
        pred_len=args.preds,
        d_model=args.d_model,
        nhead=args.nhead,
        ff_dim=args.ff_dim,
        encoder_layers=args.encoder_layers,
        social_layers=args.social_layers,
        topk_proto=args.topk_proto,
        n_micro=args.micro_per_proto,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        dropout=args.dropout,
        proto_summary_5d=torch.tensor(model_artifact["summary_5d"], dtype=torch.float32),
        proto_frequency=torch.tensor(model_artifact["frequency"], dtype=torch.float32),
        basis_bank=torch.tensor(model_artifact["basis_bank"], dtype=torch.float32),
        disable_social=args.disable_social,
        disable_router=args.disable_router,
        disable_refiner=args.disable_refiner,
    )



def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved



def init_best_records(run_dir):
    return {
        "fde20": {"value": float("inf"), "epoch": 0, "path": os.path.join(run_dir, "best_fde20.pt")},
        "ade5": {"value": float("inf"), "epoch": 0, "path": os.path.join(run_dir, "best_ade5.pt")},
        "glev_report20": {"value": float("inf"), "epoch": 0, "path": os.path.join(run_dir, "best_glev_report20.pt")},
        "rare_fde20": {"value": float("inf"), "epoch": 0, "path": os.path.join(run_dir, "best_rare_fde20.pt")},
    }



def checkpoint_payload(model, optimizer, epoch, global_step, best_records, model_artifact, config, meta):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best": {
            "fde20": best_records["fde20"]["value"],
            "ade5": best_records["ade5"]["value"],
            "glev_report20": best_records["glev_report20"]["value"],
            "rare_fde20": best_records["rare_fde20"]["value"],
        },
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "proto_summary_5d": model_artifact["summary_5d"],
        "proto_freq": model_artifact["frequency"],
        "basis_bank": model_artifact["basis_bank"],
        "config": config,
        "meta": meta,
    }



def save_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)



def evaluate(model, loader, device, enable_refiner, limit_eval_batches=0):
    metric_sums = init_metric_sums()
    total_count = 0
    rare_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if limit_eval_batches and batch_index >= limit_eval_batches:
                break
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch["obs_xyz"], batch["obs_mask"], enable_refiner=enable_refiner)
            metrics, count, rare = summarize_batch_metrics(outputs, batch)
            update_metric_sums(metric_sums, metrics, count)
            total_count += count
            rare_count += rare
    return average_metric_sums(metric_sums, total_count, rare_count), total_count, rare_count



def peak_memory_mb(device):
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))



def main():
    args = build_parser().parse_args()
    apply_protocol(args)
    args.epochs = args.phase_a_epochs + args.phase_b_epochs + args.phase_c_epochs
    set_seed(args.seed)
    device = select_device(args.device)
    project_root = os.getcwd()

    train_dir = resolve_split_dir(project_root, args.dataset_variant, args.dataset_name, "train")
    test_dir = resolve_split_dir(project_root, args.dataset_variant, args.dataset_name, "test")

    train_dataset = ProtoBasisSceneDataset(
        data_dir=train_dir,
        split_name="train",
        obs_len=args.obs,
        pred_len=args.preds,
        max_agents=args.max_agents,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        rare_threshold=args.rare_threshold,
    )
    model_artifact = train_dataset.export_model_artifact()
    eval_dataset = ProtoBasisSceneDataset(
        data_dir=test_dir,
        split_name="test",
        obs_len=args.obs,
        pred_len=args.preds,
        max_agents=args.max_agents,
        model_artifact=model_artifact,
        n_proto=args.n_proto,
        basis_dim=args.basis_dim,
        rare_threshold=args.rare_threshold,
    )

    eval_loader = build_eval_loader(eval_dataset, args)
    model = build_model(args, model_artifact).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.stage_a_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    loss_fn = ProtoBasisLoss(
        lambda_xyz=args.lambda_xyz,
        lambda_fde=args.lambda_fde,
        lambda_proto=args.lambda_proto,
        lambda_res=args.lambda_res,
        lambda_score=args.lambda_score,
        lambda_rank=args.lambda_rank,
        lambda_div=args.lambda_div,
        lambda_coeff=args.lambda_coeff,
        lambda_smooth=args.lambda_smooth,
    )

    run_dir = os.path.join(args.save_dir, args.dataset_name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    meta = {
        "protocol_name": args.protocol_name,
        "protocol_hash": protocol_hash(vars(args)),
        "artifact_hash": model_artifact_hash(model_artifact),
        "basis_hash": basis_hash(model_artifact),
        "git_commit": git_commit(project_root),
        "rare_count": int(len(model_artifact["rare_ids"])),
        "n_proto": args.n_proto,
        "basis_dim": args.basis_dim,
    }
    recorder = RunRecorder(run_dir, vars(args), extra_metadata=meta)
    best_records = init_best_records(run_dir)
    last_path = os.path.join(run_dir, "last.pt")
    global_step = 0

    try:
        for epoch in range(1, args.epochs + 1):
            cfg = stage_config(epoch, args)
            current_lr = set_epoch_lr(optimizer, epoch, args)
            train_loader = build_train_loader(train_dataset, args, rare_weight=cfg["rare_weight"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            seen_batches = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            progress = tqdm(train_loader, ncols=110, leave=False)
            for batch_index, raw_batch in enumerate(progress):
                if args.limit_train_batches and batch_index >= args.limit_train_batches:
                    break
                batch = move_batch_to_device(raw_batch, device)
                outputs = model(
                    batch["obs_xyz"],
                    batch["obs_mask"],
                    gt_proto_id=batch["gt_proto_id"],
                    force_gt_proto=cfg["force_gt_proto"],
                    enable_refiner=cfg["enable_refiner"],
                )
                loss, loss_stats = loss_fn(outputs, batch, cfg)
                loss = loss / max(args.grad_accum, 1)
                loss.backward()

                if (batch_index + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                train_loss_sum += float(loss.detach().item()) * max(args.grad_accum, 1)
                seen_batches += 1
                progress.set_description(
                    f"{cfg['name']} xyz={loss_stats['xyz']:.4f} fde={loss_stats['fde']:.4f} "
                    f"proto={loss_stats['proto']:.4f} coeff={loss_stats['coeff']:.4f}"
                )

            if seen_batches and seen_batches % args.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            metrics, _, _ = evaluate(
                model,
                eval_loader,
                device=device,
                enable_refiner=cfg["enable_refiner"],
                limit_eval_batches=args.limit_eval_batches,
            )
            memory_mb = peak_memory_mb(device)
            train_loss = train_loss_sum / max(seen_batches, 1)

            ckpt = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_records=best_records,
                model_artifact=model_artifact,
                config=vars(args),
                meta=meta,
            )
            save_checkpoint(last_path, ckpt)

            tracked = {
                "fde20": ("FDE@20", best_records["fde20"]["path"]),
                "ade5": ("ADE@5", best_records["ade5"]["path"]),
                "glev_report20": ("GLeV_report@20", best_records["glev_report20"]["path"]),
                "rare_fde20": ("rare_FDE@20", best_records["rare_fde20"]["path"]),
            }
            for best_key, (metric_key, ckpt_path) in tracked.items():
                value = metrics[metric_key]
                if math.isnan(value):
                    continue
                if value < best_records[best_key]["value"]:
                    best_records[best_key]["value"] = value
                    best_records[best_key]["epoch"] = epoch
                    save_checkpoint(ckpt_path, ckpt)

            recorder.log_epoch(epoch, cfg["name"], train_loss, metrics, current_lr, memory_mb)
            recorder.update_live_status(epoch, best_records)
            print(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] phase={cfg['name']} "
                f"loss={train_loss:.4f} ADE@5={metrics['ADE@5']:.4f} FDE@5={metrics['FDE@5']:.4f} "
                f"ADE@20={metrics['ADE@20']:.4f} FDE@20={metrics['FDE@20']:.4f} "
                f"GLeV_report@20={metrics['GLeV_report@20']:.4f}"
            )

        recorder.finalize(best_records)
        print("[Done] ProtoBasis-Flight training finished.")

    except KeyboardInterrupt:
        recorder.finalize_incomplete("interrupted", "Training interrupted by user.")
        raise
    except Exception as exc:
        recorder.finalize_incomplete("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
