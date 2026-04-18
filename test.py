import argparse
import os
import warnings
from contextlib import nullcontext
from functools import partial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
)

import torch
from torch.utils.data import DataLoader

from model import (
    ProtoBasisNet,
    ProtoBasisSceneDataset,
    average_metric_sums,
    init_metric_sums,
    metric_names_for_protocol,
    proto_basis_collate,
    summarize_batch_metrics,
    update_metric_sums,
)
from model.data import resolve_split_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate ProtoBasis-Net checkpoints")
    parser.add_argument("checkpoint", nargs="?", default="", help="Checkpoint path.")
    parser.add_argument("--checkpoint", dest="checkpoint_flag", default="", help="Checkpoint path.")
    parser.add_argument("--dataset_variant", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--limit_eval_batches", type=int, default=0)
    parser.add_argument("--measure_latency", action="store_true")
    parser.add_argument("--latency_batch_sizes", type=str, default="1,16")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU fallback when CUDA is unavailable.")
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
        dropout=config["dropout"],
        proto_summary_5d=torch.tensor(checkpoint["proto_summary_5d"], dtype=torch.float32),
        proto_frequency=torch.tensor(checkpoint["proto_freq"], dtype=torch.float32),
        basis_bank=torch.tensor(checkpoint["basis_bank"], dtype=torch.float32),
        disable_social=config.get("disable_social", False),
        disable_router=config.get("disable_router", False),
        disable_refiner=config.get("disable_refiner", False),
    )


def evaluate(model, loader, device, config, use_amp=False, limit_eval_batches=0):
    metric_names = metric_names_for_protocol(
        config.get("eval_topk_primary", 5),
        config.get("eval_topk_secondary", 20),
    )
    metric_sums = init_metric_sums(metric_names)
    total_count = 0
    rare_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if limit_eval_batches and batch_index >= limit_eval_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            with autocast_context(device, use_amp):
                outputs = model(batch["obs_xyz"], batch["obs_mask"], enable_refiner=True)
            metrics, batch_count, _ = summarize_batch_metrics(
                outputs,
                batch,
                primary_k=config.get("eval_topk_primary", 5),
                secondary_k=config.get("eval_topk_secondary", 20),
                glev_topn_primary=config.get("glev_topn_primary", 2),
                glev_topn_secondary=config.get("glev_topn_secondary", 5),
            )
            update_metric_sums(metric_sums, metrics, batch_count)
            total_count += batch_count
            rare_count += int(batch["is_rare"].sum().item())
    return average_metric_sums(metric_sums, total_count, rare_count)


@torch.no_grad()
def measure_latency_ms(model, batch, warmup=30, iters=100):
    if next(model.parameters()).device.type != "cuda":
        return None
    model.eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        _ = model(batch["obs_xyz"], batch["obs_mask"], enable_refiner=True)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        starter.record()
        _ = model(batch["obs_xyz"], batch["obs_mask"], enable_refiner=True)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
    tensor = torch.tensor(timings)
    return {
        "latency_mean_ms": float(tensor.mean().item()),
        "latency_p50_ms": float(tensor.median().item()),
        "latency_p90_ms": float(torch.quantile(tensor, 0.9).item()),
    }


def build_loader(dataset, batch_size, args, max_agents):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(proto_basis_collate, max_agents=max_agents),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )


def ordered_metric_items(config, metrics):
    ordered_names = list(
        metric_names_for_protocol(
            config.get("eval_topk_primary", 5),
            config.get("eval_topk_secondary", 20),
        )
    )
    ordered_names.extend(name for name in metrics.keys() if name not in ordered_names)
    return [(name, metrics[name]) for name in ordered_names if name in metrics]


def main():
    args = build_parser().parse_args()
    checkpoint_path = args.checkpoint or args.checkpoint_flag
    if not checkpoint_path:
        raise SystemExit("checkpoint path is required")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise SystemExit("CUDA is not available. Use --allow_cpu only if you really want to run on CPU.")
    use_amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_variant = args.dataset_variant or config["dataset_variant"]
    dataset_name = args.dataset_name or config["dataset_name"]

    project_root = os.getcwd()
    split_dir = resolve_split_dir(project_root, dataset_variant, dataset_name, args.split)
    model_artifact = {
        "summary_5d": checkpoint["proto_summary_5d"],
        "frequency": checkpoint["proto_freq"],
        "rare_ids": torch.nonzero(torch.tensor(checkpoint["proto_freq"]) < config["rare_threshold"]).view(-1).tolist(),
        "basis_bank": checkpoint["basis_bank"],
        "n_proto": config["n_proto"],
        "basis_dim": config["basis_dim"],
        "rare_threshold": config["rare_threshold"],
        "obs_len": config["obs"],
        "pred_len": config["preds"],
        "obs_stride": config.get("obs_stride", 1),
        "pred_stride": config.get("pred_stride", 1),
    }
    dataset = ProtoBasisSceneDataset(
        data_dir=split_dir,
        split_name=args.split,
        obs_len=config["obs"],
        pred_len=config["preds"],
        obs_stride=config.get("obs_stride", 1),
        pred_stride=config.get("pred_stride", 1),
        max_agents=config["max_agents"],
        model_artifact=model_artifact,
        n_proto=config["n_proto"],
        basis_dim=config["basis_dim"],
        rare_threshold=config["rare_threshold"],
    )
    loader = build_loader(dataset, args.batch_size, args, config["max_agents"])

    model = build_model(config, checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, loader, device, config, use_amp=use_amp, limit_eval_batches=args.limit_eval_batches)

    if args.measure_latency:
        for bs in [int(token) for token in args.latency_batch_sizes.split(",") if token.strip()]:
            latency_loader = build_loader(dataset, bs, args, config["max_agents"])
            batch = next(iter(latency_loader))
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            latency = measure_latency_ms(model, batch)
            if latency is not None:
                metrics[f"latency_bs{bs}_mean_ms"] = latency["latency_mean_ms"]
                metrics[f"latency_bs{bs}_p50_ms"] = latency["latency_p50_ms"]
                metrics[f"latency_bs{bs}_p90_ms"] = latency["latency_p90_ms"]

    print(" ".join(f"{name}={value:.4f}" for name, value in ordered_metric_items(config, metrics)))


if __name__ == "__main__":
    main()
