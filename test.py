import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from functools import partial

import torch
from torch.utils.data import DataLoader

from model import ProtoBasisFlight, ProtoBasisSceneDataset, proto_basis_collate, resolve_split_dir, summarize_batch_metrics



def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate ProtoBasis-Flight checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
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
    return parser



def build_model(config, checkpoint):
    meta = checkpoint.get("meta", {})
    return ProtoBasisFlight(
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



def evaluate(model, loader, device, limit_eval_batches=0):
    sums = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if limit_eval_batches and batch_index >= limit_eval_batches:
                break
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            outputs = model(batch["obs_xyz"], batch["obs_mask"], enable_refiner=True)
            metrics, batch_count, _ = summarize_batch_metrics(outputs, batch)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value * batch_count
            count += batch_count
    return {name: value / max(count, 1) for name, value in sums.items()}


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



def main():
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    }
    dataset = ProtoBasisSceneDataset(
        data_dir=split_dir,
        split_name=args.split,
        obs_len=config["obs"],
        pred_len=config["preds"],
        max_agents=config["max_agents"],
        model_artifact=model_artifact,
        n_proto=config["n_proto"],
        basis_dim=config["basis_dim"],
        rare_threshold=config["rare_threshold"],
    )
    loader = build_loader(dataset, args.batch_size, args, config["max_agents"])

    model = build_model(config, checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, loader, device, limit_eval_batches=args.limit_eval_batches)

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

    print(" ".join(f"{name}={value:.4f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
