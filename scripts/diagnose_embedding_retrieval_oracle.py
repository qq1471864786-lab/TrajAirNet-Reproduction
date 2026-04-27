import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import ProtoBasisSceneDataset, proto_basis_collate  # noqa: E402
from model.data import resolve_split_dir  # noqa: E402
from model.proto_basis_flight_model import build_global_features, build_local_features  # noqa: E402
from test import build_model, load_checkpoint_state  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Observed-representation retrieval oracle for ProtoBasis-Net.")
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--dataset_variant", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--train_count", type=int, default=50000)
    parser.add_argument("--test_count", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--query_chunk", type=int, default=256)
    parser.add_argument("--bank_chunk", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", default="")
    return parser


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_dataset(config, checkpoint, dataset_name, split):
    split_dir = resolve_split_dir(os.getcwd(), config["dataset_variant"], dataset_name, split)
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


def deterministic_indices(length, count, seed):
    count = min(int(count), int(length))
    if count <= 0 or count >= length:
        return torch.arange(length, dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(length, generator=generator)[:count].sort().values


def observed_tail_features(batch, tail_len=20):
    obs_xyz = batch["obs_xyz"]
    obs_mask = batch["obs_mask"]
    target = obs_xyz[:, 0]
    local_xyz, _, _, _, _ = batch["_model"].pose_normalizer(obs_xyz)
    target_local = local_xyz[:, 0]
    tail = target_local[:, -min(tail_len, target_local.size(1)) :]
    velocity = target_local[:, 1:] - target_local[:, :-1]
    vel_tail = velocity[:, -min(tail_len, velocity.size(1)) :]
    agent_count = obs_mask.float().sum(dim=1, keepdim=True)
    last_positions = local_xyz[:, :, -1]
    neighbor_disp = last_positions[:, 1:] - last_positions[:, :1]
    neighbor_disp = neighbor_disp.reshape(neighbor_disp.size(0), -1)
    return torch.cat([tail.flatten(1), vel_tail.flatten(1), agent_count, neighbor_disp], dim=-1)


@torch.no_grad()
def collect_features(model, dataset, indices, config, device, batch_size, desc):
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(proto_basis_collate, max_agents=config["max_agents"]),
        pin_memory=device.type == "cuda",
    )
    model.eval()
    buckets = {
        "target_ctx": [],
        "target_scene_ctx": [],
        "router_hidden": [],
        "proto_prob": [],
        "observed_tail20": [],
        "future_summary_5d": [],
    }
    futures = []
    rare = []
    for raw_batch in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False, file=sys.stdout):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        batch["_model"] = model
        obs_xyz = batch["obs_xyz"]
        obs_mask = batch["obs_mask"]
        local_xyz, _, _, _, _ = model.pose_normalizer(obs_xyz)
        feats_local = build_local_features(local_xyz)
        feats_global = build_global_features(obs_xyz)
        agent_feat = model.temporal_encoder(feats_local, feats_global)
        if model.disable_social:
            target_ctx = agent_feat[:, 0]
            scene_ctx = (agent_feat * obs_mask.unsqueeze(-1)).sum(dim=1) / obs_mask.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            target_ctx, scene_ctx = model.social_aggregator(agent_feat, obs_mask)
        router_input = torch.cat([target_ctx, scene_ctx], dim=-1)
        router_hidden = model.prototype_router.trunk(router_input)
        proto_logits = model.prototype_router.logit_head(router_hidden)

        buckets["target_ctx"].append(target_ctx.detach().float().cpu())
        buckets["target_scene_ctx"].append(router_input.detach().float().cpu())
        buckets["router_hidden"].append(router_hidden.detach().float().cpu())
        buckets["proto_prob"].append(proto_logits.softmax(dim=-1).detach().float().cpu())
        buckets["observed_tail20"].append(observed_tail_features(batch).detach().float().cpu())
        buckets["future_summary_5d"].append(batch["proto_summary_5d"].detach().float().cpu())
        futures.append(batch["fut_local"].detach().float().cpu())
        rare.append(batch["is_rare"].detach().cpu())

    features = {name: torch.cat(values, dim=0) for name, values in buckets.items()}
    return features, torch.cat(futures, dim=0), torch.cat(rare, dim=0)


def standardize(train_feat, test_feat):
    mean = train_feat.mean(dim=0, keepdim=True)
    std = train_feat.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_z = (train_feat - mean) / std
    test_z = (test_feat - mean) / std
    train_z = torch.nn.functional.normalize(train_z, dim=-1)
    test_z = torch.nn.functional.normalize(test_z, dim=-1)
    return train_z, test_z


def retrieval_metrics(train_feat, test_feat, train_future, test_future, rare_mask, topk, device, query_chunk, bank_chunk):
    train_feat, test_feat = standardize(train_feat, test_feat)
    train_feat = train_feat.to(device)
    test_feat = test_feat.to(device)
    train_future = train_future.to(device)
    test_future = test_future.to(device)
    safe_topk = min(int(topk), train_feat.size(0))
    all_indices = []
    all_distances = []

    for start in tqdm(range(0, test_feat.size(0), query_chunk), desc="retrieval", dynamic_ncols=True, leave=False, file=sys.stdout):
        query = test_feat[start : start + query_chunk]
        best_dist = None
        best_idx = None
        for bank_start in range(0, train_feat.size(0), bank_chunk):
            bank = train_feat[bank_start : bank_start + bank_chunk]
            dist = torch.cdist(query, bank)
            chunk_dist, chunk_idx = dist.topk(safe_topk, dim=1, largest=False)
            chunk_idx = chunk_idx + bank_start
            if best_dist is None:
                best_dist, best_idx = chunk_dist, chunk_idx
            else:
                merged_dist = torch.cat([best_dist, chunk_dist], dim=1)
                merged_idx = torch.cat([best_idx, chunk_idx], dim=1)
                best_dist, keep = merged_dist.topk(safe_topk, dim=1, largest=False)
                best_idx = merged_idx.gather(1, keep)
        all_indices.append(best_idx.cpu())
        all_distances.append(best_dist.cpu())

    nn_idx = torch.cat(all_indices, dim=0).to(device)
    gathered_future = train_future[nn_idx]
    l2 = torch.linalg.norm(gathered_future - test_future[:, None], dim=-1)
    ade = l2.mean(dim=-1)
    fde = l2[..., -1]
    min_ade = ade.min(dim=1).values.detach().cpu()
    min_fde = fde.min(dim=1).values.detach().cpu()
    rare_mask = rare_mask.bool()
    result = {
        f"top{safe_topk}_minADE": float(min_ade.mean().item()),
        f"top{safe_topk}_minFDE": float(min_fde.mean().item()),
        "top1_ADE": float(ade[:, 0].detach().cpu().mean().item()),
        "top1_FDE": float(fde[:, 0].detach().cpu().mean().item()),
        "ADE_p50": float(torch.quantile(min_ade, 0.5).item()),
        "ADE_p90": float(torch.quantile(min_ade, 0.9).item()),
    }
    if rare_mask.any():
        result[f"rare_top{safe_topk}_minADE"] = float(min_ade[rare_mask].mean().item())
        result[f"rare_top{safe_topk}_minFDE"] = float(min_fde[rare_mask].mean().item())
    return result


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset_name = args.dataset_name or config.get("dataset_name", "111_days")
    if args.dataset_variant:
        config["dataset_variant"] = args.dataset_variant

    train_ds = build_dataset(config, checkpoint, dataset_name, args.train_split)
    test_ds = build_dataset(config, checkpoint, dataset_name, args.test_split)
    train_indices = deterministic_indices(len(train_ds), args.train_count, args.seed)
    test_indices = deterministic_indices(len(test_ds), args.test_count, args.seed + 1)

    model = build_model(config, checkpoint).to(device)
    load_checkpoint_state(model, checkpoint, config["dropout"])
    train_features, train_future, _ = collect_features(
        model,
        train_ds,
        train_indices,
        config,
        device,
        args.batch_size,
        desc="embed:train",
    )
    test_features, test_future, test_rare = collect_features(
        model,
        test_ds,
        test_indices,
        config,
        device,
        args.batch_size,
        desc="embed:test",
    )

    results = {
        "checkpoint": args.checkpoint,
        "dataset_name": dataset_name,
        "train_count": int(train_future.size(0)),
        "test_count": int(test_future.size(0)),
        "topk": int(args.topk),
        "features": {},
    }
    for name in train_features:
        results["features"][name] = retrieval_metrics(
            train_features[name],
            test_features[name],
            train_future,
            test_future,
            test_rare,
            topk=args.topk,
            device=device,
            query_chunk=args.query_chunk,
            bank_chunk=args.bank_chunk,
        )

    text = json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
