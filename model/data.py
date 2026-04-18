import hashlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_split_dir(project_root, dataset_variant, dataset_name, split):
    if dataset_variant != "social":
        raise ValueError(f"Unsupported dataset variant: {dataset_variant}")
    candidates = [
        os.path.join(project_root, "dataset", dataset_name, "processed_data", split),
        os.path.join(project_root, "dataset", dataset_name, dataset_name, "processed_data", split),
        os.path.join(project_root, "dataset", "social", dataset_name, "processed_data", split),
    ]
    for data_dir in candidates:
        if os.path.isdir(data_dir):
            return data_dir
    raise FileNotFoundError(
        f"Split directory not found for variant={dataset_variant}, dataset={dataset_name}, split={split}. "
        f"Checked: {candidates}"
    )


def read_txt_file(path, delim=" "):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = [float(token) for token in line.strip().split(delim) if token]
            if values:
                rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def _angle_wrap(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _build_rotation(yaw, pitch):
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )
    return ry @ rz


def _future_local_summary(obs_xyz, fut_xyz):
    origin = obs_xyz[-1]
    previous = obs_xyz[-2] if obs_xyz.shape[0] > 1 else obs_xyz[-1]
    direction = origin - previous
    yaw = math.atan2(float(direction[1]), float(direction[0]) + 1e-6)
    horizontal = math.sqrt(float(direction[0] ** 2 + direction[1] ** 2) + 1e-6)
    pitch = math.atan2(float(direction[2]), horizontal)
    rotation = _build_rotation(yaw, pitch)

    future_centered = fut_xyz - origin[None, :]
    future_local = np.einsum("ij,tj->ti", rotation, future_centered).astype(np.float32)
    future_delta = np.concatenate([future_local[:1], future_local[1:] - future_local[:-1]], axis=0)

    yaw_series = np.arctan2(future_delta[:, 1], future_delta[:, 0] + 1e-6)
    horizontal_speed = np.sqrt(future_delta[:, 0] ** 2 + future_delta[:, 1] ** 2 + 1e-6)
    pitch_series = np.arctan2(future_delta[:, 2], horizontal_speed)
    delta_yaw = _angle_wrap(float(yaw_series[-1] - yaw_series[0]))
    delta_pitch = float(pitch_series[-1] - pitch_series[0])

    summary = np.asarray(
        [
            future_local[-1, 0],
            future_local[-1, 1],
            future_local[-1, 2],
            delta_yaw,
            delta_pitch,
        ],
        dtype=np.float32,
    )
    return future_local, summary


def _pairwise_last_obs_distance(track_a, track_b):
    delta = track_a[-1] - track_b[-1]
    return float(np.linalg.norm(delta))


def _cache_path(data_dir, obs_len, pred_len, max_agents):
    file_names = sorted(name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name)))
    key = f"protobasis_v1|{data_dir}|{file_names}|{obs_len}|{pred_len}|{max_agents}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(data_dir, ".cache")
    return os.path.join(cache_dir, f"protobasis_dataset_{digest}.pt")


def _artifact_path(data_dir, n_proto, basis_dim):
    cache_dir = os.path.join(data_dir, ".cache")
    return os.path.join(cache_dir, f"protobasis_artifact_p{n_proto}_b{basis_dim}.pt")


def _run_kmeans(data, n_clusters, random_state=3407, iters=50):
    rng = np.random.default_rng(random_state)
    if data.shape[0] < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} exceeds sample count={data.shape[0]}")

    initial_ids = rng.choice(data.shape[0], size=n_clusters, replace=False)
    centers = data[initial_ids].copy()

    for _ in range(iters):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                new_centers[cluster_id] = data[mask].mean(axis=0)
            else:
                new_centers[cluster_id] = data[rng.integers(0, data.shape[0])]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    final_distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = final_distances.argmin(axis=1)
    return centers.astype(np.float32), labels.astype(np.int64)


def _fit_basis_bank(future_local_bank, pred_len, basis_dim, max_samples=60000, random_state=3407):
    if len(future_local_bank) > max_samples:
        rng = np.random.default_rng(random_state)
        sample_ids = rng.choice(len(future_local_bank), size=max_samples, replace=False)
        future_local_bank = [future_local_bank[idx] for idx in sample_ids.tolist()]
    futures = np.stack(future_local_bank, axis=0).astype(np.float32)
    alpha = np.linspace(0.0, 1.0, num=pred_len, dtype=np.float32)[None, :, None]
    endpoints = futures[:, -1:, :]
    anchors = alpha * endpoints
    residual = futures - anchors
    flat_residual = residual.reshape(residual.shape[0], -1)
    _, _, vt = np.linalg.svd(flat_residual, full_matrices=False)
    rank = min(basis_dim, vt.shape[0])
    basis_flat = vt[:rank]
    if rank < basis_dim:
        pad = np.zeros((basis_dim - rank, flat_residual.shape[1]), dtype=np.float32)
        basis_flat = np.concatenate([basis_flat, pad], axis=0)
    return basis_flat.reshape(basis_dim, pred_len, 3).astype(np.float32)


@dataclass
class ProtoBasisArtifact:
    summary_5d: np.ndarray
    frequency: np.ndarray
    rare_ids: np.ndarray
    basis_bank: np.ndarray
    n_proto: int
    basis_dim: int
    rare_threshold: float

    def to_dict(self):
        return {
            "summary_5d": self.summary_5d,
            "frequency": self.frequency,
            "rare_ids": self.rare_ids,
            "basis_bank": self.basis_bank,
            "n_proto": self.n_proto,
            "basis_dim": self.basis_dim,
            "rare_threshold": self.rare_threshold,
        }


class ProtoBasisSceneDataset(Dataset):
    CACHE_VERSION = 3

    def __init__(
        self,
        data_dir,
        split_name,
        obs_len=40,
        pred_len=120,
        max_agents=7,
        delim=" ",
        model_artifact: Optional[Dict] = None,
        n_proto=64,
        basis_dim=16,
        rare_threshold=0.02,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_name = split_name
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_agents = max_agents
        self.delim = delim
        self.sequence_len = obs_len + pred_len
        self.n_proto = n_proto
        self.basis_dim = basis_dim
        self.rare_threshold = rare_threshold

        self.file_records: List[Dict] = []
        self.sample_index: List[Dict] = []

        cache_path = _cache_path(data_dir, obs_len, pred_len, max_agents)
        if os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, weights_only=False)
            except Exception:
                os.remove(cache_path)
                self._build_and_cache(cache_path)
            else:
                if cached.get("cache_version") == self.CACHE_VERSION:
                    self.file_records = [self._load_file_record(path) for path in cached["source_paths"]]
                    self.sample_index = cached["sample_index"]
                else:
                    self._build_and_cache(cache_path)
        else:
            self._build_and_cache(cache_path)

        if not self.sample_index:
            raise RuntimeError(f"No valid ProtoBasis-Flight samples built from {data_dir}")

        if model_artifact is None:
            if split_name != "train":
                raise ValueError("Model artifact must be provided for non-train splits.")
            model_artifact = self._fit_model_artifact()
            artifact_path = _artifact_path(data_dir, n_proto, basis_dim)
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            torch.save(model_artifact, artifact_path)

        self.model_artifact = model_artifact
        self.prototype_summary_5d = np.asarray(model_artifact["summary_5d"], dtype=np.float32)
        self.prototype_frequency = np.asarray(model_artifact["frequency"], dtype=np.float32)
        self.rare_proto_ids = np.asarray(model_artifact["rare_ids"], dtype=np.int64)
        self.basis_bank = np.asarray(model_artifact["basis_bank"], dtype=np.float32)
        self.rare_proto_set = set(int(idx) for idx in self.rare_proto_ids.tolist())
        self._assign_prototypes()

    def _build_and_cache(self, cache_path):
        file_names = sorted(
            name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))
        )
        for file_name in file_names:
            path = os.path.join(self.data_dir, file_name)
            data = read_txt_file(path, self.delim)
            file_record = self._build_file_record(data, path)
            if file_record is None:
                continue
            file_index = len(self.file_records)
            self.file_records.append(file_record)
            self.sample_index.extend(self._build_target_windows(file_record, file_index))

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            torch.save(
                {
                    "cache_version": self.CACHE_VERSION,
                    "source_paths": [record["source_path"] for record in self.file_records],
                    "sample_index": self.sample_index,
                },
                cache_path,
            )
        except MemoryError:
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    def _load_file_record(self, source_path):
        data = read_txt_file(source_path, self.delim)
        file_record = self._build_file_record(data, source_path)
        if file_record is None:
            raise RuntimeError(f"Cached source path no longer yields a valid file record: {source_path}")
        return file_record

    def _build_file_record(self, data, source_path):
        if data.size == 0:
            return None
        frames = np.unique(data[:, 0]).tolist()
        if len(frames) < self.sequence_len:
            return None
        return {
            "data": data,
            "frames": frames,
            "frame_map": {frame: data[data[:, 0] == frame] for frame in frames},
            "source_path": source_path,
        }

    def _build_target_windows(self, file_record, file_index):
        frames = file_record["frames"]
        frame_map = file_record["frame_map"]
        windows = []
        max_start = len(frames) - self.sequence_len + 1

        for start_idx in range(max_start):
            window_frames = frames[start_idx : start_idx + self.sequence_len]
            window_segments = [frame_map[frame] for frame in window_frames]
            window_data = np.concatenate(window_segments, axis=0)
            agent_ids = np.unique(window_data[:, 1]).tolist()

            complete_tracks = {}
            for agent_id in agent_ids:
                agent_rows = window_data[window_data[:, 1] == agent_id]
                if agent_rows.shape[0] != self.sequence_len:
                    continue
                track_xyz = agent_rows[:, 2:5].astype(np.float32)
                complete_tracks[int(agent_id)] = track_xyz

            valid_ids = sorted(complete_tracks.keys())
            if not valid_ids:
                continue

            for target_id in valid_ids:
                target_track = complete_tracks[target_id]
                obs_xyz = target_track[: self.obs_len]
                fut_xyz = target_track[self.obs_len :]
                future_local, proto_summary_5d = _future_local_summary(obs_xyz, fut_xyz)

                neighbor_ids = [agent_id for agent_id in valid_ids if agent_id != target_id]
                neighbor_ids.sort(
                    key=lambda agent_id: _pairwise_last_obs_distance(
                        complete_tracks[target_id][: self.obs_len],
                        complete_tracks[agent_id][: self.obs_len],
                    )
                )
                selected_ids = [target_id] + neighbor_ids[: self.max_agents - 1]
                windows.append(
                    {
                        "file_index": file_index,
                        "start_idx": start_idx,
                        "target_id": target_id,
                        "agent_ids": selected_ids,
                        "agent_count": len(selected_ids),
                        "proto_summary_5d": proto_summary_5d,
                        "future_local": future_local,
                    }
                )
        return windows

    def _fit_model_artifact(self):
        summaries = np.stack([sample["proto_summary_5d"] for sample in self.sample_index], axis=0)
        centers, labels = _run_kmeans(summaries, n_clusters=self.n_proto, random_state=3407)
        frequency = np.bincount(labels, minlength=self.n_proto).astype(np.float32)
        frequency = frequency / max(float(frequency.sum()), 1.0)
        rare_ids = np.nonzero(frequency < self.rare_threshold)[0].astype(np.int64)
        basis_bank = _fit_basis_bank(
            [sample["future_local"] for sample in self.sample_index],
            pred_len=self.pred_len,
            basis_dim=self.basis_dim,
        )
        return ProtoBasisArtifact(
            summary_5d=centers,
            frequency=frequency,
            rare_ids=rare_ids,
            basis_bank=basis_bank,
            n_proto=self.n_proto,
            basis_dim=self.basis_dim,
            rare_threshold=self.rare_threshold,
        ).to_dict()

    def _assign_prototypes(self):
        proto_summary = self.prototype_summary_5d
        proto_endpoints = proto_summary[:, :3]
        for sample in self.sample_index:
            summary = sample["proto_summary_5d"]
            distances = ((proto_summary - summary[None, :]) ** 2).sum(axis=1)
            proto_id = int(np.argmin(distances))
            endpoint_residual = summary[:3] - proto_endpoints[proto_id]
            sample["gt_proto_id"] = proto_id
            sample["gt_proto_residual"] = endpoint_residual.astype(np.float32)
            sample["is_rare"] = proto_id in self.rare_proto_set

    def export_model_artifact(self):
        return {
            "summary_5d": self.prototype_summary_5d.copy(),
            "frequency": self.prototype_frequency.copy(),
            "rare_ids": self.rare_proto_ids.copy(),
            "basis_bank": self.basis_bank.copy(),
            "n_proto": self.n_proto,
            "basis_dim": self.basis_dim,
            "rare_threshold": self.rare_threshold,
        }

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index):
        sample_meta = self.sample_index[index]
        file_record = self.file_records[sample_meta["file_index"]]
        frames = file_record["frames"]
        frame_map = file_record["frame_map"]
        start_idx = sample_meta["start_idx"]
        window_frames = frames[start_idx : start_idx + self.sequence_len]
        window_segments = [frame_map[frame] for frame in window_frames]
        window_data = np.concatenate(window_segments, axis=0)

        obs_agents = []
        for agent_id in sample_meta["agent_ids"]:
            agent_rows = window_data[window_data[:, 1] == agent_id]
            track_xyz = agent_rows[:, 2:5].astype(np.float32)
            obs_agents.append(track_xyz[: self.obs_len])

        target_rows = window_data[window_data[:, 1] == sample_meta["target_id"]]
        target_xyz = target_rows[:, 2:5].astype(np.float32)
        fut_target = target_xyz[self.obs_len :]

        return {
            "obs_xyz": torch.tensor(np.stack(obs_agents, axis=0), dtype=torch.float32),
            "fut_xyz": torch.tensor(fut_target, dtype=torch.float32),
            "fut_local": torch.tensor(sample_meta["future_local"], dtype=torch.float32),
            "obs_mask": torch.ones(len(sample_meta["agent_ids"]), dtype=torch.bool),
            "gt_proto_id": torch.tensor(sample_meta["gt_proto_id"], dtype=torch.long),
            "gt_proto_residual": torch.tensor(sample_meta["gt_proto_residual"], dtype=torch.float32),
            "is_rare": torch.tensor(sample_meta["is_rare"], dtype=torch.bool),
            "source_path": file_record["source_path"],
            "scene_id": f"{os.path.basename(file_record['source_path'])}:{start_idx}:{sample_meta['target_id']}",
            "split_name": self.split_name,
        }


def proto_basis_collate(batch, max_agents=7):
    batch_size = len(batch)
    obs_len = batch[0]["obs_xyz"].shape[1]
    pred_len = batch[0]["fut_xyz"].shape[0]

    obs_xyz = torch.zeros(batch_size, max_agents, obs_len, 3, dtype=torch.float32)
    obs_mask = torch.zeros(batch_size, max_agents, dtype=torch.bool)
    fut_xyz = torch.zeros(batch_size, pred_len, 3, dtype=torch.float32)
    fut_local = torch.zeros(batch_size, pred_len, 3, dtype=torch.float32)
    gt_proto_id = torch.zeros(batch_size, dtype=torch.long)
    gt_proto_residual = torch.zeros(batch_size, 3, dtype=torch.float32)
    is_rare = torch.zeros(batch_size, dtype=torch.bool)
    source_path = []
    scene_id = []
    split_name = []

    for batch_index, item in enumerate(batch):
        agent_count = min(item["obs_xyz"].shape[0], max_agents)
        obs_xyz[batch_index, :agent_count] = item["obs_xyz"][:agent_count]
        obs_mask[batch_index, :agent_count] = item["obs_mask"][:agent_count]
        fut_xyz[batch_index] = item["fut_xyz"]
        fut_local[batch_index] = item["fut_local"]
        gt_proto_id[batch_index] = item["gt_proto_id"]
        gt_proto_residual[batch_index] = item["gt_proto_residual"]
        is_rare[batch_index] = item["is_rare"]
        source_path.append(item["source_path"])
        scene_id.append(item["scene_id"])
        split_name.append(item["split_name"])

    return {
        "obs_xyz": obs_xyz,
        "obs_mask": obs_mask,
        "fut_xyz": fut_xyz,
        "fut_local": fut_local,
        "gt_proto_id": gt_proto_id,
        "gt_proto_residual": gt_proto_residual,
        "is_rare": is_rare,
        "source_path": source_path,
        "scene_id": scene_id,
        "split_name": split_name,
    }
