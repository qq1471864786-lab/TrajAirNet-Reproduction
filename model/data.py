import hashlib
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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


def _cache_path(data_dir, obs_len, pred_len, max_agents, obs_stride=1, pred_stride=1):
    file_names = sorted(name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name)))
    key = f"protobasis_v3|{data_dir}|{file_names}|{obs_len}|{pred_len}|{max_agents}|{obs_stride}|{pred_stride}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(data_dir, ".cache")
    return os.path.join(cache_dir, f"protobasis_dataset_{digest}.pt")


def _artifact_path(
    data_dir,
    n_proto,
    basis_dim,
    local_basis_dim,
    micro_per_proto,
    artifact_max_samples,
    obs_len,
    pred_len,
    obs_stride=1,
    pred_stride=1,
    rare_threshold=0.02,
):
    cache_dir = os.path.join(data_dir, ".cache")
    sample_suffix = f"_am{artifact_max_samples}" if int(artifact_max_samples) > 0 else ""
    return os.path.join(
        cache_dir,
        (
            f"protobasis_artifact_v5_o{obs_len}_p{pred_len}_os{obs_stride}_ps{pred_stride}_"
            f"n{n_proto}_b{basis_dim}_lb{local_basis_dim}_mc{micro_per_proto}"
            f"{sample_suffix}_r{rare_threshold:.4f}.pt"
        ),
    )


def _run_progress_step(description, fn):
    start = time.perf_counter()
    progress = tqdm(
        total=1,
        desc=description,
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )
    try:
        result = fn()
    except Exception:
        progress.close()
        raise
    elapsed = time.perf_counter() - start
    progress.update(1)
    progress.set_postfix_str(f"{elapsed:.1f}s", refresh=False)
    progress.close()
    print(f"[Cache] {description} done in {elapsed:.1f}s", flush=True)
    return result


def _run_kmeans(data, n_clusters, random_state=3407, iters=50, show_progress=False, progress_desc="kmeans"):
    rng = np.random.default_rng(random_state)
    if data.shape[0] < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} exceeds sample count={data.shape[0]}")

    initial_ids = rng.choice(data.shape[0], size=n_clusters, replace=False)
    centers = data[initial_ids].copy()

    iteration_iter = range(iters)
    progress = None
    if show_progress:
        progress = tqdm(
            iteration_iter,
            desc=progress_desc,
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        iteration_iter = progress

    for iteration in iteration_iter:
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
        if progress is not None:
            progress.set_postfix(iter=iteration + 1, refresh=False)

    if progress is not None:
        progress.close()

    final_distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels = final_distances.argmin(axis=1)
    return centers.astype(np.float32), labels.astype(np.int64)


def _fit_basis_bank(
    future_local_bank,
    pred_len,
    basis_dim,
    max_samples=60000,
    random_state=3407,
    progress_desc=None,
):
    if basis_dim <= 0:
        return np.zeros((0, pred_len, 3), dtype=np.float32)
    if isinstance(future_local_bank, np.ndarray):
        futures = future_local_bank.astype(np.float32, copy=False)
        if futures.shape[0] > max_samples:
            rng = np.random.default_rng(random_state)
            sample_ids = rng.choice(futures.shape[0], size=max_samples, replace=False)
            futures = futures[sample_ids]
    else:
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

    def _compute_basis():
        _, _, vt = np.linalg.svd(flat_residual, full_matrices=False)
        rank = min(basis_dim, vt.shape[0])
        basis_flat = vt[:rank]
        if rank < basis_dim:
            pad = np.zeros((basis_dim - rank, flat_residual.shape[1]), dtype=np.float32)
            basis_flat = np.concatenate([basis_flat, pad], axis=0)
        return basis_flat.reshape(basis_dim, pred_len, 3).astype(np.float32)

    if progress_desc:
        return _run_progress_step(progress_desc, _compute_basis)

    basis = _compute_basis()
    return basis


def _solve_basis_coefficients(future_local, basis_bank, pred_len):
    if basis_bank.shape[0] <= 0:
        return np.zeros((future_local.shape[0], 0), dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, num=pred_len, dtype=np.float32)[None, :, None]
    anchors = alpha * future_local[:, -1:, :]
    residual = (future_local - anchors).reshape(future_local.shape[0], -1).T
    basis_matrix = basis_bank.reshape(basis_bank.shape[0], -1).T
    coeff, *_ = np.linalg.lstsq(basis_matrix, residual, rcond=None)
    return coeff.T.astype(np.float32)


def _fit_micro_coeff_anchors(
    future_local_bank,
    labels,
    basis_bank,
    pred_len,
    n_proto,
    micro_per_proto,
    max_samples_per_proto=5000,
    random_state=3407,
):
    basis_dim = int(basis_bank.shape[0])
    if micro_per_proto <= 0 or basis_dim <= 0:
        return np.zeros((n_proto, max(int(micro_per_proto), 0), basis_dim), dtype=np.float32)

    coeff_anchors = np.zeros((n_proto, micro_per_proto, basis_dim), dtype=np.float32)
    rng = np.random.default_rng(random_state)
    progress = tqdm(
        range(n_proto),
        desc="artifact:micro_anchors",
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
    )
    for proto_id in progress:
        proto_indices = np.flatnonzero(labels == proto_id)
        if proto_indices.size == 0:
            continue
        if proto_indices.size > max_samples_per_proto:
            proto_indices = rng.choice(proto_indices, size=max_samples_per_proto, replace=False)
        proto_future = future_local_bank[proto_indices].astype(np.float32, copy=False)
        coeff_samples = _solve_basis_coefficients(
            proto_future,
            basis_bank,
            pred_len,
        )
        if coeff_samples.shape[0] >= micro_per_proto:
            centers, _ = _run_kmeans(
                coeff_samples,
                n_clusters=micro_per_proto,
                random_state=random_state + proto_id,
                iters=30,
            )
            coeff_anchors[proto_id] = centers
        else:
            coeff_anchors[proto_id, : coeff_samples.shape[0]] = coeff_samples
            coeff_anchors[proto_id, coeff_samples.shape[0] :] = coeff_samples[-1]
        progress.set_postfix(proto=proto_id + 1, refresh=False)
    progress.close()
    return coeff_anchors.astype(np.float32)


def _empty_sample_store(max_agents, pred_len):
    return {
        "file_index": np.zeros((0,), dtype=np.int32),
        "start_idx": np.zeros((0,), dtype=np.int32),
        "target_agent_idx": np.zeros((0,), dtype=np.int32),
        "agent_indices": np.full((0, max_agents), -1, dtype=np.int32),
        "agent_count": np.zeros((0,), dtype=np.int16),
        "proto_summary_5d": np.zeros((0, 5), dtype=np.float32),
        "future_local": np.zeros((0, pred_len, 3), dtype=np.float32),
        "gt_proto_id": np.zeros((0,), dtype=np.int64),
        "gt_proto_residual": np.zeros((0, 3), dtype=np.float32),
        "is_rare": np.zeros((0,), dtype=np.bool_),
    }


def _pack_sample_rows(sample_rows, max_agents, pred_len):
    if not sample_rows:
        return _empty_sample_store(max_agents, pred_len)

    sample_count = len(sample_rows)
    store = {
        "file_index": np.zeros((sample_count,), dtype=np.int32),
        "start_idx": np.zeros((sample_count,), dtype=np.int32),
        "target_agent_idx": np.zeros((sample_count,), dtype=np.int32),
        "agent_indices": np.full((sample_count, max_agents), -1, dtype=np.int32),
        "agent_count": np.zeros((sample_count,), dtype=np.int16),
        "proto_summary_5d": np.zeros((sample_count, 5), dtype=np.float32),
        "future_local": np.zeros((sample_count, pred_len, 3), dtype=np.float32),
    }

    for index, row in enumerate(sample_rows):
        store["file_index"][index] = row["file_index"]
        store["start_idx"][index] = row["start_idx"]
        store["target_agent_idx"][index] = row["target_agent_idx"]
        store["agent_count"][index] = row["agent_count"]
        store["agent_indices"][index, : row["agent_count"]] = row["agent_indices"]
        store["proto_summary_5d"][index] = row["proto_summary_5d"]
        store["future_local"][index] = row["future_local"]

    return store


@dataclass
class ProtoBasisArtifact:
    summary_5d: np.ndarray
    frequency: np.ndarray
    rare_ids: np.ndarray
    basis_bank: np.ndarray
    prototype_mean_path: np.ndarray
    local_basis_bank: np.ndarray
    micro_coeff_anchors: np.ndarray
    n_proto: int
    basis_dim: int
    local_basis_dim: int
    micro_per_proto: int
    rare_threshold: float
    obs_len: int
    pred_len: int
    obs_stride: int
    pred_stride: int

    def to_dict(self):
        return {
            "summary_5d": self.summary_5d,
            "frequency": self.frequency,
            "rare_ids": self.rare_ids,
            "basis_bank": self.basis_bank,
            "prototype_mean_path": self.prototype_mean_path,
            "local_basis_bank": self.local_basis_bank,
            "micro_coeff_anchors": self.micro_coeff_anchors,
            "n_proto": self.n_proto,
            "basis_dim": self.basis_dim,
            "local_basis_dim": self.local_basis_dim,
            "micro_per_proto": self.micro_per_proto,
            "rare_threshold": self.rare_threshold,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "obs_stride": self.obs_stride,
            "pred_stride": self.pred_stride,
        }


class ProtoBasisSceneDataset(Dataset):
    CACHE_VERSION = 5

    def __init__(
        self,
        data_dir,
        split_name,
        obs_len=40,
        pred_len=120,
        obs_stride=1,
        pred_stride=1,
        max_agents=7,
        delim=" ",
        model_artifact: Optional[Dict] = None,
        n_proto=64,
        basis_dim=16,
        local_basis_dim=0,
        micro_per_proto=0,
        artifact_max_samples=0,
        rare_threshold=0.02,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split_name = split_name
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.obs_stride = obs_stride
        self.pred_stride = pred_stride
        self.max_agents = max_agents
        self.delim = delim
        self.sequence_span = (obs_len - 1) * obs_stride + pred_len * pred_stride + 1
        self.n_proto = n_proto
        self.basis_dim = basis_dim
        self.local_basis_dim = local_basis_dim
        self.micro_per_proto = max(int(micro_per_proto), 0)
        self.artifact_max_samples = max(int(artifact_max_samples), 0)
        self.rare_threshold = rare_threshold
        self.obs_index_offsets = np.arange(obs_len, dtype=np.int32) * obs_stride
        self.pred_index_offsets = (
            (obs_len - 1) * obs_stride + (np.arange(pred_len, dtype=np.int32) + 1) * pred_stride
        )
        self.window_index_offsets = np.concatenate([self.obs_index_offsets, self.pred_index_offsets], axis=0)

        self.file_records: List[Dict] = []
        self.samples = _empty_sample_store(max_agents, pred_len)

        cache_path = _cache_path(data_dir, obs_len, pred_len, max_agents, obs_stride=obs_stride, pred_stride=pred_stride)
        if os.path.isfile(cache_path):
            print(f"[Cache] loading {split_name} split: {cache_path}", flush=True)
            try:
                cached = _run_progress_step(
                    f"load:{split_name}:split",
                    lambda: torch.load(cache_path, weights_only=False),
                )
            except Exception:
                os.remove(cache_path)
                self._build_and_cache(cache_path)
            else:
                if cached.get("cache_version") == self.CACHE_VERSION and "file_records" in cached and "samples" in cached:
                    self.file_records = cached["file_records"]
                    self.samples = cached["samples"]
                else:
                    self._build_and_cache(cache_path)
        else:
            self._build_and_cache(cache_path)

        if len(self) == 0:
            raise RuntimeError(f"No valid ProtoBasis-Net samples built from {data_dir}")
        print(
            f"[Cache] ready {split_name}: files={len(self.file_records)} samples={len(self)}",
            flush=True,
        )

        if model_artifact is None:
            if split_name != "train":
                raise ValueError("Model artifact must be provided for non-train splits.")
            artifact_path = _artifact_path(
                data_dir,
                n_proto,
                basis_dim,
                local_basis_dim,
                self.micro_per_proto,
                self.artifact_max_samples,
                obs_len=obs_len,
                pred_len=pred_len,
                obs_stride=obs_stride,
                pred_stride=pred_stride,
                rare_threshold=rare_threshold,
            )
            if os.path.isfile(artifact_path):
                print(f"[Cache] loading train artifact: {artifact_path}", flush=True)
                model_artifact = _run_progress_step(
                    "load:train:artifact",
                    lambda: torch.load(artifact_path, weights_only=False),
                )
            else:
                print(f"[Cache] fitting train artifact from {len(self)} cached samples...", flush=True)
                model_artifact = self._fit_model_artifact()
                os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
                _run_progress_step(
                    "save:train:artifact",
                    lambda: torch.save(model_artifact, artifact_path),
                )
                print(f"[Cache] saved train artifact: {artifact_path}", flush=True)
        self.model_artifact = model_artifact
        self.basis_bank = np.asarray(model_artifact["basis_bank"], dtype=np.float32)
        self.basis_dim = int(model_artifact.get("basis_dim", self.basis_dim))
        self.prototype_summary_5d = np.asarray(model_artifact["summary_5d"], dtype=np.float32)
        self.prototype_frequency = np.asarray(model_artifact["frequency"], dtype=np.float32)
        self.rare_proto_ids = np.asarray(model_artifact["rare_ids"], dtype=np.int64)
        self.local_basis_dim = int(model_artifact.get("local_basis_dim", self.local_basis_dim))
        self.micro_per_proto = int(model_artifact.get("micro_per_proto", self.micro_per_proto))
        default_mean_path = np.zeros((self.n_proto, self.pred_len, 3), dtype=np.float32)
        default_local_basis = np.zeros((self.n_proto, self.local_basis_dim, self.pred_len, 3), dtype=np.float32)
        default_micro_coeff = np.zeros((self.n_proto, self.micro_per_proto, self.basis_dim), dtype=np.float32)
        self.prototype_mean_path = np.asarray(model_artifact.get("prototype_mean_path", default_mean_path), dtype=np.float32)
        self.local_basis_bank = np.asarray(model_artifact.get("local_basis_bank", default_local_basis), dtype=np.float32)
        self.micro_coeff_anchors = np.asarray(
            model_artifact.get("micro_coeff_anchors", default_micro_coeff),
            dtype=np.float32,
        )
        self.gt_basis_coeff = _solve_basis_coefficients(
            self.samples["future_local"].astype(np.float32, copy=False),
            self.basis_bank,
            self.pred_len,
        )
        self._assign_prototypes()

    def _build_and_cache(self, cache_path):
        file_names = sorted(
            name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))
        )
        print(
            f"[Cache] building {self.split_name} split cache for {os.path.basename(self.data_dir)}...",
            flush=True,
        )
        sample_rows = []
        progress = tqdm(
            file_names,
            desc=f"cache:{self.split_name}",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        for file_name in progress:
            path = os.path.join(self.data_dir, file_name)
            data = read_txt_file(path, self.delim)
            file_record = self._build_file_record(data, path)
            if file_record is None:
                continue
            file_index = len(self.file_records)
            sample_rows.extend(self._build_target_windows(file_record, file_index))
            file_record.pop("present", None)
            file_record.pop("frame_ids", None)
            self.file_records.append(file_record)
            progress.set_postfix(files=len(self.file_records), samples=len(sample_rows), refresh=False)
        progress.close()
        self.samples = _pack_sample_rows(sample_rows, self.max_agents, self.pred_len)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            _run_progress_step(
                f"save:{self.split_name}:split",
                lambda: torch.save(
                    {
                        "cache_version": self.CACHE_VERSION,
                        "file_records": self.file_records,
                        "samples": self.samples,
                    },
                    cache_path,
                ),
            )
            print(f"[Cache] saved {self.split_name} split cache: {cache_path}", flush=True)
        except MemoryError:
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    def _build_file_record(self, data, source_path):
        if data.size == 0:
            return None
        frames = np.unique(data[:, 0]).astype(np.float32)
        if len(frames) < self.sequence_span:
            return None
        agent_ids = np.unique(data[:, 1]).astype(np.int64)
        frame_indices = np.searchsorted(frames, data[:, 0])
        agent_indices = np.searchsorted(agent_ids, data[:, 1])
        positions = np.zeros((len(agent_ids), len(frames), 3), dtype=np.float32)
        present = np.zeros((len(agent_ids), len(frames)), dtype=np.bool_)
        positions[agent_indices, frame_indices] = data[:, 2:5].astype(np.float32)
        present[agent_indices, frame_indices] = True
        return {
            "positions": positions,
            "present": present,
            "frame_ids": frames,
            "agent_ids": agent_ids,
            "source_path": source_path,
        }

    def _build_target_windows(self, file_record, file_index):
        frame_ids = file_record["frame_ids"]
        positions = file_record["positions"]
        present = file_record["present"]
        windows = []
        if positions.shape[0] == 0:
            return windows
        max_start = len(frame_ids) - self.sequence_span + 1
        if max_start <= 0:
            return windows

        for start_idx in range(max_start):
            selected_index_ids = start_idx + self.window_index_offsets
            valid_agent_indices = np.flatnonzero(present[:, selected_index_ids].all(axis=1))
            if valid_agent_indices.size == 0:
                continue
            window_tracks = positions[valid_agent_indices][:, selected_index_ids]
            last_obs = window_tracks[:, self.obs_len - 1]
            for valid_offset, target_agent_idx in enumerate(valid_agent_indices.tolist()):
                target_track = window_tracks[valid_offset]
                obs_xyz = target_track[: self.obs_len]
                fut_xyz = target_track[self.obs_len :]
                future_local, proto_summary_5d = _future_local_summary(obs_xyz, fut_xyz)
                distances = np.linalg.norm(last_obs - last_obs[valid_offset][None, :], axis=1)
                nearest_order = np.argsort(distances, kind="stable")[: self.max_agents]
                selected_indices = valid_agent_indices[nearest_order]
                windows.append(
                    {
                        "file_index": file_index,
                        "start_idx": start_idx,
                        "target_agent_idx": target_agent_idx,
                        "agent_indices": selected_indices.astype(np.int32),
                        "agent_count": int(selected_indices.shape[0]),
                        "proto_summary_5d": proto_summary_5d,
                        "future_local": future_local,
                    }
                )
        return windows

    def _fit_model_artifact(self):
        summaries = self.samples["proto_summary_5d"]
        future_bank = self.samples["future_local"]
        total_samples = int(summaries.shape[0])
        fit_samples = total_samples
        if self.artifact_max_samples > 0 and summaries.shape[0] > self.artifact_max_samples:
            rng = np.random.default_rng(3407)
            fit_indices = rng.choice(summaries.shape[0], size=self.artifact_max_samples, replace=False)
            fit_indices.sort()
            summaries = summaries[fit_indices]
            future_bank = future_bank[fit_indices]
            fit_samples = int(fit_indices.shape[0])
        print(
            f"[Cache] artifact fit samples: using {fit_samples} / {total_samples} "
            f"(micro_per_proto={self.micro_per_proto})"
        )
        centers, labels = _run_kmeans(
            summaries,
            n_clusters=self.n_proto,
            random_state=3407,
            show_progress=True,
            progress_desc="artifact:kmeans",
        )
        frequency = np.bincount(labels, minlength=self.n_proto).astype(np.float32)
        frequency = frequency / max(float(frequency.sum()), 1.0)
        rare_ids = np.nonzero(frequency < self.rare_threshold)[0].astype(np.int64)
        basis_bank = _fit_basis_bank(
            future_bank,
            pred_len=self.pred_len,
            basis_dim=self.basis_dim,
            progress_desc="artifact:global_basis",
        )
        prototype_mean_path = np.zeros((self.n_proto, self.pred_len, 3), dtype=np.float32)
        local_basis_bank = np.zeros((self.n_proto, self.local_basis_dim, self.pred_len, 3), dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, num=self.pred_len, dtype=np.float32)[:, None]
        progress = tqdm(
            range(self.n_proto),
            desc="artifact:prototype_stats",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        for proto_id in progress:
            mask = labels == proto_id
            proto_samples = future_bank[mask]
            if proto_samples.shape[0] == 0:
                prototype_mean_path[proto_id] = alpha * centers[proto_id, :3][None, :]
                continue
            prototype_mean_path[proto_id] = proto_samples.mean(axis=0)
            if self.local_basis_dim > 0:
                local_basis_bank[proto_id] = _fit_basis_bank(
                    proto_samples - prototype_mean_path[proto_id][None, :],
                    pred_len=self.pred_len,
                    basis_dim=self.local_basis_dim,
                    max_samples=20000,
                    random_state=3407 + proto_id,
                )
            progress.set_postfix(proto=proto_id + 1, refresh=False)
        progress.close()
        micro_coeff_anchors = _fit_micro_coeff_anchors(
            future_bank,
            labels,
            basis_bank,
            pred_len=self.pred_len,
            n_proto=self.n_proto,
            micro_per_proto=self.micro_per_proto,
        )
        return ProtoBasisArtifact(
            summary_5d=centers,
            frequency=frequency,
            rare_ids=rare_ids,
            basis_bank=basis_bank,
            prototype_mean_path=prototype_mean_path,
            local_basis_bank=local_basis_bank,
            micro_coeff_anchors=micro_coeff_anchors,
            n_proto=self.n_proto,
            basis_dim=self.basis_dim,
            local_basis_dim=self.local_basis_dim,
            micro_per_proto=self.micro_per_proto,
            rare_threshold=self.rare_threshold,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            obs_stride=self.obs_stride,
            pred_stride=self.pred_stride,
        ).to_dict()

    def _assign_prototypes(self):
        proto_summary = self.prototype_summary_5d
        proto_endpoints = proto_summary[:, :3]
        sample_count = len(self)
        gt_proto_id = np.zeros((sample_count,), dtype=np.int64)
        gt_proto_residual = np.zeros((sample_count, 3), dtype=np.float32)
        chunk_size = 8192
        progress = tqdm(
            range(0, sample_count, chunk_size),
            desc=f"cache:{self.split_name}:assign_proto",
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        for start in progress:
            end = min(start + chunk_size, sample_count)
            summary = self.samples["proto_summary_5d"][start:end]
            distances = ((summary[:, None, :] - proto_summary[None, :, :]) ** 2).sum(axis=2)
            chunk_ids = np.argmin(distances, axis=1).astype(np.int64)
            gt_proto_id[start:end] = chunk_ids
            gt_proto_residual[start:end] = summary[:, :3] - proto_endpoints[chunk_ids]
            progress.set_postfix(samples=end, refresh=False)
        progress.close()
        self.samples["gt_proto_id"] = gt_proto_id
        self.samples["gt_proto_residual"] = gt_proto_residual
        self.samples["is_rare"] = np.isin(gt_proto_id, self.rare_proto_ids)

    def export_model_artifact(self):
        return {
            "summary_5d": self.prototype_summary_5d.copy(),
            "frequency": self.prototype_frequency.copy(),
            "rare_ids": self.rare_proto_ids.copy(),
            "basis_bank": self.basis_bank.copy(),
            "prototype_mean_path": self.prototype_mean_path.copy(),
            "local_basis_bank": self.local_basis_bank.copy(),
            "micro_coeff_anchors": self.micro_coeff_anchors.copy(),
            "n_proto": self.n_proto,
            "basis_dim": self.basis_dim,
            "local_basis_dim": self.local_basis_dim,
            "micro_per_proto": self.micro_per_proto,
            "rare_threshold": self.rare_threshold,
            "obs_len": self.obs_len,
            "pred_len": self.pred_len,
            "obs_stride": self.obs_stride,
            "pred_stride": self.pred_stride,
        }

    def sample_weights(self, rare_weight):
        is_rare = self.samples["is_rare"].astype(np.float64, copy=False)
        return np.where(is_rare > 0.0, float(rare_weight), 1.0).astype(np.float64)

    def __len__(self):
        return int(self.samples["file_index"].shape[0])

    def __getitem__(self, index):
        file_index = int(self.samples["file_index"][index])
        file_record = self.file_records[file_index]
        start_idx = int(self.samples["start_idx"][index])
        agent_count = int(self.samples["agent_count"][index])
        target_agent_idx = int(self.samples["target_agent_idx"][index])
        obs_index_ids = start_idx + self.obs_index_offsets
        pred_index_ids = start_idx + self.pred_index_offsets
        agent_indices = self.samples["agent_indices"][index, :agent_count]
        positions = file_record["positions"]
        obs_agents = positions[agent_indices][:, obs_index_ids]
        fut_target = positions[target_agent_idx, pred_index_ids]
        target_agent_id = int(file_record["agent_ids"][target_agent_idx])

        return {
            "obs_xyz": torch.tensor(obs_agents, dtype=torch.float32),
            "fut_xyz": torch.tensor(fut_target, dtype=torch.float32),
            "fut_local": torch.tensor(self.samples["future_local"][index], dtype=torch.float32),
            "gt_basis_coeff": torch.tensor(self.gt_basis_coeff[index], dtype=torch.float32),
            "proto_summary_5d": torch.tensor(self.samples["proto_summary_5d"][index], dtype=torch.float32),
            "obs_mask": torch.ones(agent_count, dtype=torch.bool),
            "gt_proto_id": torch.tensor(int(self.samples["gt_proto_id"][index]), dtype=torch.long),
            "gt_proto_residual": torch.tensor(self.samples["gt_proto_residual"][index], dtype=torch.float32),
            "is_rare": torch.tensor(bool(self.samples["is_rare"][index]), dtype=torch.bool),
            "source_path": file_record["source_path"],
            "scene_id": f"{os.path.basename(file_record['source_path'])}:{start_idx}:{target_agent_id}",
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
    basis_dim = int(batch[0].get("gt_basis_coeff", torch.zeros(0)).numel())
    gt_basis_coeff = torch.zeros(batch_size, basis_dim, dtype=torch.float32)
    proto_summary_5d = torch.zeros(batch_size, 5, dtype=torch.float32)
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
        if basis_dim > 0:
            gt_basis_coeff[batch_index] = item["gt_basis_coeff"]
        proto_summary_5d[batch_index] = item["proto_summary_5d"]
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
        "gt_basis_coeff": gt_basis_coeff,
        "proto_summary_5d": proto_summary_5d,
        "gt_proto_id": gt_proto_id,
        "gt_proto_residual": gt_proto_residual,
        "is_rare": is_rare,
        "source_path": source_path,
        "scene_id": scene_id,
        "split_name": split_name,
    }
