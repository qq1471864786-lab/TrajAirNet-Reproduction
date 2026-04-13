import hashlib
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def resolve_split_dir(project_root, dataset_variant, dataset_name, split):
    if dataset_variant == "social":
        data_dir = os.path.join(
            project_root,
            "dataset",
            "social",
            dataset_name,
            "processed_data",
            split,
        )
    elif dataset_variant == "no_social":
        candidate_dirs = [
            os.path.join(
                project_root,
                "dataset",
                "no_social",
                dataset_name,
                "processed_data",
                split,
            ),
            os.path.join(
                project_root,
                "dataset",
                "no_social",
                dataset_name,
                split,
            ),
            os.path.join(
                project_root,
                "dataset",
                "no_social",
                f"{dataset_name}_no_social",
                split,
            ),
        ]
        data_dir = next((path for path in candidate_dirs if os.path.isdir(path)), None)
    else:
        raise ValueError(f"Unsupported dataset variant: {dataset_variant}")

    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Split directory not found for variant={dataset_variant}, "
            f"dataset={dataset_name}, split={split}"
        )
    return data_dir


def read_txt_file(path, delim=" "):
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "

    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = [float(token) for token in line.strip().split(delim) if token]
            if values:
                rows.append(values)
    return np.asarray(rows, dtype=np.float32)


class SceneTrajectoryDataset(Dataset):
    CACHE_VERSION = 2

    def __init__(
        self,
        data_dir,
        obs_len=11,
        pred_len=120,
        pred_step=10,
        skip=1,
        min_agents=1,
        delim=" ",
        show_progress=False,
        progress_desc="Load data",
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pred_step = pred_step
        self.skip = skip
        self.min_agents = min_agents
        self.delim = delim
        self.sequence_len = obs_len + pred_len
        self.future_points = int(math.ceil(pred_len / pred_step))
        self.file_records = []
        self.sample_index = []

        # --- Cache logic ---
        cache_path = self._cache_path(data_dir, obs_len, pred_len, pred_step, skip, min_agents)
        if cache_path and os.path.isfile(cache_path):
            print(f"[Cache] loading from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            sample_index = cached.get("sample_index", [])
            schema_ok = (
                cached.get("cache_version") == self.CACHE_VERSION
                and (
                    len(sample_index) == 0
                    or all(
                        "agent_count" in sample and "future_vertical_range" in sample
                        for sample in sample_index
                    )
                )
            )
            if schema_ok:
                self.file_records = cached["file_records"]
                self.sample_index = sample_index
                self.samples = self.sample_index
                print(f"[Cache] ready  samples={len(self.sample_index):,}")
                return
            print("[Cache] stale schema detected, rebuilding cache.")

        # --- Normal build ---
        file_names = sorted(os.listdir(data_dir))
        file_paths = [os.path.join(data_dir, file_name) for file_name in file_names
                      if os.path.isfile(os.path.join(data_dir, file_name))]
        use_tqdm = show_progress and sys.stderr.isatty()
        progress_bar = tqdm(
            file_paths,
            desc=progress_desc,
            unit="file",
            disable=not use_tqdm,
            leave=False,
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.5,
            bar_format="{desc:<12} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for path in progress_bar:
            data = read_txt_file(path, delim)
            file_record = self._build_file_record(data, path)
            if file_record is None:
                continue
            file_index = len(self.file_records)
            self.file_records.append(file_record)
            self.sample_index.extend(self._build_window_index(file_record, file_index))

        self.samples = self.sample_index

        if not self.sample_index:
            raise RuntimeError(f"No valid samples built from {data_dir}")

        # --- Save cache ---
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(
                {
                    "cache_version": self.CACHE_VERSION,
                    "file_records": self.file_records,
                    "sample_index": self.sample_index,
                },
                cache_path,
            )
            print(f"[Cache] saved to {cache_path}")

    @staticmethod
    def _cache_path(data_dir, obs_len, pred_len, pred_step, skip, min_agents):
        """Build a deterministic cache path based on data dir content + params."""
        file_names = sorted(os.listdir(data_dir))
        if not file_names:
            return None
        key_str = (
            f"{SceneTrajectoryDataset.CACHE_VERSION}|{data_dir}|{file_names}|"
            f"{obs_len}|{pred_len}|{pred_step}|{skip}|{min_agents}"
        )
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:12]
        cache_dir = os.path.join(data_dir, ".cache")
        return os.path.join(cache_dir, f"dataset_{key_hash}.pt")

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

    def _build_window_index(self, file_record, file_index):
        frames = file_record["frames"]
        frame_map = file_record["frame_map"]
        windows = []
        max_start = len(frames) - self.sequence_len + 1

        for start_idx in range(0, max_start, self.skip):
            window_frames = frames[start_idx : start_idx + self.sequence_len]
            window_segments = [frame_map[frame] for frame in window_frames]
            window_data = np.concatenate(window_segments, axis=0)
            agent_ids = np.unique(window_data[:, 1])

            valid_agent_ids = []
            future_vertical_ranges = []

            for agent_id in agent_ids:
                agent_rows = window_data[window_data[:, 1] == agent_id]
                if agent_rows.shape[0] != self.sequence_len:
                    continue

                track = agent_rows[:, 2:].T
                obs = track[:3, : self.obs_len]
                future = track[:3, self.obs_len + self.pred_step - 1 :: self.pred_step]
                context = track[3:, : self.obs_len]

                if future.shape[1] != self.future_points:
                    continue

                valid_agent_ids.append(agent_id)
                future_vertical_ranges.append(float(future[2].max() - future[2].min()))

            if len(valid_agent_ids) < self.min_agents:
                continue

            windows.append(
                {
                    "file_index": file_index,
                    "start_idx": start_idx,
                    "agent_ids": valid_agent_ids,
                    "agent_count": len(valid_agent_ids),
                    "future_vertical_range": max(future_vertical_ranges) if future_vertical_ranges else 0.0,
                }
            )

        return windows

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
        fut_agents = []
        ctx_agents = []
        for agent_id in sample_meta["agent_ids"]:
            agent_rows = window_data[window_data[:, 1] == agent_id]
            track = agent_rows[:, 2:].T
            obs = track[:3, : self.obs_len]
            future = track[:3, self.obs_len + self.pred_step - 1 :: self.pred_step]
            context = track[3:, : self.obs_len]
            obs_agents.append(obs.T)
            fut_agents.append(future.T)
            ctx_agents.append(context.T)

        return {
            "obs": torch.tensor(np.stack(obs_agents, axis=1), dtype=torch.float32),
            "target": torch.tensor(np.stack(fut_agents, axis=1), dtype=torch.float32),
            "context": torch.tensor(np.stack(ctx_agents, axis=1), dtype=torch.float32),
            "source_path": file_record["source_path"],
            "agent_count": sample_meta["agent_count"],
            "future_vertical_range": sample_meta["future_vertical_range"],
        }


def scene_batch_collate(batch):
    if not batch:
        raise ValueError("Empty batch is not supported.")

    obs = torch.cat([item["obs"] for item in batch], dim=1)
    target = torch.cat([item["target"] for item in batch], dim=1)
    context = torch.cat([item["context"] for item in batch], dim=1)

    scene_ids = []
    scene_slices = []
    source_paths = []
    agent_counts = []
    future_vertical_ranges = []
    start = 0
    for scene_index, item in enumerate(batch):
        agent_count = item["obs"].shape[1]
        scene_ids.append(torch.full((agent_count,), scene_index, dtype=torch.long))
        scene_slices.append((start, start + agent_count))
        source_paths.append(item["source_path"])
        agent_counts.append(item["agent_count"])
        future_vertical_ranges.append(item["future_vertical_range"])
        start += agent_count

    return {
        "obs": obs,
        "target": target,
        "context": context,
        "scene_ids": torch.cat(scene_ids, dim=0),
        "scene_slices": torch.tensor(scene_slices, dtype=torch.long),
        "scene_count": len(batch),
        "agent_counts": torch.tensor(agent_counts, dtype=torch.long),
        "future_vertical_ranges": torch.tensor(future_vertical_ranges, dtype=torch.float32),
        "source_path": source_paths,
    }
