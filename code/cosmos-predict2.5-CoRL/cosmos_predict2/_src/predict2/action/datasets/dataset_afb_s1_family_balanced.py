# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import random
import sys
import traceback
import warnings
from collections import OrderedDict
from pathlib import Path

import av
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils.dataset_utils import Resize_Preprocess, ToTensorVideo
from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm


def _import_h5py():
    try:
        import h5py as _h5py

        return _h5py
    except ModuleNotFoundError:
        extra_paths = [
            p
            for p in [
                os.environ.get("H5PY_EXTRA_PATH"),
                "/usr/local/lib/python3.10/dist-packages",
                "/usr/lib/python3/dist-packages",
            ]
            if p
        ]
        for path in extra_paths:
            if Path(path).exists() and path not in sys.path:
                sys.path.append(path)
            try:
                import h5py as _h5py

                return _h5py
            except ModuleNotFoundError:
                continue
        raise ModuleNotFoundError(
            "h5py is required to read expert HDF5 files. Set H5PY_EXTRA_PATH to a Python 3.10 site-packages "
            "directory containing h5py."
        )


h5py = _import_h5py()


def _rotm_to_6d(rotm):
    return np.asarray(rotm[:2, :], dtype=np.float32).reshape(6)


def _ee_target_from_state_vector(state):
    state = np.asarray(state, dtype=np.float32)
    left_pose = state[:, 0:6]
    left_gripper = state[:, 6]
    right_pose = state[:, 7:13]
    right_gripper = state[:, 13]
    position = np.stack([left_pose[:, 0:3], right_pose[:, 0:3]], axis=1).astype(np.float32)
    rotation_6d = np.stack(
        [
            np.stack([_rotm_to_6d(euler2rotm(rot)) for rot in left_pose[:, 3:6]], axis=0),
            np.stack([_rotm_to_6d(euler2rotm(rot)) for rot in right_pose[:, 3:6]], axis=0),
        ],
        axis=1,
    ).astype(np.float32)
    gripper = np.stack([left_gripper, right_gripper], axis=1).astype(np.float32)
    return position, rotation_6d, gripper


def _ee_target_from_h5(f, start, length):
    frame_slice = slice(start + 1, start + length + 1)
    state = np.asarray(f["delta_ee_state/vector"][frame_slice], dtype=np.float32)
    return _ee_target_from_state_vector(state)


def _h5_future_window_length(f):
    action_len = int(f["delta_ee_action/vector"].shape[0])
    state_len = int(f["delta_ee_state/vector"].shape[0])
    frame_len = int(f["observation/head_camera/rgb"].shape[0])
    return min(action_len, state_len - 1, frame_len - 1)


class AFBS1FamilyBalancedDataset(Dataset):
    """Online family-balanced S1 sampler for ActionFollowingBench delta-ee data."""

    TASKS = [
        "click_alarmclock",
        "click_bell",
        "place_object_basket",
        "open_laptop",
        "stack_blocks_two",
    ]

    FAMILY_BY_SLOT = (
        "expert",
        "expert",
        "expert",
        "pca_c8_sigma0p05",
        "raw_sigma0p0025",
        "random_feasible_300step",
    )

    TASK_CAPTIONS = {
        "click_alarmclock": "Press the top button of the alarm clock.",
        "click_bell": "Press the bell button.",
        "place_object_basket": "Place the object into the basket.",
        "open_laptop": "Open the laptop.",
        "stack_blocks_two": "Stack the two blocks.",
    }

    def __init__(
        self,
        expert_root,
        enhanced_lerobot_root,
        random_feasible_root,
        mode="train",
        num_action_per_chunk=16,
        epoch_size=68428,
        video_size=(256, 320),
        gripper_rescale_factor=1.0,
        seed=20260609,
        load_t5_embeddings=False,
        rf_video_cache_size=None,
        **unused_kwargs,
    ):
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.expert_root = Path(expert_root)
        self.enhanced_lerobot_root = Path(enhanced_lerobot_root)
        self.random_feasible_root = Path(random_feasible_root)
        self.mode = mode
        self.num_action_per_chunk = int(num_action_per_chunk)
        if self.num_action_per_chunk != 16:
            warnings.warn(
                f"AFB S1 family-balanced chunk16 dataset received num_action_per_chunk={self.num_action_per_chunk}; "
                "forcing num_action_per_chunk=16."
            )
            self.num_action_per_chunk = 16
        self.sequence_length = self.num_action_per_chunk + 1
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        self.load_t5_embeddings = load_t5_embeddings
        if rf_video_cache_size is None:
            rf_video_cache_size = int(os.environ.get("AFB_S1_RF_VIDEO_CACHE_SIZE", "0"))
        self.rf_video_cache_size = max(0, int(rf_video_cache_size))

        self.c_act_scaler = np.array(
            [20.0] * 6
            + [float(gripper_rescale_factor)]
            + [20.0] * 6
            + [float(gripper_rescale_factor)],
            dtype=np.float32,
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

        self._video_cache = OrderedDict()
        self._lerobot_data_cache = {}

        self.pools = {
            "expert": self._build_expert_pool(),
            "pca_c8_sigma0p05": self._build_lerobot_pool("perturbed_pca_gaussian/c_8_sigma_0p05"),
            "raw_sigma0p0025": self._build_lerobot_pool("perturbed_raw_gaussian/sigma_0p0025"),
            "random_feasible_300step": self._build_rf_pool(),
        }
        if self.mode != "train":
            self.pools["random_feasible_300step"] = {task: [] for task in self.TASKS}

        for family, by_task in self.pools.items():
            n = sum(len(v) for v in by_task.values())
            print(f"[AFB S1] {mode} {family}: {n} trajectories/samples")

    def __len__(self):
        return self.epoch_size

    def _split_episode_ids(self):
        if self.mode == "train":
            return set(range(40))
        return set(range(40, 50))

    def _build_expert_pool(self):
        split_ids = self._split_episode_ids()
        pool = {task: [] for task in self.TASKS}
        for task in self.TASKS:
            for ep in sorted(split_ids):
                h5_path = self.expert_root / task / "data" / f"episode{ep}.hdf5"
                if not h5_path.exists():
                    continue
                with h5py.File(h5_path, "r") as f:
                    length = _h5_future_window_length(f)
                if length >= self.num_action_per_chunk:
                    pool[task].append({"family": "expert", "task": task, "h5_path": h5_path, "length": length})
        return pool

    def _build_lerobot_pool(self, rel):
        # Enhanced LeRobot task_index is the best available source-episode split key.
        split_ids = self._split_episode_ids()
        pool = {task: [] for task in self.TASKS}
        for task in self.TASKS:
            task_root = self.enhanced_lerobot_root / rel / task
            ep_files = sorted((task_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
            if not ep_files:
                continue
            eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
            eps = eps[(eps["episode_index"].astype(int) // 100).isin(split_ids)]
            for row in eps.to_dict("records"):
                length = int(row["length"]) - 1
                if length >= self.num_action_per_chunk:
                    pool[task].append(
                        {
                            "family": rel,
                            "task": task,
                            "task_root": task_root,
                            "episode_index": int(row["episode_index"]),
                            "length": length,
                            "dataset_from_index": int(row["dataset_from_index"]),
                            "video_file_index": int(row["videos/observation.images.cam_high/file_index"]),
                            "video_from_timestamp": float(row["videos/observation.images.cam_high/from_timestamp"]),
                        }
                    )
        return pool

    def _build_rf_pool(self):
        pool = {task: [] for task in self.TASKS}
        variants = [
            "rf_5task_300step_2ep5start_formal_uniform_10seed_v1",
            "rf_5task_300step_2ep5start_formal_weighted_10seed_v1",
        ]
        for variant in variants:
            for task in self.TASKS:
                task_root = self.random_feasible_root / variant / task
                for meta_path in sorted(task_root.glob("*/metadata.json")):
                    meta = json.loads(meta_path.read_text())
                    if not meta.get("accepted", True):
                        continue
                    length = int(meta.get("trajectory_length") or meta.get("chunk_size") or 300)
                    if length >= self.num_action_per_chunk:
                        sample_dir = meta_path.parent
                        pool[task].append(
                            {
                                "family": "random_feasible_300step",
                                "task": task,
                                "sample_dir": sample_dir,
                                "length": length,
                                "variant": variant,
                            }
                        )
        return pool

    def _rng(self, index):
        # Keep family slots index-addressable, while sampling task/window online.
        return random.Random(self.seed + int(index) * 1000003 + random.randint(0, 2**31 - 1))

    def _choose_sample(self, index):
        rng = self._rng(index)
        family = self.FAMILY_BY_SLOT[int(index) % len(self.FAMILY_BY_SLOT)] if self.mode == "train" else "expert"
        by_task = self.pools[family]
        tasks = [task for task in self.TASKS if by_task[task]]
        if not tasks:
            raise RuntimeError(f"No samples available for family={family}, mode={self.mode}")
        task = rng.choice(tasks)
        sample = rng.choice(by_task[task])
        max_start = sample["length"] - self.num_action_per_chunk
        sample = dict(sample)
        sample["start"] = rng.randint(0, max_start)
        sample["chosen_family"] = family
        return sample

    def _read_expert_frames(self, h5_path, start):
        frame_ids = range(start, start + self.sequence_length)
        frames = []
        with h5py.File(h5_path, "r") as f:
            rgb = f["observation/head_camera/rgb"]
            for i in frame_ids:
                frames.append(imageio.imread(io.BytesIO(bytes(rgb[i]))))
            actions = f["delta_ee_action/vector"][start : start + self.num_action_per_chunk]
            ee_target = _ee_target_from_h5(f, start, self.num_action_per_chunk)
        return np.stack(frames), np.asarray(actions, dtype=np.float32), ee_target

    def _video_reader(self, path):
        path = str(path)
        if self.rf_video_cache_size == 0:
            return VideoReader(path, ctx=cpu(0), num_threads=1)

        reader = self._video_cache.get(path)
        if reader is None:
            reader = VideoReader(path, ctx=cpu(0), num_threads=1)
            self._video_cache[path] = reader
            while len(self._video_cache) > self.rf_video_cache_size:
                self._video_cache.popitem(last=False)
        else:
            self._video_cache.move_to_end(path)
        return reader

    def _read_video_frames(self, path, frame_ids):
        vr = self._video_reader(path)
        frames = vr.get_batch(list(frame_ids)).asnumpy()
        if self.rf_video_cache_size == 0:
            del vr
        return frames

    def _read_video_frames_av1(self, path, first_frame):
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            seek_pts = int((first_frame / 30.0) / stream.time_base)
            container.seek(seek_pts, any_frame=False, backward=True, stream=stream)
            for frame in container.decode(stream):
                frame_idx = int(round(float(frame.pts * stream.time_base) * 30.0))
                if frame_idx < first_frame:
                    continue
                if frame_idx >= first_frame + self.sequence_length:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) != self.sequence_length:
            raise RuntimeError(f"Could not read {self.sequence_length} frames from {path} at {first_frame}")
        return np.stack(frames)

    def _lerobot_data(self, task_root):
        task_root = Path(task_root)
        key = str(task_root)
        data = self._lerobot_data_cache.get(key)
        if data is None:
            parquet = task_root / "data" / "chunk-000" / "file-000.parquet"
            data = pd.read_parquet(parquet, columns=["observation.state", "action"])
            self._lerobot_data_cache[key] = data
        return data

    def _read_lerobot(self, sample):
        start = sample["start"]
        row_start = sample["dataset_from_index"] + start
        data = self._lerobot_data(sample["task_root"])
        actions = np.stack(data.iloc[row_start : row_start + self.num_action_per_chunk]["action"].to_numpy()).astype(
            np.float32
        )
        states = np.stack(
            data.iloc[row_start + 1 : row_start + self.num_action_per_chunk + 1]["observation.state"].to_numpy()
        ).astype(np.float32)
        ee_target = _ee_target_from_state_vector(states)
        file_index = sample["video_file_index"]
        video_path = (
            sample["task_root"]
            / "videos"
            / "observation.images.cam_high"
            / "chunk-000"
            / f"file-{file_index:03d}.mp4"
        )
        first_frame = int(round(sample["video_from_timestamp"] * 30.0)) + start
        frames = self._read_video_frames_av1(video_path, first_frame)
        return frames, actions, ee_target

    def _read_rf(self, sample):
        sample_dir = sample["sample_dir"]
        start = sample["start"]
        actions = np.load(sample_dir / "action.npy", mmap_mode="r")[start : start + self.num_action_per_chunk].astype(
            np.float32
        )
        frames = self._read_video_frames(sample_dir / "video.mp4", range(start, start + self.sequence_length))
        with h5py.File(sample_dir / "data.hdf5", "r") as f:
            ee_target = _ee_target_from_h5(f, start, self.num_action_per_chunk)
        return frames, actions, ee_target

    def _read_sample(self, sample):
        family = sample["chosen_family"]
        if family == "expert":
            return self._read_expert_frames(sample["h5_path"], sample["start"])
        if family in {"pca_c8_sigma0p05", "raw_sigma0p0025"}:
            return self._read_lerobot(sample)
        if family == "random_feasible_300step":
            return self._read_rf(sample)
        raise ValueError(f"Unknown family: {family}")

    def __getitem__(self, index):
        try:
            sample = self._choose_sample(index)
            frames, actions, ee_target = self._read_sample(sample)
            frames = torch.from_numpy(frames.astype(np.uint8)).permute(0, 3, 1, 2)
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            actions = torch.from_numpy(actions * self.c_act_scaler).float()
            ee_position, ee_rotation_6d, ee_gripper = ee_target

            data = {
                "video": frames.permute(1, 0, 2, 3),
                "action": actions,
                "ee_target_position": torch.from_numpy(ee_position).float(),
                "ee_target_rotation_6d": torch.from_numpy(ee_rotation_6d).float(),
                "ee_target_gripper": torch.from_numpy(ee_gripper).float(),
                "annotation_file": str(sample.get("h5_path") or sample.get("sample_dir") or sample.get("task_root")),
                "__key__": f"{sample['chosen_family']}/{sample['task']}/{sample.get('episode_index', '')}/{sample['start']}",
                "fps": 4,
                "image_size": 256 * torch.ones(4),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, 256, 256),
                "ai_caption": self.TASK_CAPTIONS.get(sample["task"], sample["task"]),
            }
            if self.load_t5_embeddings:
                raise NotImplementedError("Precomputed T5 embeddings are not implemented for AFB S1.")
            data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
            return data
        except Exception:
            warnings.warn(f"Invalid AFB S1 sample at virtual index {index}; resampling.")
            warnings.warn(traceback.format_exc())
            return self[random.randint(0, len(self) - 1)]


class AFBS1ExpertSingleTaskDataset(Dataset):
    """Single-task expert-only S1 sampler for ActionFollowingBench delta-ee data."""

    TASKS = AFBS1FamilyBalancedDataset.TASKS
    TASK_CAPTIONS = AFBS1FamilyBalancedDataset.TASK_CAPTIONS

    def __init__(
        self,
        expert_root,
        task,
        mode="train",
        num_action_per_chunk=16,
        epoch_size=None,
        video_size=(256, 320),
        gripper_rescale_factor=1.0,
        seed=20260611,
        load_t5_embeddings=False,
        **unused_kwargs,
    ):
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if task not in self.TASKS:
            raise ValueError(f"Unsupported AFB S1 task: {task}. Expected one of {self.TASKS}")

        self.expert_root = Path(expert_root)
        self.task = task
        self.mode = mode
        self.num_action_per_chunk = int(num_action_per_chunk)
        if self.num_action_per_chunk != 16:
            warnings.warn(
                f"AFB S1 single-task expert dataset received num_action_per_chunk={self.num_action_per_chunk}; "
                "forcing num_action_per_chunk=16."
            )
            self.num_action_per_chunk = 16
        self.sequence_length = self.num_action_per_chunk + 1
        self.seed = int(seed)
        self.load_t5_embeddings = load_t5_embeddings
        self.c_act_scaler = np.array(
            [20.0] * 6
            + [float(gripper_rescale_factor)]
            + [20.0] * 6
            + [float(gripper_rescale_factor)],
            dtype=np.float32,
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

        self.samples = self._build_expert_windows()
        if not self.samples:
            raise RuntimeError(f"No expert windows found for task={self.task}, mode={self.mode}")
        self.epoch_size = int(epoch_size) if epoch_size is not None else len(self.samples)
        print(
            f"[AFB S1 single-task expert] {mode} task={self.task}: "
            f"{len(self.samples)} windows, epoch_size={self.epoch_size}"
        )

    def __len__(self):
        return self.epoch_size

    def _split_episode_ids(self):
        if self.mode == "train":
            return set(range(40))
        return set(range(40, 50))

    def _build_expert_windows(self):
        samples = []
        for ep in sorted(self._split_episode_ids()):
            h5_path = self.expert_root / self.task / "data" / f"episode{ep}.hdf5"
            if not h5_path.exists():
                continue
            with h5py.File(h5_path, "r") as f:
                length = _h5_future_window_length(f)
            max_start = length - self.num_action_per_chunk
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                samples.append({"task": self.task, "h5_path": h5_path, "episode": ep, "start": start})
        return samples

    def _read_expert_frames(self, h5_path, start):
        frame_ids = range(start, start + self.sequence_length)
        frames = []
        with h5py.File(h5_path, "r") as f:
            rgb = f["observation/head_camera/rgb"]
            for i in frame_ids:
                frames.append(imageio.imread(io.BytesIO(bytes(rgb[i]))))
            actions = f["delta_ee_action/vector"][start : start + self.num_action_per_chunk]
            ee_target = _ee_target_from_h5(f, start, self.num_action_per_chunk)
        return np.stack(frames), np.asarray(actions, dtype=np.float32), ee_target

    def __getitem__(self, index):
        try:
            rng = random.Random(self.seed + int(index) * 1000003 + random.randint(0, 2**31 - 1))
            sample = self.samples[rng.randrange(len(self.samples))] if self.mode == "train" else self.samples[index % len(self.samples)]
            frames, actions, ee_target = self._read_expert_frames(sample["h5_path"], sample["start"])
            frames = torch.from_numpy(frames.astype(np.uint8)).permute(0, 3, 1, 2)
            frames = self.not_norm_preprocess(frames)
            frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
            actions = torch.from_numpy(actions * self.c_act_scaler).float()
            ee_position, ee_rotation_6d, ee_gripper = ee_target

            data = {
                "video": frames.permute(1, 0, 2, 3),
                "action": actions,
                "ee_target_position": torch.from_numpy(ee_position).float(),
                "ee_target_rotation_6d": torch.from_numpy(ee_rotation_6d).float(),
                "ee_target_gripper": torch.from_numpy(ee_gripper).float(),
                "annotation_file": str(sample["h5_path"]),
                "__key__": f"expert/{self.task}/{sample['episode']}/{sample['start']}",
                "fps": 4,
                "image_size": 256 * torch.ones(4),
                "num_frames": self.sequence_length,
                "padding_mask": torch.zeros(1, 256, 256),
                "ai_caption": self.TASK_CAPTIONS.get(self.task, self.task),
            }
            if self.load_t5_embeddings:
                raise NotImplementedError("Precomputed T5 embeddings are not implemented for AFB S1.")
            data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
            return data
        except Exception:
            warnings.warn(f"Invalid AFB S1 single-task sample at virtual index {index}; resampling.")
            warnings.warn(traceback.format_exc())
            return self[random.randint(0, len(self) - 1)]
