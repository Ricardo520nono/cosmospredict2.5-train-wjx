"""
Convert RobotWin2.0 aloha-agilex_clean_50.zip to CosmosPredict2.5 Dataset_3D format.

Output structure (per task):
  <out_dir>/
    videos/<split>/<episode_id>/rgb.mp4     ← front_camera frames
    annotation/<split>/<episode_id>.json    ← state + gripper + video path

Usage:
    python scripts/convert_robotwin_clean50_to_cosmos.py \
        --zip_path /mnt/public_ckp/.../place_container_plate/aloha-agilex_clean_50.zip \
        --out_dir  /mnt/gyc/cosmos-predict2.5-CoRL/datasets/robotwin_clean50/place_container_plate \
        --task_name place_container_plate \
        --val_ratio 0.1 \
        --fps 10
"""

import argparse
import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import h5py
import imageio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to euler angles (roll, pitch, yaw) in radians."""
    r = Rotation.from_quat(quat_xyzw)  # scipy expects xyzw
    return r.as_euler("xyz", degrees=False)


def extract_episode(hdf5_bytes: bytes, episode_id: int, out_dir: str, split: str, fps: int):
    """
    Extract one episode from HDF5 bytes:
      - Write front_camera RGB frames as mp4
      - Return annotation dict
    """
    with h5py.File(io.BytesIO(hdf5_bytes), "r") as f:
        T = f["endpose/left_endpose"].shape[0]

        # --- video: front_camera RGB (JPEG bytes) → mp4 ---
        video_rel_path = f"videos/{split}/{episode_id}/rgb.mp4"
        video_abs_path = os.path.join(out_dir, video_rel_path)
        os.makedirs(os.path.dirname(video_abs_path), exist_ok=True)

        jpeg_bytes_list = f["observation/front_camera/rgb"][:]
        frames = []
        for jpeg_bytes in jpeg_bytes_list:
            img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
            frames.append(np.array(img.convert("RGB")))

        writer = imageio.get_writer(video_abs_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

        # --- state: left+right arm xyz+euler+gripper, shape (T, 14) ---
        left_endpose  = f["endpose/left_endpose"][:]   # (T, 7): xyz(3) + quat_xyzw(4)
        left_gripper  = f["endpose/left_gripper"][:]   # (T,)
        right_endpose = f["endpose/right_endpose"][:]  # (T, 7): xyz(3) + quat_xyzw(4)
        right_gripper = f["endpose/right_gripper"][:]  # (T,)

        left_xyz   = left_endpose[:, :3]
        left_euler = np.array([quat_to_euler(q) for q in left_endpose[:, 3:7]])
        right_xyz  = right_endpose[:, :3]
        right_euler = np.array([quat_to_euler(q) for q in right_endpose[:, 3:7]])

        # state layout: [left_xyz(3), left_euler(3), left_gripper(1), right_xyz(3), right_euler(3), right_gripper(1)] = 14D
        state = np.concatenate([
            left_xyz, left_euler, left_gripper[:, None],
            right_xyz, right_euler, right_gripper[:, None],
        ], axis=1)  # (T, 14)

    annotation = {
        "episode_id": episode_id,
        "state": state.tolist(),
        "continuous_gripper_state": left_gripper.tolist(),  # kept for compat; left gripper
        "videos": [{"video_path": video_rel_path}],
        "task": "",
        "texts": [""],
        "episode_metadata": {},
    }
    return annotation


def process_zip(zip_path: str, out_dir: str, task_name: str, val_ratio: float, fps: int):
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        hdf5_names = sorted([n for n in zf.namelist() if n.endswith(".hdf5")])
        total = len(hdf5_names)
        n_val = max(1, int(total * val_ratio))
        n_train = total - n_val
        print(f"[{task_name}] {total} episodes → train:{n_train} val:{n_val}")

        for ep_idx, hdf5_name in enumerate(hdf5_names):
            split = "train" if ep_idx < n_train else "val"
            episode_id = ep_idx

            print(f"  [{task_name}] Episode {ep_idx+1}/{total} ({split}) ...", flush=True)
            hdf5_bytes = zf.read(hdf5_name)

            ann = extract_episode(hdf5_bytes, episode_id, out_dir, split, fps)

            ann_dir = os.path.join(out_dir, "annotation", split)
            os.makedirs(ann_dir, exist_ok=True)
            ann_path = os.path.join(ann_dir, f"{episode_id}.json")
            with open(ann_path, "w") as f:
                json.dump(ann, f, indent=2)

    print(f"[{task_name}] Done. Output: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    process_zip(args.zip_path, args.out_dir, args.task_name, args.val_ratio, args.fps)
