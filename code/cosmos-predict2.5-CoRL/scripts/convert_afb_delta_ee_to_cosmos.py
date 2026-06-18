"""
Convert ActionFollowingBench delta_ee dataset to CosmosPredict2.5 Dataset_3D format.

Source: /mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible/
Output: datasets/afb_delta_ee/

Structure per task:
  datasets/afb_delta_ee/<task_name>/
    videos/<split>/<episode_id>/rgb.mp4   (symlink to source mp4)
    annotation/<split>/<episode_id>.json

Action format (already delta, directly from HDF5):
  delta_ee_action/left_delta_pose  (T,6): xyz + euler delta
  delta_ee_action/left_gripper     (T,): gripper state
  delta_ee_action/right_delta_pose (T,6): xyz + euler delta
  delta_ee_action/right_gripper    (T,): gripper state
  -> state layout: [left_xyz(3), left_euler(3), left_g(1), right_xyz(3), right_euler(3), right_g(1)] = 14D
  -> scale by [20]*6 + [1] + [20]*6 + [1]

Usage:
    cd /mnt/gyc/cosmos-predict2.5-CoRL
    python scripts/convert_afb_delta_ee_to_cosmos.py
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np

SRC = Path("/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible")
OUT = Path("/mnt/gyc_ckp/datasets/afb_delta_ee")
VAL_RATIO = 0.1
ACTION_SCALER = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0,
                          20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0])


def convert_task(task_name: str):
    task_src = SRC / task_name
    task_out = OUT / task_name

    hdf5_files = sorted((task_src / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem.replace("episode", "")))
    n_total = len(hdf5_files)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val
    print(f"[{task_name}] {n_total} episodes → train:{n_train} val:{n_val}")

    for ep_idx, hdf5_path in enumerate(hdf5_files):
        ep_num = int(hdf5_path.stem.replace("episode", ""))
        split = "train" if ep_idx < n_train else "val"

        # ---- video: symlink existing head_camera mp4 ----
        src_mp4 = task_src / "video" / f"episode{ep_num}.mp4"  # head_camera 640x480
        video_rel = f"videos/{split}/{ep_idx}/rgb.mp4"
        video_abs = task_out / video_rel
        video_abs.parent.mkdir(parents=True, exist_ok=True)
        if not video_abs.exists():
            os.symlink(src_mp4.resolve(), video_abs)

        # ---- action: read delta directly from HDF5 ----
        with h5py.File(hdf5_path, "r") as f:
            left_pose  = f["delta_ee_action/left_delta_pose"][:]   # (T,6)
            left_g     = f["delta_ee_action/left_gripper"][:]       # (T,)
            right_pose = f["delta_ee_action/right_delta_pose"][:]  # (T,6)
            right_g    = f["delta_ee_action/right_gripper"][:]      # (T,)

        # state (T,14): [left_xyz(3), left_euler(3), left_g(1), right_xyz(3), right_euler(3), right_g(1)]
        state = np.concatenate([
            left_pose, left_g[:, None],
            right_pose, right_g[:, None],
        ], axis=1)  # (T, 14)
        state_scaled = (state * ACTION_SCALER).tolist()
        gripper_list = left_g.tolist()  # kept for compat; left gripper

        # ---- instruction: pick first seen instruction ----
        instr_path = task_src / "instructions" / f"episode{ep_num}.json"
        task_text = ""
        if instr_path.exists():
            with open(instr_path) as f:
                instr = json.load(f)
            seen = instr.get("seen", [])
            task_text = seen[0] if seen else ""

        annotation = {
            "episode_id": ep_idx,
            "task_name": task_name,
            "state": state_scaled,
            "continuous_gripper_state": gripper_list,
            "videos": [{"video_path": video_rel}],
            "task": task_text,
            "texts": [task_text],
        }

        ann_dir = task_out / "annotation" / split
        ann_dir.mkdir(parents=True, exist_ok=True)
        with open(ann_dir / f"{ep_idx}.json", "w") as f:
            json.dump(annotation, f)

    print(f"[{task_name}] Done → {task_out}")


if __name__ == "__main__":
    tasks = sorted([d.name for d in SRC.iterdir() if d.is_dir()])
    print(f"Found {len(tasks)} tasks")
    for task in tasks:
        convert_task(task)
    print(f"\nAll done. Output: {OUT}")
