#!/usr/bin/env python3
"""
Run one RobotWin validation episode with the three final 14D RobotWin checkpoints.

Official references:
  - cosmos_predict2/action_conditioned.py
  - examples/action_conditioned.py

The official default action loader is 7D. This script mirrors the 14D dual-arm
action construction from:
  cosmos_predict2/_src/predict2/action/datasets/dataset_local.py

Usage, from repo root on a machine with a CUDA GPU:
    cd /mnt/gyc/cosmos-predict2.5-CoRL
    VENV_CUDNN="/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
    LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PYTHONPATH=".:packages/cosmos-cuda" \
    /mnt/gyc/cosmos-predict2.5/.venv/bin/python3 scripts/infer_robotwin_three_ckpts.py

Outputs are GT | Pred comparison videos under:
    /mnt/gyc_ckp/infer_results/robotwin_three_ckpts/
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import mediapy
import numpy as np
import torch
import torchvision

try:
    import imageio_ffmpeg

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    pass

REPO_ROOT = Path("/mnt/gyc/cosmos-predict2.5-CoRL")
CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
VIDEO_BASE = Path("/mnt/gyc_ckp/datasets/robotwin_clean50")
DEFAULT_ANN = Path("/mnt/gyc_ckp/datasets/robotwin_clean50_pcp_pob/annotation/val/0.json")
DEFAULT_SAVE_ROOT = Path("/mnt/gyc_ckp/infer_results/robotwin_three_ckpts")

GUIDANCE = 7
NUM_STEPS = 35
SAVE_FPS = 10
RESOLUTION = "256,320"
ACTION_SCALE = np.array([20.0] * 6 + [1.0] + [20.0] * 6 + [1.0], dtype=np.float32)


@dataclass(frozen=True)
class RunSpec:
    name: str
    experiment: str
    ckpt_path: Path
    chunk_size: int


RUN_SPECS = [
    RunSpec(
        name="pcp_pob_chunk16",
        experiment="cosmos_predict2p5_2B_robotwin_pcp_pob_clean50_14D_chunk16",
        ckpt_path=Path(
            "/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin"
            "/robotwin_clean50/cosmos_predict2p5_2B_robotwin_pcp_pob_clean50_14D_chunk16"
            "/checkpoints/iter_000060000"
        ),
        chunk_size=16,
    ),
    RunSpec(
        name="all50_chunk16",
        experiment="cosmos_predict2p5_2B_robotwin_all50_clean50_14D_chunk16",
        ckpt_path=Path(
            "/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin"
            "/robotwin_clean50_all50/cosmos_predict2p5_2B_robotwin_all50_clean50_14D_chunk16"
            "/checkpoints/iter_000060000"
        ),
        chunk_size=16,
    ),
    RunSpec(
        name="all50_chunk32",
        experiment="cosmos_predict2p5_2B_robotwin_all50_clean50_14D_chunk32",
        ckpt_path=Path(
            "/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin"
            "/robotwin_clean50_all50/cosmos_predict2p5_2B_robotwin_all50_clean50_14D_chunk32"
            "/checkpoints/iter_000060000"
        ),
        chunk_size=32,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=Path, default=DEFAULT_ANN, help="Annotation JSON for the single episode.")
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--resolution", default=RESOLUTION)
    parser.add_argument("--save-fps", type=int, default=SAVE_FPS)
    parser.add_argument("--max-actions", type=int, default=None, help="Optional quick test limit.")
    parser.add_argument(
        "--only",
        choices=[spec.name for spec in RUN_SPECS],
        default=None,
        help="Run only one checkpoint. Useful for multi-GPU launchers.",
    )
    return parser.parse_args()


def patch_zero_text_embeddings() -> None:
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    def _patched_get_data_batch(
        self,
        video,
        prompt,
        num_conditional_frames=1,
        negative_prompt="",
        use_neg_prompt=True,
        camera=None,
        action=None,
    ):
        _, _, _, h, w = video.shape
        zero_emb = torch.zeros(1, 512, 1024, dtype=torch.bfloat16, device="cuda")
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "action": action.unsqueeze(0) if action is not None else None,
            "fps": torch.randint(16, 32, (self.batch_size,)).float(),
            "padding_mask": torch.zeros(self.batch_size, 1, h, w),
            "num_conditional_frames": num_conditional_frames,
            "t5_text_embeddings": zero_emb,
            "neg_t5_text_embeddings": zero_emb,
        }
        for key, value in data_batch.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                data_batch[key] = value.cuda().to(dtype=torch.bfloat16)
        return data_batch

    Video2WorldInference._get_data_batch_input = _patched_get_data_batch


def compute_14d_actions(states: np.ndarray) -> np.ndarray:
    from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler

    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"Expected state shape (T, 14), got {states.shape}")

    left_arm = states[:, :6]
    right_arm = states[:, 7:13]
    left_gripper = states[:, 6]
    right_gripper = states[:, 13]

    actions = np.zeros((len(states) - 1, 14), dtype=np.float32)
    for k in range(1, len(states)):
        prev_l_rotm = euler2rotm(left_arm[k - 1, 3:])
        rel_xyz_l = prev_l_rotm.T @ (left_arm[k, :3] - left_arm[k - 1, :3])
        rel_rpy_l = rotm2euler(prev_l_rotm.T @ euler2rotm(left_arm[k, 3:]))

        prev_r_rotm = euler2rotm(right_arm[k - 1, 3:])
        rel_xyz_r = prev_r_rotm.T @ (right_arm[k, :3] - right_arm[k - 1, :3])
        rel_rpy_r = rotm2euler(prev_r_rotm.T @ euler2rotm(right_arm[k, 3:]))

        actions[k - 1, :3] = rel_xyz_l
        actions[k - 1, 3:6] = rel_rpy_l
        actions[k - 1, 6] = left_gripper[k]
        actions[k - 1, 7:10] = rel_xyz_r
        actions[k - 1, 10:13] = rel_rpy_r
        actions[k - 1, 13] = right_gripper[k]

    return actions * ACTION_SCALE


def make_vid_input(frame: np.ndarray, num_frames: int, resolution: str) -> torch.Tensor:
    img_t = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0)
    vid = torch.cat([img_t, torch.zeros_like(img_t).repeat(num_frames - 1, 1, 1, 1)], dim=0)
    vid = (vid * 255.0).to(torch.uint8)
    if resolution != "none":
        h, w = [int(x) for x in resolution.split(",")]
        vid = torchvision.transforms.functional.resize(vid, [h, w], antialias=True)
        vid = torchvision.transforms.functional.center_crop(vid, [h, w])
    return vid.unsqueeze(0).permute(0, 2, 1, 3, 4)


def decode_video(video_tensor: torch.Tensor) -> np.ndarray:
    frames = (video_tensor[0].permute(1, 2, 3, 0).clamp(-1, 1) / 2 + 0.5) * 255
    return frames.to(torch.uint8).cpu().numpy()


def resize_gt(frames: np.ndarray, resolution: str) -> np.ndarray:
    if resolution == "none":
        return frames
    h, w = [int(x) for x in resolution.split(",")]
    resized = [mediapy.resize_image(frame, (h, w)) for frame in frames]
    return np.asarray(resized, dtype=np.uint8)


def side_by_side(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    n = min(len(gt), len(pred))
    return np.concatenate([gt[:n], pred[:n]], axis=2)


def resolve_video_path(annotation: dict) -> Path:
    video_entry = annotation["videos"][0]
    rel_path = video_entry["video_path"] if isinstance(video_entry, dict) else video_entry
    return VIDEO_BASE / rel_path


def run_one_model(spec: RunSpec, ann: dict, gt_frames: np.ndarray, actions: np.ndarray, args: argparse.Namespace) -> Path:
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    experiment_opts = [
        "model.config.net.use_crossattn_projection=False",
        "model.config.text_encoder_config.compute_online=False",
        "trainer.straggler_detection.enabled=false",
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "upload_reproducible_setup=False",
        "model.config.tokenizer.vae_pth=/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/tokenizer.pth",
    ]

    print(f"[load] {spec.name}: {spec.ckpt_path}")
    model = Video2WorldInference(
        experiment_name=spec.experiment,
        ckpt_path=str(spec.ckpt_path),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
        experiment_opts=experiment_opts,
    )

    pred_chunks = []
    current_frame = gt_frames[0]
    total_actions = len(actions) if args.max_actions is None else min(len(actions), args.max_actions)

    for start in range(0, total_actions, spec.chunk_size):
        action_chunk = actions[start : start + spec.chunk_size]
        valid_len = len(action_chunk)
        if valid_len < spec.chunk_size:
            pad = np.zeros((spec.chunk_size - valid_len, 14), dtype=np.float32)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)

        vid_input = make_vid_input(current_frame, spec.chunk_size + 1, args.resolution)
        print(f"[infer] {spec.name}: chunk {start // spec.chunk_size + 1}, actions {start}:{start + valid_len}")
        result = model.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(action_chunk).float(),
            guidance=args.guidance,
            num_video_frames=spec.chunk_size + 1,
            num_latent_conditional_frames=1,
            resolution=args.resolution,
            seed=42 + start,
            negative_prompt="",
            num_steps=args.num_steps,
        )

        chunk = decode_video(result)
        pred_chunks.append(chunk if start == 0 else chunk[1:])
        current_frame = chunk[-1]

    pred_video = np.concatenate(pred_chunks, axis=0)
    gt_video = resize_gt(gt_frames, args.resolution)
    combined = side_by_side(gt_video, pred_video)

    task_name = Path(resolve_video_path(ann)).parts[-4]
    out_path = args.save_root / f"{spec.name}_{task_name}_gt_vs_pred.mp4"
    mediapy.write_video(str(out_path), combined, fps=args.save_fps)
    print(f"[save] {out_path} ({len(combined)} frames, GT | Pred)")

    model.cleanup()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out_path


def main() -> None:
    os.chdir(REPO_ROOT)
    args = parse_args()
    args.save_root.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script on a GPU machine.")

    patch_zero_text_embeddings()

    with open(args.ann) as f:
        ann = json.load(f)

    video_path = resolve_video_path(ann)
    states = np.asarray(ann["state"], dtype=np.float32)
    actions = compute_14d_actions(states)
    gt_frames = mediapy.read_video(str(video_path))

    print(f"[episode] ann={args.ann}")
    print(f"[episode] video={video_path}")
    print(f"[episode] states={states.shape}, actions={actions.shape}, gt_frames={gt_frames.shape}")

    outputs = []
    specs = [spec for spec in RUN_SPECS if args.only in (None, spec.name)]
    for spec in specs:
        outputs.append(run_one_model(spec, ann, gt_frames, actions, args))

    print("[done] outputs:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
