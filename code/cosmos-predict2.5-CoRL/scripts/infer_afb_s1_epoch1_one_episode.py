#!/usr/bin/env python3
"""Run one AFB S1 head-camera episode with an AFB S1 Cosmos-Predict2.5 checkpoint."""

import argparse
import io
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mediapy
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TVF


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    "/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/checkpoints/checkpoint-epoch1-step4277"
)
DEFAULT_EPISODE = Path(
    "/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee"
    "/demo_clean_zed2i_visible/click_alarmclock/data/episode0.hdf5"
)
DEFAULT_OUTPUT = Path("/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/infer_results/epoch1_step4277_one_episode")

EXPERIMENT = "cosmos_predict2p5_2B_afb_s1_family_balanced_3_1_1_1_chunk16_headcam"
CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
CHUNK_SIZE = 16
RESOLUTION = "256,320"
NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of "
    "poor quality."
)

TASK_CAPTIONS = {
    "click_alarmclock": "Press the top button of the alarm clock.",
    "click_bell": "Press the bell button.",
    "place_object_basket": "Place the object into the basket.",
    "open_laptop": "Open the laptop.",
    "stack_blocks_two": "Stack the two blocks.",
}


def import_h5py():
    try:
        import h5py

        return h5py
    except ModuleNotFoundError:
        extra = os.environ.get("H5PY_EXTRA_PATH", "/mnt/gyc/envs/cosmos-policy/lib/python3.10/site-packages")
        if extra and Path(extra).exists() and extra not in sys.path:
            sys.path.append(extra)
        import h5py

        return h5py


def task_from_episode(path: Path) -> str:
    # .../<task>/data/episodeX.hdf5
    try:
        return path.parent.parent.name
    except Exception:
        return "click_alarmclock"


def read_expert_episode(path: Path):
    h5py = import_h5py()
    with h5py.File(path, "r") as f:
        rgb = f["observation/head_camera/rgb"]
        frames = [imageio.imread(io.BytesIO(bytes(rgb[i]))) for i in range(len(rgb))]
        actions = np.asarray(f["delta_ee_action/vector"][:], dtype=np.float32)
    return np.stack(frames), actions


def resize_frame(frame: np.ndarray, size=(256, 320)) -> np.ndarray:
    tensor = torch.from_numpy(frame).permute(2, 0, 1)
    tensor = TVF.resize(tensor, list(size), antialias=True)
    return tensor.permute(1, 2, 0).to(torch.uint8).cpu().numpy()


def make_condition_tensor(frame: np.ndarray, num_frames: int = CHUNK_SIZE + 1) -> torch.Tensor:
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(torch.uint8)
    video = torch.zeros((num_frames, 3, frame_tensor.shape[1], frame_tensor.shape[2]), dtype=torch.uint8)
    video[0] = frame_tensor
    return video.unsqueeze(0).permute(0, 2, 1, 3, 4)


def scale_actions(actions: np.ndarray) -> np.ndarray:
    scaler = np.array([20.0] * 6 + [1.0] + [20.0] * 6 + [1.0], dtype=np.float32)
    return actions.astype(np.float32) * scaler


def to_uint8_video(video: torch.Tensor) -> np.ndarray:
    frames = (video[0].permute(1, 2, 3, 0).clamp(-1, 1) / 2 + 0.5) * 255.0
    return frames.to(torch.uint8).cpu().numpy()


def write_side_by_side(gt: np.ndarray, pred: np.ndarray, path: Path, fps: int):
    n = min(len(gt), len(pred))
    both = np.concatenate([gt[:n], pred[:n]], axis=2)
    mediapy.write_video(str(path), both, fps=fps)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=22)
    parser.add_argument("--single-chunk", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only validate data/checkpoint paths and tensor shapes.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames, raw_actions = read_expert_episode(args.episode)
    task = task_from_episode(args.episode)
    prompt = TASK_CAPTIONS.get(task, task)
    frames = np.stack([resize_frame(frame) for frame in frames])
    n_actions = min(raw_actions.shape[0], frames.shape[0] - 1)
    actions = scale_actions(raw_actions[:n_actions])

    print(f"[INFO] Experiment: {EXPERIMENT}")
    print(f"[INFO] Checkpoint:  {args.checkpoint}")
    print(f"[INFO] Episode:     {args.episode}")
    print(f"[INFO] Task/prompt: {task} / {prompt}")
    print(f"[INFO] Frames:      {frames.shape}")
    print(f"[INFO] Actions:     raw={raw_actions.shape}, used={actions.shape}, chunk={CHUNK_SIZE}")
    print(f"[INFO] Output:      {args.output}")

    gt_path = args.output / "gt_headcam.mp4"
    mediapy.write_video(str(gt_path), frames[: n_actions + 1], fps=args.fps)
    print(f"[INFO] Saved GT:    {gt_path}")

    if args.dry_run:
        print("[DONE] Dry run passed.")
        return

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    model = Video2WorldInference(
        experiment_name=EXPERIMENT,
        ckpt_path=str(args.checkpoint),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )

    pred_frames = [frames[0]]
    current = frames[0]
    total_chunks = (n_actions + CHUNK_SIZE - 1) // CHUNK_SIZE
    if args.single_chunk:
        total_chunks = 1

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        chunk = actions[start : start + CHUNK_SIZE]
        if chunk.shape[0] < CHUNK_SIZE:
            pad = np.zeros((CHUNK_SIZE - chunk.shape[0], actions.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        print(f"[INFO] Generating chunk {chunk_idx + 1}/{total_chunks}, action rows {start}:{start + CHUNK_SIZE}")
        result = model.generate_vid2world(
            prompt=prompt,
            input_path=make_condition_tensor(current),
            action=torch.from_numpy(chunk).float(),
            guidance=args.guidance,
            num_video_frames=CHUNK_SIZE + 1,
            num_latent_conditional_frames=1,
            resolution=RESOLUTION,
            seed=args.seed + start,
            negative_prompt=NEGATIVE_PROMPT,
            num_steps=args.num_steps,
        )
        generated = to_uint8_video(result)
        needed = min(CHUNK_SIZE, n_actions - start)
        pred_frames.extend(list(generated[1 : needed + 1]))
        current = generated[needed]
        torch.cuda.empty_cache()

    pred = np.stack(pred_frames[: n_actions + 1])
    pred_path = args.output / "pred_headcam.mp4"
    sbs_path = args.output / "gt_vs_pred_headcam.mp4"
    mediapy.write_video(str(pred_path), pred, fps=args.fps)
    write_side_by_side(frames[: len(pred)], pred, sbs_path, fps=args.fps)
    print(f"[INFO] Saved pred:  {pred_path}")
    print(f"[INFO] Saved sbs:   {sbs_path}")
    print("[DONE] Inference complete.")


if __name__ == "__main__":
    main()
