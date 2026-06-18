#!/usr/bin/env python3
"""
Inference: CosmosPredict2.5 AFB delta_ee pcp+pob, iter_000035000.
Autoregressive rollout with GT actions. GT and pred saved separately.

Usage:
    cd /mnt/gyc/cosmos-predict2.5-CoRL
    VENV_CUDNN="/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
    LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}" \
    PYTHONPATH=".:packages/cosmos-cuda" \
    /mnt/gyc_wjx/cosmos-predict2.5/.venv/bin/python3 scripts/infer_cosmos_afb_pcp_pob.py
"""

import json
import os
import shutil
from pathlib import Path

import mediapy
import numpy as np
import torch

# ── config ─────────────────────────────────────────────────────────────────────
CKPT_PATH   = (
    "/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin"
    "/afb_delta_ee_pcp_pob/cosmos_predict2p5_2B_afb_delta_ee_pcp_pob"
    "/checkpoints/iter_000060000"
)
EXPERIMENT  = "cosmos_predict2p5_2B_afb_delta_ee_pcp_pob"
CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
DATA_BASE   = Path("/mnt/gyc_ckp/datasets/afb_delta_ee_pcp_pob")
SAVE_ROOT   = Path("/mnt/gyc_ckp/infer_results/afb_pcp_pob_60k")
CHUNK_SIZE  = 12
NUM_STEPS   = 35
GUIDANCE    = 7
RESOLUTION  = "256,320"   # H,W (model training resolution)
SAVE_FPS    = 10

EPISODES = [
    {
        "task": "place_container_plate",
        "ann":  DATA_BASE / "place_container_plate/annotation/val/45.json",
        "src_mp4": Path("/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench"
                        "/data_delta_ee/demo_clean_zed2i_visible"
                        "/place_container_plate/video/episode45.mp4"),
    },
    {
        "task": "place_object_basket",
        "ann":  DATA_BASE / "place_object_basket/annotation/val/45.json",
        "src_mp4": Path("/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench"
                        "/data_delta_ee/demo_clean_zed2i_visible"
                        "/place_object_basket/video/episode45.mp4"),
    },
]


def run():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    print(f"[INFO] Loading model: {EXPERIMENT}")
    print(f"[INFO] Checkpoint:    {CKPT_PATH}")
    model = Video2WorldInference(
        experiment_name=EXPERIMENT,
        ckpt_path=CKPT_PATH,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    print("[INFO] Model loaded.\n")

    for ep in EPISODES:
        task    = ep["task"]
        src_mp4 = ep["src_mp4"]
        ann_path = ep["ann"]

        print(f"[INFO] === {task} ===")

        # ── GT: copy original mp4 ────────────────────────────────────────────
        gt_out = SAVE_ROOT / f"{task}_gt.mp4"
        shutil.copy2(src_mp4, gt_out)
        print(f"[INFO] GT saved: {gt_out}")

        # ── load annotation ──────────────────────────────────────────────────
        with open(ann_path) as f:
            ann = json.load(f)

        prompt  = (ann.get("texts") or [""])[0] or ann.get("task", "")
        actions = np.array(ann["state"])   # (T,7) already scaled delta
        print(f"[INFO] Prompt: {prompt[:80]}")
        print(f"[INFO] Actions: {len(actions)} steps, {CHUNK_SIZE} per chunk → {len(actions)//CHUNK_SIZE} chunks")

        # ── condition frame: first frame of source mp4 ───────────────────────
        video_gt = mediapy.read_video(str(src_mp4))   # (T,H,W,3) uint8
        condition_frame_path = str(SAVE_ROOT / f"{task}_condition.png")
        import PIL.Image
        PIL.Image.fromarray(video_gt[0]).save(condition_frame_path)

        # ── autoregressive rollout ────────────────────────────────────────────
        pred_frames = []
        current_input = condition_frame_path   # first chunk uses image path

        for i in range(0, len(actions) - 1, CHUNK_SIZE):
            chunk_actions = actions[i : i + CHUNK_SIZE]
            if len(chunk_actions) < CHUNK_SIZE:
                pad = np.zeros((CHUNK_SIZE - len(chunk_actions), 7))
                chunk_actions = np.concatenate([chunk_actions, pad], axis=0)

            action_tensor = torch.from_numpy(chunk_actions).float()  # (12,7)

            result = model.generate_vid2world(
                prompt=prompt,
                input_path=current_input,
                guidance=GUIDANCE,
                num_video_frames=CHUNK_SIZE + 1,   # 1 cond + 12 future
                num_latent_conditional_frames=1,
                resolution=RESOLUTION,
                seed=42 + i,
                action=action_tensor,
                num_steps=NUM_STEPS,
            )  # (1, C, T, H, W) in [-1, 1]

            # decode to uint8 (T,H,W,3)
            frames = (result[0].permute(1, 2, 3, 0).clamp(-1, 1) / 2 + 0.5) * 255
            frames = frames.to(torch.uint8).cpu().numpy()

            pred_frames.extend(list(frames[1:]))   # skip condition frame

            # next chunk: use last generated frame as condition (save as tmp png)
            last_frame = PIL.Image.fromarray(frames[-1])
            tmp_path = str(SAVE_ROOT / f"{task}_tmp_cond.png")
            last_frame.save(tmp_path)
            current_input = tmp_path

            print(f"  chunk {i//CHUNK_SIZE+1}: +{len(frames)-1} frames (total {len(pred_frames)})")

        # ── save pred video ───────────────────────────────────────────────────
        pred_out = SAVE_ROOT / f"{task}_pred.mp4"
        mediapy.write_video(str(pred_out), pred_frames, fps=SAVE_FPS)
        print(f"[INFO] Pred saved: {pred_out} ({len(pred_frames)} frames)\n")

    # cleanup tmp files
    for ep in EPISODES:
        for f in [SAVE_ROOT / f"{ep['task']}_condition.png",
                  SAVE_ROOT / f"{ep['task']}_tmp_cond.png"]:
            if f.exists():
                f.unlink()

    print(f"[DONE] Results in {SAVE_ROOT}/")
    print(f"  GT:   {SAVE_ROOT}/<task>_gt.mp4")
    print(f"  Pred: {SAVE_ROOT}/<task>_pred.mp4")


if __name__ == "__main__":
    run()
