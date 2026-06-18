#!/usr/bin/env python3
"""
Action Following 推理脚本（修正版）。

同一初始帧 + 3 条不同 action，各自完整 autoregressive rollout → 3 个连贯 mp4。
chunk 间传递上一 chunk 最后一帧作为下一 chunk 的 condition（标准自回归）。

Usage:
    cd /mnt/gyc/cosmos-predict2.5-CoRL
    VENV_CUDNN="/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
    LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PYTHONPATH=".:packages/cosmos-cuda" \
    /mnt/gyc_wjx/cosmos-predict2.5/.venv/bin/python3 scripts/infer_action_following.py
"""

import json
from pathlib import Path

import mediapy
import numpy as np
import PIL.Image
import torch

# ── config ─────────────────────────────────────────────────────────────────────
CKPT_PATH = (
    "/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin"
    "/afb_delta_ee_pcp_pob/cosmos_predict2p5_2B_afb_delta_ee_pcp_pob"
    "/checkpoints/iter_000060000"
)
EXPERIMENT  = "cosmos_predict2p5_2B_afb_delta_ee_pcp_pob"
CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
DATA_BASE   = Path("/mnt/gyc_ckp/datasets/afb_delta_ee_pcp_pob")
SAVE_ROOT   = Path("/mnt/gyc_ckp/infer_results/action_following")
CHUNK_SIZE  = 12
NUM_STEPS   = 35
GUIDANCE    = 1
RESOLUTION  = "256,320"
SAVE_FPS    = 10

INIT_MP4 = Path(
    "/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench"
    "/data_delta_ee/demo_clean_zed2i_visible"
    "/place_container_plate/video/episode45.mp4"
)

EPISODES = [
    {
        "gid":  "gid1_pcp_ep45",
        "ann":  DATA_BASE / "place_container_plate/annotation/val/45.json",
    },
    {
        "gid":  "gid2_pcp_ep46",
        "ann":  DATA_BASE / "place_container_plate/annotation/val/46.json",
    },
    {
        "gid":  "gid3_pob_ep45",
        "ann":  DATA_BASE / "place_object_basket/annotation/val/45.json",
    },
]


def run():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    print(f"[INFO] Loading model: {EXPERIMENT}")
    model = Video2WorldInference(
        experiment_name=EXPERIMENT,
        ckpt_path=CKPT_PATH,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    print("[INFO] Model loaded.\n")

    # 保存共用初始帧
    init_video = mediapy.read_video(str(INIT_MP4))
    init_frame_png = str(SAVE_ROOT / "init_frame_pcp_ep45.png")
    PIL.Image.fromarray(init_video[0]).save(init_frame_png)
    print(f"[INFO] Initial frame saved: {init_frame_png}\n")

    for ep in EPISODES:
        gid      = ep["gid"]
        ann_path = ep["ann"]

        print(f"[INFO] === {gid} ===")

        with open(ann_path) as f:
            ann = json.load(f)

        prompt  = (ann.get("texts") or [""])[0] or ann.get("task", "")
        actions = np.array(ann["state"])  # (T, 7)
        n_chunks = (len(actions) - 1) // CHUNK_SIZE
        print(f"[INFO] Prompt: {prompt[:80]}")
        print(f"[INFO] Actions: {len(actions)} steps → {n_chunks} chunks")

        pred_frames = []
        # 第一个 chunk 用初始帧，后续 chunk 用上一 chunk 的最后一帧（标准自回归）
        current_input = init_frame_png
        tmp_cond_png = str(SAVE_ROOT / f"{gid}_tmp_cond.png")

        for i in range(0, len(actions) - 1, CHUNK_SIZE):
            chunk_actions = actions[i: i + CHUNK_SIZE]
            if len(chunk_actions) < CHUNK_SIZE:
                pad = np.zeros((CHUNK_SIZE - len(chunk_actions), 7))
                chunk_actions = np.concatenate([chunk_actions, pad], axis=0)

            action_tensor = torch.from_numpy(chunk_actions).float()

            result = model.generate_vid2world(
                prompt=prompt,
                input_path=current_input,
                guidance=GUIDANCE,
                num_video_frames=CHUNK_SIZE + 1,
                num_latent_conditional_frames=1,
                resolution=RESOLUTION,
                seed=42 + i,
                action=action_tensor,
                num_steps=NUM_STEPS,
            )  # (1, C, T, H, W) in [-1, 1]

            frames = (result[0].permute(1, 2, 3, 0).clamp(-1, 1) / 2 + 0.5) * 255
            frames = frames.to(torch.uint8).cpu().numpy()
            pred_frames.extend(list(frames[1:]))  # skip condition frame

            # 用本 chunk 最后一帧作为下一 chunk 的 condition
            PIL.Image.fromarray(frames[-1]).save(tmp_cond_png)
            current_input = tmp_cond_png

            print(f"  chunk {i//CHUNK_SIZE+1}/{n_chunks}: +{len(frames)-1} frames (total {len(pred_frames)})")

        # 保存视频
        out_path = SAVE_ROOT / f"{gid}.mp4"
        mediapy.write_video(str(out_path), pred_frames, fps=SAVE_FPS)
        print(f"[INFO] Saved: {out_path} ({len(pred_frames)} frames)\n")

        # cleanup tmp
        Path(tmp_cond_png).unlink(missing_ok=True)

    print(f"[DONE] All 3 videos in {SAVE_ROOT}/")


if __name__ == "__main__":
    run()
