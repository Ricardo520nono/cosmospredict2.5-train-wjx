#!/usr/bin/env python3
"""
Debug: 测试不同 guidance 值对 afb 模型推理质量的影响。
只跑 gid1 的第一个 chunk（12帧），对比 4 种 guidance。

Usage:
    cd /mnt/gyc/cosmos-predict2.5-CoRL
    VENV_CUDNN="/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
    LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PYTHONPATH=".:packages/cosmos-cuda" \
    /mnt/gyc_wjx/cosmos-predict2.5/.venv/bin/python3 scripts/debug_guidance.py \
    2>&1 | tee /tmp/debug_guidance.log
"""

import json
from pathlib import Path

import mediapy
import numpy as np
import PIL.Image
import torch

CKPT_PATH = (
    "/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin"
    "/afb_delta_ee_pcp_pob/cosmos_predict2p5_2B_afb_delta_ee_pcp_pob"
    "/checkpoints/iter_000060000"
)
EXPERIMENT = "cosmos_predict2p5_2B_afb_delta_ee_pcp_pob"
CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
SAVE_ROOT = Path("/mnt/gyc_ckp/infer_results/action_following/debug_guidance")
CHUNK_SIZE = 12
NUM_STEPS = 35
RESOLUTION = "256,320"

INIT_MP4 = Path(
    "/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench"
    "/data_delta_ee/demo_clean_zed2i_visible"
    "/place_container_plate/video/episode45.mp4"
)
ANN_PATH = Path(
    "/mnt/gyc_ckp/datasets/afb_delta_ee_pcp_pob"
    "/place_container_plate/annotation/val/45.json"
)

GUIDANCE_VALUES = [1, 3, 5, 7]


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

    # 读初始帧
    init_video = mediapy.read_video(str(INIT_MP4))
    init_frame_png = str(SAVE_ROOT / "init_frame.png")
    PIL.Image.fromarray(init_video[0]).save(init_frame_png)

    # 读 action
    with open(ANN_PATH) as f:
        ann = json.load(f)
    prompt = (ann.get("texts") or [""])[0] or ann.get("task", "")
    actions = np.array(ann["state"])
    chunk_actions = actions[0:CHUNK_SIZE]
    action_tensor = torch.from_numpy(chunk_actions).float()

    print(f"[INFO] Prompt: {prompt[:80]}")
    print(f"[INFO] Chunk actions shape: {chunk_actions.shape}")
    print()

    # 对 GT 第一帧也做统计
    gt_frame = init_video[0]
    print(f"[REF] GT init frame: mean={gt_frame.mean():.1f}, std={gt_frame.std():.1f}, "
          f"saturated={((gt_frame == 0) | (gt_frame == 255)).sum() / gt_frame.size * 100:.1f}%")
    print()

    for guidance in GUIDANCE_VALUES:
        print(f"[INFO] === guidance={guidance} ===")

        result = model.generate_vid2world(
            prompt=prompt,
            input_path=init_frame_png,
            guidance=guidance,
            num_video_frames=CHUNK_SIZE + 1,
            num_latent_conditional_frames=1,
            resolution=RESOLUTION,
            seed=42,
            action=action_tensor,
            num_steps=NUM_STEPS,
        )

        # decode
        frames = (result[0].permute(1, 2, 3, 0).clamp(-1, 1) / 2 + 0.5) * 255
        frames = frames.to(torch.uint8).cpu().numpy()

        # 统计
        for i in [0, 6, 12]:
            if i >= len(frames):
                continue
            f = frames[i]
            sat = ((f == 0) | (f == 255)).sum() / f.size * 100
            print(f"  frame {i}: mean={f.mean():.1f}, std={f.std():.1f}, saturated={sat:.1f}%")

        # 保存视频
        out_path = SAVE_ROOT / f"guidance_{guidance}.mp4"
        mediapy.write_video(str(out_path), list(frames[1:]), fps=10)
        print(f"  Saved: {out_path}\n")

        # 也看看 raw model output 在 clamp 前有多少值超出 [-1, 1]
        raw = result[0].permute(1, 2, 3, 0).cpu().float().numpy()
        out_of_range = ((raw < -1) | (raw > 1)).sum() / raw.size * 100
        print(f"  Raw output out-of-range: {out_of_range:.1f}%")
        print(f"  Raw range: [{raw.min():.3f}, {raw.max():.3f}]")
        print()

    print("[DONE] All guidance values tested.")
    print(f"Results in {SAVE_ROOT}/")
    print("对比 guidance_1.mp4 vs guidance_7.mp4 看哪个正常")


if __name__ == "__main__":
    run()
