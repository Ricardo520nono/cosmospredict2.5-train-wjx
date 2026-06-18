"""
Task 5: autoregressive rollout inference on pcp + pob val episode 45.

Loads the 10k-iter checkpoint trained in Task 3, runs chunk-wise autoregressive
rollout over the full trajectory using GT actions, and saves GT-vs-pred side-by-side
mp4s to results/inference_task5/.

Usage (single GPU, from repo root):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/gyc/cosmos-predict2.5-CoRL:packages/cosmos-cuda \
    python scripts/infer_cosmos_pcp_pob.py
"""

import json
import os
import sys

import mediapy
import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
REPO_ROOT = "/mnt/gyc/cosmos-predict2.5-CoRL"
CKPT_PATH = (
    "/mnt/gyc_ckp/cosmos_train_output"
    "/cosmos_predict2_action_conditioned_robotwin"
    "/robotwin_clean50"
    "/cosmos_predict2p5_2B_robotwin_pcp_pob_clean50"
    "/checkpoints/iter_000010000"
)
DATASET_ROOT = os.path.join(REPO_ROOT, "datasets/robotwin_clean50")
SAVE_ROOT = os.path.join(REPO_ROOT, "results/inference_task5")

EPISODES = [
    {
        "name": "pcp",
        "ann_json": os.path.join(
            DATASET_ROOT, "place_container_plate/annotation/val/45.json"
        ),
        "video_base": os.path.join(DATASET_ROOT, "place_container_plate"),
    },
    {
        "name": "pob",
        "ann_json": os.path.join(
            DATASET_ROOT, "place_object_basket/annotation/val/45.json"
        ),
        "video_base": os.path.join(DATASET_ROOT, "place_object_basket"),
    },
]

EXPERIMENT_NAME = "cosmos_predict2p5_2B_robotwin_pcp_pob_clean50"
CONFIG_FILE = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"

CHUNK_SIZE = 12          # num_action_per_chunk
ACTION_SCALER = 20.0
GUIDANCE = 7
NUM_STEPS = 35
SAVE_FPS = 10
RESOLUTION = "none"      # use model's native resolution (256x320)

# ---------------------------------------------------------------------------
# helpers (mirror of action_conditioned.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "packages/cosmos-cuda"))

from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler


def compute_actions(state: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Compute delta-ee-pose 7D actions from consecutive states."""
    T = len(state)
    actions = np.zeros((T - 1, 7), dtype=np.float32)
    for k in range(1, T):
        prev_xyz = state[k - 1, 0:3]
        prev_rpy = state[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = state[k, 0:3]
        curr_rpy = state[k, 3:6]
        curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = prev_rotm.T @ (curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm
        rel_euler = rotm2euler(rel_rotm)
        actions[k - 1, 0:3] = rel_xyz
        actions[k - 1, 3:6] = rel_euler
        actions[k - 1, 6] = gripper[k]
    scale = np.array([ACTION_SCALER] * 6 + [1.0], dtype=np.float32)
    return actions * scale


def make_vid_input(img_array: np.ndarray, num_frames: int) -> torch.Tensor:
    """Pack a single HWC uint8 frame into (1, C, T, H, W) uint8 tensor."""
    img_t = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)  # (1,C,H,W)
    vid = torch.cat([img_t, torch.zeros_like(img_t).repeat(num_frames - 1, 1, 1, 1)], dim=0)
    vid = (vid * 255.0).to(torch.uint8)
    return vid.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1,C,T,H,W)


def decode_video(video_tensor: torch.Tensor) -> np.ndarray:
    """Convert model output (1,C,T,H,W) in [-1,1] → uint8 numpy (T,H,W,C)."""
    v = (video_tensor - (-1)) / 2.0          # → [0,1]
    v = (torch.clamp(v[0], 0, 1) * 255).to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()  # (T,H,W,C)


def side_by_side(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Concatenate GT and pred horizontally; pad shorter one if lengths differ."""
    T = max(len(gt), len(pred))
    H, W, C = gt.shape[1], gt.shape[2], gt.shape[3]
    out_gt   = np.zeros((T, H, W, C), dtype=np.uint8)
    out_pred = np.zeros((T, H, W, C), dtype=np.uint8)
    out_gt[: len(gt)]   = gt
    out_pred[: len(pred)] = pred
    return np.concatenate([out_gt, out_pred], axis=2)   # (T, H, 2W, C)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.chdir(REPO_ROOT)

    # experiment_opts mirrors the training overrides that disable text encoder
    experiment_opts = [
        "model.config.net.use_crossattn_projection=False",
        "model.config.text_encoder_config.compute_online=False",
        "trainer.straggler_detection.enabled=false",
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "upload_reproducible_setup=False",
        f"model.config.tokenizer.vae_pth=/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/tokenizer.pth",
    ]

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    # Monkey-patch _get_data_batch_input to inject zero T5 embeddings instead of
    # downloading T5-11B from HuggingFace (text encoder is disabled in our training config).
    _orig_get_data_batch = Video2WorldInference._get_data_batch_input

    def _patched_get_data_batch(self, video, prompt, num_conditional_frames=1,
                                negative_prompt="", use_neg_prompt=True,
                                camera=None, action=None):
        B, C, T, H, W = video.shape
        # T5-11B hidden_size=1024, max_length=512
        zero_emb = torch.zeros(1, 512, 1024, dtype=torch.bfloat16, device="cuda")
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "action": action.unsqueeze(0) if action is not None else None,
            "fps": torch.randint(16, 32, (self.batch_size,)).float(),
            "padding_mask": torch.zeros(self.batch_size, 1, H, W),
            "num_conditional_frames": num_conditional_frames,
            "t5_text_embeddings": zero_emb,
            "neg_t5_text_embeddings": zero_emb,
        }
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
        return data_batch

    Video2WorldInference._get_data_batch_input = _patched_get_data_batch

    print(f"[init] Loading model from {CKPT_PATH} …")
    video2world = Video2WorldInference(
        experiment_name=EXPERIMENT_NAME,
        ckpt_path=CKPT_PATH,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
        experiment_opts=experiment_opts,
    )
    print("[init] Model loaded.")

    for ep in EPISODES:
        ep_name = ep["name"]
        print(f"\n[{ep_name}] Loading annotation …")

        with open(ep["ann_json"]) as f:
            ann = json.load(f)

        state = np.array(ann["state"], dtype=np.float32)          # (T,7)
        gripper = np.array(ann["continuous_gripper_state"], dtype=np.float32)  # (T,)
        video_rel = ann["videos"][0]["video_path"]
        video_path = os.path.join(ep["video_base"], video_rel)

        gt_frames = mediapy.read_video(video_path)                 # (T,H,W,C) uint8
        T_gt = len(gt_frames)
        actions = compute_actions(state, gripper)                  # (T-1, 7)
        T_act = len(actions)

        print(f"[{ep_name}] T_gt={T_gt}, T_act={T_act}")

        # autoregressive rollout
        img_array = gt_frames[0]                                   # HWC uint8 seed frame
        pred_frames = [img_array]
        chunk_videos = []

        for i in range(0, T_act, CHUNK_SIZE):
            actions_chunk = actions[i: i + CHUNK_SIZE]
            if len(actions_chunk) == 0:
                break
            # pad last chunk if needed
            if len(actions_chunk) < CHUNK_SIZE:
                pad = np.zeros((CHUNK_SIZE - len(actions_chunk), 7), dtype=np.float32)
                actions_chunk = np.concatenate([actions_chunk, pad], axis=0)

            num_video_frames = CHUNK_SIZE + 1
            vid_input = make_vid_input(img_array, num_video_frames)

            print(f"[{ep_name}] chunk {i//CHUNK_SIZE}: frames {i}~{i+CHUNK_SIZE}")

            video = video2world.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=torch.from_numpy(actions_chunk).float(),
                guidance=GUIDANCE,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=1,
                resolution=RESOLUTION,
                seed=i,
                negative_prompt="",
                num_steps=NUM_STEPS,
            )

            chunk_np = decode_video(video)                         # (T,H,W,C)
            # chunk[0] is the conditioned frame (already in pred_frames), append the rest
            chunk_videos.append(chunk_np)
            pred_frames.append(chunk_np[-1])
            img_array = chunk_np[-1]                               # next seed frame

        # stitch predicted video: first chunk fully, subsequent chunks skip frame 0
        chunk_list = [chunk_videos[0]] + [c[1:] for c in chunk_videos[1:]]
        pred_video = np.concatenate(chunk_list, axis=0)            # (T',H,W,C)

        # trim / pad gt to same length as pred for fair comparison
        T_pred = len(pred_video)
        gt_trim = gt_frames[:T_pred] if T_pred <= T_gt else np.concatenate(
            [gt_frames, np.zeros((T_pred - T_gt, *gt_frames.shape[1:]), dtype=np.uint8)], axis=0
        )

        combined = side_by_side(gt_trim, pred_video)

        out_path = os.path.join(SAVE_ROOT, f"{ep_name}_gt_vs_pred.mp4")
        mediapy.write_video(out_path, combined, fps=SAVE_FPS)
        print(f"[{ep_name}] Saved → {out_path}  (GT | Pred, {len(combined)} frames)")

    print("\n[done] All episodes finished.")


if __name__ == "__main__":
    main()
