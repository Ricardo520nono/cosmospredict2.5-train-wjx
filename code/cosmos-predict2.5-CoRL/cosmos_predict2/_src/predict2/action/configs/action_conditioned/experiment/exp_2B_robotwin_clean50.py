# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RobotWin2.0 clean50 (pcp+pob) experiment config for CosmosPredict2.5-2B action-conditioned.

Inherits from AC_CHUNK_MULTI_VIEW_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_13FRAME_256X320,
overrides checkpoint to local action-conditioned weights, and points data to our converted dataset.

Smoke run:
    cd /mnt/gyc/cosmos-predict2.5-CoRL
    torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
      --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
      -- experiment=cosmos_predict2p5_2B_robotwin_pcp_pob_clean50 ~dataloader_train.dataloaders

8-GPU training:
    see scripts/train_cosmos_8gpu_pcp_pob.sh
"""

import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, DistributedSampler
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

# ---- dataset paths ----
# pcp+pob merged: /mnt/gyc_ckp/datasets/robotwin_clean50_pcp_pob (90 train / 10 val)
# video base: /mnt/gyc_ckp/datasets/robotwin_clean50 (video_path prefixed with task name)
_MERGED = "/mnt/gyc_ckp/datasets/robotwin_clean50_pcp_pob"
_VIDEO_BASE = "/mnt/gyc_ckp/datasets/robotwin_clean50"

_TRAIN_ANN = os.path.join(_MERGED, "annotation/train")
_VAL_ANN   = os.path.join(_MERGED, "annotation/val")

_COMMON = dict(
    train_annotation_path=_TRAIN_ANN,
    val_annotation_path=_VAL_ANN,
    test_annotation_path=_VAL_ANN,
    video_path=_VIDEO_BASE,
    fps_downsample_ratio=1,
    num_action_per_chunk=16,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    state_key="state",
    gripper_key="continuous_gripper_state",
    gripper_rescale_factor=1.0,
)

robotwin_train_dataset = L(Dataset_3D)(mode="train", **_COMMON)
robotwin_val_dataset   = L(Dataset_3D)(mode="val",   **_COMMON)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


robotwin_train_dataloader = L(DataLoader)(
    dataset=robotwin_train_dataset,
    sampler=L(get_sampler)(dataset=robotwin_train_dataset),
    batch_size=1,
    drop_last=True,
)
robotwin_val_dataloader = L(DataLoader)(
    dataset=robotwin_val_dataset,
    sampler=L(get_sampler)(dataset=robotwin_val_dataset),
    batch_size=1,
    drop_last=True,
)


# ---- experiment config ----
_LOCAL_CKPT = (
    "/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/robot/action-cond/"
    "38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
)

COSMOS_PREDICT2P5_2B_ROBOTWIN_PCP_POB_CLEAN50 = LazyDict(
    dict(
        defaults=[
            # inherit from bridge 256x320 action-conditioned experiment
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            # override data to our robotwin dataset
            {"override /data_train": "robotwin_clean50_train"},
            {"override /data_val":   "robotwin_clean50_val"},
            # basic + wandb (no viz sampling, no cluster_speed)
            {"override /callbacks": ["basic", "wandb"]},
        ],
        job=dict(
            group="robotwin_clean50",
            name="cosmos_predict2p5_2B_robotwin_pcp_pob_clean50_14D_chunk16",
            project="cosmos_predict2_action_conditioned_robotwin",
        ),
        checkpoint=dict(
            save_iter=5000,
            load_path=_LOCAL_CKPT,
            load_training_state=False,
            strict_resume=False,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=60000,
            logging_iter=20,
        ),
        upload_reproducible_setup=False,
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1,
                    num_action_per_chunk=16,
                    fps_downsample_ratio=1,
                    video_size=[256, 320],
                )
            ),
            dataset=dict(
                gripper_rescale_factor=1,
                num_action_per_chunk=16,
                fps_downsample_ratio=1,
                video_size=[256, 320],
            ),
        ),
        model=dict(
            config=dict(
                state_t=1 + 16 // 4,
                net=dict(
                    action_dim=14,
                    num_action_per_chunk=16,
                    temporal_compression_ratio=4,
                    use_crossattn_projection=False,
                ),
                tokenizer=dict(
                    vae_pth="/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/tokenizer.pth",
                ),
                text_encoder_config=dict(
                    compute_online=False,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


def register_robotwin_clean50_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="robotwin_clean50_train",
        node=robotwin_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="robotwin_clean50_val",
        node=robotwin_val_dataloader,
    )


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=COSMOS_PREDICT2P5_2B_ROBOTWIN_PCP_POB_CLEAN50["job"]["name"],
    node=COSMOS_PREDICT2P5_2B_ROBOTWIN_PCP_POB_CLEAN50,
)
