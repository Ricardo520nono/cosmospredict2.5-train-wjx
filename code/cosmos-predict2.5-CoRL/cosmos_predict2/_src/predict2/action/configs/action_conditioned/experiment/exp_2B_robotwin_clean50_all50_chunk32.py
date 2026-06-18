# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RobotWin2.0 clean50 全量 50 任务 experiment config for CosmosPredict2.5-2B action-conditioned.

继承 Task 3 (exp_2B_robotwin_clean50.py) 的所有修复，数据换成 all50 合并目录。

8-GPU training:
    bash scripts/train_cosmos_8gpu_all50.sh
"""

import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, DistributedSampler
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

# ---- dataset paths ----
_BASE_ALL50 = "/mnt/gyc_ckp/datasets/robotwin_clean50_all50"
_VIDEO_BASE  = "/mnt/gyc_ckp/datasets/robotwin_clean50"

_COMMON = dict(
    train_annotation_path=os.path.join(_BASE_ALL50, "annotation/train"),
    val_annotation_path=os.path.join(_BASE_ALL50, "annotation/val"),
    test_annotation_path=os.path.join(_BASE_ALL50, "annotation/val"),
    video_path=_VIDEO_BASE,
    fps_downsample_ratio=1,
    num_action_per_chunk=32,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    state_key="state",
    gripper_key="continuous_gripper_state",
    gripper_rescale_factor=1.0,
)

all50_train_dataset = L(Dataset_3D)(mode="train", **_COMMON)
all50_val_dataset   = L(Dataset_3D)(mode="val",   **_COMMON)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


all50_train_dataloader = L(DataLoader)(
    dataset=all50_train_dataset,
    sampler=L(get_sampler)(dataset=all50_train_dataset),
    batch_size=1,
    drop_last=True,
)
all50_val_dataloader = L(DataLoader)(
    dataset=all50_val_dataset,
    sampler=L(get_sampler)(dataset=all50_val_dataset),
    batch_size=1,
    drop_last=True,
)

# ---- experiment config ----
_LOCAL_CKPT = (
    "/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/robot/action-cond/"
    "38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
)

COSMOS_PREDICT2P5_2B_ROBOTWIN_ALL50_CLEAN50 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            {"override /data_train": "robotwin_clean50_all50_train"},
            {"override /data_val":   "robotwin_clean50_all50_val"},
            {"override /callbacks":  ["basic", "wandb"]},
        ],
        job=dict(
            group="robotwin_clean50_all50",
            name="cosmos_predict2p5_2B_robotwin_all50_clean50_14D_chunk32",
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
            straggler_detection=dict(enabled=False),
        ),
        upload_reproducible_setup=False,
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1,
                    num_action_per_chunk=32,
                    fps_downsample_ratio=1,
                    video_size=[256, 320],
                )
            ),
            dataset=dict(
                gripper_rescale_factor=1,
                num_action_per_chunk=32,
                fps_downsample_ratio=1,
                video_size=[256, 320],
            ),
        ),
        model=dict(
            config=dict(
                state_t=1 + 32 // 4,
                net=dict(
                    action_dim=14,
                    num_action_per_chunk=32,
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


def register_robotwin_all50_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="robotwin_clean50_all50_train",
        node=all50_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="robotwin_clean50_all50_val",
        node=all50_val_dataloader,
    )


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=COSMOS_PREDICT2P5_2B_ROBOTWIN_ALL50_CLEAN50["job"]["name"],
    node=COSMOS_PREDICT2P5_2B_ROBOTWIN_ALL50_CLEAN50,
)
