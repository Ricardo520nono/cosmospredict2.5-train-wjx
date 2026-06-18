# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ActionFollowingBench delta_ee pcp+pob experiment config.
head_camera (640x480), text condition enabled.
"""

import os
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader, DistributedSampler
from megatron.core import parallel_state
from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

_BASE     = "/mnt/gyc_ckp/datasets/afb_delta_ee_pcp_pob/merged"
_VID_BASE = "/mnt/gyc_ckp/datasets/afb_delta_ee_pcp_pob"

_COMMON = dict(
    train_annotation_path=os.path.join(_BASE, "annotation/train"),
    val_annotation_path=os.path.join(_BASE, "annotation/val"),
    test_annotation_path=os.path.join(_BASE, "annotation/val"),
    video_path=_VID_BASE,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    state_key="state",
    gripper_key="continuous_gripper_state",
    gripper_rescale_factor=1.0,
)

afb_pcp_pob_train_dataset = L(Dataset_3D)(mode="train", **_COMMON)
afb_pcp_pob_val_dataset   = L(Dataset_3D)(mode="val",   **_COMMON)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


afb_pcp_pob_train_dataloader = L(DataLoader)(
    dataset=afb_pcp_pob_train_dataset,
    sampler=L(get_sampler)(dataset=afb_pcp_pob_train_dataset),
    batch_size=1,
    drop_last=True,
)
afb_pcp_pob_val_dataloader = L(DataLoader)(
    dataset=afb_pcp_pob_val_dataset,
    sampler=L(get_sampler)(dataset=afb_pcp_pob_val_dataset),
    batch_size=1,
    drop_last=True,
)

_LOCAL_CKPT   = "/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
_REASON1_CKPT = "/mnt/gyc_ckp/models/Cosmos-Reason1-7B"

COSMOS_PREDICT2P5_2B_AFB_DELTA_EE_PCP_POB = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            {"override /data_train": "afb_delta_ee_pcp_pob_train"},
            {"override /data_val":   "afb_delta_ee_pcp_pob_val"},
            {"override /callbacks":  ["basic", "wandb"]},
        ],
        job=dict(
            group="afb_delta_ee_pcp_pob",
            name="cosmos_predict2p5_2B_afb_delta_ee_pcp_pob",
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
            sampler=dict(dataset=dict(gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320])),
            dataset=dict(gripper_rescale_factor=1, num_action_per_chunk=12, fps_downsample_ratio=1, video_size=[256, 320]),
        ),
        model=dict(
            config=dict(
                state_t=1 + 12 // 4,
                net=dict(action_dim=14, num_action_per_chunk=12, temporal_compression_ratio=4),
                tokenizer=dict(vae_pth="/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/tokenizer.pth"),
                text_encoder_config=dict(compute_online=True, ckpt_path=_REASON1_CKPT),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


def register_afb_delta_ee_pcp_pob_data():
    cs = ConfigStore.instance()
    cs.store(group="data_train", package="dataloader_train", name="afb_delta_ee_pcp_pob_train", node=afb_pcp_pob_train_dataloader)
    cs.store(group="data_val",   package="dataloader_val",   name="afb_delta_ee_pcp_pob_val",   node=afb_pcp_pob_val_dataloader)


cs = ConfigStore.instance()
cs.store(group="experiment", package="_global_", name=COSMOS_PREDICT2P5_2B_AFB_DELTA_EE_PCP_POB["job"]["name"], node=COSMOS_PREDICT2P5_2B_AFB_DELTA_EE_PCP_POB)
