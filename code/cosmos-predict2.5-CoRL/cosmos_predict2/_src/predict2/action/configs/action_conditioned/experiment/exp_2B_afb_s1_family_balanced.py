# SPDX-License-Identifier: Apache-2.0

"""
ActionFollowingBench S1 family-balanced training.

Choices:
- tasks: 5 fixed S1 tasks
- view: head/high camera only
- chunk: 16 delta-ee actions
- family mix: expert:pca:raw:random-feasible = 3:1:1:1
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.embedding_concat_strategy import EmbeddingConcatStrategy
from cosmos_predict2._src.predict2.action.datasets.dataset_afb_s1_family_balanced import AFBS1FamilyBalancedDataset

_DATA_ROOT = os.environ.get(
    "AFB_DATA_ROOT", "/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench"
)
_EXPERT_ROOT = os.environ.get("AFB_EXPERT_ROOT", os.path.join(_DATA_ROOT, "data_delta_ee/demo_clean_zed2i_visible"))
_ENHANCED_LEROBOT_ROOT = (
    os.environ.get(
        "AFB_ENHANCED_LEROBOT_ROOT",
        os.path.join(_DATA_ROOT, "data_lerobot/robotwin_delta_ee/_enhanced_reconvert_wjx5_20260607"),
    )
)
_RF_ROOT = (
    os.environ.get(
        "AFB_RF_ROOT",
        os.path.join(
            _DATA_ROOT,
            "EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk",
        ),
    )
)

_CHUNK = 16
_VIRTUAL_EPOCH_SIZE = 68428
_PER_GPU_BATCH = int(os.environ.get("AFB_S1_PER_GPU_BATCH", "2"))

_COMMON = dict(
    expert_root=_EXPERT_ROOT,
    enhanced_lerobot_root=_ENHANCED_LEROBOT_ROOT,
    random_feasible_root=_RF_ROOT,
    num_action_per_chunk=_CHUNK,
    video_size=[256, 320],
    gripper_rescale_factor=1.0,
    load_t5_embeddings=False,
)

afb_s1_train_dataset = L(AFBS1FamilyBalancedDataset)(mode="train", epoch_size=_VIRTUAL_EPOCH_SIZE, **_COMMON)
afb_s1_val_dataset = L(AFBS1FamilyBalancedDataset)(mode="val", epoch_size=1000, **_COMMON)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


afb_s1_train_dataloader = L(DataLoader)(
    dataset=afb_s1_train_dataset,
    sampler=L(get_sampler)(dataset=afb_s1_train_dataset),
    batch_size=_PER_GPU_BATCH,
    drop_last=True,
)
afb_s1_val_dataloader = L(DataLoader)(
    dataset=afb_s1_val_dataset,
    sampler=L(get_sampler)(dataset=afb_s1_val_dataset),
    batch_size=1,
    drop_last=True,
)

_TRAIN_ROOT = os.environ["COSMOS_TRAIN_ROOT"]
_MODEL_ROOT = os.path.join(_TRAIN_ROOT, "models")
_PREDICT2_MODEL_ROOT = os.path.join(_MODEL_ROOT, "Cosmos-Predict2.5-2B")
_LOCAL_CKPT = os.path.join(
    _PREDICT2_MODEL_ROOT,
    "robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt",
)
_TOKENIZER = os.path.join(_PREDICT2_MODEL_ROOT, "tokenizer.pth")
_REASON1_CKPT = os.path.join(_MODEL_ROOT, "Cosmos-Reason1-7B")

COSMOS_PREDICT2P5_2B_AFB_S1_FAMILY_BALANCED_CHUNK16 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            {"override /data_train": "afb_s1_family_balanced_train"},
            {"override /data_val": "afb_s1_family_balanced_val"},
            {"override /callbacks": ["basic", "wandb"]},
        ],
        job=dict(
            group="afb_s1_family_balanced",
            name="cosmos_predict2p5_2B_afb_s1_family_balanced_3_1_1_1_chunk16_headcam",
            project="cosmos_predict2_action_conditioned_robotwin",
        ),
        checkpoint=dict(
            save_iter=int(os.environ.get("AFB_S1_EPOCH_STEP", "4277")),
            load_path=_LOCAL_CKPT,
            load_training_state=False,
            strict_resume=False,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        trainer=dict(
            callbacks=dict(
                device_monitor=dict(log_memory_detail=False, upload_every_n_mul=100),
            ),
            max_iter=int(os.environ.get("AFB_S1_MAX_ITER", "40000")),
            logging_iter=20,
            run_validation=False,
            straggler_detection=dict(enabled=False),
        ),
        upload_reproducible_setup=False,
        dataloader_train=dict(
            batch_size=_PER_GPU_BATCH,
            sampler=dict(dataset=dict(num_action_per_chunk=_CHUNK)),
            dataset=dict(num_action_per_chunk=_CHUNK),
        ),
        dataloader_val=dict(
            sampler=dict(dataset=dict(num_action_per_chunk=_CHUNK)),
            dataset=dict(num_action_per_chunk=_CHUNK),
        ),
        model=dict(
            config=dict(
                state_t=1 + _CHUNK // 4,
                ee_head=dict(
                    enabled=True,
                    loss_weight=float(os.environ.get("AFB_S1_EE_LOSS_WEIGHT", "0.05")),
                    position_loss_weight=1.0,
                    rotation_6d_loss_weight=1.0,
                    gripper_loss_weight=1.0,
                ),
                net=dict(
                    action_dim=14,
                    num_action_per_chunk=_CHUNK,
                    temporal_compression_ratio=4,
                    ee_head_enabled=True,
                    ee_head_num_frames=_CHUNK,
                    ee_head_latent_frames=1 + _CHUNK // 4,
                    ee_head_hidden_dim=int(os.environ.get("AFB_S1_EE_HEAD_HIDDEN_DIM", "1024")),
                    ee_head_dropout=0.0,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                ),
                tokenizer=dict(vae_pth=_TOKENIZER),
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path=_REASON1_CKPT,
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


def register_afb_s1_family_balanced_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="afb_s1_family_balanced_train",
        node=afb_s1_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="afb_s1_family_balanced_val",
        node=afb_s1_val_dataloader,
    )


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=COSMOS_PREDICT2P5_2B_AFB_S1_FAMILY_BALANCED_CHUNK16["job"]["name"],
    node=COSMOS_PREDICT2P5_2B_AFB_S1_FAMILY_BALANCED_CHUNK16,
)
