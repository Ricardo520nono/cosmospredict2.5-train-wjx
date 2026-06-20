# SPDX-License-Identifier: Apache-2.0

"""ActionFollowingBench S1 single-task expert-only training."""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.embedding_concat_strategy import EmbeddingConcatStrategy
from cosmos_predict2._src.predict2.action.datasets.dataset_afb_s1_family_balanced import AFBS1ExpertSingleTaskDataset

_TASKS = [
    "click_alarmclock",
    "click_bell",
    "place_object_basket",
    "open_laptop",
    "stack_blocks_two",
]
_TASK = os.environ.get("AFB_S1_TASK", "click_alarmclock")
if _TASK not in _TASKS:
    raise ValueError(f"Unsupported AFB_S1_TASK={_TASK!r}. Expected one of {_TASKS}")

_DATA_ROOT = os.environ.get(
    "AFB_DATA_ROOT", "/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench"
)
_EXPERT_ROOT = os.environ.get("AFB_EXPERT_ROOT", os.path.join(_DATA_ROOT, "data_delta_ee/demo_clean_zed2i_visible"))
_CHUNK = 16
_PER_GPU_BATCH = int(os.environ.get("AFB_S1_PER_GPU_BATCH", "2"))
_EPOCH_SIZE_BY_TASK = {
    "click_alarmclock": 2708,
    "click_bell": 2474,
    "place_object_basket": 9093,
    "open_laptop": 7822,
    "stack_blocks_two": 11917,
}
_EPOCH_STEP_BY_TASK = {
    "click_alarmclock": 170,
    "click_bell": 155,
    "place_object_basket": 569,
    "open_laptop": 489,
    "stack_blocks_two": 745,
}
_TRAIN_EPOCH_SIZE = int(os.environ.get("AFB_S1_EPOCH_SIZE", str(_EPOCH_SIZE_BY_TASK[_TASK])))
_EPOCH_STEP = int(os.environ.get("AFB_S1_EPOCH_STEP", str(_EPOCH_STEP_BY_TASK[_TASK])))

_COMMON = dict(
    expert_root=_EXPERT_ROOT,
    task=_TASK,
    num_action_per_chunk=_CHUNK,
    video_size=[256, 320],
    gripper_rescale_factor=1.0,
    load_t5_embeddings=False,
)

afb_s1_expert_single_task_train_dataset = L(AFBS1ExpertSingleTaskDataset)(
    mode="train", epoch_size=_TRAIN_EPOCH_SIZE, **_COMMON
)
afb_s1_expert_single_task_val_dataset = L(AFBS1ExpertSingleTaskDataset)(mode="val", epoch_size=1000, **_COMMON)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


afb_s1_expert_single_task_train_dataloader = L(DataLoader)(
    dataset=afb_s1_expert_single_task_train_dataset,
    sampler=L(get_sampler)(dataset=afb_s1_expert_single_task_train_dataset),
    batch_size=_PER_GPU_BATCH,
    drop_last=True,
)
afb_s1_expert_single_task_val_dataloader = L(DataLoader)(
    dataset=afb_s1_expert_single_task_val_dataset,
    sampler=L(get_sampler)(dataset=afb_s1_expert_single_task_val_dataset),
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
_REASON1_CKPT = os.environ.get("COSMOS_REASON1_CKPT", os.path.join(_MODEL_ROOT, "Cosmos-Reason1-7B"))
_EXP_NAME = f"cosmos_predict2p5_2B_afb_s1_expert_only_{_TASK}_chunk16_headcam"

COSMOS_PREDICT2P5_2B_AFB_S1_EXPERT_SINGLE_TASK_CHUNK16 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_256x320",
            {"override /data_train": "afb_s1_expert_single_task_train"},
            {"override /data_val": "afb_s1_expert_single_task_val"},
            {"override /callbacks": ["basic", "wandb"]},
        ],
        job=dict(
            group="afb_s1_expert_single_task",
            name=_EXP_NAME,
            project="cosmos_predict2_action_conditioned_robotwin",
        ),
        checkpoint=dict(
            save_iter=_EPOCH_STEP,
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
            dataset=dict(num_action_per_chunk=_CHUNK, task=_TASK, epoch_size=_TRAIN_EPOCH_SIZE),
        ),
        dataloader_val=dict(
            sampler=dict(dataset=dict(num_action_per_chunk=_CHUNK)),
            dataset=dict(num_action_per_chunk=_CHUNK, task=_TASK),
        ),
        model=dict(
            config=dict(
                state_t=1 + _CHUNK // 4,
                ee_head=dict(enabled=False),
                net=dict(
                    action_dim=14,
                    num_action_per_chunk=_CHUNK,
                    temporal_compression_ratio=4,
                    ee_head_enabled=False,
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


def register_afb_s1_expert_single_task_data():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="afb_s1_expert_single_task_train",
        node=afb_s1_expert_single_task_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="afb_s1_expert_single_task_val",
        node=afb_s1_expert_single_task_val_dataloader,
    )


cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=_EXP_NAME,
    node=COSMOS_PREDICT2P5_2B_AFB_S1_EXPERT_SINGLE_TASK_CHUNK16,
)
