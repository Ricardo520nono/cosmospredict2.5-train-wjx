#!/bin/bash
# 8-GPU Cosmos-Predict2.5-2B AFB S1 family-balanced action-conditioned finetune.
#
# Run on Baidu Cloud 8-GPU node:
#   bash scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_COSMOS_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
COSMOS_TRAIN_ROOT="${COSMOS_TRAIN_ROOT:-${DEFAULT_COSMOS_TRAIN_ROOT}}"
REPO="${REPO:-${COSMOS_TRAIN_ROOT}/code/cosmos-predict2.5-CoRL}"
cd "${REPO}"

COSMOS_VENV="${COSMOS_VENV:-${COSMOS_TRAIN_ROOT}/.venv}"
VENV_CUDNN="${COSMOS_VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="${COSMOS_VENV}/bin:${PATH}"
export PYTHONPATH=".:packages/cosmos-cuda"
export H5PY_EXTRA_PATH="${H5PY_EXTRA_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-jw10014-new-york-university}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export COSMOS_TRAIN_ROOT
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${COSMOS_TRAIN_ROOT}/outputs/cosmos_train_output}"
export AFB_S1_RF_VIDEO_CACHE_SIZE="${AFB_S1_RF_VIDEO_CACHE_SIZE:-0}"
export COSMOS_SAVE_MODEL_ONLY="${COSMOS_SAVE_MODEL_ONLY:-1}"
export AFB_DATA_ROOT="${AFB_DATA_ROOT:-/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench}"

export AFB_S1_PER_GPU_BATCH="${AFB_S1_PER_GPU_BATCH:-2}"
export AFB_S1_MAX_ITER="${AFB_S1_MAX_ITER:-40000}"
export AFB_S1_EPOCH_STEP="${AFB_S1_EPOCH_STEP:-4277}"
export AFB_S1_EE_LOSS_WEIGHT="${AFB_S1_EE_LOSS_WEIGHT:-0.05}"
export AFB_S1_EE_HEAD_HIDDEN_DIM="${AFB_S1_EE_HEAD_HIDDEN_DIM:-1024}"
export AFB_S1_SKIP_PREFLIGHT="${AFB_S1_SKIP_PREFLIGHT:-0}"
export AFB_S1_DRYRUN="${AFB_S1_DRYRUN:-0}"
export AFB_S1_NPROC="${AFB_S1_NPROC:-8}"
export COSMOS_EPOCH_CKPT_STEP="${AFB_S1_EPOCH_STEP}"
export COSMOS_FINAL_CKPT_STEP="${AFB_S1_MAX_ITER}"

EXPERT_ROOT="${AFB_EXPERT_ROOT:-${AFB_DATA_ROOT}/data_delta_ee/demo_clean_zed2i_visible}"
ENHANCED_ROOT="${AFB_ENHANCED_LEROBOT_ROOT:-${AFB_DATA_ROOT}/data_lerobot/robotwin_delta_ee/_enhanced_reconvert_wjx5_20260607}"
RF_ROOT="${AFB_RF_ROOT:-${AFB_DATA_ROOT}/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk}"
CKPT="${COSMOS_TRAIN_ROOT}/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
TOKENIZER="${COSMOS_TRAIN_ROOT}/models/Cosmos-Predict2.5-2B/tokenizer.pth"
REASON1="${COSMOS_TRAIN_ROOT}/models/Cosmos-Reason1-7B"

for path in "${EXPERT_ROOT}" "${ENHANCED_ROOT}" "${RF_ROOT}" "${CKPT}" "${TOKENIZER}" "${REASON1}"; do
    if [ ! -e "${path}" ]; then
        echo "[ERROR] Required path not found: ${path}"
        exit 1
    fi
done

LOCAL_QWEN_PROCESSOR_ROOT="/mnt/public_ckp/shijy/models"
LOCAL_QWEN_PROCESSOR="${LOCAL_QWEN_PROCESSOR_ROOT}/Qwen2.5-VL-7B-Instruct"
mkdir -p "${LOCAL_QWEN_PROCESSOR_ROOT}"
ln -sfn "${REASON1}" "${LOCAL_QWEN_PROCESSOR}"

TORCHRUN="${COSMOS_VENV}/bin/torchrun"
if [ ! -x "${TORCHRUN}" ]; then
    echo "[ERROR] torchrun not found at ${TORCHRUN}. Create ${COSMOS_VENV} with uv or set COSMOS_VENV explicitly."
    exit 1
fi

EXP="cosmos_predict2p5_2B_afb_s1_family_balanced_3_1_1_1_chunk16_headcam"
LOG_DIR="${IMAGINAIRE_OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/afb_s1_family_balanced_chunk16_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Using torchrun: ${TORCHRUN}"
echo "[INFO] Experiment: ${EXP}"
echo "[INFO] GPUs: ${AFB_S1_NPROC}"
echo "[INFO] Per-GPU batch: ${AFB_S1_PER_GPU_BATCH}"
echo "[INFO] Max iter: ${AFB_S1_MAX_ITER}"
echo "[INFO] Epoch checkpoint step: ${AFB_S1_EPOCH_STEP}"
echo "[INFO] Final checkpoint step: ${COSMOS_FINAL_CKPT_STEP}"
echo "[INFO] H5PY extra path: ${H5PY_EXTRA_PATH}"
echo "[INFO] RF video cache size: ${AFB_S1_RF_VIDEO_CACHE_SIZE}"
echo "[INFO] Save model only: ${COSMOS_SAVE_MODEL_ONLY}"
echo "[INFO] AFB data root: ${AFB_DATA_ROOT}"
echo "[INFO] EE loss weight: ${AFB_S1_EE_LOSS_WEIGHT}"
echo "[INFO] EE head hidden dim: ${AFB_S1_EE_HEAD_HIDDEN_DIM}"
echo "[INFO] Qwen processor cache: ${LOCAL_QWEN_PROCESSOR}"
echo "[INFO] WandB mode: ${WANDB_MODE}"
echo "[INFO] Skip preflight: ${AFB_S1_SKIP_PREFLIGHT}"
echo "[INFO] Dryrun: ${AFB_S1_DRYRUN}"
echo "[INFO] Log: ${LOG_FILE}"

PYTHON_BIN="${TORCHRUN%/torchrun}/python"
if ! "${PYTHON_BIN}" - <<'PY'
import os
import sys

extra = os.environ.get("H5PY_EXTRA_PATH")
if extra and os.path.exists(extra) and extra not in sys.path:
    sys.path.append(extra)
try:
    import h5py
except Exception as exc:
    raise SystemExit(
        "[ERROR] h5py is not importable. Set H5PY_EXTRA_PATH to a Python 3.10 site-packages path "
        f"that contains h5py. Current H5PY_EXTRA_PATH={extra!r}. Original error: {exc}"
    )
print(f"[INFO] h5py import OK: {h5py.__file__}")
PY
then
    UV_BIN="${COSMOS_VENV}/bin/uv"
    if [ ! -x "${UV_BIN}" ]; then
        UV_BIN="$(command -v uv || true)"
    fi
    if [ -z "${UV_BIN}" ]; then
        echo "[ERROR] h5py missing and uv not found. Please install h5py into ${PYTHON_BIN} or set H5PY_EXTRA_PATH."
        exit 1
    fi
    echo "[WARN] h5py missing; installing h5py into cosmos venv with ${UV_BIN}."
    "${UV_BIN}" pip install --python "${PYTHON_BIN}" h5py
    "${PYTHON_BIN}" - <<'PY'
import h5py
print(f"[INFO] h5py import OK after install: {h5py.__file__}")
PY
fi

if [ "${AFB_S1_SKIP_PREFLIGHT}" != "1" ]; then
    echo "[INFO] Running single-process preflight checks."
    "${PYTHON_BIN}" - <<'PY'
import os
import sys

extra = os.environ.get("H5PY_EXTRA_PATH")
if extra and os.path.exists(extra) and extra not in sys.path:
    sys.path.append(extra)

for module in ("h5py", "av", "decord", "cv2", "pandas", "pyarrow", "torch"):
    __import__(module)
print("[INFO] Preflight deps OK")

from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import make_config
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils.config_helper import override

exp = "cosmos_predict2p5_2B_afb_s1_family_balanced_3_1_1_1_chunk16_headcam"
cfg = override(
    make_config(),
    [
        "--",
        f"experiment={exp}",
        "~dataloader_train.dataloaders",
        "trainer.straggler_detection.enabled=false",
    ],
)
assert cfg.job.name == exp, cfg.job.name
assert cfg.dataloader_train.dataset.mode == "train"
assert cfg.dataloader_val.dataset.mode == "val"
assert cfg.dataloader_train.batch_size == int(os.environ.get("AFB_S1_PER_GPU_BATCH", "2"))
assert cfg.trainer.max_iter == int(os.environ.get("AFB_S1_MAX_ITER", "40000"))
assert cfg.checkpoint.save_iter == int(os.environ.get("AFB_S1_EPOCH_STEP", "4277"))
assert cfg.model.config.state_t == 5
assert cfg.model.config.net.action_dim == 14
assert cfg.model.config.net.num_action_per_chunk == 16
assert cfg.model.config.ee_head.enabled is True
assert cfg.model.config.ee_head.loss_weight == float(os.environ.get("AFB_S1_EE_LOSS_WEIGHT", "0.05"))
assert cfg.model.config.net.ee_head_enabled is True
assert cfg.model.config.net.ee_head_num_frames == 16
assert cfg.model.config.net.ee_head_latent_frames == 5
assert cfg.model.config.net.use_crossattn_projection is True
assert cfg.model.config.net.crossattn_proj_in_channels == 100352
assert cfg.model.config.net.crossattn_emb_channels == 1024
assert cfg.model.config.text_encoder_config.compute_online is True
assert cfg.model.config.text_encoder_config.embedding_concat_strategy == "full_concat"
assert "wandb" in cfg.trainer.callbacks
assert "wandb_10x" in cfg.trainer.callbacks
assert cfg.trainer.callbacks.device_monitor.log_memory_detail is False
assert cfg.dataloader_train.dataset.num_action_per_chunk == 16
assert cfg.dataloader_val.dataset.num_action_per_chunk == 16
assert cfg.dataloader_train.sampler.dataset.num_action_per_chunk == 16
assert cfg.dataloader_val.sampler.dataset.num_action_per_chunk == 16
print("[INFO] Preflight Hydra config OK")

ds = instantiate(cfg.dataloader_train.dataset)
totals = {family: sum(len(items) for items in by_task.values()) for family, by_task in ds.pools.items()}
for family in ("expert", "pca_c8_sigma0p05", "raw_sigma0p0025", "random_feasible_300step"):
    if totals.get(family, 0) <= 0:
        raise RuntimeError(f"No train samples found for {family}: {totals}")
    by_task = ds.pools[family]
    task = next(task for task, samples in by_task.items() if samples)
    sample = dict(by_task[task][0])
    sample["start"] = 0
    sample["chosen_family"] = family
    frames, actions, ee_target = ds._read_sample(sample)
    if tuple(frames.shape) != (17, 480, 640, 3):
        raise RuntimeError(f"Unexpected raw frame shape for {family}: {frames.shape}")
    if tuple(actions.shape) != (16, 14):
        raise RuntimeError(f"Unexpected raw action shape for {family}: {actions.shape}")
    ee_position, ee_rotation_6d, ee_gripper = ee_target
    if tuple(ee_position.shape) != (16, 2, 3):
        raise RuntimeError(f"Unexpected EE position shape for {family}: {ee_position.shape}")
    if tuple(ee_rotation_6d.shape) != (16, 2, 6):
        raise RuntimeError(f"Unexpected EE rotation 6D shape for {family}: {ee_rotation_6d.shape}")
    if tuple(ee_gripper.shape) != (16, 2):
        raise RuntimeError(f"Unexpected EE gripper shape for {family}: {ee_gripper.shape}")
item = ds[0]
if tuple(item["video"].shape) != (3, 17, 256, 320):
    raise RuntimeError(f"Unexpected video shape: {tuple(item['video'].shape)}")
if tuple(item["action"].shape) != (16, 14):
    raise RuntimeError(f"Unexpected action shape: {tuple(item['action'].shape)}")
if tuple(item["ee_target_position"].shape) != (16, 2, 3):
    raise RuntimeError(f"Unexpected EE target position shape: {tuple(item['ee_target_position'].shape)}")
if tuple(item["ee_target_rotation_6d"].shape) != (16, 2, 6):
    raise RuntimeError(f"Unexpected EE target rotation shape: {tuple(item['ee_target_rotation_6d'].shape)}")
if tuple(item["ee_target_gripper"].shape) != (16, 2):
    raise RuntimeError(f"Unexpected EE target gripper shape: {tuple(item['ee_target_gripper'].shape)}")
print(f"[INFO] Preflight dataset OK: {totals}")
PY
fi

DRYRUN_ARGS=()
if [ "${AFB_S1_DRYRUN}" = "1" ]; then
    DRYRUN_ARGS+=(--dryrun)
fi

${TORCHRUN} \
    --nproc_per_node="${AFB_S1_NPROC}" \
    --master_port="${MASTER_PORT:-29617}" \
    -m scripts.train \
    "${DRYRUN_ARGS[@]}" \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment="${EXP}" \
       ~dataloader_train.dataloaders \
       trainer.straggler_detection.enabled=false \
    2>&1 | tee "${LOG_FILE}"
