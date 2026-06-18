#!/bin/bash
# 16-GPU (2 nodes x 8 GPU) Cosmos-Predict2.5-2B action-conditioned finetune
# RobotWin clean50 全量 50 任务, 14D dual-arm action, chunk_size=32
#
# Baidu Cloud injects: WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT, NPROC_PER_NODE
# Submit with 2-node 8-GPU job, then:
#   bash /mnt/gyc/cosmos-predict2.5-CoRL/scripts/train_cosmos_16gpu_all50.sh

set -e
REPO="/mnt/gyc/cosmos-predict2.5-CoRL"
cd "${REPO}"

VENV_CUDNN="/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export PYTHONPATH=".:packages/cosmos-cuda"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="jw10014-new-york-university"
export WANDB_API_KEY="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"
export NCCL_DEBUG=WARN
export IMAGINAIRE_OUTPUT_ROOT="/mnt/gyc_ckp/cosmos_train_output"

DATASET_DIR="/mnt/gyc_ckp/datasets/robotwin_clean50_all50"
if [ ! -d "${DATASET_DIR}/annotation/train" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_DIR}."
    exit 1
fi

CKPT="/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi

TORCHRUN=""
for candidate in \
    "/mnt/gyc_wjx/cosmos-predict2.5/.venv/bin/torchrun" \
    "/mnt/gyc/cosmos-predict2.5/.venv/bin/torchrun" \
    "$(which torchrun 2>/dev/null)"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
        TORCHRUN="${candidate}"
        break
    fi
done
if [ -z "${TORCHRUN}" ]; then
    echo "[ERROR] torchrun not found."
    exit 1
fi

echo "[INFO] Using torchrun: ${TORCHRUN}"
echo "[INFO] WORLD_SIZE=${WORLD_SIZE:-1}, RANK=${RANK:-0}, MASTER_ADDR=${MASTER_ADDR:-localhost}, MASTER_PORT=${MASTER_PORT:-29615}"

LOG_DIR="/mnt/gyc_ckp/cosmos_train_output/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/all50_chunk32_16gpu_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Log: ${LOG_FILE}"

${TORCHRUN} \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-localhost} \
    --master_port=${MASTER_PORT:-29615} \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=cosmos_predict2p5_2B_robotwin_all50_clean50_14D_chunk32 \
       ~dataloader_train.dataloaders \
       trainer.straggler_detection.enabled=false \
    2>&1 | tee "${LOG_FILE}"
