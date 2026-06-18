#!/bin/bash
# 8-GPU Cosmos-Predict2.5-2B action-conditioned finetune on RobotWin clean50 pcp+pob
#
# Run on Baidu Cloud 8-GPU node:
#   bash scripts/train_cosmos_8gpu_pcp_pob.sh

set -e
REPO="/mnt/gyc/cosmos-predict2.5-CoRL"
cd "${REPO}"

# Put venv cuDNN9 first to avoid system cuDNN8/broken cuDNN9 being picked up
VENV_CUDNN="/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export PYTHONPATH=".:packages/cosmos-cuda"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="jw10014-new-york-university"
export WANDB_API_KEY="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"
export NCCL_DEBUG=WARN
export IMAGINAIRE_OUTPUT_ROOT="/mnt/gyc_ckp/cosmos_train_output"

DATASET_DIR="/mnt/gyc_ckp/datasets/robotwin_clean50_pcp_pob"
if [ ! -d "${DATASET_DIR}/annotation/train" ]; then
    echo "[ERROR] Dataset not found at ${DATASET_DIR}."
    exit 1
fi

CKPT="/mnt/gyc_ckp/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt"
if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT}"
    exit 1
fi

# find torchrun: prefer cosmos venv, fallback to system
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
    echo "[ERROR] torchrun not found. Activate the cosmos venv first."
    exit 1
fi
echo "[INFO] Using torchrun: ${TORCHRUN}"

echo "[INFO] Starting 8-GPU Cosmos training: cosmos_predict2p5_2B_robotwin_pcp_pob_clean50_14D_chunk16"
LOG_DIR="/mnt/gyc_ckp/cosmos_train_output/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pcp_pob_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Log: ${LOG_FILE}"
${TORCHRUN} \
    --nproc_per_node=8 \
    --master_port=29613 \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=cosmos_predict2p5_2B_robotwin_pcp_pob_clean50_14D_chunk16 \
       ~dataloader_train.dataloaders \
       trainer.straggler_detection.enabled=false \
    2>&1 | tee "${LOG_FILE}"
