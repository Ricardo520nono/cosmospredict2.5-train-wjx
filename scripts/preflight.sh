#!/bin/bash
# Fast single-process checks for paths, Python deps, Hydra config, and one dataset sample.

set -euo pipefail

export COSMOS_TRAIN_ROOT="${COSMOS_TRAIN_ROOT:-/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train}"
export REPO="${REPO:-${COSMOS_TRAIN_ROOT}/code/cosmos-predict2.5-CoRL}"
export AFB_S1_DRYRUN=1
export AFB_S1_NPROC="${AFB_S1_NPROC:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${REPO}"

echo "[INFO] Preflight: family-balanced config and data"
AFB_S1_SKIP_PREFLIGHT=0 bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh" --dryrun

echo "[INFO] Preflight: expert-only configs and data"
for task in click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two; do
    echo "[INFO] Task: ${task}"
    AFB_S1_TASK="${task}" AFB_S1_SKIP_PREFLIGHT=0 bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh" --dryrun
done
