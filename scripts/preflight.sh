#!/bin/bash
# Fast single-process checks for paths, Python deps, Hydra config, and one dataset sample.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_COSMOS_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export COSMOS_TRAIN_ROOT="${COSMOS_TRAIN_ROOT:-${DEFAULT_COSMOS_TRAIN_ROOT}}"
export REPO="${REPO:-${COSMOS_TRAIN_ROOT}/code/cosmos-predict2.5-CoRL}"
export AFB_DATA_ROOT="${AFB_DATA_ROOT:-/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench}"
export COSMOS_VENV="${COSMOS_VENV:-${COSMOS_TRAIN_ROOT}/.venv}"
export AFB_S1_DRYRUN=1
export AFB_S1_NPROC="${AFB_S1_NPROC:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${REPO}"

echo "[INFO] Preflight: EE head/loss smoke"
"${COSMOS_VENV}/bin/python" scripts/smoke_ee_head_loss.py

echo "[INFO] Preflight: family-balanced config and data"
AFB_S1_SKIP_PREFLIGHT=0 bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh" --dryrun

echo "[INFO] Preflight: expert-only configs and data"
for task in click_alarmclock click_bell place_object_basket open_laptop stack_blocks_two; do
    echo "[INFO] Task: ${task}"
    AFB_S1_TASK="${task}" AFB_S1_SKIP_PREFLIGHT=0 bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh" --dryrun
done
