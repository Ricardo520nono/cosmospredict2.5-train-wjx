#!/bin/bash
# One-command 8-GPU AFB S1 expert-only single-task training.
#
# Usage:
#   AFB_S1_TASK=click_alarmclock bash scripts/run_expert_single_task_8gpu.sh

set -euo pipefail

export COSMOS_TRAIN_ROOT="${COSMOS_TRAIN_ROOT:-/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train}"
export REPO="${REPO:-${COSMOS_TRAIN_ROOT}/code/cosmos-predict2.5-CoRL}"

exec bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh"
