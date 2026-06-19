#!/bin/bash
# One-command 8-GPU AFB S1 expert-only single-task training.
#
# Usage:
#   AFB_S1_TASK=click_alarmclock bash scripts/run_expert_single_task_8gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_COSMOS_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export COSMOS_TRAIN_ROOT="${COSMOS_TRAIN_ROOT:-${DEFAULT_COSMOS_TRAIN_ROOT}}"
export REPO="${REPO:-${COSMOS_TRAIN_ROOT}/code/cosmos-predict2.5-CoRL}"
export AFB_DATA_ROOT="${AFB_DATA_ROOT:-/mnt/dataset/public_data/cscsx_projects/data/ActionFollowingBench}"

exec bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh"
