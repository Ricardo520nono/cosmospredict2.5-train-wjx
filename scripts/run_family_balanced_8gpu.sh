#!/bin/bash
# One-command 8-GPU AFB S1 family-balanced training.

set -euo pipefail

export COSMOS_TRAIN_ROOT="${COSMOS_TRAIN_ROOT:-/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train}"
export REPO="${REPO:-${COSMOS_TRAIN_ROOT}/code/cosmos-predict2.5-CoRL}"

exec bash "${REPO}/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh"
