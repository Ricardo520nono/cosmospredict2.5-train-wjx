#!/bin/bash
set -euo pipefail
export AFB_S1_TASK=click_bell
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh" "$@"
