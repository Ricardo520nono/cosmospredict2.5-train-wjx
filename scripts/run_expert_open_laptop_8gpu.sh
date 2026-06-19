#!/bin/bash
set -euo pipefail
export AFB_S1_TASK=open_laptop
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_expert_single_task_8gpu.sh"
