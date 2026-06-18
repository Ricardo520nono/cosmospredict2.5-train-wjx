#!/bin/bash
set -euo pipefail
export AFB_S1_TASK=stack_blocks_two
exec bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_single_task_8gpu.sh
