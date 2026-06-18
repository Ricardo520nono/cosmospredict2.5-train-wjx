#!/bin/bash
set -euo pipefail
export AFB_S1_TASK=place_object_basket
exec bash /mnt/gyc/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh "$@"
