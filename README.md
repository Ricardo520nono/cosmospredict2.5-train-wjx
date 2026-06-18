# CosmosPredict2.5 AFB S1 Training Package

This directory is the clean training handoff for CosmosPredict2.5 on ActionFollowingBench S1.

Root path:

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
```

## What Is Included

- `code/cosmos-predict2.5-CoRL`: runnable CosmosPredict2.5 code with the AFB S1 dataset/config changes.
- `models/Cosmos-Predict2.5-2B`: local base CosmosPredict2.5-2B action-conditioned weights and tokenizer.
- `models/Cosmos-Reason1-7B`: local Reason1 text encoder checkpoint directory.
- `scripts`: clean one-command launchers.
- `docs`: training notes for Codex/human handoff.
- `outputs`: default output root for logs and checkpoints.

The training view is fixed to `head_camera` / `cam_high`. Do not switch to wrist or side camera unless the dataset/config is intentionally changed.

## Quick Start

Run a fast preflight first:

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/preflight.sh
```

Launch the 8-GPU family-balanced run:

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/run_family_balanced_8gpu.sh
```

Launch one expert-only task:

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
AFB_S1_TASK=click_alarmclock bash scripts/run_expert_single_task_8gpu.sh
```

Or use the task wrappers:

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_click_alarmclock_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_click_bell_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_place_object_basket_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_open_laptop_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_stack_blocks_two_8gpu.sh
```

## Training Recipes

### Family-Balanced S1

This is the current reference run:

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh
```

Config summary:

- 5 S1 tasks.
- Data mix: expert : PCA enhanced : raw enhanced : random feasible = `3:1:1:1`.
- Online family-balanced sampler.
- View: head camera only.
- Chunk size: 16.
- Action dim: 14.
- Per-GPU batch: 2 by default.
- Total steps: 40000 by default.
- Checkpoint interval: every 4277 steps by default, approximately one integer epoch.

### Expert-Only Single Task

Use clean expert data only, one task per run.

Supported tasks:

- `click_alarmclock`
- `click_bell`
- `place_object_basket`
- `open_laptop`
- `stack_blocks_two`

Default checkpoint intervals:

| Task | Windows | Save Every |
| --- | ---: | ---: |
| `click_alarmclock` | 2748 | 172 |
| `click_bell` | 2514 | 158 |
| `place_object_basket` | 9133 | 571 |
| `open_laptop` | 7862 | 492 |
| `stack_blocks_two` | 11957 | 748 |

## Common Overrides

```bash
export AFB_S1_PER_GPU_BATCH=1
export AFB_S1_MAX_ITER=40000
export AFB_S1_NPROC=8
export WANDB_MODE=online
export MASTER_PORT=29617
```

Default output root:

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/outputs/cosmos_train_output
```

## Environment

The scripts reuse the existing Cosmos Python environment by default:

```bash
/mnt/gyc/cosmos-predict2.5/.venv
```

They also add the known working h5py fallback:

```bash
/mnt/gyc/envs/cosmos-policy/lib/python3.10/site-packages
```

If this machine changes, install the requirements for `code/cosmos-predict2.5-CoRL` into a Python 3.10 CUDA environment and make sure `torchrun`, `h5py`, `av`, `decord`, `cv2`, `pandas`, `pyarrow`, and `flash_attn` import successfully.

See `docs/CODEX_HANDOFF.md` and `docs/TRAINING_DETAILS.md` before editing training behavior.
