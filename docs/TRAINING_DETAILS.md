# Training Details

## Reference Scripts

Top-level launchers:

- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh`
- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_single_task_8gpu.sh`

Internal training scripts:

- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh`
- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh`

## Important Code Files

- Dataset implementation:
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/datasets/dataset_afb_s1_family_balanced.py`
- Data registry:
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`
- Family-balanced experiment:
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_family_balanced.py`
- Expert-only experiment:
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_expert_single_task.py`

## Data

Expert clean HDF5:

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible
```

Enhanced LeRobot:

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/_enhanced_reconvert_wjx5_20260607
```

Random feasible:

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk
```

## Model Weights

Base action-conditioned checkpoint:

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt
```

Tokenizer:

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models/Cosmos-Predict2.5-2B/tokenizer.pth
```

Reason1:

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models/Cosmos-Reason1-7B
```

## Checkpoint Policy

The scripts set:

```bash
export COSMOS_SAVE_MODEL_ONLY=1
export COSMOS_EPOCH_CKPT_STEP="${AFB_S1_EPOCH_STEP}"
export COSMOS_FINAL_CKPT_STEP="${AFB_S1_MAX_ITER}"
```

Family-balanced default:

```bash
AFB_S1_MAX_ITER=40000
AFB_S1_EPOCH_STEP=4277
```

Expert-only defaults are task-specific and listed in `README.md`.

## Known Pitfalls

- Keep chunk size and model config aligned: `num_action_per_chunk=16`, `state_t=5`, `temporal_compression_ratio=4`.
- Keep action dim at 14 for delta-ee.
- Keep `use_crossattn_projection=True`, `crossattn_proj_in_channels=100352`, and `crossattn_emb_channels=1024`.
- Keep Reason1 online embedding mode: `compute_online=True`, `embedding_concat_strategy=full_concat`.
- Keep the camera fixed to head/high camera.
- Do not enable random-feasible VideoReader caching unless memory is audited; default is `AFB_S1_RF_VIDEO_CACHE_SIZE=0`.
- If `h5py` is missing from the Cosmos venv, the scripts use `H5PY_EXTRA_PATH=/mnt/gyc/envs/cosmos-policy/lib/python3.10/site-packages`.
