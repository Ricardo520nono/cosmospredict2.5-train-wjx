# 训练细节

## 参考脚本

顶层启动入口：

- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh`
- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_single_task_8gpu.sh`

内部训练脚本：

- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh`
- `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_expert_single_task_chunk16.sh`

## 关键代码文件

- 数据集实现：
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/datasets/dataset_afb_s1_family_balanced.py`
- 数据注册：
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`
- Family-balanced 实验 config：
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_family_balanced.py`
- Expert-only 单任务实验 config：
  `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_expert_single_task.py`

## 数据路径

Expert clean HDF5：

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_delta_ee/demo_clean_zed2i_visible
```

Enhanced LeRobot：

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/data_lerobot/robotwin_delta_ee/_enhanced_reconvert_wjx5_20260607
```

Random feasible：

```bash
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench/EnhancedData/random_feasible_300step_5task_2ep5start_formal_v1/random_feasible_random_walk
```

## 模型权重

基础 action-conditioned checkpoint：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models/Cosmos-Predict2.5-2B/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt
```

Tokenizer：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models/Cosmos-Predict2.5-2B/tokenizer.pth
```

Reason1：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models/Cosmos-Reason1-7B
```

## Checkpoint 策略

训练脚本会设置：

```bash
export COSMOS_SAVE_MODEL_ONLY=1
export COSMOS_EPOCH_CKPT_STEP="${AFB_S1_EPOCH_STEP}"
export COSMOS_FINAL_CKPT_STEP="${AFB_S1_MAX_ITER}"
```

Family-balanced 默认值：

```bash
AFB_S1_MAX_ITER=40000
AFB_S1_EPOCH_STEP=4277
```

Expert-only 的默认保存间隔按任务区分，见 `README.md`。

## 容易出错的点

- Chunk size 和模型 config 必须一致：`num_action_per_chunk=16`、`state_t=5`、`temporal_compression_ratio=4`。
- Delta-ee 的 action dim 保持为 14。
- 保持 `use_crossattn_projection=True`、`crossattn_proj_in_channels=100352`、`crossattn_emb_channels=1024`。
- 保持 Reason1 在线 embedding 模式：`compute_online=True`、`embedding_concat_strategy=full_concat`。
- 视角固定为 head/high camera。
- Random-feasible 的 VideoReader cache 默认关闭：`AFB_S1_RF_VIDEO_CACHE_SIZE=0`。除非重新评估内存，否则不要打开。
- 如果 Cosmos venv 里没有 `h5py`，脚本会使用 `H5PY_EXTRA_PATH=/mnt/gyc/envs/cosmos-policy/lib/python3.10/site-packages`。
