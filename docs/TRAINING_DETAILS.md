# 训练 Pipeline 细节

本文档解释当前 CosmosPredict2.5 AFB S1 训练从数据到 checkpoint 的完整链路。后续 Codex 接新需求时，应该先理解这里，再改代码。

## 训练入口

推荐入口是顶层脚本：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh
```

这个脚本只做两件事：

1. 设置 `COSMOS_TRAIN_ROOT=/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train`。
2. 调用内部训练脚本：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh
```

内部脚本负责：

- 进入 `code/cosmos-predict2.5-CoRL`。
- 设置 `LD_LIBRARY_PATH`、`PYTHONPATH`、`H5PY_EXTRA_PATH`、WandB、NCCL、输出目录等环境变量。
- 检查数据路径和模型权重路径是否存在。
- 检查 `h5py` 是否可用；如果不可用，尝试用 `uv pip install h5py` 安装到当前环境。
- 运行单进程 preflight，实例化 Hydra config 和 dataset，读取样本确认 shape。
- 通过 `torchrun -m scripts.train` 启动正式训练。

## CosmosPredict2.5 训练主流程

训练主流程如下：

```text
顶层启动脚本
  -> 内部训练脚本设置环境变量
  -> Hydra make_config / override 组装实验配置
  -> 注册 AFB S1 dataset 和 dataloader
  -> torchrun 启动 8 个训练进程
  -> 每个 rank 构造 DataLoader + DistributedSampler
  -> Dataset 在线采样 family/task/window
  -> 读取视频帧和 delta-ee action chunk
  -> 读取未来 16 帧 EE target：position / rotation 6D / gripper
  -> Resize/ToTensor，得到模型输入 video tensor
  -> Reason1 在线编码任务文本
  -> CosmosPredict2.5 action-conditioned 模型训练
  -> family-balanced 入口额外计算 EE trajectory auxiliary loss
  -> 按 save_iter 保存 checkpoint
  -> 到 max_iter 保存 final checkpoint
```

## 关键代码文件

Dataset：

```bash
code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/datasets/dataset_afb_s1_family_balanced.py
```

这个文件里有两个 dataset：

- `AFBS1FamilyBalancedDataset`：family-balanced 3:1:1:1 混合训练用。
- `AFBS1ExpertSingleTaskDataset`：单任务 expert-only 训练用。

Data registry：

```bash
code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py
```

这个文件负责把新 dataset 注册进 Hydra ConfigStore。

Family-balanced experiment config：

```bash
code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_family_balanced.py
```

Expert-only experiment config：

```bash
code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_expert_single_task.py
```

## 当前数据源

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

## 当前任务集合

S1 训练使用 5 个任务：

```text
click_alarmclock
click_bell
place_object_basket
open_laptop
stack_blocks_two
```

Family-balanced 版本会在这 5 个任务之间在线随机采样。Expert-only 版本每次只跑一个任务。

## Family-Balanced Sampler

当前 family 混合比例是：

```text
expert : pca_c8_sigma0p05 : raw_sigma0p0025 : random_feasible_300step = 3 : 1 : 1 : 1
```

代码里用 `FAMILY_BY_SLOT` 实现：

```text
expert
expert
expert
pca_c8_sigma0p05
raw_sigma0p0025
random_feasible_300step
```

训练时 dataset 根据 index 取 slot，再在该 family 内随机选 task 和 sample window。这样不需要提前写死一个巨大 manifest，也能稳定维持近似 3:1:1:1 的混合比例。

## 当前视角

当前可运行基线是单视角：

- Expert HDF5 读取 `observation/head_camera/rgb`。
- Enhanced LeRobot 读取 `observation.images.cam_high`。
- Random feasible 读取 metadata 对应视频。

数据里已经确认存在多视角字段，但当前 dataset 代码还没有切换到三视角训练。多视角主线见 `docs/MULTIVIEW_MAINLINE.md`。

## 视频和动作张量

默认 chunk size：

```text
num_action_per_chunk = 16
```

每条样本读取：

```text
17 帧视频 = 16 个 action step + 1 个 initial frame
16 条 delta-ee action
```

Preprocess 后 shape：

```text
video:  (3, 17, 256, 320)
action: (16, 14)
ee_target_position:    (16, 2, 3)
ee_target_rotation_6d: (16, 2, 6)
ee_target_gripper:     (16, 2)
```

其中 action dim = 14，表示双臂 delta-ee：

```text
left arm  6DoF + gripper
right arm 6DoF + gripper
```

## 模型配置关键点

必须保持一致的配置：

```text
num_action_per_chunk = 16
state_t = 1 + 16 // 4 = 5
temporal_compression_ratio = 4
action_dim = 14
use_crossattn_projection = True
crossattn_proj_in_channels = 100352
crossattn_emb_channels = 1024
compute_online = True
embedding_concat_strategy = full_concat
```

这些配置和官方 action-conditioned 模型、Reason1 文本编码、chunk16 输入长度是配套的。之前训练反复报错时，核心问题之一就是 chunk/action/model 配置不一致。

Family-balanced 入口额外启用：

```text
model.config.ee_head.enabled = True
model.config.ee_head.loss_weight = AFB_S1_EE_LOSS_WEIGHT, 默认 0.05
model.config.net.ee_head_enabled = True
model.config.net.ee_head_num_frames = 16
model.config.net.ee_head_latent_frames = 5
model.config.net.ee_head_hidden_dim = AFB_S1_EE_HEAD_HIDDEN_DIM, 默认 1024
```

Expert-only 单任务入口显式保持 no-head：

```text
model.config.ee_head.enabled = False
model.config.net.ee_head_enabled = False
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

训练脚本设置：

```bash
export COSMOS_SAVE_MODEL_ONLY=1
export COSMOS_EPOCH_CKPT_STEP="${AFB_S1_EPOCH_STEP}"
export COSMOS_FINAL_CKPT_STEP="${AFB_S1_MAX_ITER}"
```

Family-balanced 默认：

```bash
AFB_S1_MAX_ITER=40000
AFB_S1_EPOCH_STEP=4277
```

Expert-only 默认保存间隔。训练窗口数按同时能读到 16 个 action、未来 16 帧 EE target 和 17 帧视频的有效窗口计算：

| 任务 | 训练窗口数 | 保存间隔 step |
| --- | ---: | ---: |
| `click_alarmclock` | 2708 | 170 |
| `click_bell` | 2474 | 155 |
| `place_object_basket` | 9093 | 569 |
| `open_laptop` | 7822 | 489 |
| `stack_blocks_two` | 11917 | 745 |

默认输出目录：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/outputs/cosmos_train_output
```

## 预检覆盖内容

运行：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/preflight.sh
```

预检会覆盖：

- `h5py`、`torch`、`av`、`decord`、`cv2`、`pandas`、`pyarrow` import。
- `scripts/smoke_ee_head_loss.py` 检查 EE head prediction shape、三项 loss 和 no-head 开关。
- Hydra experiment 能否正确 compose。
- Dataset 能否实例化。
- Family-balanced 四类数据是否都有样本。
- 5 个 expert-only 单任务窗口数量是否符合预期。
- 单条样本 `video/action/ee_target_*` shape 是否正确。
- `torchrun --dryrun` 能否启动到训练框架。

## 常见风险

- 不要把 chunk size 改成 32 但忘记改 `state_t`、`num_action_per_chunk` 和 dataset。
- 不要把 action dim 从 14 改掉，除非数据本身不再是双臂 delta-ee。
- 不要把 `compute_online=True` 改成离线 embedding，除非已经准备好对应 embedding 文件。
- 不要随便打开 random-feasible 视频缓存；默认 `AFB_S1_RF_VIDEO_CACHE_SIZE=0` 是为了避免内存膨胀。
- 如果换机器，先解决环境，再跑 preflight，不要直接提交长训练。
