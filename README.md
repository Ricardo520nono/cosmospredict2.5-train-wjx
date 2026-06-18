# CosmosPredict2.5 AFB S1 训练包

这是给 ActionFollowingBench S1 训练整理出来的一份干净交接目录，目标是让人或 Codex 读完后，可以直接在公共路径下一键预检和启动训练。

根目录：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
```

## 目录内容

- `code/cosmos-predict2.5-CoRL`：可运行的 CosmosPredict2.5 代码，已包含 AFB S1 数据集和训练 config 修改。
- `models/Cosmos-Predict2.5-2B`：本地 CosmosPredict2.5-2B action-conditioned 基础权重和 tokenizer。
- `models/Cosmos-Reason1-7B`：本地 Reason1 文本编码器 checkpoint 目录。
- `scripts`：干净的一键启动脚本。
- `docs`：给人和 Codex 看的训练交接文档。
- `outputs`：默认日志和 checkpoint 输出目录。

当前训练视角固定为 `head_camera` / `cam_high`。除非明确要改数据集和 config，否则不要切到腕部或侧视角。

## 快速开始

先跑一次预检：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/preflight.sh
```

启动 8 卡 family-balanced 训练：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/run_family_balanced_8gpu.sh
```

启动一个单任务 expert-only 训练：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
AFB_S1_TASK=click_alarmclock bash scripts/run_expert_single_task_8gpu.sh
```

也可以直接用 5 个任务的快捷脚本：

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_click_alarmclock_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_click_bell_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_place_object_basket_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_open_laptop_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_stack_blocks_two_8gpu.sh
```

## 训练方案

### Family-Balanced S1

这是目前的参考训练版本：

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh
```

配置摘要：

- 固定 5 个 S1 任务。
- 数据混合比例：expert : PCA enhanced : raw enhanced : random feasible = `3:1:1:1`。
- 使用在线 family-balanced sampler。
- 视角：只用 head camera。
- Chunk size：16。
- Action dim：14。
- 默认每卡 batch：2。
- 默认总步数：40000。
- 默认每 4277 step 保存一次 checkpoint，约等于一个整数 epoch。

### Expert-Only 单任务

每次只用一个任务的干净 expert 数据训练。

支持的 5 个任务：

- `click_alarmclock`
- `click_bell`
- `place_object_basket`
- `open_laptop`
- `stack_blocks_two`

默认 checkpoint 保存间隔如下：

| 任务 | 训练窗口数 | 保存间隔 step |
| --- | ---: | ---: |
| `click_alarmclock` | 2748 | 172 |
| `click_bell` | 2514 | 158 |
| `place_object_basket` | 9133 | 571 |
| `open_laptop` | 7862 | 492 |
| `stack_blocks_two` | 11957 | 748 |

## 常用环境变量

```bash
export AFB_S1_PER_GPU_BATCH=1
export AFB_S1_MAX_ITER=40000
export AFB_S1_NPROC=8
export WANDB_MODE=online
export MASTER_PORT=29617
```

默认输出目录：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/outputs/cosmos_train_output
```

## Python 环境

脚本默认复用当前机器上已经跑通的 Cosmos Python 环境：

```bash
/mnt/gyc/cosmos-predict2.5/.venv
```

同时会加入已验证可用的 `h5py` fallback 路径：

```bash
/mnt/gyc/envs/cosmos-policy/lib/python3.10/site-packages
```

如果换机器，需要在 Python 3.10 + CUDA 环境里安装 `code/cosmos-predict2.5-CoRL` 所需依赖，并确认 `torchrun`、`h5py`、`av`、`decord`、`cv2`、`pandas`、`pyarrow`、`flash_attn` 都能正常 import。

修改训练逻辑前，建议先读：

- `docs/CODEX_HANDOFF.md`
- `docs/TRAINING_DETAILS.md`
