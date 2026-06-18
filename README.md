# CosmosPredict2.5 AFB S1 训练包

这是给 ActionFollowingBench S1 训练整理出来的 CosmosPredict2.5 训练交接包。它的目标不是只保存几个启动脚本，而是让后续负责训练的 Codex 读完后，能理解训练 pipeline、知道哪些路径必须存在、知道如何预检、如何启动训练，以及下一阶段 head/left/right 多视角主线应该从哪里改。

公共根目录：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
```

GitHub 仓库：

```bash
https://github.com/Ricardo520nono/cosmospredict2.5-train-wjx.git
```

## 重要结论

当前这个包已经把训练必需的代码、脚本、文档和基础模型权重放到了 `/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train`。

如果翔哥能访问下面这些公共路径，就可以不依赖 `/mnt/gyc_ckp` 来训练：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
/mnt/public_ckp/cscsx_projects/data/ActionFollowingBench
```

如果翔哥访问不到你的 `/mnt/gyc`，也可以训练，但需要他自己的 Python/CUDA 环境。启动脚本支持用 `COSMOS_VENV` 指向新的环境：

```bash
export COSMOS_VENV=/path/to/cosmos-python-env
```

只看 GitHub 仓库可以理解和接手代码，但 GitHub 不包含 4.5G 模型权重和训练数据。实际训练仍然需要公共目录里的 `models/` 和 AFB 数据集，或者把这些资源在新机器上按文档放到同样路径。

## 当前可运行基线

当前已经预检通过、可直接启动的基线是：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/run_family_balanced_8gpu.sh
```

它是 AFB S1 family-balanced 单视角版本：

- 任务：5 个 S1 任务。
- 视角：当前代码固定读取 `head_camera` / `cam_high`。
- 数据混合：expert : PCA enhanced : raw enhanced : random feasible = `3:1:1:1`。
- Sampler：在线 family-balanced sampler。
- Chunk size：16。
- Action dim：14。
- 默认 batch：8 卡训练时每卡 2。
- 默认总步数：40000。
- 默认保存间隔：4277 step，约等于一个整数 epoch。

下一阶段主线是 head/left/right 多视角训练。当前包已经确认数据里存在多视角字段，但当前训练代码还没有切换成三视角版本。多视角改法见：

```bash
docs/MULTIVIEW_MAINLINE.md
```

## 目录内容

```text
cosmospredict2.5_train/
  README.md
  docs/
    CODEX_HANDOFF.md
    TRAINING_DETAILS.md
    MULTIVIEW_MAINLINE.md
  scripts/
    preflight.sh
    run_family_balanced_8gpu.sh
    run_expert_single_task_8gpu.sh
    run_expert_click_alarmclock_8gpu.sh
    run_expert_click_bell_8gpu.sh
    run_expert_place_object_basket_8gpu.sh
    run_expert_open_laptop_8gpu.sh
    run_expert_stack_blocks_two_8gpu.sh
  code/
    cosmos-predict2.5-CoRL/
  models/
    Cosmos-Predict2.5-2B/
    Cosmos-Reason1-7B/
  outputs/
```

说明：

- `code/cosmos-predict2.5-CoRL`：可运行的 CosmosPredict2.5 代码，包含 AFB S1 数据集和训练 config 修改。
- `models/Cosmos-Predict2.5-2B`：基础 action-conditioned 权重和 tokenizer。
- `models/Cosmos-Reason1-7B`：Reason1 文本编码器 checkpoint 目录。
- `scripts`：干净的一键启动脚本。
- `docs`：给人和 Codex 看的交接文档。
- `outputs`：默认日志和 checkpoint 输出目录。

## 快速开始

先跑预检。这个预检会检查依赖、模型路径、Hydra config、数据读取和样本 shape。它会走 dryrun，不会进入真实训练。

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

也可以直接用 5 个任务快捷脚本：

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_click_alarmclock_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_click_bell_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_place_object_basket_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_open_laptop_8gpu.sh
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_expert_stack_blocks_two_8gpu.sh
```

## Python 环境

脚本默认会优先找：

```bash
${COSMOS_VENV}/bin/torchrun
```

如果没有设置 `COSMOS_VENV`，默认值是历史已跑通环境：

```bash
/mnt/gyc/cosmos-predict2.5/.venv
```

如果翔哥没有 `/mnt/gyc`，请在自己的环境里安装依赖，然后设置：

```bash
export COSMOS_VENV=/path/to/your/cosmos-predict2.5-venv
```

最低要求：

- Python 3.10。
- CUDA/PyTorch 环境能运行 8 卡 `torchrun`。
- `torch`、`flash_attn`、`h5py`、`av`、`decord`、`cv2`、`pandas`、`pyarrow` 可 import。
- `packages/cosmos-cuda` 能通过 `PYTHONPATH` 被加载。

如果 `h5py` 不在主环境里，可以设置：

```bash
export H5PY_EXTRA_PATH=/path/to/python3.10/site-packages
```

## 常用环境变量

```bash
export COSMOS_VENV=/path/to/venv
export AFB_S1_PER_GPU_BATCH=2
export AFB_S1_MAX_ITER=40000
export AFB_S1_NPROC=8
export WANDB_MODE=online
export MASTER_PORT=29617
```

默认输出目录：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/outputs/cosmos_train_output
```

## 后续阅读顺序

建议让 Codex 按这个顺序读文档：

1. `README.md`
2. `docs/CODEX_HANDOFF.md`
3. `docs/TRAINING_DETAILS.md`
4. `docs/MULTIVIEW_MAINLINE.md`
