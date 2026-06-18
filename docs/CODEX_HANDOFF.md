# Codex 交接说明

本文档是给后续接手 CosmosPredict2.5 训练和测试闭环的 Codex 看的。接手时先读 `README.md`，再读本文档，然后读 `TRAINING_DETAILS.md` 和 `MULTIVIEW_MAINLINE.md`。

## 当前状态

公共训练包已经整理到：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
```

GitHub 仓库：

```bash
https://github.com/Ricardo520nono/cosmospredict2.5-train-wjx.git
```

当前已经验证通过的基线是 AFB S1 family-balanced 单视角训练：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/run_family_balanced_8gpu.sh
```

注意：新版主线是 head/left/right 多视角训练，但当前已跑通基线仍是 head/high 单视角。后续接新需求时，如果需求里提到“主线训练”，应优先考虑 `docs/MULTIVIEW_MAINLINE.md` 中的三视角方向，而不是继续深挖旧单视角实验。

## 对 `/mnt/gyc` 和 `/mnt/gyc_ckp` 的依赖结论

模型权重已经复制到公共目录，不再依赖 `/mnt/gyc_ckp`：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models
```

代码和启动脚本也已经复制到公共目录，不再依赖 `/mnt/gyc/cosmos-predict2.5-CoRL`：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/code/cosmos-predict2.5-CoRL
```

Python 环境默认值仍指向历史环境：

```bash
/mnt/gyc/cosmos-predict2.5/.venv
```

但脚本支持用 `COSMOS_VENV` 覆盖：

```bash
export COSMOS_VENV=/path/to/xiangge/cosmos-venv
```

所以，如果翔哥没有 `/mnt/gyc`，他需要先准备自己的 Python/CUDA 环境，然后设置 `COSMOS_VENV`。只看 GitHub 可以让 Codex 理解怎么接手，但实际训练还必须能访问公共数据和模型权重。

## 接手后第一件事

任何训练前先跑：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/preflight.sh
```

如果 preflight 失败，先修 preflight。不要直接提交 8 卡训练任务去赌，因为过去的失败大多来自路径、依赖、Hydra config、chunk/action 配置不一致，这些都能在 preflight 里提前发现。

## 当前不要随便改的约束

- Chunk size：16。
- Action dim：14。
- `state_t=5`。
- `temporal_compression_ratio=4`。
- `use_crossattn_projection=True`。
- `crossattn_proj_in_channels=100352`。
- `crossattn_emb_channels=1024`。
- Reason1 在线编码：`compute_online=True`。
- 文本 embedding concat：`embedding_concat_strategy=full_concat`。
- WandB 默认开启。
- Random feasible 视频缓存默认关闭：`AFB_S1_RF_VIDEO_CACHE_SIZE=0`。

## 后续 Codex 修改代码的规则

代码修改放在：

```bash
code/cosmos-predict2.5-CoRL
```

顶层 `scripts/` 只放清晰的一键启动脚本，不放临时 debug 脚本。

只要改了以下内容，就同步更新 Markdown：

- 任务集合。
- 数据路径。
- 视角。
- 数据混合比例。
- Chunk size。
- Action dim。
- Checkpoint 保存间隔。
- 模型权重路径。
- 环境变量或启动方式。

## GitHub 规则

GitHub 上推代码、脚本、文档，不推大模型权重和训练输出。

这些目录留在公共机器上，不进 GitHub：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/models
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/outputs
```

如果未来确实要让 GitHub 仓库在另一台机器上完全复现训练，需要额外准备模型下载脚本或 Git LFS 方案，但这不是当前版本的假设。
