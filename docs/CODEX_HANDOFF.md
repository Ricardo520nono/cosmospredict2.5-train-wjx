# Codex 交接说明

目标：让后续负责训练的 Codex 能直接维护并启动 CosmosPredict2.5 在 ActionFollowingBench S1 上的训练，不需要反复重新排查路径、依赖和 config 问题。

## 当前稳定基线

稳定基线入口：

```bash
bash /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train/scripts/run_family_balanced_8gpu.sh
```

它对应之前已经跑通的训练脚本：

```bash
bash /mnt/gyc/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh
```

公共目录版本保持同样训练行为，但代码、权重和输出默认都指向：

```bash
/mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
```

## 不要随便改的约束

- Camera：只用 head camera。
- Family-balanced 混合比例：`expert:pca:raw:random-feasible = 3:1:1:1`。
- Chunk size：16。
- Action dim：14。
- 默认 batch：8 卡训练时每卡 2。
- WandB 默认开启。
- 默认输出位置：`outputs/cosmos_train_output`。

## 启动前必须先预检

运行：

```bash
cd /mnt/public_ckp/cscsx_projects/cosmospredict2.5_train
bash scripts/preflight.sh
```

如果 preflight 失败，先修 preflight，不要直接提交云端训练任务。大多数失败都是路径、import、Hydra config 或数据读取问题，本地就能提前发现。

## 后续编辑规则

- 代码修改放在 `code/cosmos-predict2.5-CoRL` 下。
- 顶层 `scripts/` 只保留薄而清晰的一键启动脚本。
- 只要改任务、路径、视角、chunk size、action dim、checkpoint 间隔或模型权重，就同步更新文档。
- 不要把旧实验草稿脚本、临时 debug 文档或无关 dirty 文件放进顶层交接包。

## GitHub 注意事项

本地公共目录包含可直接训练的模型权重，但不要把 `models/` 和 `outputs/` 当普通 git 文件推到 GitHub。GitHub 上只推代码、脚本和文档；模型权重保留在 `/mnt/public_ckp/...` 公共路径，除非仓库 owner 明确要求使用 Git LFS 管理权重。
