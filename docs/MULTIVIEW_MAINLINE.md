# Head/Left/Right 多视角主线

新版主线应该是 head/left/right 多视角训练。当前公共包已经包含可运行的单视角 baseline，但还没有把训练 dataset 和 experiment config 切到三视角。本文件记录已经确认的信息、需要确认的问题、以及后续 Codex 应该怎么改。

## 已确认的数据字段

Expert HDF5 数据里存在这些图像字段：

```text
observation/head_camera/rgb
observation/left_camera/rgb
observation/right_camera/rgb
observation/front_camera/rgb
third_view_rgb
```

当前单视角 baseline 只读取：

```text
observation/head_camera/rgb
```

Enhanced LeRobot metadata 里存在这些视频字段：

```text
observation.images.cam_high
observation.images.cam_front
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

当前单视角 baseline 只读取：

```text
observation.images.cam_high
```

## 需要先确认的命名问题

用户说的 head/left/right 主线，需要先确认 left/right 的含义：

1. Expert HDF5 里的 `left_camera` / `right_camera` 是外部左右视角。
2. Enhanced LeRobot 里的 `cam_left_wrist` / `cam_right_wrist` 是左右腕部视角。

这两套命名不完全一致。后续实现三视角训练前，必须确认要不要把它们映射为：

```text
head  -> expert: head_camera, enhanced: cam_high
left  -> expert: left_camera, enhanced: cam_left_wrist
right -> expert: right_camera, enhanced: cam_right_wrist
```

如果翔哥说的 left/right 是外部左右相机，而 enhanced 数据只有 wrist left/right，那么要么需要新的 enhanced 多视角数据，要么只在 expert 数据上做三视角。

## 推荐实现方向

不要在旧的读取函数里硬编码更多 if/else。建议把视角抽象成配置：

```python
views = ["head", "left", "right"]
```

再定义不同数据源的 view mapping：

```python
EXPERT_VIEW_TO_H5_KEY = {
    "head": "observation/head_camera/rgb",
    "left": "observation/left_camera/rgb",
    "right": "observation/right_camera/rgb",
}

LEROBOT_VIEW_TO_VIDEO_KEY = {
    "head": "observation.images.cam_high",
    "left": "observation.images.cam_left_wrist",
    "right": "observation.images.cam_right_wrist",
}
```

Dataset 返回值要明确设计。有两种选择：

### 方案 A：把多视角当作 batch 内随机视角增强

每个 sample 随机选一个 view，仍然返回：

```text
video: (3, 17, 256, 320)
```

优点：

- 对模型结构改动最小。
- 仍然兼容当前 action-conditioned 模型输入。
- 可以快速起第一版多视角数据训练。

缺点：

- 模型每次只看一个视角，不是真正同时融合三视角。

### 方案 B：三视角同时输入

每个 sample 返回：

```text
video: (V, 3, 17, 256, 320)
```

或者把 view 合并到 channel/batch 维度。

优点：

- 真正多视角融合。

缺点：

- 需要确认 CosmosPredict2.5 action-conditioned 模型是否支持这种输入。
- 可能需要改网络输入、tokenizer/vae 处理、collate、config 和显存预算。
- OOM 风险比单视角高很多。

建议第一版主线先做方案 A：随机 head/left/right 单视角采样，把数据覆盖面拉起来；等训练和推理稳定后，再评估方案 B 的模型结构改造。

## 需要改的代码位置

核心 dataset：

```bash
code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/datasets/dataset_afb_s1_family_balanced.py
```

需要改：

- `__init__` 增加 `views` 参数。
- `_build_lerobot_pool` 不再只记录 `cam_high`，而是按 view 记录对应 video file index/from timestamp。
- `_read_expert_frames` 根据 sample/view 读取对应 HDF5 key。
- `_read_lerobot_frames` 根据 sample/view 读取对应 video key。
- `_read_rf_frames` 需要确认 random feasible 数据是否有三视角；如果没有，先只用 head 或暂时从多视角训练中排除 RF。
- `__getitem__` 里记录 `view`，最好把 view 写入返回 dict，方便 debug 和日志分析。

Experiment config：

```bash
code/cosmos-predict2.5-CoRL/cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_afb_s1_family_balanced.py
```

需要改：

- 给 dataset 传入 `views=["head", "left", "right"]`。
- 实验名改成能看出多视角，例如：

```text
cosmos_predict2p5_2B_afb_s1_family_balanced_3_1_1_1_chunk16_head_left_right
```

启动脚本：

```bash
scripts/run_family_balanced_8gpu.sh
code/cosmos-predict2.5-CoRL/scripts/train_cosmos_8gpu_afb_s1_family_balanced_chunk16.sh
```

需要改：

- 新增独立多视角 launcher，不要覆盖当前已验证的单视角 baseline。
- Preflight 增加三视角样本读取检查。

## 多视角 preflight 应该检查什么

至少检查：

- Expert 数据的 head/left/right 三个 HDF5 key 都存在。
- Enhanced 数据的 cam_high/cam_left_wrist/cam_right_wrist 三个 metadata key 都存在。
- 每个 view 都能读取 17 帧。
- 每个 view 的输出 shape 都是 `(3, 17, 256, 320)`。
- Action shape 仍然是 `(16, 14)`。
- Family-balanced 四类数据都有样本。
- 如果 RF 没有三视角，要明确打印 RF 当前使用 head-only 或被排除。

## 显存和 batch 建议

如果采用方案 A，单次模型输入仍是单视角，显存接近当前 baseline，可以先沿用：

```bash
AFB_S1_PER_GPU_BATCH=2
```

如果采用方案 B，三视角同时输入，显存会显著增加。建议第一版从：

```bash
AFB_S1_PER_GPU_BATCH=1
```

开始，并先跑 smoke/dryrun，再提交长训练。

## 当前建议

当前要发给翔哥的版本，可以作为“单视角稳定 baseline + 多视角主线开发起点”。不要把它描述成已经完成 head/left/right 三视角训练。准确说法是：

```text
当前包已经能稳定启动 head/high 单视角 family-balanced 训练；
数据里已确认存在 head/left/right 相关字段；
下一阶段主线应基于该包实现 head/left/right 多视角训练。
```
