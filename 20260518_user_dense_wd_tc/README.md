# User-Dense UE + Int-Dense Pair Projector + Warmup/Weight Decay + torch.compile

## 改动概述

在 `20260518_user_dense_wd`（UserDenseUEPairProjector + warmup + weight decay）基础上，叠加 `torch.compile(mode="reduce-overhead")` 加速训练。

### 核心改动

**`train.py`**：
- 新增 `--use_torch_compile` CLI 参数
- 启用后通过 `torch.compile(model, mode="reduce-overhead")` 编译模型

**`run.sh`**：
- 默认启用 `--use_torch_compile`

### 涉及文件

| 文件 | 改动 |
|------|------|
| `train.py` | 新增 --use_torch_compile 参数；编译模型 |
| `run.sh` | 添加 --use_torch_compile 标志 |

其余文件（model.py, trainer.py, dataset.py, inference/）与 `20260518_user_dense_wd` 一致。

### 基线

本实验基于 `20260518_user_dense_wd`（UserDenseUEPairProjector + weight_decay 0.01 + warmup 100 steps）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
|             |         |               |          |                |
