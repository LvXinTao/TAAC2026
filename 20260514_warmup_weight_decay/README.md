# Warmup + Weight Decay

## 改动概述

在 `20260510_timestamp_features` 基础上添加两项正则化手段以缓解过拟合：
- **Weight Decay**: dense 参数加入 L2 正则化（默认 0.0 → 实验用 0.01）
- **LR Warmup**: dense optimizer 线性预热学习率（默认 0 → 实验用 100/500 steps）

### 动机

实验数据显示 val AUC 比 test AUC 稳定高出 ~0.022，存在明显的过拟合/分布偏移问题。`time_split` 实验进一步验证时间泛化是核心瓶颈。weight decay 直接约束 dense 参数范数，warmup 让 embedding 先稳定学习再逐步提高 LR，两者联合可缩小 train/val 间的泛化差距。

### 核心改动

**`train.py`**：
- 新增 `--weight_decay` 参数（float，默认 0.0）
- 新增 `--warmup_steps` 参数（int，默认 0）
- 将两个参数透传至 trainer

**`trainer.py`**：
- `__init__` 接收 `weight_decay` 并传入 dense optimizer (AdamW)
- `__init__` 接收 `warmup_steps` 并创建 LambdaLR warmup scheduler
- `_train_step` 之后调用 warmup scheduler（仅在前 warmup_steps 内生效）
- TensorBoard 新增 `LR/train` 指标，可视化学习率曲线

### 涉及文件

| 文件 | 改动 |
|------|------|
| `train.py` | 新增 `--weight_decay`、`--warmup_steps` CLI 参数并透传 |
| `trainer.py` | weight decay 接入 AdamW；LR warmup scheduler；TensorBoard LR 记录 |

### 基础

本实验基于 `20260510_timestamp_features`（hour + day_of_week 时间特征）代码。
