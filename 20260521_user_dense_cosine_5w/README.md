# User-Dense-Cosine-5w (Warmup + Cosine Annealing LR, max_lr_steps=50000)

## 改动概述

在 `20260520_user_dense_cosine` 基础上，将 `--max_lr_steps` 从 10000 调整为 **50000**。

### 动机

`max_lr_steps=10000` 远小于实际训练步数（~91320），cosine 过早归零。
`max_lr_steps=100000` 覆盖了全部训练步数，lr 在 best step 时仍有余量。
折中方案 `max_lr_steps=50000`：
- T_max = 49900（warmup 后），cosine 在 step ~50000 时衰减到 0
- 前半段（~55% 训练步数）dense 参数有学习率，后半段 lr 归零
- 可用于验证 dense 参数是否主要在前中期学习，后期是否还需要更新

### LR 调度策略

- **Phase 1 (step 1 → 100)**: 线性 warmup，lr 从 0 → target lr
- **Phase 2 (step 101 → 50000)**: CosineAnnealingLR，lr 从 target lr 衰减到 0

### 核心改动

- `run.sh`: `--max_lr_steps 50000`（其余与 `20260520_user_dense_cosine` 相同）

### 涉及文件

| 文件 | 改动 |
|------|------|
| `run.sh` | `--max_lr_steps 50000` |

### 基线

本实验基于 `20260520_user_dense_cosine`（cosine annealing，max_lr_steps=10000），间接基于 `20260519_user_dense_ema`。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| 0.22330 (step 53270) | 0.87176 (step 53270) | +0.00054 | 0.84735 | -0.00187 |
