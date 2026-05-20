# User-Dense-Cosine-10w (Warmup + Cosine Annealing LR, max_lr_steps=100000)

## 改动概述

在 `20260520_user_dense_cosine` 基础上，将 `--max_lr_steps` 从 10000 调整为 **100000**。

### 动机

`max_lr_steps=10000` 远小于实际训练步数（~91320），导致 cosine 衰减到 0 后 dense 参数停止更新，best model（step 68490）时 lr 早已归零。调整为 100000 后：
- T_max = 99900（warmup 后），best step ~68490 对应 cosine 进度 ~69%
- lr ≈ 0.08 * base_lr，仍在精细调优但已充分衰减

### LR 调度策略

- **Phase 1 (step 1 → 100)**: 线性 warmup，lr 从 0 → target lr
- **Phase 2 (step 101 → 100000)**: CosineAnnealingLR，lr 从 target lr 衰减到 0

### 核心改动

- `run.sh`: `--max_lr_steps 100000`（其余与 `20260520_user_dense_cosine` 相同）

### 涉及文件

| 文件 | 改动 |
|------|------|
| `run.sh` | `--max_lr_steps 100000` |

### 基线

本实验基于 `20260520_user_dense_cosine`（cosine annealing，max_lr_steps=10000），间接基于 `20260519_user_dense_ema`。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| - | - | - | - | - |
