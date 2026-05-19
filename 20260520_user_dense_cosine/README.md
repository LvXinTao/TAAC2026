# User-Dense-Cosine (Warmup + Cosine Annealing LR)

## 改动概述

在 `20260519_user_dense_ema`（EMA + warmup）基础上，新增 **Cosine Annealing LR** 衰减策略：warmup 阶段结束后，dense optimizer 的学习率从 full lr 开始按余弦曲线衰减至 0，而非一直保持 full lr。

### 动机

之前的实验（warmup + EMA）在 warmup 后学习率一直保持 full lr，没有衰减。Cosine Annealing 可以让训练后期以更小的学习率精细调优，通常能提升泛化性能。

### LR 调度策略

- **Phase 1 (step 1 → warmup_steps)**: 线性 warmup，lr 从 0 线性增长到 target lr
- **Phase 2 (step warmup_steps+1 → max_lr_steps)**: CosineAnnealingLR，lr 从 target lr 衰减到 0
- **max_lr_steps = 0**: 不启用衰减（退化为原有行为，warmup 后保持 full lr）

### 核心改动

**`trainer.py`**：
- `__init__` 新增 `max_lr_steps` 参数
- 新增 `self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_lr_steps - warmup_steps, eta_min=0)`
- `train()` 循环中：warmup 阶段结束后自动切换到 cosine scheduler

**`train.py`**：
- 新增 `--max_lr_steps` CLI 参数（默认 `0` = 不衰减）

**`run.sh`**：
- 添加 `--max_lr_steps 10000`（warmup 100 步后，从第 101 步开始 cosine 衰减到第 10000 步）

### 涉及文件

| 文件 | 改动 |
|------|------|
| `trainer.py` | 新增 cosine_scheduler；train loop 中 warmup → cosine 切换逻辑 |
| `train.py` | 新增 --max_lr_steps CLI 参数 |
| `run.sh` | 添加 --max_lr_steps 10000 |
| `inference/` | 无需改动 |

### 基线

本实验基于 `20260519_user_dense_ema`（UserDenseUEPairProjector + weight_decay 0.01 + warmup 100 + EMA 0.999）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| - | - | - | - | - |
