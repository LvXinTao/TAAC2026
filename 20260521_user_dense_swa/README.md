# User-Dense-UE-WD + SWA (Stochastic Weight Averaging)

## 改动概述

在 `20260519_user_dense_ema`（UserDenseUEPairProjector + warmup/weight decay + EMA）基础上，新增 **SWA（Stochastic Weight Averaging）** 对 dense 参数维护随机权重平均。推理时使用 SWA 权重。

### 动机

SWA 是深度学习中的经典优化 trick。与 EMA 每一步都做指数滑动平均不同，SWA 在训练后期每隔固定步数收集一次模型权重快照，对这些快照取算术平均。SWA 通常能找到更平坦的损失极小值，带来泛化能力的提升。

### 核心改动

**`trainer.py`**：
- 新增 `SWA` 类，维护 dense 参数的 running average
  - `__init__`: 通过 `model.get_dense_params()` 获取 dense 参数并初始化 swa_state
  - `should_update(step)`: 判断当前步是否应该收集快照（step > swa_start 且 step % swa_freq == 0）
  - `step(step)`: 收集快照并更新 running average: `new_avg = (old_avg * n + param) / (n + 1)`
  - `apply_to_model()`: 将 SWA 权重写入 model，返回原值用于恢复
  - `restore()`: 从 apply_to_model 返回值恢复训练权重
- `__init__` 新增 `swa_start`, `swa_freq` 参数，若 swa_freq > 0 则创建 SWA 实例
- `_train_step`: `dense_optimizer.step()` 后调用 `swa.step(total_step)`
- `_handle_validation_result`: 保存 checkpoint 前 swap 到 SWA 权重（优先 EMA，其次 SWA），保存后恢复训练权重

**`train.py`**：
- 新增 `--swa_start` 参数（默认 `0`，表示使用 warmup_steps）
- 新增 `--swa_freq` 参数（默认 `0` = 禁用）
- 推荐配置: `--swa_start 5000 --swa_freq 500`

### SWA vs EMA 对比

| 特性 | EMA | SWA |
|------|-----|-----|
| 更新频率 | 每一步 | 固定间隔 |
| 权重方式 | 指数衰减 (decay * shadow + (1-decay) * param) | 算术平均 (sum / n) |
| 起始时机 | 训练开始即生效 | swa_start 步后开始 |
| 超参数 | decay (0.999, 0.9999) | swa_start, swa_freq |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `trainer.py` | 新增 SWA 类；集成 SWA step 和 checkpoint swap |
| `train.py` | 新增 --swa_start, --swa_freq CLI 参数 |
| `run.sh` | 配置 swa_start=5000, swa_freq=500 |
| `inference/` | 无需改动（best_model 已经是 SWA 权重） |

### 基线

本实验基于 `20260519_user_dense_ema`（UserDenseUEPairProjector + weight_decay 0.01 + warmup 100 steps + EMA 0.999）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| 0.22436 | 0.87093 | +0.00188 | 0.84510 | -0.00414 |
