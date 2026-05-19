# User-Dense-UE-WD + EMA (Exponential Moving Average)

## 改动概述

在 `20260518_user_dense_wd`（UserDenseUEPairProjector + warmup/weight decay）基础上，新增 **EMA（Exponential Moving Average）** 对 dense 参数维护滑动平均权重。推理时使用 EMA 权重。

### 动机

EMA 是 CTR 领域的经典 trick。训练过程中对 dense 参数维护指数滑动平均，推理时使用 EMA 权重可以平滑优化轨迹中的噪声，通常带来 free gain。

### 核心改动

**`trainer.py`**：
- 新增 `EMA` 类，维护 dense 参数的 shadow 副本
  - `__init__`: 通过 `model.get_dense_params()` 获取 dense 参数并拷贝 shadow
  - `step()`: 每次 `dense_optimizer.step()` 后更新 shadow: `shadow = decay * shadow + (1 - decay) * param`
  - `apply_to_model()`: 将 EMA 权重写入 model，返回原值用于恢复
  - `restore()`: 从 apply_to_model 返回值恢复训练权重
- `__init__` 新增 `ema_decay` 参数，若 > 0 则创建 EMA 实例
- `_train_step`: `dense_optimizer.step()` 后调用 `ema.step()`
- `_handle_validation_result`: 保存 checkpoint 前 swap 到 EMA 权重，保存后恢复训练权重

**`train.py`**：
- 新增 `--ema_decay` 参数（默认 `0.0` = 禁用）
- 推荐值: `0.999` 或 `0.9999`

### 涉及文件

| 文件 | 改动 |
|------|------|
| `trainer.py` | 新增 EMA 类；集成 EMA step 和 checkpoint swap |
| `train.py` | 新增 --ema_decay CLI 参数 |
| `inference/` | 无需改动（best_model 已经是 EMA 权重） |

### 基线

本实验基于 `20260518_user_dense_wd`（UserDenseUEPairProjector + weight_decay 0.01 + warmup 100 steps）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| TBD | TBD | TBD | TBD | TBD |
