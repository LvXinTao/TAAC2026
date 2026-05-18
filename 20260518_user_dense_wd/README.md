# User-Dense UE + Int-Dense Pair Projector + Warmup/Weight Decay

## 改动概述

在 `20260514_warmup_weight_decay`（per-token independent FFN + LR warmup + weight decay）基础上，叠加 `20260518_user_dense_ue_pair` 的 **UserDenseUEPairProjector** 模块，将 user dense features 拆分为两类并分别处理。

### 动机

user dense features 中包含两类不同语义的特征：
1. **UE features** (fid 61,87,89,90,91): 仅有 dense 值，适合直接线性投影
2. **Int-Dense Pair features** (fid 62,63,64,65,66): 同时有 int embedding 和 dense 值，可用 dense 值作为权重对 int embedding 做加权池化，融合两种表征

两者联合 warmup/weight decay 的正则化手段，期望在提升特征表达能力的同时控制过拟合。

### 核心改动

**`model.py`**：
- 新增 `UserDenseUEPairProjector` 类，包含：
  - `user_dense_ue_embeddings`: UE 特征的独立 Embedding，直接投影为 d_model 维 token
  - `pair_int_embeddings`: Pair 特征的 int embedding，用 dense 值做加权池化（softmax 权重 × embedding 求和）
  - `user_dense_proj`: 剩余 dense 特征的 fallback 投影（保持兼容性）
- 新增 `_normalize_fid_list` 辅助函数，规范化 fid 配置
- `PCVRHyFormer.__init__` 新增 4 个参数：`user_int_feature_ids`, `user_dense_feature_specs`, `user_dense_ue_fids`, `user_int_dense_pair_fids`
- 新增 `_make_user_dense_token` 方法，在 forward/predict 中构造 user dense token

**`train.py`**：
- 新增 `--user_dense_ue_fids` 参数（默认 `''`）
- 新增 `--user_int_dense_pair_fids` 参数（默认 `''`）
- 将两个参数透传至 `PCVRHyFormer` 构造函数

**`run.sh`**：
- 配置 `USER_DENSE_UE_FIDS=61,87,89,90,91`
- 配置 `USER_INT_DENSE_PAIR_FIDS=62,63,64,65,66`
- 保留 `--weight_decay 0.01`、`--warmup_steps 100`

### 涉及文件

| 文件 | 改动 |
|------|------|
| `model.py` | 新增 UserDenseUEPairProjector；PCVRHyFormer 新增 4 参数；新增 _make_user_dense_token |
| `train.py` | 新增 --user_dense_ue_fids、--user_int_dense_pair_fids CLI 参数 |
| `run.sh` | 配置 UE/Pair fid 列表，保留 weight_decay/warmup |
| `inference/model.py` | 同 model.py 改动 + 替换 RankMixerBlock 为 per-token independent FFN |
| `inference/infer.py` | 新增 fallback 参数 + build_model 传入 user_int_feature_ids/user_dense_feature_specs |
| `inference/dataset.py` | 无 UE 特定改动（来自 amp_warmup_wd） |

### 基线

本实验基于 `20260514_warmup_weight_decay`（per-token independent FFN + weight_decay 0.01 + warmup 100 steps）代码。trainer.py 来自 `20260518_user_dense_ue_pair`（无 AMP）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| 0.22437 | 0.87082 | +0.00181 | 0.84825 | +0.00178 |

对比 `warmup-weight-decay`（val AUC 0.86901, test AUC 0.84647）：
- **val AUC** 提升 +0.00181（0.86901 → 0.87082）
- **test AUC** 提升 +0.00178（0.84647 → 0.84825）
- **val LogLoss** 降低 0.00137（0.22574 → 0.22437）

对比 `baseline`（val AUC 0.86427, test AUC 0.84186）：
- **val AUC** 提升 +0.00655
- **test AUC** 提升 +0.00640

最佳 checkpoint: `global_step38050.layer=2.head=4.hidden=64.best_model`

UE+Pair projector 在 warmup/weight decay 正则化基础上进一步提升了模型对用户 dense 特征的建模能力，是目前所有实验中提升最大的结果。
