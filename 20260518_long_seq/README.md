# 20260518_long_seq — 长序列建模（LongerEncoder）

## 改动概述

在 `20260514_warmup_weight_decay`（per-token independent FFN + weight_decay 0.01 + warmup 100 steps）基础上，引入 **LongerEncoder** 对行为序列进行更长上下文的建模，同时扩大各序列域的最大长度上限。代码层面同步引入了 `UserDenseUEPairProjector`（来自 `20260518_user_dense_wd`）以使用 user dense UE / int-dense pair 特征。

### 动机

baseline 的序列编码器（Transformer）在默认长度限制下对长尾行为信息利用不足。通过引入支持更大感受野的 `LongerEncoder` 并放宽 `seq_max_lens`，期望充分利用用户更长的历史行为，提升模型对长序列的建模能力。

### 核心改动

**`run.sh`**：
- 新增 `--seq_max_lens "seq_a:256,seq_b:1024,seq_c:1024,seq_d:2048"`，放宽各域最大序列长度
- 新增 `--seq_encoder_type longer`，对所有序列域启用 LongerEncoder
- 新增 `--seq_top_k 128`，LongerEncoder 压缩后保留的 token 数
- 保留 `--user_dense_ue_fids`、`--user_int_dense_pair_fids`、`--weight_decay 0.01`、`--warmup_steps 100`

**其余文件**（`model.py`、`train.py`、`dataset.py`、`trainer.py`、`utils.py`、`inference/`）在 `20260514_warmup_weight_decay` 基础上叠加了 `UserDenseUEPairProjector` 相关代码（来自 `20260518_user_dense_wd`），无其他额外修改。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `run.sh` | 新增长序列参数：seq_max_lens、seq_encoder_type longer、seq_top_k 128 |
| `model.py` | 在 warmup_weight_decay 基础上增加 UserDenseUEPairProjector（来自 user_dense_wd） |
| `train.py` | 在 warmup_weight_decay 基础上增加 --user_dense_ue_fids / --user_int_dense_pair_fids 参数 |
| `inference/model.py` | 同 model.py，增加 UserDenseUEPairProjector 支持 |
| `inference/infer.py` | 同步更新 _FALLBACK_MODEL_CFG |

### 基线

本实验基于 `20260514_warmup_weight_decay`（per-token independent FFN + weight_decay 0.01 + warmup 100 steps）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| 0.22449 | 0.87035 | +0.00134 | 0.83924 | -0.00723 |

对比 `warmup-weight-decay`（val AUC 0.86901, test AUC 0.84647）：
- **val AUC** 提升 +0.00134（0.86901 → 0.87035）
- **test AUC** 下降 -0.00723（0.84647 → 0.83924）
- **val LogLoss** 降低 0.00125（0.22574 → 0.22449）

最佳 checkpoint: step 38050

**结论**：引入 LongerEncoder + 长序列配置后，验证集 AUC 略有提升（+0.00134），但测试集 AUC 明显下降（-0.00723），表明更长的序列编码在当前参数规模下存在**过拟合**风险，模型泛化能力下降。在此基础上 `20260518_long_seq_v2` 进一步尝试 per-domain encoder + d_model=128，但过拟合问题更为严重。
