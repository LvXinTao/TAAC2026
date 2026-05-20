# User-Dense-AutoToken: AutoNSTokenizer for Data-Driven Feature Grouping

## 改动概述

在 `20260519_user_dense_ema`（EMA + UserDenseUEPairProjector）基础上，将 NS Tokenizer 从 RankMixer 替换为 **AutoNSTokenizer**（AutoToken），实现数据驱动的离散特征动态分组。

### 动机

传统的离散特征分组方式（GroupNSTokenizer / RankMixerNSTokenizer）依赖人工规则：
- GroupNSTokenizer：每个分组产生恰好 1 个 NS token，token 数不可调
- RankMixerNSTokenizer：所有特征 embedding 拼接后机械等分，切割点忽视语义边界

AutoNSTokenizer 的思路（参考 MTMixAtt 论文 Section 3.2 AutoToken）：让模型通过 **可学习的特征选择矩阵 W** 从数据中自动学习特征分组，消除人工偏差。

### 核心改动

**`model.py`**：
- 新增 `AutoNSTokenizer` 类（约 100 行）
  - 可学习选择矩阵 `W ∈ R^{n_g × n_f}`，初始化为全零
  - Warm-start：当提供 ns_groups 时，将对应 group 的 fid 位置设为 1.0
  - Forward 时对每个 token 做 Top-k 选择 + Softmax 加权 + 投影
- `PCVRHyFormer.__init__` 新增 `auto_top_k`、`auto_temperature` 参数
  - 新增 `ns_tokenizer_type='auto'` 分支，构建 `AutoNSTokenizer`

**`train.py`**：
- `--ns_tokenizer_type` 新增 `'auto'` 选项
- 新增 `--auto_top_k`（默认 10，0 = 自动 ceil(n_f / num_ns_tokens)）
- 新增 `--auto_temperature`（默认 1.0，越大权重越均匀）

**`run.sh`**：
- 主配置改为 `--ns_tokenizer_type auto`
- 新增 `AUTO_TOP_K`、`AUTO_TEMPERATURE` 环境变量
- RankMixer 配置保留为注释备选

**`inference/model.py`**：与 `model.py` 完全同步
**`inference/infer.py`**：`_FALLBACK_MODEL_CFG` 补齐 `auto_top_k: 10`、`auto_temperature: 1.0`

### 涉及文件

| 文件 | 改动 |
|------|------|
| `model.py` | 新增 AutoNSTokenizer 类；PCVRHyFormer 新增 auto 分支和参数 |
| `train.py` | 新增 --auto_top_k、--auto_temperature CLI 参数；加入 model_args |
| `run.sh` | 主配置改为 auto，新增 AUTO_TOP_K/AUTO_TEMPERATURE 环境变量 |
| `inference/model.py` | 与 model.py 完全同步 |
| `inference/infer.py` | _FALLBACK_MODEL_CFG 补齐新参数 |

### 基线

本实验基于 `20260519_user_dense_ema`（UserDenseUEPairProjector + EMA decay 0.999 + weight_decay 0.01 + warmup 100 steps）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| - | - | - | - | - |

（待训练完成后更新）
