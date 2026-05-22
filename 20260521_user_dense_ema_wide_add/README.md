# User-Dense-EMA + Wide&Deep Additive Fusion

## 改动概述

在 `20260519_user_dense_ema`（UserDenseUEPairProjector + warmup/weight decay + EMA）基础上，新增 **Wide&Deep 式加法融合分支**，将强特征（user/item NS tokens + dense tokens）以残差捷径方式直接输送到 classifier 前。

### 动机

当前模型中，user/item 强特征经过 NS Tokenizer → QueryGenerator → HyFormer Blocks → output_proj → clsfier，路径较深，原始信号在多层非线性变换中可能被稀释。Wide&Deep 结构通过独立的 wide 分支让强特征以低阶方式直接参与最终决策，兼顾特征记忆能力和泛化能力。

### 核心改动

**`model.py`**：
- `PCVRHyFormer.__init__` 新增 `use_wide_branch` 参数（默认 `False`）
- 当启用时，创建 `self.wide_branch`：
  ```python
  nn.Sequential(
      nn.Linear(num_ns * d_model, d_model),
      nn.LayerNorm(d_model),
      nn.SiLU(),
      nn.Dropout(dropout_rate),
  )
  ```
- `forward` / `predict` 中，在 HyFormer output 之后：
  ```python
  wide_feat = self.wide_branch(ns_tokens.view(B, -1))
  output = output + wide_feat  # 加法融合
  ```

**`train.py`**：
- 新增 `--use_wide_branch` 参数（默认 `False`）

### Wide 分支输入

`ns_tokens` 包含：
- user_int 特征经过 tokenizer 的 NS tokens
- user_dense 特征经过 UEPairProjector 的 token
- item_int 特征经过 tokenizer 的 NS tokens
- item_dense 特征经过 projection 的 token

这些 token 已经是强特征的浓缩表示，比直接用 raw features 更紧凑。

### 融合方式

采用 **加法融合**：`final_output = deep_output + wide_feat`

优势：
- 不改变 classifier 输入维度，零额外推理开销
- wide 分支学习一个 residual 偏移，类似残差连接
- 如果 wide 分支学到零，退化回原模型

### 涉及文件

| 文件 | 改动 |
|------|------|
| `model.py` | 新增 wide_branch 模块；forward/predict 中加法融合 |
| `train.py` | 新增 --use_wide_branch CLI 参数 |
| `trainer.py` | 无需改动（wide branch 参数自动归入 get_dense_params） |
| `inference/` | 无需改动 |

### 基线

本实验基于 `20260519_user_dense_ema`（UserDenseUEPairProjector + weight_decay 0.01 + warmup 100 steps + EMA 0.999）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| | | | | |
