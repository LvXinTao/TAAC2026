# Token-Specific Q for Cross-Attention

## 改动概述

在 query decoding 的 cross-attention 中，将共享的 Q 矩阵改为 token-specific（每个 global token 位置拥有独立的 Q 矩阵）。

### 动机

目前 query decoding 时，global token 作为 Q，sequence 作为 KV，所有 global token 共享同一个 Q 矩阵。不同 token 位置（如 q1、q2）可能关注序列的不同方面，token-specific Q 允许每个位置学习专属的查询投影。

### 核心改动

**`model.py`** — 新增 `TokenSpecificQProjection` 类：
- 为每个 token 位置创建独立的 `[num_tokens, d_model, d_model]` 权重矩阵和偏置
- 使用 `torch.einsum('bnd,ndm->bnm', query, W_q) + bias` 高效批处理

**`model.py`** — 修改 `CrossAttention` 类：
- 新增 `num_queries` 和 `token_specific_q` 参数
- 当 `token_specific_q=True` 时，在 attention 前对 query 做 token-specific 投影

**`model.py`** — 透传至 `MultiSeqHyFormerBlock` 和 `PCVRHyFormer`：
- `MultiSeqHyFormerBlock` 新增 `token_specific_q` 参数，传递给 `CrossAttention`
- `PCVRHyFormer` 新增 `token_specific_q` 参数，传递给各 block

**`train.py`** — 新增 `--token_specific_q` CLI 参数

### 实验配置

- 基于 `20260510_timestamp_features` 代码
- `num_epochs=8`（原为 999，受 early stopping 控制）
- 其余配置与 timestamp_features 一致

### 涉及文件

| 文件 | 改动 |
|------|------|
| `model.py` | 新增 TokenSpecificQProjection；修改 CrossAttention/MultiSeqHyFormerBlock/PCVRHyFormer |
| `train.py` | 新增 --token_specific_q 参数 |
| `run.sh` | 添加 --num_epochs 8 和 --token_specific_q |

### 基础

本实验基于 `20260510_timestamp_features` 代码。
