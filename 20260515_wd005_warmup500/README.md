# Weight Decay 0.05 + Warmup 500 Steps

## 改动概述

在 `20260514_warmup_weight_decay`（wd=0.01, warmup=100）基础上，将正则化强度进一步调大：
- **Weight Decay**: 0.01 → **0.05**
- **LR Warmup**: 100 → **500 steps**

### 动机

上一轮实验（wd=0.01, warmup=100）已取得 test AUC +0.00146 的收益，说明 weight decay + warmup 的方向正确。本实验尝试加大正则化力度，看是否能进一步压缩 val/test gap、提升 test 泛化。

### 核心改动

仅调整 `run.sh` 中的超参数，代码逻辑与 `20260514_warmup_weight_decay` 完全一致。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `run.sh` | `--weight_decay 0.05 --warmup_steps 500` |

### 基础

本实验基于 `20260514_warmup_weight_decay`（warmup + weight decay 代码）代码。
