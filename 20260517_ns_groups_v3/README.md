# NS Groups v2

## 改动概述

在 `20260514_warmup_weight_decay` 基础上，引入**基于 EDA 的语义化 NS 分组**（GroupNSTokenizer），将 user/item 离散特征按语义相关性分组，替代默认的每个特征独立成组方案。

### 动机

EDA 报告（`20260516_server_eda`）揭示了各特征的缺失率、基数、语义归属等信息。将语义相近的特征聚合到同一 NS token，有望让模型更好地学习特征间的协同关系，减少 NS token 数量的同时提高信息密度。

### 核心改动

**`ns_groups.json`**（新增）：
- 根据 EDA 报告设计的语义化分组方案
- **user 7 组**：U1_static_profile [1,3,4,49]、U2_activity [48,50,82]、U3_interests [51,52,53,54,55,57]、U4_beh_context [56,58,59,93,95,97,98]、U5_multival [15,80]、U6_aligned_ids [62,63,64,65,66,89,90]、U7_high_missing [60,86,91,92,94,96,99,100,101,102,103,104,105,106,107,108,109]
- **item 4 组**：I1_category [5,6,9,10,13]、I2_fine_category [7,8,12]、I3_ad_identity [16,81,11]、I4_high_missing [83,84,85]
- T = 1×4 + (7+1+4) = 16，d_model=64 满足整除约束

**`run.sh`**：
- 切换为 `--ns_tokenizer_type group` + `--ns_groups_json ns_groups.json`
- 保持 `--num_queries 1`、`--weight_decay 0.01`、`--warmup_steps 100`

### 涉及文件

| 文件 | 改动 |
|------|------|
| `ns_groups.json` | 新增语义化 NS 分组配置 |
| `run.sh` | 切换为 GroupNSTokenizer，传入 ns_groups.json |
| `inference/ns_groups.json` | 同步训练侧分组配置 |

### 基础

本实验基于 `20260514_warmup_weight_decay`（weight_decay=0.01 + warmup_steps=100）。

### 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| 0.22759 | 0.86526 | -0.00375 | 0.83781 | -0.00866 |

对比 `warmup-weight-decay`（val AUC 0.86901, test AUC 0.84647）：
- **val AUC** 下降 -0.00375（0.86901 → 0.86526）
- **test AUC** 下降 -0.00866（0.84647 → 0.83781）

GroupNSTokenizer + 语义化分组效果不及默认的 RankMixerNSTokenizer，未能带来提升。可能原因：分组方案将高缺失率特征孤立为一组（U7/I4），导致这些特征的信息被稀释；或 GroupNSTokenizer 本身表达能力弱于 RankMixerNSTokenizer。
