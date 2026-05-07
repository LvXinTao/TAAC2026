# Baseline 代码解读

## 概述

Baseline 实现了一个名为 **PCVRHyFormer** 的深度学习模型，用于腾讯广告算法竞赛（TAAC2026）中的**后点击转化率预测**（Post-Click Conversion Rate Prediction）任务。整体架构基于 HyFormer（Hybrid Transformer），核心思路是将用户/物品的离散特征和稠密特征编码为 Non-Sequence (NS) Token，与多域序列特征（seq_a/b/c/d）通过多轮交叉注意力和 Token Mixing 进行融合，最终输出点击/转化的预测 logit。

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `model.py` | 模型定义：RoPE 注意力、序列编码器、HyFormer Block、NS Tokenizer、PCVRHyFormer 主模型 |
| `dataset.py` | 数据加载：Parquet 数据集读取、Feature Schema 管理、分桶时间编码 |
| `trainer.py` | 训练循环：双优化器、Early Stopping、Checkpoint 管理、评估 |
| `train.py` | 训练入口：参数解析、模型/数据构建、启动训练 |
| `utils.py` | 工具函数：日志、随机种子、Focal Loss、EarlyStopping |
| `ns_groups.json` | NS 特征分组配置（GroupNSTokenizer 使用） |
| `run.sh` | 启动脚本（默认使用 RankMixer NSTokenizer） |

---

## 一、数据层（`dataset.py`）

### 1.1 FeatureSchema

`FeatureSchema` 记录每个特征的 `(feature_id, offset, length)`，用于在展平的特征张量中定位特定特征的片段。支持四类特征：

- `int_value`：标量整数特征（length=1）
- `int_array`：多值离散特征（length=数组长度）
- `float_value`：标量稠密特征（length=1）
- `float_array`：变长稠密特征（length=数组长度）

### 1.2 PCVRParquetDataset

基于 `IterableDataset`，直接从多列 Parquet 文件读取数据，主要优化：

- **预分配 numpy buffer**：消除 `np.zeros` + `np.stack` 开销
- **融合序列 padding 循环**：直接写入 3D buffer
- **预计算列索引查找表**：避免逐行字符串查找
- **file_system tensor sharing**：避免多 DataLoader worker 时 `/dev/shm` 耗尽
- **shuffle buffer**：在 `buffer_batches` 大小的窗口内打乱 RowGroup 级别数据

数据读取流程：
```
Parquet文件 → 遍历RowGroup → iter_batches → _convert_batch → 预分配buffer填充 → 返回Dict[str, Tensor]
```

### 1.3 时间分桶编码

序列特征中包含时间戳列，代码通过 `BUCKET_BOUNDARIES`（64 个边界）将时间差映射到 65 个桶（含 padding=0），每个桶对应一个可学习的时间 embedding，叠加到序列 token embedding 上。

### 1.4 训练/验证切分

按 RowGroup 切分：最后 `valid_ratio` 比例的 RowGroup 作为验证集，其余作为训练集。支持 `train_ratio` 控制训练数据使用量。

---

## 二、模型层（`model.py`）

### 2.1 整体架构

```
输入特征 ──┬── 用户离散特征 ──→ NS Tokenizer ──→ User NS Tokens
          ├── 物品离散特征 ──→ NS Tokenizer ──→ Item NS Tokens
          ├── 用户稠密特征 ──→ Linear+LN ──────→ User Dense Token
          ├── 物品稠密特征 ──→ Linear+LN ──────→ Item Dense Token
          └── 多域序列 ──────→ Embedding+TimeBucket ──→ Seq Tokens
                                    │
                    ┌───────────────┼───────────────┐
                    │    QueryGenerator              │
                    │  (NS + MeanPool(Seq) → Q)     │
                    └───────────────┼───────────────┘
                                    ▼
              ┌─────────── MultiSeqHyFormerBlock × N ───────────┐
              │  1. Sequence Evolution (各域独立编码)              │
              │  2. Query Decoding (Cross Attention Q↔Seq)       │
              │  3. Token Fusion (所有Q + NS concat)             │
              │  4. Query Boosting (RankMixer Token Mixing+FFN)  │
              └─────────────────────────┬────────────────────────┘
                                        ▼
                              Output Project → Classifier
                                        ▼
                                  Logits (B, 1)
```

### 2.2 NS Tokenizer（Non-Sequence Token 构造）

提供两种模式：

**GroupNSTokenizer**（`ns_tokenizer_type=group`）：
- 按 `ns_groups.json` 中的分组，每组内的离散特征各自 embedding 后拼接
- 每组通过一个 `Linear + LayerNorm` 投影为一个 NS Token
- NS Token 数量 = 用户组数 + 物品组数

**RankMixerNSTokenizer**（`ns_tokenizer_type=rankmixer`，默认）：
- 所有特征的 embedding 拼接为一个长向量
- 均分为 `num_ns_tokens` 段，每段独立投影到 `d_model`
- Token 数量可自由设定（不受组数限制）

此外，稠密特征（user_dense/item_dense）各自通过 `Linear + LayerNorm` 投影为一个单独的 NS Token。

### 2.3 RoPE 旋转位置编码

`RotaryEmbedding` 预计算并缓存 cos/sin 值，应用于 Q 和 K 的注意力计算中。支持：
- 自注意力中 Q/K 独立编码
- Cross Attention 中 Q 使用独立的位置索引（LongerEncoder 场景）

### 2.4 序列编码器（3 种可选）

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| `swiglu` | 无注意力，纯 SwiGLU FFN + Residual | 轻量、快速 |
| `transformer` | 标准 Transformer Encoder Layer（Pre-LN）+ RoPE 自注意力 | 高容量、通用 |
| `longer` | Top-K 压缩编码器：长序列用 Cross Attention 压缩到 top_k，短序列用自注意力 | 超长序列 |

### 2.5 MultiSeqQueryGenerator

为每个序列独立生成 Query Token：
```
GlobalInfo_i = Concat(NS_Tokens, MeanPool(Seq_i))
Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,Nq}(GlobalInfo_i)]
```

每个序列有自己的 Nq 个独立 FFN，从全局信息中生成专属 Query Token。

### 2.6 MultiSeqHyFormerBlock

每个 Block 包含四个阶段：

1. **Sequence Evolution**：各序列通过独立的序列编码器独立演化
2. **Query Decoding**：各序列的 Q Token 通过 Cross Attention 从对应序列中解码信息
3. **Token Fusion**：所有序列的解码 Q Token + NS Token 拼接
4. **Query Boosting**：通过 RankMixerBlock 进行 Token Mixing 和 FFN 增强

### 2.7 RankMixerBlock

提供三种模式：

- `full`：Token Mixing（无参数张量重排）+ 共享参数 FFN + Residual
- `ffn_only`：仅共享参数 FFN
- `none`：Identity 直通

Token Mixing 的核心操作：将 `(B, T, D)` 重塑为 `(B, T, T, d_sub)`，交换 token 和 subspace 轴后再展平，实现参数免费的跨 token 信息交互。要求 `d_model % T == 0`。

### 2.8 双优化器策略

- **Adagrad**：优化所有 Embedding 表参数（稀疏参数）
- **AdamW**：优化所有非 Embedding 参数（稠密参数）

### 2.9 高基数 Embedding 冷重启

从指定 epoch 开始，每轮结束时重新初始化基数（vocab_size）超过阈值的 Embedding，并重建 Adagrad 优化器状态（保留低基数 Embedding 的优化器状态）。这是参考快手 MultiEpoch 论文的防过拟合技巧。

---

## 三、训练层（`trainer.py` + `train.py`）

### 3.1 训练流程

```
for epoch in 1..num_epochs:
    for batch in train_loader:
        _train_step(batch)              # 前向 → loss → 反向 → 双优化器 step
        if eval_every_n_steps > 0:      # 可选的 step 级别验证
            evaluate()
            _handle_validation_result()

    evaluate()                          # epoch 级别验证
    _handle_validation_result()         # EarlyStopping checkpoint 管理

    if epoch >= reinit_sparse_after_epoch:
        reinit_high_cardinality_params()  # 高基数 Embedding 冷重启
```

### 3.2 损失函数

- **BCE**：`F.binary_cross_entropy_with_logits`
- **Focal Loss**：`sigmoid_focal_loss`（可配置 alpha/gamma，默认 alpha=0.1, gamma=2.0）

### 3.3 EarlyStopping

监控验证集 AUC（越高越好），连续 `patience` 次验证无提升则终止训练。每次提升时保存模型 checkpoint（含 schema.json、ns_groups.json、train_config.json 等 sidecar 文件）。

### 3.4 Checkpoint 管理

每次验证产生新最优模型时：
1. 创建 `global_stepN.layer=X.head=Y.hidden=Z.best_model/` 目录
2. 保存 `model.pt`
3. 复制 `schema.json`、`ns_groups.json`、`train_config.json` 作为 sidecar
4. 删除旧的 `*.best_model` 目录

### 3.5 默认配置（`run.sh`）

```bash
--ns_tokenizer_type rankmixer   # 使用 RankMixer NSTokenizer
--user_ns_tokens 5              # 用户侧 5 个 NS Token
--item_ns_tokens 2              # 物品侧 2 个 NS Token
--num_queries 2                 # 每个序列生成 2 个 Query Token
--emb_skip_threshold 1000000    # 基数 > 100万的特征不创建 Embedding
--num_workers 8                 # DataLoader worker 数
```

同时注释了一个备选配置（GroupNSTokenizer + ns_groups.json + num_queries=1）。

---

## 四、关键约束

1. **d_model 整除约束**：当 `rank_mixer_mode=full` 时，`d_model` 必须能被 `T = num_queries * num_sequences + num_ns` 整除
2. **num_heads 约束**：`d_model % num_heads == 0`
3. **时间桶数量**：由 `BUCKET_BOUNDARIES` 长度唯一确定为 65

---

## 五、快速上手

```bash
# 默认配置启动（RankMixer 模式）
cd baseline && bash run.sh \
    --data_dir /path/to/data \
    --ckpt_dir /path/to/checkpoints \
    --log_dir /path/to/logs

# GroupNSTokenizer 模式（需修改 run.sh 注释切换）
```

环境变量优先级高于 CLI 参数：
- `TRAIN_DATA_PATH` → `--data_dir`
- `TRAIN_CKPT_PATH` → `--ckpt_dir`
- `TRAIN_LOG_PATH` → `--log_dir`
- `TRAIN_TF_EVENTS_PATH` → TensorBoard 日志目录
