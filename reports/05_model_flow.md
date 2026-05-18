# PCVRHyFormer 核心逻辑梳理

> 基于 `20260516_ns_groups_v2` 版本代码

---

## 一、特征类型与处理流程

### 1. 用户/商品整型离散特征（`user_int_feats` / `item_int_feats`）

**数据来源与 Shape 变化：**

```
原始 Parquet 列（scalar int 或 list<int>）
  ↓ _convert_batch
  [B, total_int_dim]  (int64, 拼接所有 fid)
  ↑ 包含合成特征: hour(+1) 和 day_of_week(+1)
```

**处理细节：**

- **scalar int**：直接读取，值 ≤ 0 → 0（padding）
- **list/multi-hot**：`_pad_varlen_int_column` → `[B, dim]`，截断到 `dim`，值 ≤ 0 → 0
- **时间戳合成特征**：从 `timestamp` 列提取 `hour`（0-23，+1 写入）和 `day_of_week`（0-6，+1 写入），追加在 `user_int` 末尾

**模型内（NS Tokenizer）：**

```
[B, total_int_dim]
  ↓ Embedding lookup（每个 fid 独立 Embedding table）
  ↓ 单值 fid:   [B, emb_dim]
    多值 fid:   [B, length, emb_dim] → masked mean pool → [B, emb_dim]
  ↓ 按 NS tokenizer 类型分两路：
```

**GroupNSTokenizer**（`ns_tokenizer_type='group'`）：

```
每组:  [B, num_fids*emb_dim] → Linear + LN + SiLU → [B, D] → unsqueeze → [B, 1, D]
所有组 cat → [B, num_groups, D]
```

**RankMixerNSTokenizer**（`ns_tokenizer_type='rankmixer'`，当前版本默认）：

```
所有 fid emb 拼接: [B, total_fids * emb_dim]
  ↓ 若需要，pad 至可被 num_ns_tokens 整除
  [B, padded_total_dim]
  ↓ split 成 num_ns_tokens 个 chunk
  [B, chunk_dim] × num_ns_tokens
  ↓ 每 chunk 独立 Linear + LN + SiLU → [B, 1, D]
  ↓ cat → [B, num_ns_tokens, D]
```

---

### 2. 用户 Dense 特征（`user_dense_feats`）

```
原始: list<float>，变长
  ↓ _pad_varlen_float_column → [B, user_dense_dim]  (float32)
  ↓ model forward: Linear(user_dense_dim → D) + LN + SiLU
  → [B, D] → unsqueeze(1) → [B, 1, D]  （1 个 NS token）
```

---

### 3. 序列特征（`seq_data` / `seq_lens` / `seq_time_buckets`）

**数据来源与 Shape 变化：**

```
原始 Parquet: 每个 domain 有 S 个 sideinfo 列（list<int64>）+ 1 个 ts 列
  ↓ _convert_batch 写入 3D buffer
  seq_data[domain]:         [B, S, max_len]   (int64, S = sideinfo fid 数)
  seq_lens[domain]:         [B]
  seq_time_buckets[domain]: [B, max_len]       ← 时间差分桶 id（1~64，0=padding）
```

**时间桶计算（64 条边界 → 65 个桶）：**

```
time_diff = max(request_timestamp - seq_timestamps, 0)  → [B, max_len]
  ↓ np.searchsorted(BUCKET_BOUNDARIES, ...)
  ↓ clip + +1 → bucket_id in [1, 64]
  ↓ seq_timestamps == 0 的位置 → bucket_id = 0（padding）
```

**模型内（`_embed_seq_domain`）：**

```
seq_data[domain]: [B, S, max_len]
  ↓ 对每个 sideinfo fid: Embedding lookup（高基数 fid 训练时额外 2x dropout）
    每个 fid: [B, max_len, emb_dim]
  ↓ cat over S → [B, max_len, S*emb_dim]
  ↓ Linear(S*emb_dim → D) + LN → GELU → [B, max_len, D]

seq_time_buckets[domain]: [B, max_len]
  ↓ time_embedding（nn.Embedding(65, D, padding_idx=0)）
  → [B, max_len, D]

token_emb = token_emb + time_bucket_emb  → [B, max_len, D]
```

**Padding mask 生成：**

```
seq_lens[domain]: [B]
  ↓ _make_padding_mask
  → [B, max_len]  （True = padding 位置）
```

---

## 二、模型各模块核心作用与 Shape 变化

### 整体 Forward 流水线

```
user_int [B, U_int]  →  user_ns_tokenizer  →  [B, M_u, D]
item_int [B, I_int]  →  item_ns_tokenizer  →  [B, M_i, D]
user_dense [B, U_d]  →  user_dense_proj    →  [B, 1,   D]
item_dense (空)

ns_tokens = cat([user_ns, user_dense_tok, item_ns])  →  [B, num_ns, D]
  ↑ num_ns = M_u + 1 + M_i

seq_data[d] [B, S, L] → _embed_seq_domain → [B, L, D]  （每个 domain d）
  ↓
MultiSeqQueryGenerator  ← ns_tokens
  → q_tokens_list: S × [B, Nq, D]

MultiSeqHyFormerBlock × num_blocks
  → q_tokens_list: S × [B, Nq, D]  （更新）
  → ns_tokens: [B, num_ns, D]       （更新）

all_q = cat(q_tokens_list, dim=1) → [B, Nq*S, D]
  ↓ view → [B, Nq*S*D]
  ↓ output_proj（Linear + LN）→ [B, D]
  ↓ clsfier（Linear + LN + SiLU + Dropout + Linear）→ [B, action_num]
```

---

### 模块 1：NS Tokenizer

| | GroupNSTokenizer | RankMixerNSTokenizer |
|---|---|---|
| 输入 | `[B, total_int_dim]` | `[B, total_int_dim]` |
| 每 fid embed | `[B, emb_dim]` | `[B, emb_dim]` |
| 聚合方式 | 按 group 拼接，每 group 一个 proj | 全部 fid 拼接，split 成 T 段 |
| 输出 | `[B, num_groups, D]` | `[B, num_ns_tokens, D]` |

**核心作用**：将高维离散特征（用户/商品画像）压缩成若干语义 token，作为非序列侧的主干特征表示，也是后续 Q token 生成和 QueryBoosting 的全局上下文来源。

---

### 模块 2：`_embed_seq_domain`（序列嵌入）

```
输入: seq [B, S, L],  time_buckets [B, L]
  ↓ 每个 sideinfo fid: Embedding → [B, L, emb_dim]
  ↓ cat over S → [B, L, S*emb_dim]
  ↓ Linear + LN → GELU → [B, L, D]
  ↓ + time_embedding(time_buckets) [B, L, D]
输出: [B, L, D]
```

**核心作用**：把序列中每个 item 的多字段侧信息（id、类目等）拼接成 item token 表示，并通过时间差分桶 embedding 融入时间感知（用户行为距当前请求的时间间隔），让模型区分近期和远期行为。

---

### 模块 3：MultiSeqQueryGenerator（查询生成器）

```
输入:
  ns_tokens: [B, num_ns, D]
  seq_tokens_list: S × [B, L_i, D]
  seq_padding_masks: S × [B, L_i]

对每个 sequence i:
  MeanPool(seq_tokens_i, mask) → [B, D]
  GlobalInfo_i = cat(ns_tokens.view(B,-1), seq_pooled_i)
               = [B, (num_ns+1)*D]  → LayerNorm
  每个 query n: FFN_{i,n}(GlobalInfo_i) → [B, D]
  q_tokens_i = stack(queries) → [B, Nq, D]

输出: q_tokens_list: S × [B, Nq, D]
```

**核心作用**：为每条序列定制化生成 query token——融合"全局静态信息（NS tokens）+ 当前序列均值表示"，作为后续 cross-attention 的 Query 初始值，让每条序列的 Q 与其自身语义对齐，避免跨域干扰。

---

### 模块 4：MultiSeqHyFormerBlock（核心迭代块，执行 N 次）

每个 block 内部执行四步：

#### Step 1 — Sequence Evolution（序列自演化）

每条序列独立经过 `seq_encoder`，支持三种类型：

| encoder_type | 机制 | Shape 变化 |
|---|---|---|
| `swiglu` | LN + SwiGLU FFN + residual，无 attention | `[B, L, D] → [B, L, D]` |
| `transformer` | Pre-LN + RoPE self-attention + FFN | `[B, L, D] → [B, L, D]` |
| `longer` | 自适应：L > top_k 则 cross-attn 压缩；L ≤ top_k 则 causal self-attn | `[B, L, D] → [B, top_k, D]`（首次压缩） |

**LongerEncoder 压缩逻辑（L > top_k 时）：**

```
x [B, L, D]
  ↓ _gather_top_k: 选最近 top_k 个有效 token → [B, top_k, D]
    同时收集各 token 的原始位置索引 → [B, top_k]（用于 Q-side RoPE）
  ↓ Q = top_k tokens（各自带原始位置的精确 RoPE cos/sin，via torch.gather）
    K/V = 全部 L tokens（带顺序 RoPE）
  ↓ Cross-Attention（Pre-LN）→ [B, top_k, D]
  ↓ FFN → [B, top_k, D]
输出: [B, top_k, D]  （序列被压缩，后续 block 走 self-attn 路径）
```

#### Step 2 — Query Decoding（查询解码）

每条序列独立：

```
q_tokens_i [B, Nq, D]    （Query）
seq_tokens_i [B, L', D]  （Key/Value，已经过 Seq Evolution）
  ↓ CrossAttention（Pre-LN，rope_on_q=False → 只给 K 施加 RoPE，Q 不加位置编码）
  → residual + attn_out → [B, Nq, D]
```

**核心作用**：Q token 通过 cross-attention 从更新后的序列中"读取"与自身相关的序列信息，实现序列感知的 Q 更新。

#### Step 3 — Token Fusion（融合）

```
decoded_qs: S × [B, Nq, D]
ns_tokens:  [B, num_ns, D]
  ↓ cat along dim=1 → combined [B, T, D]
    T = Nq*S + num_ns
```

#### Step 4 — RankMixerBlock / Query Boosting（信息混合与增强）

```
combined [B, T, D]

  ── token_mixing（mode='full'，参数无关）──
  [B, T, D] → view [B, T, T, d_sub]（d_sub = D // T）
            → transpose(1, 2)         [B, T, T, d_sub]
            → view [B, T, D]          （各 token 的子空间相互交换）

  ── per-token independent FFN ──
  对每个位置 i（共 T 个，权重互不共享）:
    LN(Q[:, i, :]) → Linear(D→4D) → GELU → Dropout → Linear(4D→D)
    + residual（原始 Q[:, i, :]）→ post-LN → [B, D]
  stack → [B, T, D]

  ↓ split 回:
    next_q_list: S × [B, Nq, D]
    next_ns:     [B, num_ns, D]
```

**核心作用**：`RankMixerBlock` 是跨序列、跨 NS 的全局信息交互核心：
- **token_mixing**：无参数的 reshape/transpose，让所有 Q token 和 NS token 的不同子空间相互交换，实现全局信息流通
- **per-token FFN**：每个 token 位置独立学习如何提炼混合后的信息，参数不共享，保留 token 的专属语义

---

### 模块 5：RotaryEmbedding + RoPEMultiheadAttention

**RoPE 位置编码：**

```
head_dim = D // num_heads
inv_freq: [head_dim//2]
cos/sin cache: [1, max_seq_len, head_dim]

Q/K [B, num_heads, L, head_dim]
  ↓ rotate_half + cos/sin 相乘
  → 位置感知的 Q/K，相对位置通过内积自然体现
```

**Gated Output（`W_g`，初始化偏置为 1）：**

```
out = SDPA(Q, K, V)  → [B, num_heads, L, head_dim]
  ↓ reshape → [B, L, D]
  ↓ * sigmoid(W_g(query))   （软门控，初始接近 1）
  ↓ W_o → [B, L, D]
```

**LongerEncoder 中的精确 Q-side RoPE：**

Q token 来自序列中的 top_k 位置（非连续），通过 `torch.gather` 从全局 cos/sin cache 中按原始位置索引提取对应的 cos/sin，确保压缩后各 Q token 保留原始绝对位置信息。

---

### 模块 6：Output Projection + Classifier

```
all_q = cat(curr_qs, dim=1)  → [B, Nq*S, D]
  ↓ view  → [B, Nq*S*D]
  ↓ Linear(Nq*S*D → D) + LN  → [B, D]
  ↓ Linear(D → D) + LN + SiLU + Dropout + Linear(D → action_num)
  → [B, action_num]  （CVR 预测 logit）
```

---

## 三、整体架构总览

```
user_int [B,U]   item_int [B,I]   user_dense [B,Ud]
    ↓                 ↓                 ↓
NS Tokenizer      NS Tokenizer      Dense Proj
    ↓                 ↓                 ↓
[B, Mu, D]        [B, Mi, D]        [B, 1, D]
         ↘              ↓            ↙
          ns_tokens = [B, num_ns, D]
                         │
         ┌───────────────┘
         │
seq[d1] [B,S1,L1]   seq[d2] [B,S2,L2]  ...
    ↓ embed               ↓ embed
[B, L1, D]            [B, L2, D]
    │                     │
    └──── MultiSeqQueryGenerator ←── ns_tokens
               ↓
    q_list: [[B,Nq,D], [B,Nq,D], ...]  （每条序列一组 Q）
               ↓
    ┌──────────────────────────────────────────┐
    │   MultiSeqHyFormerBlock × N              │
    │   ┌──────────────────────────────────┐   │
    │   │ Step1: Seq Evolution（独立 per seq）│   │
    │   │   swiglu / transformer / longer  │   │
    │   │   [B, L, D] → [B, L', D]        │   │
    │   │                                  │   │
    │   │ Step2: Query Decoding（独立 per seq）│   │
    │   │   CrossAttn: Q ← seq             │   │
    │   │   [B, Nq, D]                     │   │
    │   │                                  │   │
    │   │ Step3: Token Fusion              │   │
    │   │   cat all Q + NS → [B, T, D]    │   │
    │   │                                  │   │
    │   │ Step4: RankMixerBlock            │   │
    │   │   token_mixing（参数无关 reshape） │   │
    │   │   per-token FFN（独立 per 位置）  │   │
    │   │   → [B, T, D] → split Q, NS     │   │
    │   └──────────────────────────────────┘   │
    └──────────────────────────────────────────┘
               ↓
    cat Q → [B, Nq*S, D] → view → [B, Nq*S*D]
               ↓ output_proj → [B, D]
               ↓ classifier  → [B, 1]   （CVR 预测 logit）
```

---

## 四、关键约束与设计说明

| 约束 / 设计 | 说明 |
|---|---|
| `d_model % T == 0`（T = Nq×S + num_ns） | RankMixerBlock token_mixing 需要将 D 等分成 T 份（`d_sub = D/T`） |
| `num_time_buckets == 65` | 由 `BUCKET_BOUNDARIES`（64 条边界）固定，+1 是 `padding_idx=0` 占用的槽 |
| `seq_id_threshold` | 高基数 id 特征（vocab > threshold）训练时施加额外 `2×dropout_rate`，防止过拟合 |
| `emb_skip_threshold` | 超大 vocab 特征跳过建表，用零向量代替，节省显存 |
| RoPE `rope_on_q=False`（CrossAttention） | Query 不加位置编码，避免与序列侧 RoPE 冲突；K/V 侧保留顺序信息 |
| `W_g` 初始化偏置=1 | Gated Output 初始近似恒等，保证训练初期梯度稳定 |
| LongerEncoder Q-side `torch.gather` RoPE | top_k token 非连续采样，需精确按原始位置取 cos/sin，而非连续切片 |
| `ns_tokenizer_type='rankmixer'` vs `'group'` | rankmixer 允许 num_ns_tokens 与 group 数解耦，更灵活地控制 T 以满足 `d_model % T == 0` |
