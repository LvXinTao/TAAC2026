# Combined Timestamp Features (non-seq + seq)

将 `timestamp` 转化为四类时间特征，同时覆盖**非序列**和**序列**两个维度。

## 改动说明

### 非序列时间特征（user_int 级别）

从 parquet 的 `timestamp`（用户请求时间戳）提取两个标量特征，追加到 user_int 特征向量：

| 特征 | fid | vocab | 含义 |
|------|-----|-------|------|
| hour | 201 | 24 | 一天中的小时（0-23） |
| day_of_week | 202 | 7 | 一周中的星期几（0=周一~6=周日） |

通过 NS tokenizer（group 或 rankmixer）作为 singleton token 进入模型。

### 序列时间特征（sequence sideinfo 级别）

从每个序列事件的 timestamp 提取两个离散特征，作为序列 side-info 的一部分：

| 特征 | fid | vocab | 含义 |
|------|-----|-------|------|
| hour | -1 | 24 | 事件发生的小时（0-23） |
| day_of_week | -2 | 7 | 事件发生的星期几（0=周一~6=周日） |

使用负 FID 避免与真实特征碰撞，自动追加到 sideinfo 末尾 slot。

### 与已有实验的关系

| 实验 | 非序列 | 序列 |
|------|--------|------|
| `20260510_timestamp_features` | hour + day_of_week | - |
| `20260511_seq_timestamp_sideinfo` | - | hour + day_of_week |
| `20260512_combined_timestamp` | hour + day_of_week | hour + day_of_week |

本实验同时启用两组特征，验证非序列 + 序列时间特征的互补效果。

### 与 time bucket 的关系

| 特征类型 | 来源 | 捕获信息 |
|----------|------|----------|
| time_bucket | 当前请求时间 - 序列事件时间 | 相对时间差（近期 vs 远期） |
| seq hour/day_of_week | 序列事件绝对时间 | 周期性模式（早晚高峰、工作日/周末） |
| non-seq hour/day_of_week | 当前请求绝对时间 | 用户活跃时间模式 |

三者正交互补：time_bucket 编码**多久以前**，seq timestamp 编码**事件发生在何时**，non-seq timestamp 编码**当前请求发生在何时**。

## 核心改动

**`dataset.py`** — `_load_schema`：
- user_int 追加 hour (fid=201, vocab=24) 和 day_of_week (fid=202, vocab=7)
- 每个序列 domain 的 sideinfo 追加 hour (fid=-1, vocab=24) 和 day_of_week (fid=-2, vocab=7)

**`dataset.py`** — `_convert_batch`：
- 非序列：从 `timestamp` 列提取 hour + day_of_week，+1 存入 user_int buffer
- 序列：从序列 timestamp 列提取 hour + day_of_week，写入 sideinfo 最后两个 slot

**`inference/dataset.py`**：同步上述改动

**模型侧无需改动**：`user_int_vocab_sizes` 和 `seq_domain_vocab_sizes` 自动包含新增 vocab，模型为其创建对应的 Embedding 表。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `dataset.py` | `_load_schema` + `_convert_batch` 两处改动（非序列 + 序列） |
| `inference/dataset.py` | 同步上述改动 |

## 运行方式

```bash
cd 20260512_combined_timestamp && bash run.sh \
    --data_dir /path/to/data \
    --ckpt_dir /path/to/checkpoints \
    --log_dir /path/to/logs
```

## 基础

本实验基于 `20260511_seq_timestamp_sideinfo` 代码（已包含序列时间特征），在此基础上新增非序列时间特征。
