# Label Time Features (hour + day_of_week)

## 改动概述

将 parquet 中的 `label_time`（转化行为发生的时间戳）转化为两个非序列特征加入 user_int 特征：
- **hour**（fid=201）：一天中的小时，0-23，vocab=24
- **day_of_week**（fid=202）：一周中的星期几，0=周一~6=周日，vocab=7

### 动机

parquet 数据包含 `label_time` 字段，但 baseline 和 per-token-ffn 均未使用该信息。将其转化为 hour 和 day_of_week 特征，捕捉转化行为的时间周期模式（如早晚高峰、工作日/周末差异）。

### 核心改动

**`dataset.py`** — `PCVRParquetDataset._load_schema`：
- 在加载 schema.json 的 user_int 特征后，追加 hour (fid=201, vocab=24) 和 day_of_week (fid=202, vocab=7) 两个合成特征到 user_int_schema

**`dataset.py`** — `PCVRParquetDataset._convert_batch`：
- 从 `label_time` 列提取 hour 和 day_of_week（纯整数运算）
- 值 +1 存入 user_int buffer（0 表示缺失，对应 embedding padding_idx=0）
- label_time <= 0 的行，hour 和 day_of_week 均设为 0

**`inference/dataset.py`**：同步上述改动

### NS Tokenizer 处理

两个新特征作为 singleton 特征组各自独立 embedding，通过 NS tokenizer（group 或 rankmixer）进入模型。无需修改 `ns_groups.json`。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `dataset.py` | `_load_schema` 追加合成特征；`_convert_batch` 从 label_time 提取 hour + day_of_week |
| `inference/dataset.py` | 同步上述改动 |

### 基础

本实验基于 `20260508_per_token_ffn`（per-token independent FFN）代码。
