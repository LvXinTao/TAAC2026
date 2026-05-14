# Time-Ordered Train-Val Split

## 改动概述

将 train/val 的分割从 row-group 顺序改为时间顺序，消除 train/valid 之间 ~94.75h 的时间重叠。

### 动机

之前的 EDA 发现（`20260513_server_eda/result/eda_report.md`），按 row-group 顺序分割导致 train 和 valid 时间范围有 94.75h 重叠，可能造成 val 指标虚高。改为时间顺序分割：按 `timestamp` 排序，前 90% 为 train，后 10% 为 valid，保证无时间泄漏。

### 核心改动

**`make_time_split.py`** — 新增脚本：
- 扫描所有 parquet 文件的 `timestamp` 列（逐行读取，~2M 行，约 10-30s）
- 按 timestamp 排序，计算 90/10 切分点
- 按文件构建布尔掩码（ts < cutoff → train，其余 → valid），合并为连续 range
- 输出 `time_split_map.json`，训练时复用无需重复生成

**`dataset.py`** — `PCVRParquetDataset.__init__`：
- 新增 `row_ranges` 参数，接受 `[(file_path, start_row, end_row), ...]` 列表
- 当提供 `row_ranges` 时，忽略 `row_group_range`，按精确行范围读取数据
- 新增 `num_rows` 属性，用于 `__len__` 返回 batch 数

**`dataset.py`** — `PCVRParquetDataset` 迭代（性能优化）：
- `__iter__` 根据 `_row_ranges` 是否提供，分发到 `_iter_row_groups()` 或 `_iter_row_ranges()`
- `_iter_row_ranges`：按文件分组 ranges，调用 `_read_file_ranges`
- `_read_file_ranges`：对同一文件的所有 ranges，先按 row group 分组，每个 row group 只读一次，通过 `pa.concat_tables` 拼接需要的 slice —— I/O 从 189K 次降至 ~1044 次（= row group 总数）
- 删除 `_read_row_range`（每个 range 独立读取，导致大量重复 I/O）

**`dataset.py`** — `get_pcvr_data`：
- 新增 `time_split_map_path` 参数
- 若提供，读取 JSON 构造 `row_ranges`
- 否则保持原有 row-group 顺序分割（向后兼容）

**`train.py`** — 新增 `--time_split_map` CLI 参数，传入 `get_pcvr_data`

**`inference/dataset.py`** — 同步 `dataset.py` 的 `row_ranges` 和 `_read_file_ranges` 性能优化

**`run.sh`** — 自动检测并生成 `time_split_map.json`（写入 SCRIPT_DIR，因 DATA_DIR 在平台只读），然后传给 train.py

### 使用方式

```bash
# 1. 生成时间分割映射（只需跑一次，数据不变可复用）
python make_time_split.py --data_dir /path/to/parquet --output time_split_map.json

# 2. 训练时传入 split map
python train.py --time_split_map time_split_map.json

# 3. 不传入 split map 时，自动使用原有的 row-group 顺序分割（向后兼容）
python train.py
```

### 涉及文件

| 文件 | 改动 |
|------|------|
| `make_time_split.py` | **新建**：时间序分割脚本 |
| `dataset.py` | `row_ranges` 参数；`_iter_row_ranges` / `_read_file_ranges`（每 RG 只读一次）；`__len__` 支持 row_ranges 模式；`get_pcvr_data` 支持 split map |
| `train.py` | `--time_split_map` CLI 参数 |
| `inference/dataset.py` | 同步 `row_ranges` 和 `_read_file_ranges` 性能优化 |
| `run.sh` | 自动生成 time_split_map.json 并传入 train.py |

### 训练速度对比

| 方案 | Range 数 | I/O 次数 | 速度 |
|------|---------|---------|------|
| row-group 顺序（baseline） | ~2 | ~1000 | 1.19s/step |
| 时间分割（无优化） | 189K | 189K | 5.7s/step |
| 时间分割（RG-merged 优化） | 189K | ~1044 | **0.53s/step** |

优化后比原始 baseline 还快，因为本训练使用 2 卡（baseline 为单卡）。

### 实验结果

| date | name | val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC | baseline |
|------|------|-------------|---------|---------------|----------|----------------|----------|
| 2026-05-14 | time-split-v6 | 0.279 | 0.837 | -0.031 | 0.83447 | -0.011 | timestamp-features |

注：
- val AUC 降低 0.031 是因为消除了 94.75h 时间泄漏，之前的 val 指标虚高。
- test AUC 比 baseline (0.842) 低 0.008，比 timestamp-features (0.845) 低 0.011，时间序分割未带来收益。可能原因是时间序分割改变了 train/val 数据分布，后 10% 时间的数据分布与测试集存在差异。

### 基础

本实验基于 `20260510_timestamp_features` 代码。
