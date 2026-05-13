# Time-Ordered Train-Val Split

## 改动概述

将 train/val 的分割从 row-group 顺序改为时间顺序，消除 train/valid 之间 ~94.75h 的时间重叠。

### 动机

之前的 EDA 发现（`20260513_server_eda/result/eda_report.md`），按 row-group 顺序分割导致 train 和 valid 时间范围有 94.75h 重叠，可能造成 val 指标虚高。改为时间顺序分割：按 `timestamp` 排序，前 90% 为 train，后 10% 为 valid，保证无时间泄漏。

### 核心改动

**`make_time_split.py`** — 新增脚本：
- 扫描所有 parquet 文件的 `timestamp` 列（逐行读取，~2M 行，约 10-30s）
- 按 timestamp 排序，计算 90/10 切分点
- 按文件合并连续行为 `(start_row, end_row)` 范围
- 输出 `time_split_map.json`

**`dataset.py`** — `PCVRParquetDataset.__init__`：
- 新增 `row_ranges` 参数，接受 `[(file_path, start_row, end_row), ...]` 列表
- 当提供 `row_ranges` 时，忽略 `row_group_range`，按精确行范围读取数据

**`dataset.py`** — `PCVRParquetDataset` 迭代：
- `__iter__` 根据 `_row_ranges` 是否提供，分发到 `_iter_row_groups()` 或 `_iter_row_ranges()`
- 新增 `_read_row_range(file, start, end)`：通过 `read_row_group` + `slice` 精确读取指定行范围

**`dataset.py`** — `get_pcvr_data`：
- 新增 `time_split_map_path` 参数
- 若提供，读取 JSON 构造 `row_ranges`，使用 `TimeSplitDataset` 路径
- 否则保持原有 row-group 顺序分割（向后兼容）

**`train.py`** — 新增 `--time_split_map` CLI 参数，传入 `get_pcvr_data`

**`inference/dataset.py`** — 同步 `dataset.py` 的 `row_ranges` 参数支持

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
| `dataset.py` | `row_ranges` 参数；`_iter_row_ranges` / `_read_row_range`；`get_pcvr_data` 支持 split map |
| `train.py` | `--time_split_map` CLI 参数 |
| `inference/dataset.py` | 同步 `row_ranges` 支持 |

### 基础

本实验基于 `20260510_timestamp_features` 代码。
