# Server-Side EDA (2026-05-16)

## 实验目标

对服务器上完整的 parquet 数据进行探索性数据分析（EDA），结果输出到训练日志中。

替代 `train.py` 作为 `run.sh` 的入口，不执行训练，仅分析数据。

## 涉及文件

- **新建**: `eda.py` — 独立的 EDA 脚本，流式读取 parquet 并输出 6 个分析 section
- **修改**: `run.sh` — 将 `train.py` 替换为 `eda.py`
- 其余文件（`dataset.py`, `model.py`, `trainer.py`, `utils.py`, `ns_groups.json`, `inference/`）均为 baseline 的未改动副本

## EDA 输出内容

| Section | 内容 |
|---------|------|
| 1 | 数据规模 — 文件数、row group 数、总行数、内存估算 |
| 2 | 时间范围 — 全局/训练集/验证集时间范围、重叠检查 |
| 3 | 数据集统计 — 唯一 user_id/item_id 数、曝光频率分布 |
| 4 | 标签统计 — 点击/转化分布、转化延迟 |
| 5 | 特征质量 — 缺失率、cardinality、值域范围（按 feature group） |
| 6 | 序列特征 — 各 domain 序列长度、缺失率、词表大小 |

## 使用方式

提交训练任务时，`run.sh` 自动执行 EDA。查看日志即可看到分析结果。

```bash
taac2026 train prepare --name server-eda --source 20260513_server_eda --template-id <taskId>
taac2026 train submit --bundle submit-bundle --template-id <jobInternalId>
taac2026 train run --task-id <taskId>
taac2026 train logs --job-id <taskId>   # EDA 结果在日志中
```

## 设计要点

- 按 row-group 流式处理，避免全量加载导致 OOM
- 纯文本日志输出（服务器无 display，不适合画图）
- 复用 `dataset.py` 的 schema.json 解析和 train/val 划分逻辑
- 可选输出 `eda_summary.json` 到 log_dir

## 相对于 20260513_server_eda 的改动

### eda.py — Bug 修复：Section 5 N-unique 统计错误

**改动文件**：`eda.py` 第 402、405 行

**问题描述**：

原代码用 `value_count`（非空行数，最大≈210万）作为判断条件，导致几乎所有 scalar 列都走 fallback 分支，将总非空行数误输出为唯一值数（N-unique）。

例如 `user_int_feats_1` 真实取值范围仅 1~5（唯一值=5），但原代码输出 `~2,099,664`。

**修复内容**：

```python
# 修复前（错误）
if s['value_count'] <= MAX_SEEN_VALUES:
    nunique = f"{len(s['seen_values']):,}"
else:
    nunique = f"~{s['value_count']:,}"

# 修复后（正确）
if len(s['seen_values']) < MAX_SEEN_VALUES:
    nunique = f"{len(s['seen_values']):,}"
else:
    nunique = f">{MAX_SEEN_VALUES:,}"
```

**修复逻辑**：
- 判断条件改为 `len(seen_values) < MAX_SEEN_VALUES`：若 set 未溢出（未达到上限 50000），说明已收集到全部唯一值，直接输出 `len(seen_values)`
- 若 set 已溢出（真高基数特征），输出 `>50,000` 表示"唯一值数超过采样上限"，而非误导性的总行数
