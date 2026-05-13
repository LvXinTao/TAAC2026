# Server-Side EDA (2026-05-13)

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
