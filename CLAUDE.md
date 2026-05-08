`baseline/`目录下为baseline代码，后续改动以baseline为基础进行改动。

## 项目开发规范

- 每次进行新的算法改动或新的模型，需要**新建`feat/xxx`分支，并新建`yyyymmdd_xxx`目录**，所有改动都应在`yyyymmdd_xxx`目录下进行。**禁止修改其他算法模型的内容**。

## 训练与评估工具

所有训练和评估操作**必须**使用 `taac2026` CLI，不要手动调用 API 或使用其他方式。

### 认证

Cookie 保存在 `.taac2026/secrets/taiji-cookie.txt`。

### 训练任务标准流程

```
train prepare → train submit → train run
```

```bash
# 准备提交包（扫描源码，创建 manifest.json）
taac2026 train prepare --name <task-name> --source <source-dir>

# 上传到 COS 并创建任务
taac2026 train submit --bundle submit-bundle --template-id <template-id>

# 启动训练
taac2026 train run --task-id <taskId>
```

### 常用命令

| 操作 | 命令 |
|------|------|
| 列出训练任务 | `taac2026 train list [--incremental]` |
| 查看任务详情+指标+日志 | `taac2026 train describe --job-id <taskId>` |
| 仅查看日志 | `taac2026 train logs --job-id <taskId>` |
| 仅查看指标 | `taac2026 train metrics --job-id <taskId>` |
| 停止训练 | `taac2026 train stop --task-id <taskId>` |
| 删除任务(需内部ID) | `taac2026 train delete --job-internal-id <numericId>` |
| 列出评估任务 | `taac2026 eval list` |
| 评估日志 | `taac2026 eval logs --task-id <taskId>` |
| 评估指标 | `taac2026 eval metrics --task-id <taskId>` |

### ID 类型注意

- `taskId`（字符串，如 `angel_training_ams_...`）：用于 `run`、`stop`、`logs`、`metrics`、`describe`
- `jobInternalId`（数字，如 `74958`）：仅用于 `delete`

两者均可在 `taiji-output/jobs.json` 中查到。

### 输出目录

所有输出默认写入 `taiji-output/`：
- `jobs.json` / `jobs-summary.csv` — 训练任务映射
- `submit-live/<timestamp>/` — 提交结果
- `train-jobs/job-{taskId}.json` — 任务详情
- `train-jobs/logs/{taskId}/` — Pod 日志
- `train-jobs/metrics/` — 训练指标
- `eval-tasks.json` — 评估任务
- `eval-jobs/` — 评估日志和指标