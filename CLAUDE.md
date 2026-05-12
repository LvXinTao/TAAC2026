`baseline/`目录下为baseline代码，后续改动以baseline为基础进行改动。

## 项目开发规范

- 每次进行新的算法改动或新的模型，需要**新建`feat/xxx`分支，并新建`yyyymmdd_xxx`目录**，所有改动都应在`yyyymmdd_xxx`目录下进行。**禁止修改其他算法模型的内容**。
- 每个模型文件夹下的 `README.md` 用于记录本次改动的内容及涉及的文件列表，便于后续阅读和追溯。

## 实验结果

实验结果记录在 `exp_result.csv`（date, name, val/LogLoss, val/AUC, val/delta_AUC, test/AUC, test/delta_AUC）。

| date | name | val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC | baseline |
|------|------|-------------|---------|---------------|----------|----------------|----------|
| 2026-05-08 | baseline | 0.22810 | 0.86427 | - | 0.84186 | - | - |
| 2026-05-09 | per-token-ffn | 0.22810 | 0.86490 | +0.00063 | 0.84270 | +0.00084 | baseline |
| 2026-05-11 | timestamp-features | 0.22584 | 0.86837 | +0.00347 | 0.84501 | +0.00231 | per-token-ffn |
| 2026-05-10 | TAAC_focal_dynamic | 0.25680 | 0.86528 | +0.00038 | | | per-token-ffn |
| 2026-05-10 | TAAC_poly | 0.25543 | 0.86480 | -0.00010 | 0.83648 | -0.00622 | per-token-ffn |
| 2026-05-12 | seq-timestamp-sideinfo | 0.22614 | 0.86749 | +0.00259 | 0.84216 | -0.00054 | per-token-ffn |
| 2026-05-12 | token-specific-q | 0.22592 | 0.86862 | +0.00026 | | | timestamp_features |

每次新实验完成后，将结果追加到 `exp_result.csv` 并同时更新本节表格。

## 训练与评估工具

所有训练和评估操作**必须**使用 `taac2026` CLI，不要手动调用 API 或使用其他方式。

### 认证

Cookie 保存在 `.taac2026/secrets/taiji-cookie.txt`。

```bash
taac2026 login                        # 浏览器登录
taac2026 login --cookie-string "..."  # 直接粘贴 cookie
taac2026 login --stdin                # 从 stdin 读取
```

### 训练任务标准流程

```
train prepare → train submit → train run → train publish
```

```bash
# 准备提交包（扫描源码，创建 manifest.json）
taac2026 train prepare --name <task-name> --source <source-dir>

# 上传到 COS 并创建任务
taac2026 train submit --bundle submit-bundle --template-id <template-id> [--dry-run]

# 启动训练
taac2026 train run --task-id <taskId>

# 训练完成后发布 checkpoint 为模型（获取 mould_id 用于评估任务）
taac2026 train publish --task-id <taskId> [--name <name>] [--desc <desc>]
```

### 完整训练→评估流程

```
train prepare → train submit → train run → train publish → eval prepare → eval submit
```

1. 训练完成后先 `train publish`，输出 `publish-<taskId>.json` 中包含 `mouldId`
2. 使用 `eval prepare --name <name> --source <dir>` 准备评估代码包（只需要inference/下的代码）
3. 使用 `eval submit --bundle eval-bundle --mould-id <mouldId>` 提交评估

### 常用命令

| 操作 | 命令 |
|------|------|
| 列出训练任务 | `taac2026 train list [--incremental] [--page-size <n>]` |
| 查看任务详情+指标+日志 | `taac2026 train describe --job-id <taskId>` |
| 描述所有任务 | `taac2026 train describe --all` |
| 仅查看日志 | `taac2026 train logs --job-id <taskId>` |
| 仅查看指标 | `taac2026 train metrics --job-id <taskId> [--json]` |
| 停止训练 | `taac2026 train stop --task-id <taskId>` |
| 删除任务(需内部ID) | `taac2026 train delete --job-internal-id <numericId>` |
| 发布 checkpoint | `taac2026 train publish --task-id <taskId>` |
| 列出评估任务 | `taac2026 eval list [--page-size <n>]` |
| 评估日志 | `taac2026 eval logs --task-id <taskId>` |
| 评估指标 | `taac2026 eval metrics --task-id <taskId> [--json]` |
| 评估准备 | `taac2026 eval prepare --name <name> --source <dir>` |
| 评估提交 | `taac2026 eval submit --bundle eval-bundle --mould-id <id>` |

### ID 类型注意

- `taskId`（字符串，如 `angel_training_ams_...`）：用于 `run`、`stop`、`logs`、`metrics`、`describe`
- `jobInternalId`（数字，如 `74958`）：仅用于 `delete`

两者均可在 `taiji-output/jobs.json` 中查到。

### 输出目录

所有输出默认写入 `taiji-output/`：
- `jobs.json` / `jobs-summary.csv` — 训练任务映射
- `submit-live/<timestamp>/` — 提交结果
- `train-jobs/job-{taskId}.json` — 任务详情
- `train-jobs/job-{taskId}-metrics.csv` — 训练指标
- `train-jobs/job-{taskId}-checkpoints.csv` — checkpoint 列表
- `train-jobs/logs/{taskId}/` — Pod 日志
- `train-jobs/metrics/` — 训练指标 CSV/JSON
- `train-jobs/ckpt/publish-{taskId}.json` — 发布结果（含 mouldId）
- `eval-bundle/` — eval prepare 输出
- `eval-submit-live/<timestamp>/` — eval submit 结果
- `eval-tasks.json` / `eval-tasks-summary.csv` — 评估任务
- `eval-jobs/` — 评估日志和指标