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
| 2026-05-12 | token-specific-q | 0.22592 | 0.86862 | +0.00026 | 0.84152 | -0.00348 | timestamp_features |
| 2026-05-13 | combined-timestamp | 0.22628 | 0.86843 | +0.00354 | 0.84099 | -0.00171 | per-token-ffn |
| 2026-05-13 | torch-compile | 0.22628 | 0.86773 | -0.00063 | 0.84514 | +0.00013 | timestamp-features |
| 2026-05-13 | amp-training | 0.22588 | 0.86862 | +0.00025 | 0.84492 | -0.00008 | timestamp-features |
| 2026-05-14 | time-split | 0.279 | 0.837 | -0.031 | 0.83447 | -0.011 | timestamp-features |
| 2026-05-14 | amp-torch-compile | 0.22555 | 0.86848 | +0.00011 | 0.83943 | -0.00558 | timestamp-features |
| 2026-05-14 | amp-torch-compile-default | 0.22640 | 0.86846 | +0.00009 | 0.84091 | -0.00410 | timestamp-features |
| 2026-05-15 | muon-optimizer | 0.23348 | 0.86111 | -0.00726 | 0.82279 | -0.02222 | timestamp-features |
| 2026-05-15 | warmup-weight-decay | 0.22574 | 0.86901 | +0.00064 | 0.84647 | +0.00146 | timestamp-features |
| 2026-05-15 | wd005-warmup500 | 0.22704 | 0.86815 | +0.00025 | 0.84602 | +0.00102 | timestamp-features |
| 2026-05-16 | amp-warmup-wd | 0.22593 | 0.86823 | +0.00025 | 0.84372 | -0.00129 | timestamp-features |
| 2026-05-18 | user-dense-ue-pair | 0.22702 | 0.86701 | +0.00274 | 0.84191 | +0.00006 | baseline |
| 2026-05-18 | user-dense-ue-wd | 0.22437 | 0.87082 | +0.00181 | 0.84825 | +0.00178 | warmup-weight-decay |
| 2026-05-18 | long-seq | 0.22449 | 0.87035 | +0.00134 | 0.83924 | -0.00723 | warmup-weight-decay |
| 2026-05-19 | long-seq-v2 | 0.22460 | 0.87041 | +0.00140 | 0.83771 | -0.00876 | warmup-weight-decay |
| 2026-05-19 | user-dense-wd-tc | 0.22475 | 0.87014 | -0.00068 | 0.84385 | -0.00440 | user-dense-ue-wd |
| 2026-05-19 | user-dense-ema | 0.22525 | 0.86905 | -0.00177 | 0.84924 | +0.00099 | user-dense-ue-wd |
| 2026-05-20 | user-dense-cosine | 0.22319 | 0.87196 | +0.00291 | 0.84815 | -0.00109 | user-dense-ema |
| 2026-05-20 | user-dense-autotoken | 0.22471 | 0.87022 | +0.00117 | 0.84544 | -0.00380 | user-dense-ema |
| 2026-05-20 | user-dense-cosine-10w | 0.22394 | 0.87122 | +0.00216 | 0.84922 | -0.00002 | user-dense-cosine |
| 2026-05-21 | user-dense-swa | 0.22436 | 0.87093 | +0.00188 | 0.84510 | -0.00414 | user-dense-ema |
| 2026-05-21 | user-dense-cosine-5w | 0.22330 | 0.87176 | +0.00054 | 0.84735 | -0.00187 | user-dense-cosine-10w |

每次新实验完成后，将结果追加到 `exp_result.csv`、`CLAUDE.md`、模型文件夹下的`README.md`，以及项目根目录下的`README.md`。

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
# --template-id 为完整 taskId 字符串（如 angel_training_ams_...）
taac2026 train prepare --name <task-name> --source <source-dir> --template-id <taskId>

# 上传到 COS 并创建任务
# --template-id 为内部数字 ID（jobInternalId，如 92380），NOT the full taskId string
taac2026 train submit --bundle submit-bundle --template-id <jobInternalId> [--dry-run]

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

- `taskId`（字符串，如 `angel_training_ams_...`）：用于 `prepare` 的 `--template-id`、`run`、`stop`、`logs`、`metrics`、`describe`
- `jobInternalId`（数字，如 `92380`）：用于 `submit` 的 `--template-id`、`delete`
- 查找 `jobInternalId` 的方式：
  1. `taiji-output/jobs.json` → `jobsById[taskId].jobInternalId`
  2. `taiji-output/submit-live/*/plan.json` → `templateJobInternalId`
  3. `taiji-output/submit-bundle/manifest.json` → `templateJobId`（然后在 `jobs.json` 中查其 `jobInternalId`）
  4. 已知示例：`token-specific-q` → `92380`，`TAAC_stat_cross` → `92437`

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