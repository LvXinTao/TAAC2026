# TAAC2026 Codebase

`baseline/`目录下为baseline代码，后续改动以baseline为基础进行改动。

## 项目开发规范

- 每次进行新的算法改动或新的模型，需要**新建`feat/xxx`分支，并新建`yyyymmdd_xxx`目录**，所有改动都应在`yyyymmdd_xxx`目录下进行。**禁止修改其他算法模型的内容**。
- 每个模型文件夹下的 `README.md` 用于记录本次改动的内容及涉及的文件列表，便于后续阅读和追溯。

## 实验结果

实验结果记录在 `exp_result.csv`，包含 `date`、`name`、`val/LogLoss`、`val/AUC`、`test/AUC`、`test/delta_AUC` 列。

| date | name | val/LogLoss | val/AUC | test/AUC | test/delta_AUC |
|------|------|-------------|---------|----------|----------------|
| 2026-05-08 | baseline | 0.22809796035289764 | 0.8642705082893372 | 0.841855 | - |
| 2026-05-08 | 20260508_per_token_ffn | 0.22810348868370056 | 0.8648973703384399 | 0.842699 | +0.000844 |

## 训练与评估工具

项目使用 [taac2026](https://github.com/LvXinTao/TAAC2026-CLI) CLI 作为训练、评估全流程工具。


### 认证

将 Taiji cookie 保存为 `.taac2026/secrets/taiji-cookie.txt`。

### 训练任务生命周期

```bash
# 1. 准备提交包
taac2026 train prepare --name my-model --source <source-dir>

# 2. 上传并创建任务
taac2026 train submit --bundle submit-bundle --template-id <template-id>

# 3. 启动训练
taac2026 train run --task-id <taskId>

# 4. 训练完成后发布 checkpoint 为模型（获取 mould_id 用于评估）
taac2026 train publish --task-id <taskId> --name <publish-name>
```

### 监控

```bash
taac2026 train describe --job-id <taskId>    # 详情+指标+日志
taac2026 train describe --all                 # 描述所有 jobs.json 中的任务
taac2026 train logs --job-id <taskId>         # 仅 pod 日志
taac2026 train metrics --job-id <taskId>      # 训练指标（默认输出 CSV）
taac2026 train metrics --job-id <taskId> --json  # 训练指标（输出 JSON）
taac2026 train list       # 获取所有训练任务
taac2026 train stop --task-id <taskId>        # 停止训练
taac2026 train delete --job-internal-id <id>  # 删除任务（需数字内部 ID）
```

### 评估

```bash
# 评估完整流程
taac2026 eval prepare --name my-eval --source <source-dir>
taac2026 eval submit --bundle eval-bundle --mould-id <mouldId>
```

```bash
taac2026 eval list                            # 列出评估任务
taac2026 eval logs --task-id <taskId>         # 评估日志
taac2026 eval metrics --task-id <taskId>      # 评估指标
taac2026 eval metrics --task-id <taskId> --json  # 评估指标（输出 JSON）
```

### 完整训练→评估流程

```
train prepare → train submit → train run → train publish → eval prepare → eval submit
```

1. 训练完成后先 `train publish` 发布 checkpoint，输出中会包含 `mould_id`
2. 使用 `eval prepare` 准备评估代码包
3. 使用 `eval submit --mould-id <mouldId>` 提交评估任务

输出默认写入 `taiji-output/` 目录。完整参考见 `taac2026 --help`。

