# TAAC2026 Codebase

`baseline/`目录下为baseline代码，后续改动以baseline为基础进行改动。

## 项目开发规范

- 每次进行新的算法改动或新的模型，需要**新建`feat/xxx`分支，并新建`yyyymmdd_xxx`目录**，所有改动都应在`yyyymmdd_xxx`目录下进行。**禁止修改其他算法模型的内容**。
- 每个模型文件夹下的 `README.md` 用于记录本次改动的内容及涉及的文件列表，便于后续阅读和追溯。

## 实验结果

实验结果记录在 `exp_result.csv`，包含 `date`、`name`、`val/LogLoss`、`val/AUC`、`val/delta_AUC`、`test/AUC`、`test/delta_AUC`、`baseline` 列。

| date | name | val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC | baseline |
|------|------|-------------|---------|---------------|----------|----------------|----------|
| 2026-05-08 | baseline | 0.22810 | 0.86427 | - | 0.84186 | - | - |
| 2026-05-08 | per-token-ffn | 0.22810 | 0.86490 | +0.00063 | 0.84270 | +0.00084 | baseline |
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
| 2026-05-14 | warmup-weight-decay | 0.22574 | 0.86901 | +0.00064 | 0.84647 | +0.00146 | timestamp-features |
| 2026-05-15 | muon-optimizer | 0.23348 | 0.86111 | -0.00726 | 0.82279 | -0.02222 | timestamp-features |
| 2026-05-15 | wd005-warmup500 | | | | | | timestamp-features |

## 训练与评估工具

项目使用 [taac2026](https://github.com/LvXinTao/TAAC2026-CLI) CLI 作为训练、评估全流程工具。


### 认证

将 Taiji cookie 保存为 `.taac2026/secrets/taiji-cookie.txt`。

### 训练任务生命周期

```bash
# 1. 准备提交包
taac2026 train prepare --name my-model --source <source-dir>

# 2. 上传并创建任务
taac2026 train submit --bundle submit-bundle --template-id <template-id,位于taijiout/train-jobs> --gpu-num <gpu-num> --yes

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
taac2026 eval prepare --name my-eval --source <source-dir> # 只需要inference/下的代码
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

