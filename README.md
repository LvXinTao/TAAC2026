# TAAC2026 Codebase

`baseline/`目录下为baseline代码，后续改动以baseline为基础进行改动。

## 项目开发规范

- 每次进行新的算法改动或新的模型，需要**新建`feat/xxx`分支，并新建`yyyymmdd_xxx`目录**，所有改动都应在`yyyymmdd_xxx`目录下进行。**禁止修改其他算法模型的内容**。

## 训练与评估工具

项目使用 `taac2026` CLI 作为训练、评估全流程工具。

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
```

### 监控

```bash
taac2026 train describe --job-id <taskId>    # 详情+指标+日志
taac2026 train logs --job-id <taskId>         # 仅 pod 日志
taac2026 train metrics --job-id <taskId>      # 训练指标
taac2026 train list       # 获取所有训练任务
```

### 评估

```bash
taac2026 eval list                            # 列出评估任务
taac2026 eval logs --task-id <taskId>         # 评估日志
taac2026 eval metrics --task-id <taskId>      # 评估指标
```

输出默认写入 `taiji-output/` 目录。完整参考见 `taac2026 --help`。

