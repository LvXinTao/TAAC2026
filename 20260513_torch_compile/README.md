# torch.compile (2026-05-13)

## 实验目标

使用 `torch.compile(mode="reduce-overhead")` 加速模型训练，预期减少每个 batch 的 forward pass overhead。

## 使用方式

```bash
# 带 torch.compile 训练
python train.py --use_torch_compile --num_epochs 10

# 不带 torch.compile（对照组）
python train.py --num_epochs 10
```

## torch.compile 集成说明

**`train.py`** 核心改动：

1. 新增 `--use_torch_compile` CLI 参数（action='store_true', default=False）
2. 模型创建后，在参数数量 log 之后调用 `torch.compile(model, mode="reduce-overhead")`

### 编译模式选择

- `mode="reduce-overhead"`：适合 batch size 固定、序列长度相对固定的场景，减少 CPU overhead
- 不选 `mode="max-autotune"`：首次编译时间过长
- 不选 `mode="default"`：不够激进，收益较小

### 兼容性

- `RotaryEmbedding` 预分配 cos/sin 缓存，forward 仅做静态 slicing — 完全兼容
- `self.training` 控制流为 Python bool — torch.compile 正确处理
- 默认 `seq_encoder_type='transformer'`，无 shape-dependent 分支 — 兼容
- 双优化器（Adagrad + AdamW）不影响 compile，compile 仅作用于 forward 图
- 模型 save/load 正常（compile 图为运行时附加，不影响权重序列化）

## timestamp 特征

本实验基于 `20260510_timestamp_features`，保留了 hour + day_of_week 合成特征（从 parquet 的 `timestamp` 列提取，fid=201/202）。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `train.py` | 新增 `--use_torch_compile` 参数，添加 `torch.compile()` 调用 |
| `dataset.py` | 同 timestamp-features |
| `inference/dataset.py` | 同 timestamp-features |
| 其他文件 | 从 `20260510_timestamp_features/` 复制，无修改 |
