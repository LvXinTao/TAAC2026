# AMP (Automatic Mixed Precision) Training

## 改动概述

在训练循环中添加 `torch.cuda.amp` 混合精度训练支持，降低显存占用并加速训练。

## 核心改动

**`train.py`**：
- 新增 `--amp` 布尔参数（默认关闭，向后兼容）

**`trainer.py`**：
- `__init__`: 新增 `use_amp` 参数，初始化 `torch.amp.GradScaler('cuda', enabled=use_amp)`
- `_train_step`: forward + loss 包裹在 `torch.amp.autocast('cuda')` 中；`backward` 改用 `scaler.scale(loss)`；`step` 改用 `scaler.step()` + `scaler.update()`
- `evaluate`: `with torch.no_grad()` 中加入 `torch.amp.autocast('cuda')`

## 使用方式

```bash
python train.py --amp
# 或在 run.sh 中添加 --amp
```

## 涉及文件

| 文件 | 改动 |
|------|------|
| `train.py` | 新增 `--amp` CLI 参数 |
| `trainer.py` | AMP scaler 初始化、autocast、scaler 训练循环 |

本模型基于 [20260508_per_token_ffn](../20260508_per_token_ffn/) 开发。
