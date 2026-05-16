# 20260516_amp_warmup_wd

结合 AMP (BF16 混合精度训练) + weight decay + warmup 三种优化策略。

基线为 `timestamp-features`，同时引入：
- **AMP BF16**：训练速度提升 ~2.3x
- **Weight Decay**：dense 参数正则化，减少过拟合
- **LR Warmup**：前 N 步线性预热学习率，避免初期梯度不稳定

## Changes

- **train.py**: 新增 `--weight_decay` 和 `--warmup_steps` 两个 CLI 参数
- **trainer.py**:
  - `__init__` 新增 `weight_decay` 和 `warmup_steps` 参数
  - AdamW optimizer 加入 `weight_decay` 参数（sparse 参数保持 Adagrad + `sparse_weight_decay` 独立控制）
  - 新增 `LambdaLR` warmup scheduler，前 `warmup_steps` 步线性从 0 升到目标 lr
  - `train()` 循环中每步调用 warmup scheduler 并记录 LR 到 TensorBoard
  - 日志输出新增 `warmup_steps` 和 `weight_decay` 信息

## 预期效果

| 策略 | 作用 |
|------|------|
| AMP BF16 | 训练速度 ~2.3x 提升，精度基本无损 |
| Weight Decay | dense 参数正则化，缩小 train/val gap |
| LR Warmup | 训练初期稳定，避免大 lr 导致梯度爆炸 |

三个策略正交，理论上可以叠加收益：AMP 提速 + warmup+wd 提泛化。

## Files

- `train.py` — training entry point（新增 --weight_decay, --warmup_steps）
- `trainer.py` — training loop（AMP + warmup scheduler + weight decay）
- `model.py` — PCVRHyFormer model (unchanged)
- `dataset.py` — dataset with timestamp features (unchanged)
- `utils.py` — utilities (unchanged)
- `run.sh` — launch script
- `ns_groups.json` — NS tokenizer config (unchanged)
