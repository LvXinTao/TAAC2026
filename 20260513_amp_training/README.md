# 20260513_amp_training

AMP (Automatic Mixed Precision) training with BF16 for training speedup.

## Results

| Metric | timestamp-features (FP32) | amp-training (BF16) | delta |
|--------|--------------------------|---------------------|-------|
| val/LogLoss | 0.22584 | 0.22588 | -0.00004 |
| val/AUC | 0.86837 | 0.86862 | +0.00025 |
| test/AUC | 0.84501 | 0.84492 | -0.00008 |

### 训练速度对比

| | timestamp-features (FP32) | amp-training (BF16) | 提升 |
|---|---|---|---|
| 完成 epochs | 9 | 8 | - |
| 总训练时间 | 13h21m | 5h03m | **2.6x** |
| 每 epoch 时间 | ~88.7 min | ~38.2 min | **2.3x** |

AMP BF16 训练速度提升约 **2.3x**（每 epoch），精度基本无损（test AUC 差距仅 0.00008）。

## Changes

- **train.py**: Added `--amp` CLI flag to enable BF16 autocast training
- **trainer.py**:
  - Added `amp` parameter to `__init__`
  - `_train_step`: wraps forward pass + loss computation in `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` when AMP is enabled
  - `_evaluate_step`: wraps forward pass in autocast for consistent inference
- **run.sh**: Added `--amp` flag to enable AMP by default

## Why BF16

BF16 has the same dynamic range as FP32, so it doesn't need GradScaler to prevent gradient underflow. This simplifies the implementation — only `autocast` is needed, no gradient scaling. BF16 is natively supported on Ampere+ GPUs (A100/A800/RTX 30 series+).

## Files

- `train.py` — training entry point
- `trainer.py` — training loop with AMP support
- `model.py` — PCVRHyFormer model (unchanged)
- `dataset.py` — dataset with timestamp features (unchanged)
- `utils.py` — utilities (unchanged)
- `run.sh` — launch script
- `ns_groups.json` — NS tokenizer config (unchanged)
