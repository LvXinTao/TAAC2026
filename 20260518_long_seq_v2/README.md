# 20260518_long_seq_v2

## 实验目的
在前一次长序列实验 (`20260518_long_seq`) 效果不如预期的基础上，进一步调整长序列建模策略，尝试针对不同长度的序列域采用不同的编码器（per-domain encoder），并增大模型容量和压缩保留的 top_k 数量。

## 具体改动
1. **模型结构参数调整**:
   - `d_model` 从 64 增大到 **128**。
   - `seq_top_k` (LongerEncoder 压缩后保留的 token 数) 从 128 增大到 **512**。
2. **Per-domain Encoder (独立域编码器)**:
   - `seq_a` (较短) 保持使用 `transformer`。
   - `seq_b`, `seq_c`, `seq_d` (较长) 均改为使用 `longer` 编码器。
3. **代码支持**:
   - `model.py` / `inference/model.py`: 修改 `MultiSeqHyFormerBlock` 和 `PCVRHyFormer`，使其 `seq_encoder_type` 参数支持 `Union[str, List[str]]`，并实现了各序列域独立初始化的逻辑。
   - `train.py`: 新增 `--seq_domain_encoder_types` 命令行参数。
   - `run.sh`: 更新了对应的训练启动参数 (`--d_model 128`, `--seq_domain_encoder_types "transformer,longer,longer,longer"`, `--seq_top_k 512`)。
   - `inference/infer.py`: 同步更新 `_FALLBACK_MODEL_CFG` 以匹配新的配置。

## 涉及文件
- `20260518_long_seq_v2/model.py`
- `20260518_long_seq_v2/train.py`
- `20260518_long_seq_v2/run.sh`
- `20260518_long_seq_v2/inference/model.py`
- `20260518_long_seq_v2/inference/infer.py`

## 基线

本实验基于 `20260514_warmup_weight_decay`（per-token independent FFN + weight_decay 0.01 + warmup 100 steps），在此基础上叠加了 `UserDenseUEPairProjector`（来自 `20260518_user_dense_wd`）以及前一次长序列实验（`20260518_long_seq`）的 LongerEncoder 改动，并进一步调整了序列编码策略和模型容量。

## 实验结果

| val/LogLoss | val/AUC | val/delta_AUC | test/AUC | test/delta_AUC |
|-------------|---------|---------------|----------|----------------|
| 0.22460 | 0.87041 | +0.00140 | 0.83771 | -0.00876 |

最佳 checkpoint: step 30440

**结论**: 
相比于 baseline (`warmup-weight-decay`, val/AUC=0.86901, test/AUC=0.84647），验证集 AUC 略有提升（+0.00140），测试集 AUC 大幅下降（-0.00876，0.84647 → 0.83771）。参数规模增大（`d_model=128`）以及更激进的 `LongerEncoder` 配置导致了**严重的过拟合**，模型完全失去了泛化能力。
