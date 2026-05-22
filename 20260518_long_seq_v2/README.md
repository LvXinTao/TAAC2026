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

## 实验结果
- **val/AUC**: 0.87041 (Step 30440)
- **val/LogLoss**: 0.22460
- **test/AUC**: 0.83771

**结论**: 
相比于 baseline (`user_dense_wd`, val/AUC=0.87082, test/AUC=0.84825)，本次实验在验证集上取得了很高的分数（0.87041），但在测试集上分数暴跌至 0.83771。这说明参数规模增大（`d_model=128`）以及更激进的 `LongerEncoder` 配置导致了**严重的过拟合**，模型失去了泛化能力。
