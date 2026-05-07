# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

TAAC2026 (腾讯广告算法大赛) — Post-Click Conversion Rate (PCVR) prediction baseline. The model is **PCVRHyFormer**, a hybrid transformer that fuses user/item discrete & dense features with multi-domain behavioral sequences.

## Codebase Structure

All code lives in `baseline/`:

| File | Responsibility |
|------|---------------|
| `model.py` | PCVRHyFormer model: RoPE attention, sequence encoders (swiglu/transformer/longer), MultiSeqHyFormerBlock, RankMixerBlock, NS Tokenizers (group/rankmixer), classifier |
| `dataset.py` | `PCVRParquetDataset` (IterableDataset): reads raw Parquet, manages FeatureSchema, time-bucket encoding, pre-allocated numpy buffers |
| `trainer.py` | `PCVRHyFormerRankingTrainer`: training loop, dual optimizer (Adagrad for sparse, AdamW for dense), BCE/Focal loss, AUC evaluation, checkpoint management |
| `train.py` | Entry point: argparse, model/data construction, TensorBoard logging |
| `utils.py` | Logging, seeding, Focal Loss, EarlyStopping |
| `ns_groups.json` | NS feature grouping config (used by GroupNSTokenizer) |
| `run.sh` | Launch script (default: RankMixer mode) |

## Key Architecture Concepts

- **NS Tokenizer**: Discrete features → embedding → grouped/chunked → projected to NS tokens. Two modes: `group` (one token per group from ns_groups.json) and `rankmixer` (all embeddings concatenated then equally split).
- **Multi-domain sequences**: seq_a/b/c/d, each independently encoded then fused via Cross Attention + RankMixer token mixing.
- **Dual optimizer**: Embedding tables use Adagrad; all other params use AdamW.
- **High-cardinality embedding cold restart**: After a configured epoch, embeddings with vocab_size > threshold are re-initialized (MultiEpoch technique).
- **Constraint**: When `rank_mixer_mode=full`, `d_model` must be divisible by `T = num_queries * num_sequences + num_ns`.

## Common Commands

```bash
# Train with default config (RankMixer mode)
cd baseline && bash run.sh \
    --data_dir /path/to/data \
    --ckpt_dir /path/to/checkpoints \
    --log_dir /path/to/logs

# Train with GroupNSTokenizer (comment out default block in run.sh, uncomment alternative)
cd baseline && bash run.sh \
    --data_dir /path/to/data \
    --ckpt_dir /path/to/checkpoints \
    --log_dir /path/to/logs

# Environment variables override CLI flags
TRAIN_DATA_PATH=/path/to/data TRAIN_CKPT_PATH=/path/to/ckpt TRAIN_LOG_PATH=/path/to/logs bash baseline/run.sh

# TensorBoard
tensorboard --logdir /path/to/tf_events
```

## Data Format

- Input: Parquet files with columns like `user_int_feats_{fid}`, `item_int_feats_{fid}`, `user_dense_feats_{fid}`, `seq_{domain}_{fid}`, `label_type`, `user_id`, `timestamp`
- Schema: `schema.json` in the data directory defines feature layout `(fid, vocab_size, dim)` per group
- Label: `label_type == 2` maps to positive class
- Validation split: last `valid_ratio` fraction of RowGroups (file order)

## Development Notes

- Always work on a feature branch, never on `main`
- For parallel dev tasks, use separate worktrees
- Commit with descriptive messages after each feature is complete
