# EDA Report — Server-Side Full Dataset

> **Task ID**: `angel_training_ams_2026_1029731852466346144_20260513172343_88d6f52b`
> **Date**: 2026-05-13
> **Data Source**: `/data_ams/industrial_training_data` (server parquet)
> **Script**: `20260513_server_eda/eda.py`

---

## 1. Data Scale

| Metric | Value |
|--------|-------|
| Parquet files | 1,000 |
| Row groups | 1,044 |
| Total rows | 2,099,956 |
| Memory estimate | ~777 GB (388 KB/row) |
| File avg size | ~2,100 rows/file |

Each file contains approximately 1 row group with ~2,100 rows.
A few files (e.g. part-00007) have 2 row groups.

## 2. Time Range

| Split | Row Groups | Rows | Time Range | Duration |
|-------|-----------|------|------------|----------|
| **Global** | 1,044 | 2,099,956 | 2026-02-28 08:35 → 2026-03-04 09:49 | 97.24h |
| **Train** (90% RG) | 940 | 1,895,650 | 2026-02-28 08:35 → 2026-03-04 09:49 | 97.24h |
| **Valid** (10% RG) | 104 | 204,306 | 2026-02-28 11:04 → 2026-03-04 09:49 | 94.75h |

**WARNING**: Train/valid split has significant time overlap (94.75h). This is because
the split is done by row group order, not by timestamp. The validation set is not
temporally separated from the training set, which may lead to over-optimistic val metrics.

## 3. Dataset Counts

| Metric | Value |
|--------|-------|
| Unique user_id | 2,099,956 |
| Unique item_id | 231,479 |
| Total rows | 2,099,956 |
| Rows per user | mean=1.00, median=1, max=1 |
| Rows per item | mean=9.07, median=2, max=7,591 |
| Users with >1 exposure | 0 (0.0%) |
| Items with >1 exposure | 126,294 (54.6%) |

**Key finding**: Each user appears exactly once — this is a one-exposure-per-user dataset.
Items are heavily skewed: 54.6% of items appear more than once, with the top item
appearing 7,591 times.

### Top-10 Most Exposed Items

| Rank | item_id | Exposures |
|------|---------|-----------|
| 1 | 53440080 | 7,591 |
| 2 | 100182015 | 5,921 |
| 3 | 34512628 | 5,893 |
| 4 | 66438683 | 5,071 |
| 5 | 69930692 | 5,055 |
| 6 | 29643702 | 4,792 |
| 7 | 41553391 | 3,836 |
| 8 | 96068810 | 3,756 |
| 9 | 98714338 | 3,430 |
| 10 | 38291975 | 3,198 |

## 4. Label Statistics

| Metric | Value |
|--------|-------|
| Click (label_type=1) | 1,886,438 (89.8%) |
| Conversion (label_type=2) | 213,518 (10.2%) |
| Positive rate | 10.17% |
| Class imbalance ratio | ~1:8.8 |

### Conversion Delay (label_time − timestamp)

| Metric | Value |
|--------|-------|
| Mean | 1,546.7s (~25.8 min) |
| Std | 7,355.0s (~122.6 min) |
| Min | 0s |
| Max | 267,254s (~74.2h) |

Most conversions happen relatively quickly (mean ~26 min), but the long tail extends
to ~3 days, suggesting a wide attribution window.

## 5. Feature Quality & Missingness

### 5.1 User Int Features (46 cols: 35 scalar, 11 list)

#### Scalar features

| Feature | Null% | N-unique | Min | Max |
|---------|-------|----------|-----|-----|
| user_int_feats_1 | 0.0% | ~2.1M | 1 | 5 |
| user_int_feats_3 | 2.6% | ~2.0M | 1 | 1,839 |
| user_int_feats_4 | 2.6% | ~2.0M | 0 | 991 |
| user_int_feats_48 | 0.1% | ~2.1M | 1 | 100 |
| user_int_feats_49 | 0.4% | ~2.1M | 1 | 2 |
| user_int_feats_50 | 0.1% | ~2.1M | 0 | 3 |
| user_int_feats_51 | 0.0% | ~2.1M | 1 | 155 |
| user_int_feats_52 | 0.0% | ~2.1M | 1 | 188 |
| user_int_feats_53 | 0.0% | ~2.1M | 1 | 558 |
| user_int_feats_54 | 31.8% | ~1.4M | 1 | 2,858 |
| user_int_feats_55 | 1.4% | ~2.1M | 1 | 43 |
| user_int_feats_56 | 1.4% | ~2.1M | 1 | 1,436 |
| user_int_feats_57 | 2.0% | ~2.1M | 0 | 252 |
| user_int_feats_58 | 15.8% | ~1.8M | 1 | 2 |
| user_int_feats_59 | 15.8% | ~1.8M | 1 | 14 |
| user_int_feats_82 | 22.2% | ~1.6M | 1 | 23 |
| user_int_feats_86 | 71.0% | ~0.6M | 1 | 245 |
| user_int_feats_92 | 50.1% | ~1.0M | 1 | 2 |
| user_int_feats_93 | 20.5% | ~1.7M | 1 | 37 |
| user_int_feats_94 | 52.8% | ~0.99M | 1 | 6 |
| user_int_feats_95 | 36.9% | ~1.3M | 1 | 3 |
| user_int_feats_96 | 71.8% | ~0.59M | 1 | 3 |
| user_int_feats_97 | 30.4% | ~1.5M | 1 | 3 |
| user_int_feats_98 | 9.1% | ~1.9M | 1 | 3 |
| user_int_feats_99 | 81.1% | ~0.40M | 1 | 3 |
| user_int_feats_100 | 84.8% | ~0.32M | 1 | 3 |
| user_int_feats_101 | 91.7% | ~0.17M | 1 | 3 |
| user_int_feats_102 | 89.4% | ~0.22M | 1 | 3 |
| user_int_feats_103 | 88.4% | ~0.24M | 1 | 3 |
| user_int_feats_104 | 38.8% | ~1.3M | 1 | 3 |
| user_int_feats_105 | 32.6% | ~1.4M | 1 | 3 |
| user_int_feats_106 | 13.9% | ~1.8M | 1 | 3 |
| user_int_feats_107 | 34.9% | ~1.4M | 1 | 3 |
| user_int_feats_108 | 52.5% | ~1.0M | 1 | 7 |
| user_int_feats_109 | 87.5% | ~0.26M | 1 | 7 |

High-cardinality features (unique ≈ total_rows): 1, 3, 4, 48, 51, 52, 53 — these
are essentially user identifiers or very fine-grained categorical features.

Low-cardinality features (range 1-3): 49, 50, 58, 92, 94, 95, 96, 97, 98, 100-103,
106, 107 — binary or ternary features.

#### List features

| Feature | Null% | Len mean | Len max |
|---------|-------|----------|---------|
| user_int_feats_15 | 13.6% | 3.9 | 26 |
| user_int_feats_60 | 68.6% | 1.5 | 2 |
| user_int_feats_62 | 6.2% | 2.3 | 7 |
| user_int_feats_63 | 6.2% | 2.6 | 20 |
| user_int_feats_64 | 6.2% | 4.3 | 33 |
| user_int_feats_65 | 7.1% | 6.4 | 116 |
| user_int_feats_66 | 7.5% | 9.2 | 201 |
| user_int_feats_80 | 24.8% | 1.3 | 6 |
| user_int_feats_89 | 3.9% | 10.0 | 10 |
| user_int_feats_90 | 7.0% | 10.0 | 10 |
| user_int_feats_91 | 49.1% | 10.0 | 10 |

Note: 62/63/64/65/66/89/90/91 are also in user_dense — int and float parts
jointly describe the same signal.

### 5.2 Item Int Features (14 cols: 13 scalar, 1 list)

#### Scalar features

| Feature | Null% | N-unique | Min | Max |
|---------|-------|----------|-----|-----|
| item_int_feats_5 | 0.1% | ~2.1M | 1 | 294 |
| item_int_feats_6 | 0.1% | ~2.1M | 0 | 896 |
| item_int_feats_7 | 0.1% | ~2.1M | -1 | 2,379 |
| item_int_feats_8 | 0.1% | ~2.1M | -1 | 2,110 |
| item_int_feats_9 | 0.1% | ~2.1M | 1 | 41 |
| item_int_feats_10 | 0.1% | ~2.1M | 1 | 306 |
| item_int_feats_12 | 0.1% | ~2.1M | -1 | 2,379 |
| item_int_feats_13 | 0.1% | ~2.1M | 1 | 8 |
| item_int_feats_16 | 0.1% | ~2.1M | 0 | 22,878 |
| item_int_feats_81 | 0.1% | ~2.1M | 0 | 2 |
| item_int_feats_83 | 88.7% | ~0.24M | 1 | 31 |
| item_int_feats_84 | 88.7% | ~0.24M | 1 | 227 |
| item_int_feats_85 | 88.7% | ~0.24M | 1 | 1,019 |

Most item int features have very low missingness (0.1%). Features 83/84/85 are
highly sparse (~89% null).

#### List features

| Feature | Null% | Len mean | Len max |
|---------|-------|----------|---------|
| item_int_feats_11 | 52.2% | 3.8 | 20 |

### 5.3 User Dense Features (10 cols, all list)

| Feature | Null% | Len mean | Len max |
|---------|-------|----------|---------|
| user_dense_feats_61 | 0.1% | 256.0 | 256 |
| user_dense_feats_62 | 6.2% | 2.3 | 7 |
| user_dense_feats_63 | 6.2% | 2.6 | 20 |
| user_dense_feats_64 | 6.2% | 4.3 | 33 |
| user_dense_feats_65 | 7.1% | 6.4 | 116 |
| user_dense_feats_66 | 7.5% | 9.2 | 201 |
| user_dense_feats_87 | 0.7% | 320.0 | 320 |
| user_dense_feats_89 | 3.9% | 10.0 | 10 |
| user_dense_feats_90 | 7.0% | 10.0 | 10 |
| user_dense_feats_91 | 49.1% | 10.0 | 10 |

Feature 61 is a fixed-length 256-dim vector (nearly complete).
Feature 87 is a fixed-length 320-dim vector (nearly complete).

### 5.4 High-Missingness Alert (>50%)

| Feature | Missing % |
|---------|-----------|
| user_int_feats_101 | 91.7% |
| user_int_feats_102 | 89.4% |
| item_int_feats_84 | 88.7% |
| item_int_feats_83 | 88.7% |
| item_int_feats_85 | 88.7% |
| user_int_feats_103 | 88.4% |
| user_int_feats_109 | 87.5% |
| user_int_feats_100 | 84.8% |
| user_int_feats_99 | 81.1% |
| user_int_feats_96 | 71.8% |
| user_int_feats_86 | 71.0% |
| user_int_feats_60 | 68.6% |
| user_int_feats_94 | 52.8% |
| user_int_feats_108 | 52.5% |
| item_int_feats_11 | 52.2% |
| user_int_feats_92 | 50.1% |

16 out of 70 features have >50% missing values. Most are in the 100-109 range
(sparse user features).

## 6. Key Findings & Recommendations

### Findings
1. **One-exposure-per-user**: Every user has exactly 1 row. This is a session-level
   rather than user-level prediction problem.
2. **Item exposure imbalance**: Top items appear 7,000+ times while median is 2.
   This may require item-frequency normalization or sampling.
3. **Train/valid time overlap**: The current row-group-based split does not prevent
   temporal leakage between train and valid. Items/users can appear in both splits.
4. **Class imbalance**: 10.2% positive rate (~1:8.8). Not extremely severe but
   worth considering for loss function tuning.
5. **Feature sparsity**: 16 features are >50% missing. Consider whether these should
   be dropped, imputed, or treated as sparse indicators.
6. **Long-tail conversion delay**: Mean ~26 min but max ~74 hours. A windowed
   attribution approach may be more appropriate than fixed delay.

### Recommendations
- Consider time-based train/valid split to avoid temporal leakage
- Investigate whether high-missingness features (100-109 range) carry useful signal
- Consider item-frequency-aware modeling for the exposure-skewed item distribution
