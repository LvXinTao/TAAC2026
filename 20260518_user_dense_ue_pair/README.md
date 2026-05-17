# User Dense UE + Int-Dense Pair Projector (2026-05-18)

## 实验目标

参考 [Singularity_TAAC_2026/commit/98919518](https://github.com/VeraMayLin/Singularity_TAAC_2026/commit/98919518e0f826c8d08da85b6637d52afeffe0b3) 的改法，将用户稠密特征分为两部分处理：

1. **UE 特征**（User Experience）：仅包含 dense 的特征（fid 61, 87, 89, 90, 91），直接投影为 NS token
2. **Int-Dense Pair 特征**：同时有 int 和 dense 表示的特征（fid 62, 63, 64, 65, 66），用 dense 值对 int embedding 做加权平均后投影

## 涉及文件

| 文件 | 改动 |
|------|------|
| `baseline/model.py` | 新增 `UserDenseUEPairProjector` 类、`_normalize_fid_list` 辅助函数；修改 `PCVRHyFormer.__init__` 支持新的参数；添加 `_make_user_dense_token` 方法替换原有的 user_dense_proj 调用；更新 `_init_params` 和 `reinit_high_cardinality_params` |
| `baseline/inference/model.py` | 同步 `baseline/model.py` |
| `baseline/inference/infer.py` | 新增 `user_dense_ue_fids` / `user_int_dense_pair_fids` 到 fallback 配置；传入 `user_int_feature_ids` 和 `user_dense_feature_specs` |
| `baseline/train.py` | 新增 `--user_dense_ue_fids` 和 `--user_int_dense_pair_fids` CLI 参数；模型构建时传入 schema 元数据 |
| `baseline/run.sh` | 默认启用 UE fids `61,87,89,90,91` 和 pair fids `62,63,64,65,66` |

## 设计要点

- **向后兼容**：当 `user_dense_ue_fids` 为空时，自动回退到原有的 `Linear+LayerNorm` 投影，不影响已有实验
- **加权池化**：pair 特征的 dense 值作为权重，对 int embedding 做加权平均，使 dense 信号（如行为强度）影响 embedding 表示
- **融合投影**：UE 特征和 pair 特征分别投影后通过 `out_proj` 融合为单一的 user dense NS token

## 实验配置

```bash
USER_DENSE_UE_FIDS=61,87,89,90,91
USER_INT_DENSE_PAIR_FIDS=62,63,64,65,66
```

可通过环境变量覆盖：
```bash
USER_DENSE_UE_FIDS="" USER_INT_DENSE_PAIR_FIDS="" bash run.sh  # 回退到原始 baseline
```
