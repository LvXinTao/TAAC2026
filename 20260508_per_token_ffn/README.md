# Per-token Independent FFN

## 改动概述

将 `RankMixerBlock` 中的 FFN 从**共享参数**改为**每个 token 位置独立参数**。

### 动机

baseline 中 `RankMixerBlock` 的 FFN 使用一组 `norm → fc1 → fc2 → post_norm` 对所有 T 个 token 位置共享。改为每个 token 位置拥有独立的 FFN 参数，使每个位置有专属的变换能力。

### 核心改动

**`model.py`** — `RankMixerBlock.__init__` 和 `forward`：

| 改动前（共享） | 改动后（独立） |
|---|---|
| `self.norm = nn.LayerNorm(d_model)` | `self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_total)])` |
| `self.fc1 = nn.Linear(d_model, d_model * hidden_mult)` | `self.fc1s = nn.ModuleList([...])` |
| `self.fc2 = nn.Linear(d_model * hidden_mult, d_model)` | `self.fc2s = nn.ModuleList([...])` |
| `self.post_norm = nn.LayerNorm(d_model)` | `self.post_norms = nn.ModuleList([...])` |

forward 从向量化 `self.norm(Q_hat) → self.fc1 → ...` 改为逐 token 循环：
```python
for i in range(self.T):
    x = self.norms[i](Q_hat[:, i, :])
    x = self.fc1s[i](x)
    x = F.gelu(x)
    x = self.dropout(x)
    x = self.fc2s[i](x)
    x = self.post_norms[i](Q[:, i, :] + x)
```

### 参数量变化

- baseline（共享 FFN）：~132K params per RankMixerBlock（d_model=128, T=8）
- 独立 FFN：~1.06M params per RankMixerBlock（8 倍增长）
- 总参数量（d_model=64, T=16）：~199.7M

### 涉及文件

| 文件 | 改动 |
|------|------|
| `model.py` | `RankMixerBlock` 类：FFN 从共享改为 per-token 独立 |

### 训练任务

| 项目 | 值 |
|------|-----|
| 任务 ID | `angel_training_ams_2026_1029731852466346144_20260508195004_3313eb5e` |
| 分支 | `feat/per-token-ffn` |
| 模板 | `20260506_baseline` (ID 61841) |
