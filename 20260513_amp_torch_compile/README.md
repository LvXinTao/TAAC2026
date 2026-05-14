# AMP + torch.compile

Combine Automatic Mixed Precision (AMP, BF16) with `torch.compile` for faster training.

Two variants were tested:
1. **`mode="reduce-overhead"`** — aggressive kernel fusion
2. **`mode="default"`** — basic graph optimization, closer to eager mode

## Results

| Experiment | val/AUC | val/LogLoss | val/delta | test/AUC | test/delta |
|------------|---------|-------------|-----------|----------|------------|
| amp-training | 0.86862 | 0.22588 | +0.00025 | 0.84492 | -0.00008 |
| amp + torch.compile (reduce-overhead) | 0.86848 | 0.22555 | +0.00011 | 0.83943 | **-0.00558** |
| amp + torch.compile (default) | 0.86846 | 0.22640 | +0.00009 | 0.84091 | **-0.00410** |

AMP alone is safe — no accuracy loss. Adding torch.compile (either mode) causes a ~0.4-0.5% test AUC drop despite identical validation AUC. The kernel fusion likely changes floating-point accumulation order in attention/embedding operations, and the effect compounds over 50k+ training steps.

## Changes

- **trainer.py**: copied from `20260513_amp_training/` — already has AMP autocast support in `_train_step` and `_evaluate_step`. No additional changes needed because `torch.compile` works transparently on top of the autocast wrapper.
- **train.py**: based on `20260513_amp_training/train.py`, added `--use_torch_compile` flag and `torch.compile(model, mode="default")` call after model construction. (Previously used `mode="reduce-overhead"` but that performed worse.)
- **inference/infer.py**: based on `20260513_torch_compile/inference/infer.py` (has `_orig_mod.` key stripping), added AMP inference autocast driven by `train_config.get('amp', False)`. Added `logits.float()` after autocast to convert BF16 to FP32 for sigmoid/numpy compatibility.
- **inference/**: added `dataset.py`, `model.py`, `ns_groups.json` (previously only `infer.py` was present, causing eval failures).
- **run.sh**: enables both `--amp` and `--use_torch_compile`.

## Rationale

AMP and torch.compile are orthogonal:
- AMP uses `torch.autocast` to cast intermediate activations to BF16 during forward/backward passes, reducing memory and improving compute throughput on tensor cores.
- torch.compile fuses and optimizes the computation graph at the Python/ATen level.

In theory they should compose cleanly. In practice, kernel fusion changes floating-point accumulation order which compounds over ~50k training steps.

## Files

| File | Source | Change |
|------|--------|--------|
| model.py | 20260513_amp_training | Unchanged |
| trainer.py | 20260513_amp_training | Unchanged (AMP autocast already present) |
| train.py | 20260513_amp_training | Added `--use_torch_compile` flag + compile call |
| dataset.py | 20260513_amp_training | Unchanged |
| utils.py | 20260513_amp_training | Unchanged |
| ns_groups.json | 20260513_amp_training | Unchanged |
| inference/infer.py | 20260513_torch_compile | Added AMP inference autocast + BF16→FP32 cast |
| inference/dataset.py | 20260513_amp_training | Copied for eval compatibility |
| inference/model.py | 20260513_amp_training | Copied for eval compatibility |
| inference/ns_groups.json | 20260513_amp_training | Copied for eval compatibility |
| run.sh | — | New, both flags enabled |

## Usage

```bash
# Local
bash 20260513_amp_torch_compile/run.sh

# Or individual flags
python train.py --amp --use_torch_compile
