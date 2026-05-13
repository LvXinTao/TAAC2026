# AMP + torch.compile

Combine Automatic Mixed Precision (AMP, BF16) with `torch.compile(mode="reduce-overhead")` for faster training.

## Changes

- **trainer.py**: copied from `20260513_amp_training/` — already has AMP autocast support in `_train_step` and `_evaluate_step`. No additional changes needed because `torch.compile` works transparently on top of the autocast wrapper.
- **train.py**: based on `20260513_amp_training/train.py`, added `--use_torch_compile` flag and `torch.compile(model, mode="reduce-overhead")` call after model construction.
- **inference/infer.py**: based on `20260513_torch_compile/inference/infer.py` (has `_orig_mod.` key stripping), added AMP inference autocast driven by `train_config.get('amp', False)`.
- **run.sh**: enables both `--amp` and `--use_torch_compile`.

## Rationale

AMP and torch.compile are orthogonal:
- AMP uses `torch.autocast` to cast intermediate activations to BF16 during forward/backward passes, reducing memory and improving compute throughput on tensor cores.
- torch.compile fuses and optimizes the computation graph at the Python/ATen level, with `mode="reduce-overhead"` prioritizing fewer kernel launches.

They should work together — AMP handles the precision, torch.compile handles the execution graph.

## Files

| File | Source | Change |
|------|--------|--------|
| model.py | 20260513_amp_training | Unchanged (same as torch_compile) |
| trainer.py | 20260513_amp_training | Unchanged (AMP autocast already present) |
| train.py | 20260513_amp_training | Added `--use_torch_compile` flag + compile call |
| dataset.py | 20260513_amp_training | Unchanged |
| utils.py | 20260513_amp_training | Unchanged |
| ns_groups.json | 20260513_amp_training | Unchanged |
| inference/infer.py | 20260513_torch_compile | Added AMP inference autocast |
| run.sh | — | New, both flags enabled |

## Usage

```bash
# Local
bash 20260513_amp_torch_compile/run.sh

# Or individual flags
python train.py --amp --use_torch_compile
