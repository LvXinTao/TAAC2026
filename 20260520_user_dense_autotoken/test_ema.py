"""Minimal tests for the EMA class."""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from trainer import EMA


class TinyModel(nn.Module):
    """Minimal model with both dense and sparse (Embedding) params."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)  # dense
        self.emb = nn.Embedding(100, 10)  # sparse (should be excluded)

    def forward(self, x):
        return self.linear(x)

    def get_dense_params(self):
        sparse_ptrs = {module.weight.data_ptr() for module in self.modules() if isinstance(module, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]


def test_ema_init_copies_dense_only():
    model = TinyModel()
    ema = EMA(model, decay=0.999, device='cpu')
    # EMA shadow should match the number of dense params (weight + bias = 2 tensors)
    assert len(ema.shadow) == len(model.get_dense_params()), \
        f"Expected {len(model.get_dense_params())} shadows, got {len(ema.shadow)}"
    # Embedding weight should NOT be in shadow
    emb_ptr = model.emb.weight.data_ptr()
    assert emb_ptr not in ema.shadow, "Embedding params should not be shadowed"


def test_ema_step_updates_shadow():
    model = TinyModel()
    ema = EMA(model, decay=0.9, device='cpu')  # Use 0.9 for easy math

    # Simulate a parameter update
    with torch.no_grad():
        model.linear.weight.data.add_(1.0)

    ema.step()

    # shadow should have moved toward the new param value
    # shadow_new = 0.9 * shadow_old + 0.1 * param_new
    assert not torch.allclose(ema.shadow[model.linear.weight.data_ptr()], model.linear.weight.data), \
        "Shadow should lag behind current params with decay=0.9"


def test_ema_apply_and_restore():
    model = TinyModel()
    ema = EMA(model, decay=0.999, device='cpu')

    # Make a param change and step
    with torch.no_grad():
        model.linear.weight.data.add_(1.0)
    ema.step()

    # Record weights right before apply (these are what restore should return to)
    pre_apply = {p.data_ptr(): p.data.clone() for p in model.get_dense_params()}

    # Apply EMA weights to model
    old_values = ema.apply_to_model()
    assert model.linear.weight.data_ptr() in old_values, \
        "apply_to_model should return old values keyed by data_ptr"

    # Model params should now equal EMA shadow
    assert torch.allclose(model.linear.weight.data, ema.shadow[model.linear.weight.data_ptr()])

    # Restore training weights
    ema.restore(old_values)
    # After restore, params should match pre-apply values (not initial values)
    for p in model.get_dense_params():
        ptr = p.data_ptr()
        if ptr in pre_apply:
            assert torch.allclose(p.data, pre_apply[ptr]), \
                f"After restore, param at {ptr} should match pre-apply value"


if __name__ == '__main__':
    test_ema_init_copies_dense_only()
    print("PASS: test_ema_init_copies_dense_only")
    test_ema_step_updates_shadow()
    print("PASS: test_ema_step_updates_shadow")
    test_ema_apply_and_restore()
    print("PASS: test_ema_apply_and_restore")
    print("\nAll EMA tests passed!")
