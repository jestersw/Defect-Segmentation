import pytest
import torch

from src.losses import build_loss


@pytest.mark.parametrize("name", ["bce", "dice", "dice_focal"])
def test_loss_is_finite_and_prefers_correct_logits(training_cfg, name):
    training_cfg["loss"] = name
    criterion = build_loss(training_cfg)
    target = torch.zeros(2, 1, 32, 32)
    target[..., :8, :8] = 1
    correct = ((target * 2 - 1) * 8).requires_grad_()
    good, bad = criterion(correct, target), criterion(-correct, target)
    assert good.ndim == 0 and torch.isfinite(good) and good < bad
    good.backward()
    assert torch.isfinite(correct.grad).all()


def test_unknown_loss_fails(training_cfg):
    training_cfg["loss"] = "unknown"
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(training_cfg)
