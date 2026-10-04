import pytest
import torch

from src.losses import DiceFocalLoss, DiceLoss, FocalLoss


def make_inputs(batch=2, size=32, positive_fraction=0.1):
    torch.manual_seed(42)
    logits = torch.randn(batch, 1, size, size, requires_grad=True)
    targets = (torch.rand(batch, 1, size, size) < positive_fraction).float()
    return logits, targets


def test_dice_loss_perfect_prediction():
    # very confident positive logits + all-ones mask -> dice should be near 0
    targets = torch.ones(1, 1, 16, 16)
    logits = torch.full((1, 1, 16, 16), 20.0)
    loss = DiceLoss()(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_empty_prediction():
    # confident negative logits + all-ones mask -> dice should be near 1
    targets = torch.ones(1, 1, 16, 16)
    logits = torch.full((1, 1, 16, 16), -20.0)
    loss = DiceLoss()(logits, targets)
    assert loss.item() == pytest.approx(1.0, abs=1e-3)


def test_dice_loss_no_defects_is_finite():
    # all-background mask: dice is still well-defined and finite
    targets = torch.zeros(1, 1, 16, 16)
    logits = torch.full((1, 1, 16, 16), -20.0)
    loss = DiceLoss()(logits, targets)
    assert torch.isfinite(loss)


def test_focal_loss_is_finite_and_non_negative():
    logits, targets = make_inputs()
    loss = FocalLoss()(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_loss_perfect_prediction_is_small():
    # very confident correct predictions -> focal should be near 0
    targets = torch.ones(1, 1, 16, 16)
    logits = torch.full((1, 1, 16, 16), 20.0)
    loss = FocalLoss()(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_dice_focal_forward_backward():
    logits, targets = make_inputs()
    criterion = DiceFocalLoss()
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dice_focal_default_weights_match_config():
    # configs/base.yaml: dice_weight 0.5, focal_weight 0.5, gamma 2.0, alpha 0.25
    criterion = DiceFocalLoss()
    assert criterion.dice_weight == 0.5
    assert criterion.focal_weight == 0.5
    assert criterion.focal.alpha == 0.25
    assert criterion.focal.gamma == 2.0


def test_dice_focal_weighting():
    # with dice_weight=1, focal_weight=0 -> equals DiceLoss
    logits, targets = make_inputs()
    loss_dice_only = DiceFocalLoss(dice_weight=1.0, focal_weight=0.0)(logits, targets)
    loss_plain_dice = DiceLoss()(logits, targets)
    assert loss_dice_only.item() == pytest.approx(loss_plain_dice.item(), abs=1e-6)

    # with dice_weight=0, focal_weight=1 -> equals FocalLoss
    loss_focal_only = DiceFocalLoss(dice_weight=0.0, focal_weight=1.0)(logits, targets)
    loss_plain_focal = FocalLoss()(logits, targets)
    assert loss_focal_only.item() == pytest.approx(loss_plain_focal.item(), abs=1e-6)