import pytest
import torch

from src.losses import build_loss


@pytest.mark.parametrize("name", ["bce", "dice", "dice_focal"])
def test_loss_is_finite_and_prefers_correct_logits(training_cfg, name):
    cfg = {**training_cfg, "loss": name}
    criterion = build_loss(cfg)

    target = torch.zeros(2, 1, 32, 32)
    target[..., :8, :8] = 1.0

    correct = ((target * 2 - 1) * 8).requires_grad_()
    good = criterion(correct, target)
    bad = criterion(-correct, target)

    assert good.ndim == 0
    assert torch.isfinite(good)
    assert good < bad

    good.backward()
    assert correct.grad is not None
    assert torch.isfinite(correct.grad).all()


def test_unknown_loss_fails(training_cfg):
    cfg = {**training_cfg, "loss": "unknown"}
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(cfg)


def test_tiny_defect_relative_penalty(training_cfg):
    """Compare relative loss for a missed tiny defect and a correct prediction."""
    target = torch.zeros(1, 1, 512, 512)
    target[:, :, 100:110, 100:110] = 1.0

    wrong_logits = torch.full_like(target, -10.0)
    correct_logits = torch.where(
        target.bool(),
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )

    bce = build_loss({**training_cfg, "loss": "bce"})
    dice_focal = build_loss({**training_cfg, "loss": "dice_focal"})

    bce_wrong = bce(wrong_logits, target).item()
    bce_correct = bce(correct_logits, target).item()
    df_wrong = dice_focal(wrong_logits, target).item()
    df_correct = dice_focal(correct_logits, target).item()

    bce_ratio = bce_wrong / max(bce_correct, 1e-12)
    df_ratio = df_wrong / max(df_correct, 1e-12)

    print(f"BCE: wrong={bce_wrong:.8f}, correct={bce_correct:.8f}")
    print(f"Dice+Focal: wrong={df_wrong:.8f}, correct={df_correct:.8f}")
    print(f"Relative ratios: BCE={bce_ratio:.2f}, Dice+Focal={df_ratio:.2f}")

    assert bce_wrong < 0.01
    assert df_wrong > 0.1