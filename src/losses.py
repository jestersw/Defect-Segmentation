"""Loss functions for defect segmentation.

Baseline: BCEWithLogitsLoss.
Condition B: 0.5 * DiceLoss + 0.5 * FocalLoss(gamma=2, alpha=0.25).

Interface contract (see docs/plan_report.tex, Section 4):
    logits:  [B, 1, H, W] float tensor, raw model output (no sigmoid)
    targets: [B, 1, H, W] float tensor in {0, 1}
    loss:    scalar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Insensitive to foreground/background imbalance, so small defects are
    not drowned out the way they are by per-pixel BCE.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).float()
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017).

    Down-weights easy pixels so training focuses on hard ones: defect
    boundaries and low-contrast scratches.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """0.5 * DiceLoss + 0.5 * FocalLoss (condition B).

    Defaults match configs/base.yaml:
        dice_weight  = 0.5
        focal_weight = 0.5
        focal.gamma  = 2.0
        focal.alpha  = 0.25
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
        )