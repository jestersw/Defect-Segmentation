"""Loss factories from the stage 2 interface (#25)."""

import segmentation_models_pytorch as smp
from torch import nn


class DiceFocalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(
            mode="binary", alpha=cfg["focal"]["alpha"], gamma=cfg["focal"]["gamma"]
        )
        self.dice_weight = cfg["dice_weight"]
        self.focal_weight = cfg["focal_weight"]

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets)
                + self.focal_weight * self.focal(logits, targets))


def build_loss(cfg):
    if cfg["loss"] == "bce":
        return nn.BCEWithLogitsLoss()
    if cfg["loss"] == "dice":
        return smp.losses.DiceLoss(mode="binary", from_logits=True)
    if cfg["loss"] == "dice_focal":
        return DiceFocalLoss(cfg)
    raise ValueError(f"Unknown loss: {cfg['loss']}")
