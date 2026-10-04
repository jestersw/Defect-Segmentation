"""Visualize the worst failure cases for a trained checkpoint.

Sorts test images by IoU, picks the N worst defective images and
plots input, ground truth, prediction and probability heatmap.

Usage:
    python -m scripts.failure_case \
        --checkpoint checkpoints/dice_focal_best.pt \
        --config configs/dice_focal.yaml \
        --output results/failure_case_dice_focal.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data import build_dataset
from src.model import build_model
from src.metrics import dice_score, iou_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num", type=int, default=6)
    return parser.parse_args()


def to_display(image: torch.Tensor) -> np.ndarray:
    arr = image.permute(1, 2, 0).numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    torch.set_num_threads(8)

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dataset = build_dataset(cfg, split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    records = []
    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            prob = torch.sigmoid(model(image.to(device))).cpu().numpy()[0, 0]
            pred = (prob >= cfg["eval"]["threshold"]).astype(np.uint8)
            gt = mask.numpy()[0, 0].astype(np.uint8)
            if gt.sum() == 0:
                continue  # only defective images
            records.append({
                "idx": idx,
                "image": to_display(image[0]),
                "gt": gt,
                "pred": pred,
                "prob": prob,
                "iou": iou_score(pred, gt),
                "dice": dice_score(pred, gt),
            })

    records.sort(key=lambda r: r["iou"])
    worst = records[: args.num]

    fig, axes = plt.subplots(len(worst), 4, figsize=(12, 3 * len(worst)))
    if len(worst) == 1:
        axes = axes[None, :]
    titles = ["Input", "Ground truth", "Prediction", "Probability"]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title)

    for i, rec in enumerate(worst):
        axes[i, 0].imshow(rec["image"]); axes[i, 0].axis("off")
        axes[i, 1].imshow(rec["gt"], cmap="gray"); axes[i, 1].axis("off")
        axes[i, 2].imshow(rec["pred"], cmap="gray"); axes[i, 2].axis("off")
        axes[i, 3].imshow(rec["prob"], cmap="hot", vmin=0, vmax=1)
        axes[i, 3].axis("off")
        axes[i, 0].set_ylabel(
            f"idx={rec['idx']}\nIoU={rec['iou']:.3f}\nDice={rec['dice']:.3f}",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()