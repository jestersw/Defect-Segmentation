# ruff: noqa: E402
"""Plot selected manifest images with ground truth and checkpoint comparisons."""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from textwrap import fill

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MVTecSegDataset, read_image
from src.evaluate import load_model, predict_probabilities
from src.metrics import dice_score


def prepare_cases(cfg, split, paths):
    """Load only the requested images, preserving the caller's path order."""
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if not paths:
        raise ValueError("Provide at least one image path")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate image paths are not allowed")
    dataset_cfg = deepcopy(cfg)
    # Avoid eagerly decoding every validation image when the run enabled caching.
    dataset_cfg["data"]["cache_images"] = False
    dataset = MVTecSegDataset(split, dataset_cfg, train=False)
    indices = {record["path"]: i for i, record in enumerate(dataset.records)}
    missing = [path for path in paths if path not in indices]
    if missing:
        raise ValueError(f"Paths not in the {split} manifest: {', '.join(missing)}")

    inputs, masks, rgb = [], [], []
    for path in paths:
        image, mask = dataset[indices[path]]
        inputs.append(image)
        masks.append(mask)
        # Display the original RGB image at evaluation resolution, without first
        # reducing it to model input size (small defects must remain visible).
        original = read_image(dataset.root / path)
        rgb.append(cv2.resize(original, (mask.shape[-1], mask.shape[-2]),
                              interpolation=cv2.INTER_LINEAR))
    loader = DataLoader(TensorDataset(torch.stack(inputs), torch.stack(masks)),
                        batch_size=1, shuffle=False, num_workers=0)
    return loader, rgb


def add_overlay(ax, rgb, mask):
    ax.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., :3] = (1.0, 0.35, 0.0)
    overlay[..., 3] = mask * 0.55
    ax.imshow(overlay)


def case_figure(rgb, targets, predictions, paths, labels, threshold, split):
    """Return a figure; good images have undefined Dice, as in per-image CSVs."""
    fig, axes = plt.subplots(len(paths), 2 + len(labels),
                             figsize=(3.2 * (2 + len(labels)), 3.4 * len(paths)),
                             squeeze=False, layout="constrained")
    fig.suptitle(f"{split} | threshold = {threshold:g} | orange: defect mask", fontsize=11)
    for row, (image, target, path) in enumerate(zip(rgb, targets, paths, strict=True)):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(fill(path, width=36), fontsize=9)
        add_overlay(axes[row, 1], image, target)
        axes[row, 1].set_title("Ground truth", fontsize=10)
        for column, (label, masks) in enumerate(zip(labels, predictions, strict=True), start=2):
            prediction = masks[row]
            dice = f"{dice_score(prediction, target):.3f}" if target.any() else "N/A (good)"
            add_overlay(axes[row, column], image, prediction)
            axes[row, column].set_title(f"{fill(label, width=28)}\nDice = {dice}", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")
    return fig


def plot_cases(cfg, checkpoints, labels, split, paths, output):
    if not checkpoints or len(checkpoints) != len(labels):
        raise ValueError("Provide one matching --label for each --checkpoint")
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Labels must be nonempty and unique")
    threshold = float(cfg["eval"]["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("eval.threshold must be finite and between 0 and 1")
    for checkpoint in checkpoints:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    loader, rgb = prepare_cases(cfg, split, paths)
    predictions = []
    for checkpoint in checkpoints:
        model = load_model(cfg, checkpoint)
        # Reuse evaluation's bilinear logit resizing BEFORE sigmoid, including
        # align_corners=False and the config's independent input/evaluation sizes.
        probabilities, targets = predict_probabilities(model, loader)
        if not all(np.isfinite(probability).all() for probability in probabilities):
            raise ValueError(f"Non-finite predictions from {checkpoint}")
        predictions.append([probability >= threshold for probability in probabilities])
        del model

    fig = case_figure(rgb, targets, predictions, paths, labels, threshold, split)
    output = Path(output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Shared saved config: model architecture, input/eval sizes, threshold")
    parser.add_argument("--checkpoint", type=Path, nargs="+", action="extend", required=True,
                        help="One or more checkpoints, in column order; may be repeated")
    parser.add_argument("--label", nargs="+", action="extend", required=True,
                        help="One display label per checkpoint, in matching order")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--paths", nargs="+", action="extend", required=True,
                        help="Dataset-relative POSIX paths in the selected manifest, in row order")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output figure, e.g. results/s5/failure_cases.png (also PDF/SVG)")
    args = parser.parse_args(argv)
    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if cfg["train"].get("cpu_threads"):
            torch.set_num_threads(int(cfg["train"]["cpu_threads"]))
        output = plot_cases(cfg, args.checkpoint, args.label, args.split, args.paths, args.output)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(args.paths)} cases with {len(args.checkpoint)} checkpoints to {output}")


if __name__ == "__main__":
    main()
