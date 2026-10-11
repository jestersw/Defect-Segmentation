"""Save eight augmented training samples with ground-truth mask overlays."""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import IMAGENET_MEAN, IMAGENET_STD, MVTecSegDataset  # noqa: E402


def render_samples(cfg, output):
    dataset = MVTecSegDataset("train", cfg, train=True)
    if len(dataset) < 8:
        raise ValueError("The training split must contain at least eight samples")
    indices = np.random.default_rng(cfg["seed"]).choice(len(dataset), size=8, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), squeeze=False)
    try:
        for ax, index in zip(axes.flat, indices):
            image, mask = dataset[int(index)]
            rgb = image.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
            foreground = mask[0].numpy() > 0
            ax.imshow(np.clip(rgb, 0, 1))
            overlay = np.zeros((*foreground.shape, 4), dtype=np.float32)
            overlay[foreground] = (1.0, 0.15, 0.0, 0.45)
            ax.imshow(overlay)
            record = dataset.records[int(index)]
            ax.set_title(f"{record['category']}/{record['defect_type']}\n"
                         f"{Path(record['path']).stem} | mask: {foreground.sum()} px", fontsize=9)
            ax.axis("off")
        fig.suptitle("Training samples after augmentation | orange: ground truth", fontsize=12)
        fig.tight_layout()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/s2/data_samples.png")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(f"Saved {render_samples(cfg, args.output)}")


if __name__ == "__main__":
    main()
