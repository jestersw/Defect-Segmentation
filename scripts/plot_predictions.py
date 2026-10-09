import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import yaml

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def choose_examples(targets, count=6):
    areas = [int(t.sum()) for t in targets]
    defective = sorted((a, i) for i, a in enumerate(areas) if a > 0)
    good = [i for i, a in enumerate(areas) if a == 0]
    if not defective:
        return good[:count]
    picks = []
    positions = np.linspace(0, len(defective) - 1, num=min(count - 1, len(defective)))
    for position in positions:
        index = defective[int(round(position))][1]
        if index not in picks:
            picks.append(index)
    picks.extend(good[: count - len(picks)])
    return picks


def to_rgb(image, size):
    tensor = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]
    array = tensor.permute(1, 2, 0).cpu().numpy() * STD + MEAN
    return np.clip(array, 0.0, 1.0)


def render(images, targets, probabilities, records, picks, threshold, output):
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.5 * len(picks)), squeeze=False)
    for row, index in enumerate(picks):
        target = targets[index]
        prediction = probabilities[index] >= threshold
        rgb = to_rgb(images[index], target.shape)
        record = records[index] if records else {"category": "", "defect_type": ""}
        title = f"{record['category']}/{record['defect_type']}, {int(target.sum())} px"
        panels = [(rgb, title), (target, "ground truth"), (prediction, "prediction")]
        for column, (data, label) in enumerate(panels):
            ax = axes[row][column]
            if column == 0:
                ax.imshow(data)
            else:
                ax.imshow(rgb)
                ax.imshow(np.ma.masked_where(~data, data), cmap="autumn", alpha=0.6)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot validation predictions of a checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "results/s2/baseline_examples.png")
    args = parser.parse_args()
    from src.data import make_loaders
    from src.evaluate import load_model, predict_probabilities

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    loader = dict(zip(("train", "val", "test"), make_loaders(cfg)))[args.split]
    model = load_model(cfg, args.checkpoint)
    probabilities, targets = predict_probabilities(model, loader)
    picks = choose_examples(targets, args.count)
    images = [loader.dataset[i][0] for i in picks]
    by_index = dict(zip(picks, images))
    records = getattr(loader.dataset, "records", None)
    output = render(
        by_index, targets, probabilities, records, picks, cfg["eval"]["threshold"], args.output
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
