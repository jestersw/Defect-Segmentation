# ruff: noqa: E402
"""Export per-image segmentation metrics for a trained checkpoint."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import make_loaders
from src.evaluate import load_model, metric_settings, predict_probabilities
from src.metrics import (
    dice_score,
    iou_score,
    is_false_positive,
    pixel_counts,
    small_defect_hits,
)

CSV_FIELDS = [
    "path",
    "category",
    "defect_type",
    "defect_area_px",
    "dice",
    "iou",
    "tp",
    "fp",
    "fn",
    "small_components_total",
    "small_components_detected",
    "false_positive",
]


def per_image_rows(records, probabilities, targets, threshold, settings):
    """Calculate one CSV row per image using the project's metric functions."""
    if not (len(records) == len(probabilities) == len(targets)):
        raise ValueError("Records, predictions and targets must have equal lengths")
    if not records:
        raise ValueError("Cannot calculate metrics for an empty dataset")

    rows = []

    for record, probability, target in zip(records, probabilities, targets):
        target = np.asarray(target).astype(bool)
        prediction = np.asarray(probability) >= threshold

        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for {record['path']}: "
                f"{prediction.shape} vs {target.shape}"
            )

        defect_area = int(target.sum())
        tp, fp, fn = pixel_counts(prediction, target)

        # The official aggregate excludes good images from Dice and IoU.
        # Keep these fields empty for good images to preserve that convention.
        if defect_area:
            dice = dice_score(prediction, target)
            iou = iou_score(prediction, target)
        else:
            dice = None
            iou = None

        small_detected, small_total = small_defect_hits(
            prediction,
            target,
            max_area=settings["max_area"],
            coverage=settings["coverage"],
        )

        # The false-positive flag is defined only for good images.
        false_positive = (
            int(is_false_positive(prediction, settings["min_component"]))
            if defect_area == 0
            else None
        )

        rows.append(
            {
                "path": record["path"],
                "category": record["category"],
                "defect_type": record["defect_type"],
                "defect_area_px": defect_area,
                "dice": dice,
                "iou": iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "small_components_total": small_total,
                "small_components_detected": small_detected,
                "false_positive": false_positive,
            }
        )

    return rows


def aggregate_per_image_rows(rows):
    """Reconstruct the aggregate metrics from the per-image CSV rows."""
    if not rows:
        raise ValueError("Cannot aggregate an empty list of rows")

    defective = [row for row in rows if row["defect_area_px"] > 0]
    good = [row for row in rows if row["defect_area_px"] == 0]

    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)

    small_total = sum(row["small_components_total"] for row in rows)
    small_detected = sum(
        row["small_components_detected"] for row in rows
    )
    good_fp = sum(row["false_positive"] for row in good)

    return {
        "dice": (
            float(np.mean([row["dice"] for row in defective]))
            if defective
            else float("nan")
        ),
        "iou": (
            float(np.mean([row["iou"] for row in defective]))
            if defective
            else float("nan")
        ),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "small_recall": (
            small_detected / small_total if small_total else float("nan")
        ),
        "good_fpr": (
            good_fp / len(good) if good else float("nan")
        ),
    }


def write_csv(path, rows):
    """Write rows to CSV, leaving undefined values empty."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def json_safe(metrics):
    """Convert undefined metrics to JSON null."""
    return {
        key: value if math.isfinite(value) else None
        for key, value in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"Config file not found: {args.config}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    loaders = dict(zip(("train", "val", "test"), make_loaders(cfg)))
    loader = loaders[args.split]

    model = load_model(cfg, args.checkpoint)
    probabilities, targets = predict_probabilities(model, loader)

    rows = per_image_rows(
        records=loader.dataset.records,
        probabilities=probabilities,
        targets=targets,
        threshold=float(cfg["eval"]["threshold"]),
        settings=metric_settings(cfg),
    )

    output = args.output or args.config.parent / f"per_image_{args.split}.csv"
    write_csv(output, rows)

    metrics = aggregate_per_image_rows(rows)
    print(json.dumps(json_safe(metrics), indent=2))
    print(f"Images: {len(rows)}")
    print(f"Per-image metrics saved to: {output}")


if __name__ == "__main__":
    main()
