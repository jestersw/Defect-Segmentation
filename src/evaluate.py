import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.metrics import aggregate

METRIC_KEYS = ("dice", "iou", "precision", "recall", "small_recall", "good_fpr")


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _collect_predictions(model, loader, criterion=None, limit_batches=None):
    device = model_device(model)
    was_training = model.training
    model.eval()
    probabilities, targets = [], []
    loss_sum, count = 0.0, 0
    batches = loader if limit_batches is None else islice(loader, limit_batches)
    try:
        with torch.no_grad():
            for images, masks in batches:
                logits = model(images.to(device))
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                if criterion is not None:
                    loss = criterion(logits, masks.to(device))
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite validation loss")
                    loss_sum += loss.item() * len(images)
                count += len(images)
                probabilities.extend(torch.sigmoid(logits)[:, 0].float().cpu().numpy())
                targets.extend((masks[:, 0] > 0.5).cpu().numpy())
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("Empty evaluation loader")
    return probabilities, targets, loss_sum / count


def predict_probabilities(model, loader):
    probabilities, targets, _ = _collect_predictions(model, loader)
    return probabilities, targets


def metric_settings(cfg):
    settings = cfg["eval"]
    if settings.get("small_defect_area_px") is None:
        raise ValueError("eval.small_defect_area_px is not set in the config")
    return {
        "max_area": settings["small_defect_area_px"],
        "coverage": settings.get("small_detect_coverage", 0.25),
        "min_component": settings.get("good_fp_min_component_px", 20),
    }


def score(probabilities, targets, threshold, settings):
    predictions = [p >= threshold for p in probabilities]
    return aggregate(predictions, targets, **settings)


def per_category(probabilities, targets, records, threshold, settings):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["category"]].append(index)
    rows = []
    for category in sorted(groups):
        indices = groups[category]
        result = score(
            [probabilities[i] for i in indices], [targets[i] for i in indices], threshold, settings
        )
        rows.append({"category": category, "images": len(indices), **result})
    return rows


def evaluate(model, loader, cfg, threshold=None, *, criterion=None, limit_batches=None):
    threshold = cfg["eval"]["threshold"] if threshold is None else threshold
    probabilities, targets, loss = _collect_predictions(model, loader, criterion, limit_batches)
    result = score(probabilities, targets, threshold, metric_settings(cfg))
    if criterion is not None:
        result["loss"] = loss
    return result


def full_report(model, loader, cfg):
    settings = metric_settings(cfg)
    threshold = cfg["eval"]["threshold"]
    probabilities, targets = predict_probabilities(model, loader)
    overall = score(probabilities, targets, threshold, settings)
    records = getattr(loader.dataset, "records", None)
    categories = (
        per_category(probabilities, targets, records, threshold, settings) if records else []
    )
    sweep = [
        {"threshold": t, **score(probabilities, targets, t, settings)}
        for t in cfg["eval"].get("threshold_sweep", [threshold])
    ]
    return {
        "threshold": threshold,
        "images": len(targets),
        "defective_images": int(sum(t.any() for t in targets)),
        "metrics": overall,
        "per_category": categories,
        "threshold_sweep": sweep,
    }


def clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return clean(value.item())
    return value


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: clean(v) for k, v in row.items()} for row in rows])


def save_report(report, output_dir, split):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": split,
        "threshold": report["threshold"],
        "images": report["images"],
        "defective_images": report["defective_images"],
        **{key: clean(report["metrics"][key]) for key in METRIC_KEYS},
    }
    (output_dir / f"metrics_{split}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_dir / f"metrics_{split}_per_category.csv", report["per_category"])
    write_csv(output_dir / f"threshold_sweep_{split}.csv", report["threshold_sweep"])
    return summary


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model(cfg, checkpoint):
    from src.model import build_model

    model_cfg = {**cfg, "model": {**cfg["model"], "encoder_weights": None}}
    model = build_model(model_cfg)
    model.load_state_dict(load_state_dict(checkpoint))
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    from src.data import make_loaders

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    loaders = dict(zip(("train", "val", "test"), make_loaders(cfg)))
    model = load_model(cfg, args.checkpoint)
    report = full_report(model, loaders[args.split], cfg)
    output_dir = args.output_dir or args.config.parent
    summary = save_report(report, output_dir, args.split)
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {output_dir}")


if __name__ == "__main__":
    main()
