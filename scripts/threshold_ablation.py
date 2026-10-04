"""Threshold sensitivity sweep on the validation split.

Evaluates a trained checkpoint at thresholds 0.3, 0.5 and 0.7
(see configs/base.yaml: eval.threshold_sweep) and saves the metrics to JSON.

Usage:
    python -m scripts.threshold_ablation \
        --checkpoint checkpoints/dice_focal_best.pt \
        --config configs/dice_focal.yaml \
        --output results/threshold_ablation_dice_focal.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.data import build_dataset  # placeholders, will be filled by psandersi
from src.model import build_model
from src.metrics import aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    torch.set_num_threads(8)

    # Build model and load weights
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    # Dataset: uses the fixed split manifests
    dataset = build_dataset(cfg, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    thresholds = cfg["eval"]["threshold_sweep"]  # [0.3, 0.5, 0.7]
    small_area = cfg["eval"]["small_defect_area_px"]
    coverage = cfg["eval"]["small_detect_coverage"]
    fp_min_px = cfg["eval"]["good_fp_min_component_px"]

    # Collect all predictions once, then sweep thresholds
    probs_all, gts_all = [], []
    with torch.no_grad():
        for images, masks in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu()
            probs_all.append(probs)
            gts_all.append(masks)

    probs_all = torch.cat(probs_all).numpy()
    gts_all = torch.cat(gts_all).numpy()

    results = {}
    for t in thresholds:
        preds = (probs_all >= t).astype("uint8")
        gts = gts_all.astype("uint8")
        metrics = aggregate(
            preds, gts,
            max_area=small_area,
            coverage=coverage,
            fp_min_px=fp_min_px,
        )
        results[str(t)] = metrics
        print(f"threshold={t}: {metrics}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()