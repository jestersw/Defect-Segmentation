"""Inference runtime and peak process RAM benchmark on CPU.

Measures time per image (after warmup) and peak process RSS using psutil,
following the approach of scripts/smoke_test.py.

Usage:
    python -m scripts.benchmark \
        --checkpoint checkpoints/dice_focal_best.pt \
        --config configs/dice_focal.yaml \
        --output results/benchmark_dice_focal.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader
import yaml

from src.data import build_dataset
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def rss_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    torch.set_num_threads(8)
    torch.manual_seed(cfg["seed"])

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dataset = build_dataset(cfg, split="test")
    loader = DataLoader(
        dataset,
        batch_size=1,  # per-image measurement
        shuffle=False,
        num_workers=0,
    )

    # Warmup
    with torch.no_grad():
        for i, (image, _) in enumerate(loader):
            if i >= args.warmup:
                break
            _ = model(image.to(device))

    # Timed runs
    times = []
    rss_samples = [rss_mib()]
    with torch.no_grad():
        for i, (image, _) in enumerate(loader):
            if i >= args.runs:
                break
            start = time.perf_counter()
            _ = model(image.to(device))
            times.append(time.perf_counter() - start)
            rss_samples.append(rss_mib())

    avg_s = sum(times) / len(times)
    peak_rss = max(rss_samples)
    result = {
        "checkpoint": args.checkpoint,
        "runs": len(times),
        "avg_inference_s": avg_s,
        "avg_inference_ms": avg_s * 1000,
        "peak_rss_mib": peak_rss,
        "note": "RSS is whole-process; measured with psutil.Process().memory_info().rss.",
    }
    print(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()