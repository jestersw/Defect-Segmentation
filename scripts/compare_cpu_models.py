"""Compare synthetic training steps for the baseline and CPU candidate at 256px."""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.model import build_model  # noqa: E402
from src.train import configure, environment  # noqa: E402


def measure(filename):
    cfg = yaml.safe_load((root / filename).read_text())
    configure(42, 8)
    model = build_model(cfg).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    image = torch.randn(8, 3, 256, 256)
    mask = torch.randint(0, 2, (8, 1, 256, 256)).float()
    times = []
    for index in range(23):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        assert logits.shape == mask.shape
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        seconds = time.perf_counter() - start
        assert torch.isfinite(loss)
        if index >= 3:
            times.append(seconds)
        print(f"{cfg['model']['encoder']} {index + 1}/23: {seconds:.4f} s", flush=True)
    return {"config": filename, "model": cfg["model"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "step_seconds": times, "mean_step_seconds": statistics.mean(times)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=root / "results/s2/cpu_model_comparison.json")
    args = parser.parse_args()
    configure(42, 8)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
        "batch_size": 8, "size": 256, "warmup": 3, "measured_steps": 20,
        "protocol": "Synthetic FP32 BCE forward/backward/AdamW; excludes validation and loading",
        "results": [],
    }
    for filename in ("configs/base.yaml", "configs/cpu_light.yaml"):
        report["results"].append(measure(filename))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
