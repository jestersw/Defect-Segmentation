"""CPU latency for model forward plus logit resizing to the evaluation resolution."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import MVTecSegDataset  # noqa: E402
from src.evaluate import load_model  # noqa: E402

CPU_THREADS = 8
WARMUP_IMAGES = 5
MIN_TIMED_IMAGES = 50


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def positive_size(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare_images(cfg, num_images, input_size=None):
    """Load only the selected validation samples, outside the timed region."""
    if num_images < MIN_TIMED_IMAGES:
        raise ValueError("At least 50 timed images are required")
    positive_size(cfg["train"]["eval_size"], "eval_size")
    benchmark_cfg = deepcopy(cfg)
    benchmark_cfg["train"]["input_size"] = positive_size(
        cfg["train"]["input_size"] if input_size is None else input_size, "input_size"
    )
    benchmark_cfg["data"]["cache_images"] = False
    dataset = MVTecSegDataset("val", benchmark_cfg, train=False)
    count = WARMUP_IMAGES + num_images
    if len(dataset) < count:
        raise ValueError(f"Need {count} distinct validation images, found {len(dataset)}")
    images = [dataset[index][0].unsqueeze(0).contiguous() for index in range(count)]
    paths = [record["path"] for record in dataset.records[:count]]
    return images, paths


def forward_logits(model, image, eval_size):
    """Match src.evaluate: resize logits with bilinear interpolation before sigmoid."""
    logits = model(image)
    if logits.shape[-2:] != (eval_size, eval_size):
        logits = F.interpolate(logits, size=(eval_size, eval_size),
                               mode="bilinear", align_corners=False)
    return logits


def benchmark_model(model, images, eval_size):
    """Time synchronous FP32 CPU forward and logit resizing; exclude sigmoid/metrics."""
    positive_size(eval_size, "eval_size")
    if len(images) < WARMUP_IMAGES + MIN_TIMED_IMAGES:
        raise ValueError("Need five warm-up images and at least 50 timed images")
    input_size = images[0].shape[-1]
    for image in images:
        if (image.shape != (1, 3, input_size, input_size) or input_size <= 0
                or image.dtype != torch.float32
                or image.device.type != "cpu" or not image.is_contiguous()):
            raise ValueError("Inputs must be matching contiguous float32 CPU tensors [1, 3, s, s]")
        if not torch.isfinite(image).all():
            raise ValueError("Input contains non-finite values")
    model = model.to(device="cpu", dtype=torch.float32)
    was_training = model.training
    previous_threads = torch.get_num_threads()
    durations = []
    try:
        torch.set_num_threads(CPU_THREADS)
        model.eval()
        with torch.inference_mode():
            for index, image in enumerate(images):
                if index < WARMUP_IMAGES:
                    logits = forward_logits(model, image, eval_size)
                else:
                    start = time.perf_counter_ns()
                    logits = forward_logits(model, image, eval_size)
                    durations.append((time.perf_counter_ns() - start) / 1_000_000)
                # Validation is outside the timed pipeline; do not report invalid outputs.
                if logits.shape != (1, 1, eval_size, eval_size):
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
                if logits.device.type != "cpu" or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite CPU logits")
                del logits
    finally:
        model.train(was_training)
        torch.set_num_threads(previous_threads)
    return {
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95, method="linear")),
        "per_image_ms": durations,
    }


def environment():
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif platform.system() == "Linux" and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(), "cpu": cpu, "python": platform.python_version(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(), "system_ram_bytes": psutil.virtual_memory().total,
        "cpu_threads": CPU_THREADS, "interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "segmentation-models-pytorch", "numpy",
            "opencv-python-headless", "albumentations",
        )},
    }


def run_benchmark(cfg, checkpoint, num_images=MIN_TIMED_IMAGES, input_size=None):
    # Fail before image loading if the checkpoint is missing; never download encoder weights.
    checkpoint_hash = file_sha256(checkpoint)
    images, paths = prepare_images(cfg, num_images, input_size)
    model = load_model(cfg, checkpoint)
    eval_size = cfg["train"]["eval_size"]
    timing = benchmark_model(model, images, eval_size)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "source_config": deepcopy(cfg),
        "protocol": {
            "split": "val", "device": "cpu", "dtype": "float32", "batch_size": 1,
            "input_size": images[0].shape[-1], "eval_size": eval_size,
            "warmup_images": WARMUP_IMAGES, "timed_images": num_images,
            "scope": "Model forward plus bilinear logit resizing to eval_size when needed "
                     "(align_corners=False); inputs preloaded in RAM. Excludes decoding, "
                     "input resizing, normalisation, checkpoint loading, sigmoid, thresholding "
                     "and metrics.",
            "selection": "First five validation entries warm up; next N distinct entries timed",
            "percentile_method": "linear",
        },
        "val_manifest_sha256": file_sha256(ROOT / cfg["data"]["splits_dir"] / "val.txt"),
        "warmup_paths": paths[:WARMUP_IMAGES], "timed_paths": paths[WARMUP_IMAGES:],
        "parameters": sum(p.numel() for p in model.parameters()),
        "environment": environment(), **timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Saved run configuration")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=MIN_TIMED_IMAGES)
    parser.add_argument("--input-size", type=int,
                        help="Override model input size (default: saved train.input_size)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_images < MIN_TIMED_IMAGES:
        parser.error("--num-images must be at least 50")
    if args.input_size is not None and args.input_size <= 0:
        parser.error("--input-size must be positive")
    if args.output.resolve() in (args.config.resolve(), args.checkpoint.resolve()):
        parser.error("--output must differ from the configuration and checkpoint paths")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    torch.manual_seed(int(cfg["seed"]))
    report = run_benchmark(cfg, args.checkpoint, args.num_images, args.input_size)
    report["config_sha256"] = file_sha256(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mean_ms", "median_ms", "p95_ms")}))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
