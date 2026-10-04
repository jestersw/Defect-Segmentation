"""Measure synthetic U-Net CPU training steps and peak process RAM."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import platform
import random
import statistics
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty

import numpy as np
import psutil
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parents[1]


def cpu_name():
    if platform.system() == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def configure(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker(size, args, queue):
    """A fresh process per resolution prevents allocator/high-water-mark carryover."""
    try:
        configure(args.threads)
        process = psutil.Process()
        peak_rss = process.memory_info().rss
        stop = threading.Event()

        def sample():
            nonlocal peak_rss
            while not stop.wait(0.01):
                peak_rss = max(peak_rss, process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            result = benchmark(size, args, torch.device(args.device))
        finally:
            stop.set()
            sampler.join()
        memory = process.memory_info()
        # Windows exposes the exact lifetime peak working set. Else sample RSS at 10 ms.
        result["peak_process_ram_bytes"] = getattr(memory, "peak_wset", peak_rss)
        result["peak_process_ram_gib"] = result["peak_process_ram_bytes"] / 2**30
        result["ram_measurement"] = (
            "OS peak working set" if hasattr(memory, "peak_wset") else "RSS sampled every 10 ms"
        )
        queue.put((result, None))
    except Exception:
        queue.put((None, traceback.format_exc()))


def estimates(step_seconds, samples, batch_size):
    steps = math.ceil(samples / batch_size)
    return {
        "samples": samples,
        "steps_per_epoch": steps,
        "minutes_per_epoch": step_seconds * steps / 60,
        "hours_per_60_epochs": step_seconds * steps * 60 / 3600,
    }


def benchmark(size, args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = smp.Unet(
        "resnet34", encoder_weights=args.encoder_weights, classes=1
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    images = torch.randn(args.batch_size, 3, size, size, device=device)
    masks = torch.randint(0, 2, (args.batch_size, 1, size, size), device=device).float()
    expected_shape = [args.batch_size, 1, size, size]

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if list(logits.shape) != expected_shape:
            raise RuntimeError(f"Unexpected output shape: {list(logits.shape)}")
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(args.warmup):
        loss = step()
        print(f"{size}: warm-up {_ + 1}/{args.warmup} complete", flush=True)
    times = []
    for index in range(args.steps):
        start = time.perf_counter()
        loss = step()
        times.append(time.perf_counter() - start)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss")
        print(f"{size}: step {index + 1}/{args.steps}: {times[-1]:.4f} s", flush=True)
    mean = statistics.mean(times)
    manifest = ROOT / "splits/train.txt"
    current_samples = len([p for p in manifest.read_text().splitlines() if p.strip()])
    result = {
        "size": size,
        "output_shape": expected_shape,
        "mean_step_seconds": mean,
        "step_seconds": times,
        "last_loss": loss.item(),
        "issue_estimate": estimates(mean, 582, args.batch_size),
        "current_split_estimate": estimates(mean, current_samples, args.batch_size),
        "train_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def markdown(report):
    env = report["environment"]
    lines = [
        "# CPU compute smoke test", "",
        f"- Recorded (UTC): {report['timestamp_utc']}",
        f"- Platform: {env['platform']}; device: {env['device_name']}",
        f"- System RAM: {env['system_ram_gib']:.2f} GiB; limits: {env['session_limits']}",
        f"- Python: {env['python']}; torch: {env['packages']['torch']}",
        f"- CPU task protocol satisfied: {report['cpu_protocol_satisfied']}",
        f"- CPU threads: {env['cpu_threads']}; interop threads: 1.",
        f"- Seed: {report['settings']['seed']}; FP32; deterministic algorithms enabled.",
        "- Step includes zero_grad, forward, BCEWithLogitsLoss, backward and AdamW update.",
        "- Inputs stay on device; excludes loading, augmentation, validation and checkpoint I/O.",
        "- Estimates are synthetic training-only projections, not measured epoch durations.",
        "", "| Size | Output | Mean s/step | Peak process RAM GiB | Train N | Steps/epoch |"
        " min/epoch | h/60 epochs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        memory = f"{row['peak_process_ram_gib']:.3f}"
        for key in ("issue_estimate", "current_split_estimate"):
            estimate = row[key]
            lines.append(
                f"| {row['size']} | {row['output_shape']} | {row['mean_step_seconds']:.4f} |"
                f" {memory} | {estimate['samples']} | {estimate['steps_per_epoch']} |"
                f" {estimate['minutes_per_epoch']:.3f} | {estimate['hours_per_60_epochs']:.3f} |"
            )
    lines += ["", "582 images is the issue reference; the current committed split is counted"
              " independently. RAM is the whole worker process peak, including model setup"
              " and warm-up, measured in a fresh process per resolution.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=min(8, psutil.cpu_count(logical=False) or 1))
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--platform", required=True, help="Actual host, e.g. local-laptop")
    parser.add_argument("--session-limits", required=True, help="Limits observed for this session")
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke_test.json")
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.warmup, args.threads) < 1:
        parser.error("batch-size, steps, warmup and threads must be positive")
    if any(size < 32 or size % 32 for size in args.sizes):
        parser.error("sizes must be positive multiples of 32")
    args.encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    configure(args.threads)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = ["torch", "torchvision", "segmentation-models-pytorch", "albumentations",
                "opencv-python", "opencv-python-headless", "numpy", "scikit-image",
                "scikit-learn", "pyyaml", "matplotlib", "tqdm", "psutil"]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "requirements_sha256": hashlib.sha256((ROOT / "requirements.txt").read_bytes()).hexdigest(),
        "environment": {
            "platform": args.platform, "os": platform.platform(),
            "device": args.device,
            "device_name": cpu_name(),
            "system_ram_gib": psutil.virtual_memory().total / 2**30,
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(),
            "session_limits": args.session_limits, "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(),
            "packages": {name: importlib.metadata.version(name) for name in packages},
        },
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "cpu_protocol_satisfied": (
            args.device == "cpu" and args.batch_size == 8 and args.warmup == 3
            and args.steps == 20 and set(args.sizes) == {512, 256}
            and args.encoder_weights == "imagenet" and platform.python_version_tuple()[:2]
            == ("3", "11")
        ),
        "results": [],
    }
    print(json.dumps(report, indent=2), flush=True)
    for size in args.sizes:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=worker, args=(size, args, queue))
        process.start()
        # The payload is small; drain it before joining to avoid Queue feeder deadlocks.
        while True:
            try:
                result, error = queue.get(timeout=1)
                break
            except Empty:
                if not process.is_alive():
                    raise RuntimeError(
                        f"Smoke worker exited without results (code {process.exitcode})"
                    ) from None
        process.join()
        queue.close()
        if error:
            raise RuntimeError(error)
        report["results"].append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
