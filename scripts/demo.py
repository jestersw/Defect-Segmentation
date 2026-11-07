# ruff: noqa: E402
"""Run CPU defect segmentation on arbitrary images, without manifests or masks."""

import argparse
import json
import sys
import time
from pathlib import Path
from textwrap import fill

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import IMAGENET_MEAN, IMAGENET_STD, read_image
from src.evaluate import load_model
from src.metrics import is_false_positive

TIMING_SCOPE = (
    "CPU FP32, batch 1: model forward, bilinear logit resizing and sigmoid. "
    "Excludes loading, image preprocessing, thresholding, component analysis and saving. "
    "No warm-up; the first image includes first-forward overhead."
)


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare_image(path, input_size):
    """Match MVTecSegDataset evaluation RGB resizing and ImageNet normalization."""
    original = read_image(path)
    image = cv2.resize(original, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
    return original, tensor


def predict_image(model, image, eval_size):
    """Time synchronous CPU inference; resize logits before applying sigmoid."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            start = time.perf_counter_ns()
            logits = model(image)
            if logits.shape[-2:] != (eval_size, eval_size):
                logits = F.interpolate(logits, size=(eval_size, eval_size),
                                       mode="bilinear", align_corners=False)
            probabilities = torch.sigmoid(logits)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            if logits.shape != (1, 1, eval_size, eval_size):
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
            if logits.device.type != "cpu" or not torch.isfinite(logits).all():
                raise ValueError("Expected finite CPU logits")
            return probabilities[0, 0].float().numpy(), elapsed_ms
    finally:
        model.train(was_training)


def prediction_summary(probabilities, threshold, min_component):
    mask = probabilities >= threshold
    return mask, {
        "predicted_defect_area_px": int(mask.sum()),
        "max_probability": float(probabilities.max()),
        # The same 8-connected component rule used for good-image FPR.
        "has_defect": bool(is_false_positive(mask, min_component)),
    }


def render_result(original, probabilities, mask, report, output):
    size = probabilities.shape
    rgb = cv2.resize(original, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    try:
        fig.suptitle(fill(Path(report["image"]).name, width=90), fontsize=11)
        axes[0].imshow(rgb)
        axes[0].set_title(f"Input ({size[1]} x {size[0]})")
        heatmap = axes[1].imshow(probabilities, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("Defect probability")
        fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
        axes[2].imshow(rgb)
        overlay = np.zeros((*size, 4), dtype=np.float32)
        overlay[..., :3] = (1.0, 0.35, 0.0)
        overlay[..., 3] = mask * 0.55
        axes[2].imshow(overlay, interpolation="nearest")
        status = "Defect" if report["has_defect"] else "No defect"
        axes[2].set_title(
            f"{status} | threshold {report['threshold']:g}\n"
            f"Orange: prediction ({report['predicted_defect_area_px']} px)"
        )
        for ax in axes:
            ax.axis("off")
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def run_demo(cfg, checkpoint, images, output_dir, threshold=None):
    input_size = positive_int(cfg["train"]["input_size"], "train.input_size")
    eval_size = positive_int(cfg["train"]["eval_size"], "train.eval_size")
    threads = positive_int(cfg["train"].get("cpu_threads", 8), "train.cpu_threads")
    min_component = positive_int(cfg["eval"]["good_fp_min_component_px"],
                                 "eval.good_fp_min_component_px")
    threshold = float(cfg["eval"]["threshold"] if threshold is None else threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1")
    checkpoint = Path(checkpoint).resolve()
    images = [Path(path).resolve() for path in images]
    if not images:
        raise ValueError("Provide at least one image")
    for path in [checkpoint, *images]:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
    output_dir = Path(output_dir)
    outputs = []
    for index, path in enumerate(images, start=1):
        # Numbering preserves order and distinguishes identical basenames.
        stem = f"{index:03d}_{path.stem}"
        outputs.append((output_dir / f"{stem}.png", output_dir / f"{stem}.json"))
    protected = {checkpoint, *images}
    if any(path.resolve() in protected for pair in outputs for path in pair):
        raise ValueError("Output files must not overwrite input images or the checkpoint")

    previous_threads = torch.get_num_threads()
    reports = []
    try:
        torch.set_num_threads(threads)
        model = load_model(cfg, checkpoint).to(device="cpu", dtype=torch.float32)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, (figure_path, json_path) in zip(images, outputs, strict=True):
            original, image = prepare_image(path, input_size)
            probabilities, elapsed_ms = predict_image(model, image, eval_size)
            mask, summary = prediction_summary(probabilities, threshold, min_component)
            report = {
                "image": str(path), "checkpoint": str(checkpoint),
                "original_size": {"height": original.shape[0], "width": original.shape[1]},
                "input_size": input_size, "eval_size": eval_size,
                "threshold": threshold, "good_fp_min_component_px": min_component,
                **summary, "inference_time_ms": elapsed_ms,
                "device": "cpu", "cpu_threads": threads, "timing_scope": TIMING_SCOPE,
                "figure": str(figure_path),
            }
            render_result(original, probabilities, mask, report, figure_path)
            json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                 encoding="utf-8")
            reports.append(report)
            print(f"Saved {figure_path} and {json_path}")
    finally:
        torch.set_num_threads(previous_threads)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Saved run configuration")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", action="extend", required=True,
                        help="Image files: working-directory-relative or absolute paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float,
                        help="Override probability threshold for this demo")
    args = parser.parse_args(argv)
    try:
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        run_demo(cfg, args.checkpoint, args.images, args.output_dir, args.threshold)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
