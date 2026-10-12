"""CPU training with validation-selected checkpoints and per-epoch artifacts (#27)."""

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import matplotlib
import numpy as np
import psutil
import torch
import yaml

from src.data import ROOT, make_loaders, project_path
from src.evaluate import evaluate
from src.losses import build_loss
from src.model import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIELDS = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision",
          "val_recall", "val_small_recall", "val_good_fpr", "lr", "epoch_seconds"]


def write_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def configure(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def prevent_sleep(enabled):
    active = enabled and os.name == "nt"
    if active and not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise RuntimeError("Could not request a Windows system sleep inhibitor")
    try:
        yield
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def environment():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "git_commit": commit, "python": platform.python_version(), "os": platform.platform(),
        "cpu": platform.processor(), "ram_gib": psutil.virtual_memory().total / 2**30,
        "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "segmentation-models-pytorch", "albumentations", "numpy")},
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"):
                          hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").glob("*.py"))},
    }


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_dice"] for row in history])
    axes[1].set_ylabel("Validation Dice (defective images)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=130)
    plt.close(fig)
    temporary.replace(path)


def train_epoch(model, loader, criterion, optimizer, limit_batches=None):
    model.train()
    loss_sum, count = 0.0, 0
    batches = loader if limit_batches is None else islice(loader, limit_batches)
    for index, (images, masks) in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != masks.shape:
            raise ValueError(f"Training shapes differ: {logits.shape} vs {masks.shape}")
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(images)
        count += len(images)
        if index % 10 == 0:
            print(f"  train batch {index}/{len(loader)}, loss={loss_sum / count:.6f}", flush=True)
    if count == 0:
        raise ValueError("Empty training loader")
    return loss_sum / count


def validate_config(cfg):
    train = cfg["train"]
    if train.get("device", "cpu") != "cpu":
        raise ValueError("This training entry point is CPU-only")
    for key in ("batch_size", "input_size", "eval_size", "max_epochs", "patience"):
        if int(train[key]) <= 0:
            raise ValueError(f"train.{key} must be positive")
    if train.get("cpu_threads", 8) <= 0 or train.get("num_workers", 0) < 0:
        raise ValueError("Invalid thread/worker count")
    if train.get("optimizer", "adamw") != "adamw" or train.get("scheduler", "cosine") != "cosine":
        raise ValueError("Expected AdamW and cosine schedule")
    if train.get("early_stopping_metric", "val_dice") != "val_dice":
        raise ValueError("Checkpoint selection must use val_dice")
    if not train.get("seeds"):
        raise ValueError("train.seeds must contain a seed")


def run_training(cfg, run_name, limit_batches=None, keep_awake=False):
    cfg = deepcopy(cfg)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("run-name must contain only letters, digits, dots, underscores or hyphens")
    if limit_batches is not None and limit_batches <= 0:
        raise ValueError("limit-batches must be positive")
    cfg["train"].setdefault("device", "cpu")
    cfg["train"].setdefault("cpu_threads", 8)
    cfg["train"]["limit_batches"] = limit_batches
    validate_config(cfg)
    configure(int(cfg["train"]["seeds"][0]), int(cfg["train"]["cpu_threads"]))
    output = project_path(cfg["output"]["results_dir"]) / "runs" / run_name
    checkpoints = project_path(cfg["output"]["checkpoints_dir"]) / run_name
    if output.exists() or checkpoints.exists():
        raise FileExistsError(f"Run already exists: {run_name}; choose a new name")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n").writeheader()
    plot_history([], output / "curves.png")
    summary = {
        "run_name": run_name, "status": "initializing", "started_utc":
        datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "epochs_completed": 0,
        "best_epoch": None, "best_val_dice": None, "total_training_time_seconds": 0,
        "limited_run": limit_batches is not None, "test_evaluated": False,
        "environment": environment(),
    }
    write_json(output / "summary.json", summary)
    started = time.perf_counter()
    history, best, bad_epochs = [], -math.inf, 0
    try:
        with prevent_sleep(keep_awake):
            train_loader, val_loader, _ = make_loaders(cfg)
            model, criterion = build_model(cfg).cpu(), build_loss(cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["train"]["max_epochs"]
            )
            summary.update({
                "status": "running", "parameters": sum(p.numel() for p in model.parameters()),
                "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
                "steps_per_epoch": len(train_loader),
                "split_sha256": {name: hashlib.sha256(
                    (project_path(cfg["data"]["splits_dir"]) / f"{name}.txt").read_bytes()
                ).hexdigest() for name in ("train", "val", "test")},
            })
            write_json(output / "summary.json", summary)
            for epoch in range(1, cfg["train"]["max_epochs"] + 1):
                budget = cfg["train"].get("max_seconds")
                if budget and history:
                    reserve = max(row["epoch_seconds"] for row in history) * 1.25
                    if time.perf_counter() - started + reserve >= budget:
                        summary["stop_reason"] = "time_budget"
                        print("Stopping before next epoch: time budget reached.", flush=True)
                        break
                print(f"Epoch {epoch}/{cfg['train']['max_epochs']}", flush=True)
                epoch_start = time.perf_counter()
                lr = optimizer.param_groups[0]["lr"]
                train_loss = train_epoch(model, train_loader, criterion, optimizer, limit_batches)
                print("  validating...", flush=True)
                metrics = evaluate(model, val_loader, cfg, criterion=criterion,
                                   limit_batches=limit_batches)
                val_dice = metrics["dice"]
                if not math.isfinite(val_dice):
                    raise ValueError("Validation Dice is undefined; validation needs defect masks")
                improved = val_dice > best
                if improved:
                    best, bad_epochs = val_dice, 0
                    summary.update(best_epoch=epoch, best_val_dice=best,
                                   best_validation_metrics=dict(metrics))
                else:
                    bad_epochs += 1
                scheduler.step()
                state = {"model_state_dict": model.state_dict(), "epoch": epoch,
                         "best_val_dice": best, "config": cfg,
                         "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict()}
                temporary = checkpoints / "last.tmp.pth"
                torch.save(state, temporary)
                temporary.replace(checkpoints / "last.pth")
                if improved:
                    temporary = checkpoints / "best.tmp.pth"
                    torch.save(state, temporary)
                    temporary.replace(checkpoints / "best.pth")
                row = {
                    "epoch": epoch, "train_loss": train_loss, "val_loss": metrics.pop("loss"),
                    **{f"val_{key}": value for key, value in metrics.items()},
                    "lr": lr, "epoch_seconds": time.perf_counter() - epoch_start,
                }
                history.append(row)
                plot_history(history, output / "curves.png")
                # Include plotting, validation and checkpoint I/O in actual epoch duration.
                row["epoch_seconds"] = time.perf_counter() - epoch_start
                with (output / "history.csv").open("a", newline="", encoding="utf-8") as stream:
                    csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n").writerow(row)
                summary.update(epochs_completed=epoch, epochs_without_improvement=bad_epochs,
                               total_training_time_seconds=time.perf_counter() - started,
                               last_epoch_seconds=row["epoch_seconds"], last_metrics=row)
                write_json(output / "summary.json", summary)
                print(f"Epoch {epoch}: loss={train_loss:.5f}, val_dice={val_dice:.5f}, "
                      f"small_recall={metrics['small_recall']:.5f}, "
                      f"seconds={row['epoch_seconds']:.1f}, best={best:.5f}", flush=True)
                if bad_epochs >= cfg["train"]["patience"]:
                    summary["stop_reason"] = "early_stopping"
                    break
            else:
                summary["stop_reason"] = "max_epochs"
            summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["total_training_time_seconds"] = time.perf_counter() - started
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "summary.json", summary)
    print(f"Finished: {output}; best checkpoint: {checkpoints / 'best.pth'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--loss", choices=("bce", "dice", "dice_focal"))
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--keep-awake", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key, value in (("input_size", args.input_size), ("max_epochs", args.max_epochs)):
        if value is not None:
            cfg["train"][key] = value
    if args.loss is not None:
        cfg["loss"] = args.loss
    run_training(cfg, args.run_name, args.limit_batches, args.keep_awake)


if __name__ == "__main__":
    main()
