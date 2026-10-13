import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.evaluate import load_model
from src.train import run_training


@pytest.mark.parametrize("loss", ["bce", "dice"])
def test_cli_writes_artifacts_and_loadable_checkpoint(training_cfg, tmp_path, loss):
    training_cfg["loss"] = loss
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(training_cfg))
    subprocess.run([
        sys.executable, "-m", "src.train", "--config", str(config), "--run-name", "synthetic",
        "--max-epochs", "1", "--input-size", "64", "--limit-batches", "1",
    ], check=True, capture_output=True, text=True, timeout=120)
    output = Path(training_cfg["output"]["results_dir"]) / "runs/synthetic"
    assert all((output / name).is_file()
               for name in ("config.yaml", "history.csv", "curves.png", "summary.json"))
    saved = yaml.safe_load((output / "config.yaml").read_text())
    assert saved["train"]["limit_batches"] == 1 and saved["train"]["input_size"] == 64
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["epochs_completed"] == 1
    assert summary["test_evaluated"] is False and summary["limited_run"] is True
    rows = list(csv.DictReader((output / "history.csv").open()))
    assert len(rows) == 1 and float(rows[0]["epoch_seconds"]) > 0
    checkpoint = Path(training_cfg["output"]["checkpoints_dir"]) / "synthetic/best.pth"
    model = load_model(saved, checkpoint).eval()
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 1, 64, 64)


def test_early_stopping_preserves_best_checkpoint(training_cfg, monkeypatch):
    values = iter([0.7, 0.6, 0.5])
    training_cfg["train"].update(max_epochs=5, patience=2)

    def evaluate(*args, **kwargs):
        return dict(dice=next(values), iou=0.5, precision=0.5, recall=0.5,
                    small_recall=0.5, good_fpr=0.0, loss=0.4)

    monkeypatch.setattr("src.train.evaluate", evaluate)
    result = run_training(training_cfg, "early", limit_batches=1)
    assert result["stop_reason"] == "early_stopping" and result["epochs_completed"] == 3
    assert result["best_epoch"] == 1 and result["best_val_dice"] == 0.7
    root = Path(training_cfg["output"]["checkpoints_dir"]) / "early"
    assert torch.load(root / "best.pth", weights_only=True)["epoch"] == 1
    assert torch.load(root / "last.pth", weights_only=True)["epoch"] == 3
    with pytest.raises(FileExistsError):
        run_training(training_cfg, "early", limit_batches=1)


def test_time_budget_preserves_completed_epoch(training_cfg):
    training_cfg["loss"] = "dice"
    training_cfg["train"].update(max_epochs=3, max_seconds=0.001)
    result = run_training(training_cfg, "budget", limit_batches=1)
    assert result["stop_reason"] == "time_budget"
    assert result["epochs_completed"] == 1
    root = Path(training_cfg["output"]["checkpoints_dir"]) / "budget"
    assert (root / "best.pth").is_file()
