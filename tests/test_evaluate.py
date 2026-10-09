import json

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.evaluate import evaluate, full_report, predict_probabilities, save_report

CFG = {
    "eval": {
        "threshold": 0.5,
        "threshold_sweep": [0.3, 0.5, 0.7],
        "small_defect_area_px": 50,
        "small_detect_coverage": 0.25,
        "good_fp_min_component_px": 20,
    }
}


class MaskEcho(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] * 20 - 10


class AlwaysDefect(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x[:, :1], 10.0)


class SyntheticSet(Dataset):
    def __init__(self, input_size=64, mask_size=64):
        self.items, self.records = [], []
        layouts = [("tile", 16, 16), ("tile", 4, 4), ("wood", None, None), ("wood", 30, 8)]
        for category, offset, side in layouts:
            small = torch.zeros(1, input_size, input_size)
            if offset is not None:
                small[:, offset:offset + side, offset:offset + side] = 1
            image = small.repeat(3, 1, 1)
            mask = F.interpolate(small[None], size=(mask_size, mask_size), mode="nearest")[0]
            self.items.append((image, mask))
            self.records.append(
                {"path": f"{category}/x.png", "category": category, "defect_type": "t"}
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def loader(**kwargs):
    return DataLoader(SyntheticSet(**kwargs), batch_size=2, shuffle=False)


def test_perfect_predictor_scores_one():
    result = evaluate(MaskEcho(), loader(), CFG)
    assert result["dice"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["small_recall"] == pytest.approx(1.0)
    assert result["good_fpr"] == pytest.approx(0.0)


def test_logits_are_upsampled_to_mask_size():
    probabilities, targets = predict_probabilities(MaskEcho(), loader(mask_size=128))
    assert probabilities[0].shape == (128, 128)
    assert targets[0].shape == (128, 128)
    result = evaluate(MaskEcho(), loader(mask_size=128), CFG)
    assert result["dice"] == pytest.approx(1.0)


def test_predicting_defects_everywhere_is_a_false_positive():
    result = evaluate(AlwaysDefect(), loader(), CFG)
    assert result["good_fpr"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] < 0.1


def test_threshold_override():
    strict = evaluate(MaskEcho(), loader(), CFG, threshold=0.999999)
    assert strict["recall"] == pytest.approx(0.0)


def test_evaluate_restores_training_mode():
    model = MaskEcho().train()
    evaluate(model, loader(), CFG)
    assert model.training


def test_full_report_and_saved_files(tmp_path):
    report = full_report(MaskEcho(), loader(), CFG)
    assert [row["category"] for row in report["per_category"]] == ["tile", "wood"]
    assert [row["threshold"] for row in report["threshold_sweep"]] == [0.3, 0.5, 0.7]
    assert report["images"] == 4 and report["defective_images"] == 3
    summary = save_report(report, tmp_path, "val")
    saved = json.loads((tmp_path / "metrics_val.json").read_text())
    assert saved == summary and saved["dice"] == pytest.approx(1.0)
    assert (tmp_path / "metrics_val_per_category.csv").read_text().startswith("category,")
    assert (tmp_path / "threshold_sweep_val.csv").read_text().startswith("threshold,")


def test_missing_threshold_in_config_fails():
    cfg = {"eval": {**CFG["eval"], "small_defect_area_px": None}}
    with pytest.raises(ValueError, match="small_defect_area_px"):
        evaluate(MaskEcho(), loader(), cfg)
