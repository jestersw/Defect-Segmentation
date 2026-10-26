
"""Tests for per-image segmentation metrics."""

import csv
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.per_image_metrics import (
    CSV_FIELDS,
    aggregate_per_image_rows,
    per_image_rows,
    write_csv,
)
from src.evaluate import evaluate, metric_settings, predict_probabilities


CFG = {
    "eval": {
        "threshold": 0.5,
        "small_defect_area_px": 50,
        "small_detect_coverage": 0.25,
        "good_fp_min_component_px": 20,
    }
}


class MaskEcho(torch.nn.Module):
    """Perfectly predicts the mask encoded in the first input channel."""

    def forward(self, x):
        return x[:, :1] * 20 - 10


class AlwaysDefect(torch.nn.Module):
    """Predicts a defect at every pixel."""

    def forward(self, x):
        return torch.full_like(x[:, :1], 10.0)


class SyntheticSet(Dataset):
    """Three defective images and one defect-free image."""

    def __init__(self):
        self.items = []
        self.records = []

        layouts = [
            ("tile", 16, 16),
            ("tile", 4, 4),
            ("wood", None, None),
            ("wood", 30, 8),
        ]

        for index, (category, offset, side) in enumerate(layouts):
            mask = torch.zeros(1, 64, 64)

            if offset is not None:
                mask[:, offset:offset + side, offset:offset + side] = 1

            image = mask.repeat(3, 1, 1)
            self.items.append((image, mask))
            self.records.append(
                {
                    "path": f"{category}/image_{index}.png",
                    "category": category,
                    "defect_type": "good" if offset is None else "scratch",
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_loader():
    return DataLoader(SyntheticSet(), batch_size=2, shuffle=False)


def get_per_image_rows(model):
    loader = make_loader()
    probabilities, targets = predict_probabilities(model, loader)

    return per_image_rows(
        records=loader.dataset.records,
        probabilities=probabilities,
        targets=targets,
        threshold=CFG["eval"]["threshold"],
        settings=metric_settings(CFG),
    )


def assert_metric_equal(actual, expected):
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected)


def test_aggregated_per_image_metrics_match_evaluate():
    """Aggregating CSV rows must reproduce evaluate() metrics."""
    rows = get_per_image_rows(MaskEcho())
    actual = aggregate_per_image_rows(rows)
    expected = evaluate(MaskEcho(), make_loader(), CFG)

    for key in (
        "dice",
        "iou",
        "precision",
        "recall",
        "small_recall",
        "good_fpr",
    ):
        assert_metric_equal(actual[key], expected[key])


def test_per_image_rows_contain_expected_fields():
    rows = get_per_image_rows(MaskEcho())

    assert len(rows) == 4
    assert set(rows[0]) == set(CSV_FIELDS)

    defective = next(row for row in rows if row["defect_area_px"] > 0)
    assert defective["dice"] == pytest.approx(1.0)
    assert defective["iou"] == pytest.approx(1.0)
    assert defective["tp"] == defective["defect_area_px"]
    assert defective["fp"] == 0
    assert defective["fn"] == 0


def test_good_image_has_empty_dice_and_iou():
    rows = get_per_image_rows(MaskEcho())
    good = next(row for row in rows if row["defect_area_px"] == 0)

    assert good["dice"] is None
    assert good["iou"] is None
    assert good["false_positive"] == 0


def test_small_defect_is_counted_and_detected():
    rows = get_per_image_rows(MaskEcho())
    small_defect = next(
        row for row in rows if row["defect_area_px"] == 16
    )

    assert small_defect["small_components_total"] == 1
    assert small_defect["small_components_detected"] == 1


def test_false_positive_is_detected_on_good_image():
    rows = get_per_image_rows(AlwaysDefect())
    good = next(row for row in rows if row["defect_area_px"] == 0)

    assert good["false_positive"] == 1

    # The false-positive flag is only defined for defect-free images.
    defective = next(row for row in rows if row["defect_area_px"] > 0)
    assert defective["false_positive"] is None


def test_csv_has_expected_header_and_all_rows(tmp_path):
    rows = get_per_image_rows(MaskEcho())
    output = tmp_path / "per_image_val.csv"

    write_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        saved_rows = list(reader)

    assert reader.fieldnames == CSV_FIELDS
    assert len(saved_rows) == 4
    assert saved_rows[0]["path"] == rows[0]["path"]
