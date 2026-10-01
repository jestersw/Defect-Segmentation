import numpy as np
import pytest

from src.metrics import (
    aggregate,
    dice_score,
    iou_score,
    is_false_positive,
    pixel_counts,
    small_defect_hits,
)


def square(size, top, left, side):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:top + side, left:left + side] = 1
    return mask


def test_perfect_prediction():
    target = square(32, 4, 4, 8)
    assert dice_score(target, target) == pytest.approx(1.0)
    assert iou_score(target, target) == pytest.approx(1.0)


def test_no_overlap():
    pred = square(32, 0, 0, 4)
    target = square(32, 20, 20, 4)
    assert dice_score(pred, target) == pytest.approx(0.0, abs=1e-6)
    assert iou_score(pred, target) == pytest.approx(0.0, abs=1e-6)


def test_half_overlap():
    target = np.zeros((10, 10), dtype=np.uint8)
    target[:, :4] = 1
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[:, 2:6] = 1
    assert dice_score(pred, target) == pytest.approx(0.5)
    assert iou_score(pred, target) == pytest.approx(1 / 3)


def test_pixel_counts():
    target = square(10, 0, 0, 4)
    pred = square(10, 2, 2, 4)
    assert pixel_counts(pred, target) == (4, 12, 12)


def test_small_defect_detected_and_missed():
    target = square(64, 5, 5, 3) + square(64, 40, 40, 3) + square(64, 15, 15, 15)
    pred = square(64, 5, 5, 3)
    assert small_defect_hits(pred, target, max_area=50) == (1, 2)


def test_small_defect_coverage_threshold():
    target = square(32, 0, 0, 4)
    pred = np.zeros_like(target)
    pred[0, :3] = 1
    assert small_defect_hits(pred, target, max_area=50, coverage=0.25) == (0, 1)
    pred[1, :1] = 1
    assert small_defect_hits(pred, target, max_area=50, coverage=0.25) == (1, 1)


def test_false_positive_on_good_image():
    assert not is_false_positive(np.zeros((32, 32)))
    assert not is_false_positive(square(32, 0, 0, 3), min_component=20)
    assert is_false_positive(square(32, 0, 0, 5), min_component=20)


def test_aggregate_keys_and_values():
    defect = square(32, 4, 4, 4)
    good = np.zeros((32, 32), dtype=np.uint8)
    result = aggregate([defect, good], [defect, good], max_area=50)
    assert set(result) == {"dice", "iou", "precision", "recall", "small_recall", "good_fpr"}
    assert result["dice"] == pytest.approx(1.0)
    assert result["small_recall"] == pytest.approx(1.0)
    assert result["good_fpr"] == pytest.approx(0.0)
