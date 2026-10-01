from pathlib import Path

import pytest

SPLITS = Path(__file__).resolve().parents[1] / "splits"
NAMES = ("train", "val", "test")


def read(name):
    path = SPLITS / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"{path.name} not created yet")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_splits_not_empty():
    for name in NAMES:
        assert read(name)


def test_no_duplicates_inside_split():
    for name in NAMES:
        items = read(name)
        assert len(items) == len(set(items))


def test_no_overlap_between_splits():
    sets = {name: set(read(name)) for name in NAMES}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
