import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

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


def test_saved_reports_match_shared_manifests():
    manifests = {name: read(name) for name in NAMES}
    results = SPLITS.parent / "results"
    metadata = json.loads((results / "split_metadata.json").read_text())
    threshold = json.loads((results / "small_defect_threshold.json").read_text())
    config = yaml.safe_load((SPLITS.parent / "configs/base.yaml").read_text())
    for name, paths in manifests.items():
        digest = hashlib.sha256((SPLITS / f"{name}.txt").read_bytes()).hexdigest()
        assert digest == metadata["split_sha256"][name] == threshold["split_sha256"][name]
        assert len(paths) == metadata["split_sizes"][name]
    assert threshold["train_images"] == len(manifests["train"])
    assert threshold["small_defect_area_px"] == config["eval"]["small_defect_area_px"]
    assert threshold["percentile"] == config["eval"]["small_defect_percentile"]
    assert metadata["seed"] == config["seed"]
    assert metadata["ratios"] == config["data"]["split_ratios"]
    with (results / "split_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["category"] for row in rows} == set(config["data"]["categories"])
    for row in rows:
        for name, paths in manifests.items():
            types = [p.split("/")[2] for p in paths if p.split("/")[0] == row["category"]]
            assert int(row[f"{name}_good"]) == sum(kind == "good" for kind in types)
            assert int(row[f"{name}_defective"]) == sum(kind != "good" for kind in types)
