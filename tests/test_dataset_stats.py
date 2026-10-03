import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.dataset_stats import (
    component_areas,
    inspect_dataset,
    read_splits,
    summarize,
    train_percentile,
)
from scripts.download_data import archive_ready, extract_categories, selected_path


def write_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[5, 5] = 255
    for split, kind, value in (("train", "good", 10), ("test", "good", 20),
                               ("test", "crack", 30)):
        write_png(root / "tile" / split / kind / "000.png",
                  np.full((8, 8, 3), value, dtype=np.uint8))
    write_png(root / "tile/ground_truth/crack/000_mask.png", mask)
    return root


def test_resize_uses_nearest_and_diagonal_connectivity():
    mask = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    assert component_areas(mask, 4).tolist() == [8]
    assert component_areas(np.zeros((2, 2), dtype=np.uint8), 4).tolist() == []


def test_inventory_counts_components_and_area_percent(dataset):
    records, areas, counts, differences = inspect_dataset(dataset, ["tile"], 8)
    assert [counts[0][k] for k in ("train_good", "test_good", "test_defective", "masks")] == [
        1, 1, 1, 1,
    ]
    assert len(differences) == 4
    row = next(r for r in summarize(records, areas, 8) if r["defect_type"] == "crack")
    assert row["component_count"] == 2
    assert row["area_min_px"] == 1
    assert row["area_median_px"] == 2.5
    assert row["area_max_pct"] == 6.25


def test_missing_and_orphan_masks_fail(dataset):
    mask = dataset / "tile/ground_truth/crack/000_mask.png"
    mask.rename(mask.with_name("001_mask.png"))
    with pytest.raises(ValueError, match="Missing mask"):
        inspect_dataset(dataset, ["tile"], 8)
    mask.with_name("001_mask.png").rename(mask)
    write_png(mask.with_name("orphan_mask.png"), np.ones((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="Orphan masks"):
        inspect_dataset(dataset, ["tile"], 8)


def test_empty_and_mismatched_masks_fail(dataset):
    mask = dataset / "tile/ground_truth/crack/000_mask.png"
    write_png(mask, np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="Empty defect mask"):
        inspect_dataset(dataset, ["tile"], 8)
    write_png(mask, np.ones((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape mismatch"):
        inspect_dataset(dataset, ["tile"], 8)


def test_defects_lost_during_resize_are_reported(dataset):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 255
    write_png(dataset / "tile/ground_truth/crack/000_mask.png", mask)
    records, areas, _, _ = inspect_dataset(dataset, ["tile"], 2)
    row = next(r for r in summarize(records, areas, 2) if r["defect_type"] == "crack")
    assert row["masks_vanished_after_resize"] == 1
    assert row["component_count"] == 0
    assert row["area_min_px"] is None


def test_train_percentile_excludes_held_out_masks():
    areas = {"train_defect": np.array([1, 4]), "train_good": np.array([]),
             "held_out": np.array([100_000])}
    value, count = train_percentile(areas, ["train_defect", "train_good"], 33)
    assert value == pytest.approx(1.99)
    assert count == 2
    with pytest.raises(ValueError, match="no defect components"):
        train_percentile(areas, ["train_good"], 33)


def make_splits(directory):
    directory.mkdir()
    paths = {"train": "tile/test/crack/000.png", "val": "tile/train/good/000.png",
             "test": "tile/test/good/000.png"}
    for name, path in paths.items():
        (directory / f"{name}.txt").write_text(path + "\n", encoding="utf-8")
    return paths


def test_split_path_and_content_leakage(dataset, tmp_path):
    directory = tmp_path / "splits"
    paths = make_splits(directory)
    read_splits(directory, set(paths.values()), dataset)
    (directory / "val.txt").write_text(paths["train"] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate split image"):
        read_splits(directory, set(paths.values()), dataset)
    (directory / "val.txt").write_text(paths["val"] + "\n", encoding="utf-8")
    (dataset / paths["val"]).write_bytes((dataset / paths["train"]).read_bytes())
    with pytest.raises(ValueError, match="Identical image bytes"):
        read_splits(directory, set(paths.values()), dataset)


def test_cli_produces_reports_and_train_threshold(dataset, tmp_path):
    config = {
        "data": {"root": str(dataset), "categories": ["tile"], "image_size": 512},
        "output": {"results_dir": str(tmp_path / "results")},
        "eval": {"small_defect_percentile": 33},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    splits = tmp_path / "splits"
    make_splits(splits)
    script = Path(__file__).resolve().parents[1] / "scripts/dataset_stats.py"
    command = [sys.executable, str(script),
               "--config", str(config_path), "--splits-dir", str(splits)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    output = tmp_path / "results"
    threshold = json.loads((output / "small_defect_threshold.json").read_text())
    assert threshold["small_defect_area_px"] == pytest.approx(8151.04)
    assert threshold["train_components"] == 2
    assert set(threshold["split_sha256"]) == {"train", "val", "test"}
    with (output / "dataset_stats.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert (output / "defect_area_hist.png").read_bytes().startswith(b"\x89PNG")
    report = json.loads((output / "dataset_verification.json").read_text())
    assert len(report["count_differences"]) == 4


@pytest.mark.parametrize("name", ["../tile/test/crack/0.png", "/tile/test/crack/0.png",
                                 "C:/tile/test/crack/0.png", "tile\\test\\crack\\0.png"])
def test_archive_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        selected_path(name, ["tile"])


def test_archive_extracts_only_selected_categories(tmp_path):
    from zipfile import ZipFile

    archive = tmp_path / "input.zip"
    with ZipFile(archive, "w") as stream:
        for category in ("tile", "bottle"):
            for split in ("train", "test", "ground_truth"):
                stream.writestr(f"wrapper/{category}/{split}/good/000.png", b"example")
    root = tmp_path / "out"
    assert extract_categories(archive, root, ["tile"]) == 3
    assert (root / "tile/train/good/000.png").read_bytes() == b"example"
    assert not (root / "bottle").exists()


def test_archive_rejects_ambiguous_layout_before_writing(tmp_path):
    from zipfile import ZipFile

    archive = tmp_path / "input.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("first/tile/train/good/000.png", b"first")
        stream.writestr("second/tile/train/good/000.png", b"second")
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="Duplicate destination"):
        extract_categories(archive, root, ["tile"])
    assert not root.exists()


def test_interrupted_archive_is_not_reused(tmp_path):
    from zipfile import ZipFile

    archive = tmp_path / "mvtec-ad.zip"
    assert not archive_ready(archive)
    archive.write_bytes(b"partial download")
    assert not archive_ready(archive)
    with ZipFile(archive, "w") as stream:
        stream.writestr("example.txt", "complete")
    assert archive_ready(archive)
    Path(str(archive) + ".kaggle-partial").write_text("{}", encoding="utf-8")
    assert not archive_ready(archive)
