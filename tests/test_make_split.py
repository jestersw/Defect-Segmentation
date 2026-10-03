import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.make_split import build_pool, split_pool, stratum, validate_splits


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    # Only filenames and file bytes are used by the split builder.
    for category in ("tile", "wood"):
        for split, kind, count in (("train", "good", 20), ("test", "good", 4),
                                   ("test", "crack", 8), ("test", "hole", 8)):
            for index in range(count):
                path = root / category / split / kind / f"{index:03}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.relative_to(root).as_posix().encode())
                if kind != "good":
                    mask = root / category / "ground_truth" / kind / f"{index:03}_mask.png"
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    mask.write_bytes(b"mask")
    return root


def test_pool_balances_categories_and_retains_all_official_test_images(dataset):
    pool, composition = build_pool(dataset, ["wood", "tile"], 42)
    assert len(pool) == len(set(pool)) == 64
    for category in ("tile", "wood"):
        selected = [p for p in pool if p.startswith(category + "/")]
        assert sum(stratum(p)[1] == "good" for p in selected) == 16
        assert sum(stratum(p)[1] != "good" for p in selected) == 16
        for path in (dataset / category / "test").glob("*/*.png"):
            assert path.relative_to(dataset).as_posix() in pool
    assert all(row["sampled_train_good"] == 12 for row in composition)
    assert build_pool(dataset, ["tile", "wood"], 42)[0] == pool


def test_stratification_preserves_groups_and_partitions_pool(dataset):
    pool, _ = build_pool(dataset, ["tile", "wood"], 42)
    splits = split_pool(pool, 42)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [44, 10, 10]
    for paths in splits.values():
        assert set(map(stratum, paths)) == set(map(stratum, pool))
    validate_splits(splits, pool, dataset)
    assert splits == split_pool(list(reversed(pool)), 42)


def test_insufficient_good_images_fail(dataset):
    for path in (dataset / "tile/train/good").glob("*.png"):
        path.unlink()
    with pytest.raises(ValueError, match="Cannot balance tile"):
        build_pool(dataset, ["tile"], 42)


def test_missing_mask_fails(dataset):
    (dataset / "tile/ground_truth/crack/000_mask.png").unlink()
    with pytest.raises(ValueError, match="Missing defect mask"):
        build_pool(dataset, ["tile"], 42)


def test_small_group_warning():
    pool = [f"tile/test/crack/{i:03}.png" for i in range(6)]
    pool += [f"tile/train/good/{i:03}.png" for i in range(50)]
    with pytest.warns(UserWarning, match="Small stratum.*6 images"):
        split_pool(pool, 42)


@pytest.mark.parametrize("ratios", [(0.7, 0.15), (0.7, 0.15, 0.2), (1, 0, 0)])
def test_invalid_ratios_fail(ratios):
    with pytest.raises(ValueError, match="Split ratios"):
        split_pool([], 42, ratios)


def test_path_and_content_leakage_fail(dataset):
    pool, _ = build_pool(dataset, ["tile", "wood"], 42)
    splits = split_pool(pool, 42)
    duplicate = {name: list(paths) for name, paths in splits.items()}
    duplicate["val"][0] = splits["train"][0]
    with pytest.raises(ValueError, match="Duplicate paths"):
        validate_splits(duplicate, pool, dataset)
    (dataset / splits["val"][0]).write_bytes((dataset / splits["train"][0]).read_bytes())
    with pytest.raises(ValueError, match="Identical image bytes across splits"):
        validate_splits(splits, pool, dataset)


def test_cli_reproducibility_and_overwrite_protection(dataset, tmp_path):
    config = {
        "seed": 42,
        "data": {"root": str(dataset), "categories": ["tile", "wood"],
                 "splits_dir": str(tmp_path / "splits"), "split_ratios": [0.7, 0.15, 0.15]},
        "output": {"results_dir": str(tmp_path / "results")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/make_split.py"
    command = [sys.executable, str(script), "--config", str(config_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    outputs = list((tmp_path / "splits").glob("*.txt"))
    outputs += list((tmp_path / "results").iterdir())
    before = {p: p.read_bytes() for p in outputs}
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert before == {p: p.read_bytes() for p in outputs}
    metadata = json.loads((tmp_path / "results/split_metadata.json").read_text())
    for name in ("train", "val", "test"):
        content = (tmp_path / "splits" / f"{name}.txt").read_bytes()
        assert b"\r" not in content
        assert metadata["split_sha256"][name] == hashlib.sha256(content).hexdigest()
    # A different seed must not silently replace the shared manifests or reports.
    config["seed"] = 7
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    attempt = subprocess.run(command, capture_output=True, text=True)
    assert attempt.returncode != 0
    assert "Refusing to change fixed split" in attempt.stderr
    assert before == {p: p.read_bytes() for p in outputs}


def test_committed_manifest_composition():
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "splits/train.txt").exists():
        pytest.skip("Project manifests have not been generated yet")
    manifests = {name: (repo / "splits" / f"{name}.txt").read_text().splitlines()
                 for name in ("train", "val", "test")}
    assert [len(paths) for paths in manifests.values()] == [582, 125, 125]
    paths = [p for split in manifests.values() for p in split]
    counts = Counter((stratum(p)[0], stratum(p)[1] == "good") for p in paths)
    for category, expected in {"tile": 84, "hazelnut": 70, "wood": 60,
                               "metal_nut": 93, "capsule": 109}.items():
        assert counts[category, True] == counts[category, False] == expected
