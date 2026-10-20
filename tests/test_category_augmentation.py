"""Category policy and exact compatibility with the pre-Stage-4 random sequence."""

from copy import deepcopy
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
import yaml

from src.data import MVTecSegDataset, make_loaders

CATEGORIES = ("tile", "wood", "hazelnut", "capsule", "metal_nut")


def raw_sample(good):
    marker = np.zeros((16, 16), dtype=np.uint8)
    marker[:6, :2] = 1
    marker[:2, :5] = 1
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = marker * 255  # RGB red marker even in good images.
    image[:, :, 1:] = 80
    return image, np.zeros_like(marker) if good else marker


@pytest.fixture
def category_cfg(tmp_path):
    root, splits = tmp_path / "data", tmp_path / "splits"
    splits.mkdir()
    paths = []
    for category in CATEGORIES:
        for kind in ("good", "crack"):
            relative = f"{category}/test/{kind}/000.png"
            image, mask = raw_sample(kind == "good")
            files = {root / relative: cv2.cvtColor(image, cv2.COLOR_RGB2BGR)}
            if kind != "good":
                files[root / category / "ground_truth" / kind / "000_mask.png"] = mask * 255
            for path, pixels in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".png", pixels)
                assert ok
                encoded.tofile(path)
            paths.append(relative)
    # Synthetic fixtures exercise each interface; real project manifests stay untouched.
    for split in ("train", "val", "test"):
        (splits / f"{split}.txt").write_text("\n".join(paths) + "\n")
    return {
        "seed": 42,
        "data": {"root": str(root), "splits_dir": str(splits),
                 "category_aware_augmentation": True},
        "train": {"input_size": 16, "eval_size": 32, "batch_size": 4, "num_workers": 0},
    }


@pytest.mark.parametrize("category", CATEGORIES)
@pytest.mark.parametrize("cache", [False, True])
def test_category_policy_for_good_and_defective_images(category_cfg, category, cache):
    category_cfg["data"]["cache_images"] = cache
    dataset = MVTecSegDataset("train", category_cfg, True)
    original_marker = raw_sample(False)[1].astype(bool)
    geometry_allowed = category in ("tile", "wood", "hazelnut")
    indices = [i for i, r in enumerate(dataset.records) if r["category"] == category]
    for index in indices:
        good = dataset.records[index]["defect_type"] == "good"
        moved, colour_changed = False, False
        for _ in range(12):
            image, mask = dataset[index]
            red = image[0].numpy() * 0.229 + 0.485
            marker = red > 0.5
            moved |= not np.array_equal(marker, original_marker)
            colour_changed |= not np.allclose(image[1].numpy(), (80 / 255 - 0.456) / 0.224)
            assert image.dtype == mask.dtype == torch.float32
            assert set(mask.unique().tolist()) <= {0.0, 1.0}
            if good:
                assert not mask.any()
            else:
                assert np.array_equal(marker, mask[0].numpy() > 0)
            if not geometry_allowed:
                assert np.array_equal(marker, original_marker)
        assert moved == geometry_allowed
        assert colour_changed  # Photometric augmentation remains active for every category.


@pytest.mark.parametrize("flag", [None, False])
def test_disabled_or_missing_flag_preserves_legacy_sequence(category_cfg, flag):
    if flag is None:
        category_cfg["data"].pop("category_aware_augmentation")
    else:
        category_cfg["data"]["category_aware_augmentation"] = flag
    dataset = MVTecSegDataset("train", category_cfg, True)
    # Frozen pre-#39 policy: compare full tensors over mixed categories and multiple passes.
    legacy = A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ], seed=42)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for _ in range(4):
        for index, record in enumerate(dataset.records):
            image, mask = raw_sample(record["defect_type"] == "good")
            expected = legacy(image=image, mask=mask)
            normalized = (expected["image"].astype(np.float32) / 255 - mean) / std
            actual_image, actual_mask = dataset[index]
            assert np.array_equal(actual_image.numpy(), normalized.transpose(2, 0, 1))
            assert np.array_equal(actual_mask.numpy(), expected["mask"][None])


def test_enabled_policy_cache_matches_uncached_and_is_not_mutated(category_cfg):
    plain = MVTecSegDataset("train", category_cfg, True)
    category_cfg["data"]["cache_images"] = True
    cached = MVTecSegDataset("train", category_cfg, True)
    originals = [(image.copy(), mask.copy()) for image, mask in cached.cache]
    for _ in range(3):
        for index in range(len(plain)):
            assert all(torch.equal(a, b) for a, b in zip(plain[index], cached[index]))
    for before, after in zip(originals, cached.cache):
        assert all(np.array_equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_flag_has_no_effect_in_evaluation_mode(category_cfg, split):
    enabled = MVTecSegDataset(split, category_cfg, False)
    category_cfg["data"]["category_aware_augmentation"] = False
    disabled = MVTecSegDataset(split, category_cfg, False)
    for index in range(len(enabled)):
        for _ in range(2):
            assert all(torch.equal(a, b) for a, b in zip(enabled[index], disabled[index]))


@pytest.mark.parametrize("workers", [0, 2])
def test_enabled_policy_reproducible_including_spawn_workers(category_cfg, workers):
    category_cfg["train"]["num_workers"] = workers
    category_cfg["data"]["cache_images"] = True
    first, second = make_loaders(category_cfg)[0], make_loaders(deepcopy(category_cfg))[0]
    if workers:
        first.multiprocessing_context = second.multiprocessing_context = "spawn"
    a, b = list(first), list(second)
    assert sum(len(images) for images, _ in a) == 10
    assert all(torch.equal(x, y) for left, right in zip(a, b) for x, y in zip(left, right))


@pytest.mark.parametrize("name", ["base.yaml", "cpu_light.yaml"])
def test_shared_configs_keep_stage_three_augmentation(name):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / name).read_text())
    assert cfg["data"].get("category_aware_augmentation", False) is False
