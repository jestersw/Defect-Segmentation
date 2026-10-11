import copy
import subprocess
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import RandomSampler, SequentialSampler

from src.data import MVTecSegDataset, make_loaders
from src.evaluate import full_report


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(path)


@pytest.fixture
def cfg(tmp_path):
    root, splits = tmp_path / "данные", tmp_path / "splits"
    splits.mkdir()
    # Red L-shaped corner makes geometric alignment observable; green identifies items.
    mask = np.zeros((16, 16), np.uint8)
    mask[:6, :2] = 17
    mask[:2, :5] = 17
    for split_index, split in enumerate(("train", "val", "test")):
        paths = []
        for index in range(9):
            category = "tile" if index % 2 == 0 else "wood"
            kind = "crack" if index % 3 else "good"
            relative = f"{category}/test/{kind}/{split_index * 10 + index:03}.png"
            image = np.zeros((16, 16, 3), np.uint8)
            image[:, :, 0] = 30  # BGR: distinguish RGB conversion from normalisation.
            image[:, :, 1] = index * 20 + 10
            if kind != "good":
                image[:, :, 2] = (mask > 0) * 255
                write_png(root / category / "ground_truth" / kind
                          / f"{Path(relative).stem}_mask.png", mask)
            write_png(root / relative, image)
            paths.append(relative)
        (splits / f"{split}.txt").write_text("\n".join(reversed(paths)) + "\n", encoding="utf-8")
    return {
        "seed": 42, "data": {"root": str(root), "splits_dir": str(splits)},
        "train": {"input_size": 16, "eval_size": 32, "batch_size": 4, "num_workers": 0},
        "eval": {"threshold": 0.5, "small_defect_area_px": 859.28,
                 "small_detect_coverage": 0.25, "good_fp_min_component_px": 20},
    }


@pytest.mark.parametrize("split,train", [("train", True), ("train", False),
                                         ("val", False), ("test", False)])
@pytest.mark.parametrize("size", [16, 32])
def test_shapes_types_binary_masks_and_good(cfg, split, train, size):
    cfg["train"]["input_size"] = size
    dataset = MVTecSegDataset(split, cfg, train)
    for index, record in enumerate(dataset.records):
        image, mask = dataset[index]
        assert image.shape == (3, size, size)
        target_size = size if train else 32
        assert mask.shape == (1, target_size, target_size)
        assert image.dtype == mask.dtype == torch.float32
        assert image.is_contiguous() and mask.is_contiguous()
        assert torch.isfinite(image).all()
        assert set(mask.unique().tolist()) <= {0.0, 1.0}
        assert bool(mask.any()) == (record["defect_type"] != "good")


def test_records_preserve_manifest_order(cfg):
    dataset = MVTecSegDataset("val", cfg, False)
    paths = (Path(cfg["data"]["splits_dir"]) / "val.txt").read_text().splitlines()
    assert [r["path"] for r in dataset.records] == paths
    assert len(dataset) == 9
    for record in dataset.records:
        assert set(record) == {"path", "category", "defect_type"}
        assert record["category"] == record["path"].split("/")[0]
        assert record["defect_type"] == record["path"].split("/")[2]


def test_rgb_normalization_and_eval_determinism(cfg):
    dataset = MVTecSegDataset("val", cfg, False)
    for index in range(len(dataset)):
        first, again = dataset[index], dataset[index]
        assert all(torch.equal(a, b) for a, b in zip(first, again))
    image, _ = dataset[0]
    assert image[0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229)
    assert image[2, 0, 0].item() == pytest.approx((30 / 255 - 0.406) / 0.225)


@pytest.mark.parametrize("transform", [A.HorizontalFlip(p=1), A.VerticalFlip(p=1),
                                       A.RandomRotate90(p=1)])
def test_geometric_augmentation_moves_image_and_mask_together(cfg, transform):
    dataset = MVTecSegDataset("train", cfg, True)
    dataset.transform = A.Compose([transform], seed=42)
    original = np.zeros((16, 16), dtype=bool)
    original[:6, :2] = True
    original[:2, :5] = True
    moved = False
    for _ in range(8):
        image, mask = dataset[0]
        red = image[0].numpy() * 0.229 + 0.485
        assert np.array_equal(red > 0.5, mask[0].numpy() > 0)
        moved |= not np.array_equal(mask[0].numpy() > 0, original)
    assert moved


def test_brightness_changes_only_image(cfg):
    dataset = MVTecSegDataset("train", cfg, True)
    dataset.transform = None
    before_image, before_mask = dataset[0]
    dataset.transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0, 0), p=1)
    ], seed=42)
    image, mask = dataset[0]
    assert torch.equal(mask, before_mask)
    assert not torch.equal(image, before_image)


def test_eval_resize_preserves_small_component(cfg):
    cfg["train"].update(input_size=8, eval_size=16)
    dataset = MVTecSegDataset("val", cfg, False)
    record = dataset.records[0]
    mask_path = (Path(cfg["data"]["root"]) / record["category"] / "ground_truth"
                 / record["defect_type"] / f"{Path(record['path']).stem}_mask.png")
    mask = np.zeros((16, 16), np.uint8)
    mask[3, 3] = 1
    write_png(mask_path, mask)
    image, target = dataset[0]
    assert image.shape == (3, 8, 8)
    assert target.sum() == 1 and target[0, 3, 3] == 1


def collect(loader):
    batches = list(loader)
    return tuple(torch.cat([batch[i] for batch in batches]) for i in (0, 1))


@pytest.mark.parametrize("workers", [0, 2])
def test_loaders_reproducible_with_seed_and_worker_count(cfg, workers):
    cfg["train"]["num_workers"] = workers
    first = make_loaders(cfg)[0]
    second = make_loaders(cfg)[0]
    if workers:
        # Exercise Windows-compatible pickling on Linux CI as well.
        first.multiprocessing_context = second.multiprocessing_context = "spawn"
    a, b = collect(first), collect(second)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert len(a[0]) == 9  # Last partial batch is not dropped.


def test_loader_order_and_train_rng_independent_of_validation(cfg):
    train, val, test = make_loaders(cfg)
    assert isinstance(train.sampler, RandomSampler)
    for loader in (val, test):
        assert isinstance(loader.sampler, SequentialSampler)
        images, masks = collect(loader)
        for index in range(len(loader.dataset)):
            expected_image, expected_mask = loader.dataset[index]
            assert torch.equal(images[index], expected_image)
            assert torch.equal(masks[index], expected_mask)
    fresh = make_loaders(cfg)[0]
    epoch_one = collect(train)
    assert all(torch.equal(x, y) for x, y in zip(epoch_one, collect(fresh)))
    epoch_two = collect(train)
    assert not torch.equal(epoch_one[0], epoch_two[0])
    other_cfg = copy.deepcopy(cfg)
    other_cfg["seed"] += 1
    assert not torch.equal(epoch_one[0], collect(make_loaders(other_cfg)[0])[0])


class AlwaysDefect(torch.nn.Module):
    def forward(self, images):
        return torch.full_like(images[:, :1], 10)


def test_evaluation_integration_with_different_resolutions(cfg):
    loader = make_loaders(cfg)[1]
    report = full_report(AlwaysDefect(), loader, cfg)
    assert report["images"] == 9 and report["defective_images"] == 6
    assert report["metrics"]["recall"] == 1
    assert report["metrics"]["good_fpr"] == 1
    assert {r["category"]: r["images"] for r in report["per_category"]} == {"tile": 5, "wood": 4}


@pytest.mark.parametrize("problem", ["missing", "corrupt", "mask_shape"])
def test_bad_files_fail_with_path(cfg, problem):
    dataset = MVTecSegDataset("val", cfg, False)
    record = dataset.records[0]
    path = Path(cfg["data"]["root"]) / record["path"]
    if problem == "missing":
        path.unlink()
    elif problem == "corrupt":
        path.write_bytes(b"not a png")
    else:
        path = (Path(cfg["data"]["root"]) / record["category"] / "ground_truth"
                / record["defect_type"] / f"{path.stem}_mask.png")
        write_png(path, np.ones((7, 7), np.uint8))
    with pytest.raises((FileNotFoundError, ValueError)) as error:
        dataset[0]
    assert path.name in str(error.value)


@pytest.mark.parametrize("content", ["", "../test/crack/a.png", "tile/test/crack/a.png\n" * 2])
def test_invalid_manifests_fail(cfg, content):
    (Path(cfg["data"]["splits_dir"]) / "val.txt").write_text(content)
    with pytest.raises(ValueError):
        MVTecSegDataset("val", cfg, False)


def test_show_batch_cli(cfg, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    output = tmp_path / "samples.png"
    script = Path(__file__).resolve().parents[1] / "scripts/show_batch.py"
    subprocess.run([sys.executable, str(script), "--config", str(config), "--output", str(output)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    image = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (900, 1800, 3)
