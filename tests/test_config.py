from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def load():
    return yaml.safe_load(CONFIG.read_text())


def test_required_sections():
    cfg = load()
    for key in ("seed", "data", "model", "loss", "train", "eval"):
        assert key in cfg


def test_values_are_valid():
    cfg = load()
    assert cfg["loss"] in {"bce", "dice_focal"}
    assert cfg["data"]["image_size"] in {256, 512}
    assert set(cfg["data"]["categories"]) == {"tile", "hazelnut", "wood", "metal_nut", "capsule"}
    assert 0.0 < cfg["eval"]["threshold"] < 1.0
    assert cfg["train"]["batch_size"] > 0
    assert cfg["train"]["lr"] > 0


def test_category_aware_run_changes_only_augmentation_flag():
    baseline = yaml.safe_load((CONFIG.parent / "cpu_light.yaml").read_text())
    category_aware = yaml.safe_load((CONFIG.parent / "cpu_light_cataug.yaml").read_text())
    assert category_aware["data"].pop("category_aware_augmentation") is True
    assert baseline["data"].pop("category_aware_augmentation", False) is False
    assert category_aware == baseline
