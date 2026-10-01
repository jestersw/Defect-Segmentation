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
