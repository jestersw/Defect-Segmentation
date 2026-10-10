from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


@pytest.fixture
def training_cfg(tmp_path):
    torch.set_num_threads(2)
    config = Path(__file__).resolve().parents[1] / "configs/cpu_light.yaml"
    cfg = yaml.safe_load(config.read_text())
    root, splits = tmp_path / "images", tmp_path / "splits"
    splits.mkdir()
    for split in ("train", "val", "test"):
        paths = []
        for kind in ("good", "crack"):
            path = f"tile/test/{kind}/{split}.png"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            if kind == "crack":
                image[4:20, 4:20] = 255
                mask[4:20, 4:20] = 255
                target = root / "tile/ground_truth/crack" / f"{split}_mask.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(target)
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(target)
            paths.append(path)
        (splits / f"{split}.txt").write_text("\n".join(paths) + "\n")
    cfg["data"].update(root=str(root), splits_dir=str(splits), categories=["tile"])
    cfg["model"]["encoder_weights"] = None
    cfg["train"].update(input_size=64, eval_size=128, batch_size=2, cpu_threads=2,
                        max_epochs=1, num_workers=0)
    cfg["output"] = {"results_dir": str(tmp_path / "results"),
                     "checkpoints_dir": str(tmp_path / "checkpoints")}
    return cfg
