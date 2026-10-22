import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts import benchmark_inference as benchmark


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 1, kernel_size=1)
        self.calls = []

    def forward(self, image):
        self.calls.append({"training": self.training, "grad": torch.is_grad_enabled(),
                           "inference": torch.is_inference_mode_enabled(),
                           "threads": torch.get_num_threads(), "shape": tuple(image.shape)})
        return self.head(image)


@pytest.fixture
def benchmark_cfg(tmp_path):
    root, splits = tmp_path / "images", tmp_path / "splits"
    splits.mkdir()
    paths = []
    for index in range(55):
        relative = f"tile/test/good/{index:03}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (index, 80, 160)).save(path)
        paths.append(relative)
    # Unselected files are not read, even when the source run configuration enables caching.
    paths.append("tile/test/good/not_loaded.png")
    (splits / "val.txt").write_text("\n".join(paths) + "\n")
    # Deliberately no train/test manifests: only validation may be touched.
    return {
        "seed": 42,
        "data": {"root": str(root), "splits_dir": str(splits), "cache_images": True},
        "train": {"input_size": 256, "eval_size": 512},
        "model": {"arch": "unet", "encoder": "resnet18", "encoder_weights": "imagenet"},
    }


def test_prepares_only_selected_validation_images_at_512(benchmark_cfg):
    original = deepcopy(benchmark_cfg)
    images, paths = benchmark.prepare_images(benchmark_cfg, 50)
    assert benchmark_cfg == original
    assert len(images) == len(paths) == len(set(paths)) == 55
    assert all(image.shape == (1, 3, 512, 512) and image.dtype == torch.float32
               and image.device.type == "cpu" and image.is_contiguous() for image in images)
    assert images[0][0, 0, 0, 0].item() == pytest.approx((0 - 0.485) / 0.229)
    assert images[0][0, 2, 0, 0].item() == pytest.approx((160 / 255 - 0.406) / 0.225)
    assert paths[5] == "tile/test/good/005.png"


def test_tiny_model_warmup_threads_eval_mode_and_statistics(monkeypatch):
    model = TinyModel().train()
    image = torch.zeros(1, 3, 512, 512)
    # Fifty forward durations of 1..50 ms. Warm-up must not consult the clock.
    ticks = iter(value for milliseconds in range(1, 51) for value in (0, milliseconds * 1_000_000))
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(ticks))
    threads = torch.get_num_threads()
    result = benchmark.benchmark_model(model, [image] * 55)
    assert model.training and torch.get_num_threads() == threads
    assert len(model.calls) == 55
    assert all(c == {"training": False, "grad": False, "inference": True,
                     "threads": 8, "shape": (1, 3, 512, 512)} for c in model.calls)
    assert result["per_image_ms"] == list(range(1, 51))
    assert result["mean_ms"] == result["median_ms"] == 25.5
    assert result["p95_ms"] == pytest.approx(47.55)
    assert next(ticks, None) is None


def test_loads_tiny_checkpoint_and_writes_cli_json(benchmark_cfg, tmp_path, monkeypatch):
    import yaml

    import src.model

    original = TinyModel()
    checkpoint = tmp_path / "best.pth"
    torch.save({"model_state_dict": original.state_dict()}, checkpoint)
    loaded = []

    def build(cfg):
        assert cfg["model"]["encoder_weights"] is None  # Offline checkpoint loading.
        model = TinyModel()
        loaded.append(model)
        return model

    monkeypatch.setattr(src.model, "build_model", build)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(benchmark_cfg))
    output = tmp_path / "output/inference.json"
    monkeypatch.setattr(benchmark.sys, "argv", ["benchmark_inference.py", "--config", str(config),
                                              "--checkpoint", str(checkpoint),
                                              "--output", str(output)])
    # CLI normally runs in a fresh process; keep the test runner's global pools intact.
    interop = []
    monkeypatch.setattr(torch, "set_num_interop_threads", interop.append)
    threads = torch.get_num_threads()
    try:
        benchmark.main()
    finally:
        torch.set_num_threads(threads)
    report = json.loads(output.read_text())
    assert interop == [1]
    assert len(loaded) == 1 and len(loaded[0].calls) == 55
    assert all(torch.equal(weight, loaded[0].state_dict()[name])
               for name, weight in original.state_dict().items())
    assert report["checkpoint_sha256"] == benchmark.file_sha256(checkpoint)
    assert report["config_sha256"] == benchmark.file_sha256(config)
    assert report["val_manifest_sha256"] == benchmark.file_sha256(
        Path(benchmark_cfg["data"]["splits_dir"]) / "val.txt")
    assert report["source_config"] == benchmark_cfg
    assert report["protocol"]["split"] == "val"
    assert report["protocol"]["input_size"] == report["protocol"]["eval_size"] == 512
    assert report["protocol"]["batch_size"] == 1
    assert report["protocol"]["warmup_images"] == len(report["warmup_paths"]) == 5
    assert report["protocol"]["timed_images"] == len(report["timed_paths"]) == 50
    assert not set(report["warmup_paths"]) & set(report["timed_paths"])
    assert report["environment"]["cpu_threads"] == 8
    assert report["environment"]["cpu"] and report["environment"]["packages"]["torch"]
    assert len(report["per_image_ms"]) == 50
    assert min(report["per_image_ms"]) > 0
    assert report["mean_ms"] == pytest.approx(np.mean(report["per_image_ms"]))
    assert report["median_ms"] == pytest.approx(np.median(report["per_image_ms"]))
    assert report["p95_ms"] == pytest.approx(np.percentile(report["per_image_ms"], 95))


@pytest.mark.parametrize("number", [0, 49, 100])
def test_rejects_too_few_or_unavailable_images(benchmark_cfg, number):
    with pytest.raises(ValueError, match="At least 50|distinct validation"):
        benchmark.prepare_images(benchmark_cfg, number)


def test_rejects_wrong_evaluation_resolution(benchmark_cfg):
    benchmark_cfg["train"]["eval_size"] = 256
    with pytest.raises(ValueError, match="eval_size=512"):
        benchmark.prepare_images(benchmark_cfg, 50)


def test_restores_state_after_model_failure():
    class BrokenModel(torch.nn.Module):
        def forward(self, image):
            raise RuntimeError("broken forward")

    model = BrokenModel().train()
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="broken forward"):
        benchmark.benchmark_model(model, [torch.zeros(1, 3, 512, 512)] * 55)
    assert model.training and torch.get_num_threads() == threads


@pytest.mark.parametrize("shape", [(2, 3, 512, 512), (1, 3, 256, 256)])
def test_rejects_inputs_outside_protocol(shape):
    with pytest.raises(ValueError, match="512, 512"):
        benchmark.benchmark_model(TinyModel(), [torch.zeros(shape)] * 55)
