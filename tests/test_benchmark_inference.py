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


@pytest.mark.parametrize("saved_size,override,expected_size", [
    (256, None, 256), (512, None, 512), (256, 128, 128),
])
def test_prepares_validation_images_at_selected_size(
        benchmark_cfg, saved_size, override, expected_size):
    benchmark_cfg["train"]["input_size"] = saved_size
    original = deepcopy(benchmark_cfg)
    images, paths = benchmark.prepare_images(benchmark_cfg, 50, input_size=override)
    assert benchmark_cfg == original
    assert len(images) == len(paths) == len(set(paths)) == 55
    assert all(image.shape == (1, 3, expected_size, expected_size) and image.dtype == torch.float32
               and image.device.type == "cpu" and image.is_contiguous() for image in images)
    assert images[0][0, 0, 0, 0].item() == pytest.approx((0 - 0.485) / 0.229)
    assert images[0][0, 2, 0, 0].item() == pytest.approx((160 / 255 - 0.406) / 0.225)
    assert paths[5] == "tile/test/good/005.png"


@pytest.mark.parametrize("input_size", [256, 512])
def test_timing_includes_logit_upsampling_but_excludes_warmup(monkeypatch, input_size):
    model = TinyModel().train()
    image = torch.zeros(1, 3, input_size, input_size)
    # Fifty pipeline durations of 1..50 ms. Verify both operations lie between clock calls.
    ticks = iter(value for milliseconds in range(1, 51) for value in (0, milliseconds * 1_000_000))
    events = []
    original_forward, original_interpolate = model.forward, benchmark.F.interpolate

    def forward(image):
        events.append("forward")
        return original_forward(image)

    def interpolate(logits, *, size, mode, align_corners):
        events.append("resize")
        assert logits.shape[-2:] == (input_size, input_size)
        assert size == (512, 512) and mode == "bilinear" and align_corners is False
        return original_interpolate(logits, size=size, mode=mode, align_corners=align_corners)

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(benchmark.F, "interpolate", interpolate)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", clock)
    threads = torch.get_num_threads()
    result = benchmark.benchmark_model(model, [image] * 55, eval_size=512)
    assert model.training and torch.get_num_threads() == threads
    assert len(model.calls) == 55
    assert all(c == {"training": False, "grad": False, "inference": True,
                     "threads": 8, "shape": (1, 3, input_size, input_size)} for c in model.calls)
    operations = ["forward", "resize"] if input_size != 512 else ["forward"]
    assert events == operations * 5 + (["clock"] + operations + ["clock"]) * 50
    assert result["per_image_ms"] == list(range(1, 51))
    assert result["mean_ms"] == result["median_ms"] == 25.5
    assert result["p95_ms"] == pytest.approx(47.55)
    assert next(ticks, None) is None


@pytest.mark.parametrize("saved_size,override,eval_size", [
    (256, None, 512), (512, None, 512), (256, 128, 512), (256, None, 256),
])
def test_loads_tiny_checkpoint_and_writes_cli_json(
        benchmark_cfg, tmp_path, monkeypatch, saved_size, override, eval_size):
    import yaml

    import src.model

    benchmark_cfg["train"].update(input_size=saved_size, eval_size=eval_size)
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
    arguments = ["benchmark_inference.py", "--config", str(config),
                 "--checkpoint", str(checkpoint), "--output", str(output)]
    if override is not None:
        arguments.extend(["--input-size", str(override)])
    monkeypatch.setattr(benchmark.sys, "argv", arguments)
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
    expected_size = saved_size if override is None else override
    assert all(call["shape"] == (1, 3, expected_size, expected_size) for call in loaded[0].calls)
    assert all(torch.equal(weight, loaded[0].state_dict()[name])
               for name, weight in original.state_dict().items())
    assert report["checkpoint_sha256"] == benchmark.file_sha256(checkpoint)
    assert report["config_sha256"] == benchmark.file_sha256(config)
    assert report["val_manifest_sha256"] == benchmark.file_sha256(
        Path(benchmark_cfg["data"]["splits_dir"]) / "val.txt")
    assert report["source_config"] == benchmark_cfg
    assert report["protocol"]["split"] == "val"
    assert report["protocol"]["input_size"] == expected_size
    assert report["protocol"]["eval_size"] == eval_size
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


@pytest.mark.parametrize("name", ["input_size", "eval_size"])
def test_rejects_invalid_resolution(benchmark_cfg, name):
    benchmark_cfg["train"][name] = 0
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        benchmark.prepare_images(benchmark_cfg, 50)


def test_restores_state_after_model_failure():
    class BrokenModel(torch.nn.Module):
        def forward(self, image):
            raise RuntimeError("broken forward")

    model = BrokenModel().train()
    threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="broken forward"):
        benchmark.benchmark_model(model, [torch.zeros(1, 3, 256, 256)] * 55, eval_size=512)
    assert model.training and torch.get_num_threads() == threads


@pytest.mark.parametrize("shape", [(2, 3, 512, 512), (1, 3, 256, 128)])
def test_rejects_inputs_outside_protocol(shape):
    with pytest.raises(ValueError, match="1, 3, s, s"):
        benchmark.benchmark_model(TinyModel(), [torch.zeros(shape)] * 55, eval_size=512)
