import json
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from scripts import demo
from src.data import MVTecSegDataset
from src.evaluate import predict_probabilities


class PatternModel(torch.nn.Module):
    def __init__(self, size=8):
        super().__init__()
        pattern = torch.tensor([[[[-4.0, 1.0], [-1.0, 4.0]]]])
        self.register_buffer("pattern", F.interpolate(pattern, size=(size, size), mode="nearest"))
        self.calls = 0

    def forward(self, image):
        assert image.device.type == "cpu" and image.dtype == torch.float32
        assert not self.training and not torch.is_grad_enabled()
        self.calls += 1
        return self.pattern


@pytest.fixture
def demo_cfg():
    # Inference must not need dataset paths, splits, ground truth or small-defect statistics.
    return {
        "model": {"encoder_weights": "imagenet"},
        "train": {"input_size": 8, "eval_size": 16, "cpu_threads": 2},
        "eval": {"threshold": 0.7, "good_fp_min_component_px": 20},
    }


def test_preprocessing_matches_evaluation(training_cfg):
    path = Path(training_cfg["data"]["root"]) / "tile/test/crack/val.png"
    original, actual = demo.prepare_image(path, training_cfg["train"]["input_size"])
    dataset = MVTecSegDataset("val", training_cfg, train=False)
    expected, _ = dataset[1]
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(original, np.asarray(Image.open(path)))
    assert actual.is_contiguous() and actual.dtype == torch.float32


@pytest.mark.parametrize("mode,suffix", [("RGB", ".jpg"), ("L", ".bmp"), ("RGBA", ".png")])
def test_arbitrary_image_formats_and_unicode_paths(tmp_path, mode, suffix):
    path = tmp_path / f"изображение с пробелами{suffix}"
    Image.new(mode, (17, 9), color=100).save(path)
    original, tensor = demo.prepare_image(path, 8)
    assert original.shape == (9, 17, 3) and tensor.shape == (1, 3, 8, 8)
    assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("eval_size", [8, 16])
def test_inference_matches_evaluation_and_restores_mode(eval_size):
    model = PatternModel().train()
    image = torch.zeros(1, 3, 8, 8)
    actual, elapsed = demo.predict_image(model, image, eval_size)
    assert model.training and elapsed >= 0
    loader = DataLoader(TensorDataset(image, torch.zeros(1, 1, eval_size, eval_size)))
    expected, _ = predict_probabilities(model, loader)
    np.testing.assert_array_equal(actual, expected[0])
    if eval_size != 8:
        wrong = F.interpolate(torch.sigmoid(model.pattern), size=(eval_size, eval_size),
                              mode="bilinear", align_corners=False)[0, 0].numpy()
        assert not np.array_equal(actual >= 0.5, wrong >= 0.5)


def test_timer_covers_forward_resize_and_sigmoid(monkeypatch):
    events = []
    ticks = iter([1_000_000, 4_250_000])
    model = PatternModel()
    original_forward, original_resize = model.forward, F.interpolate
    original_sigmoid = torch.sigmoid

    def clock():
        events.append("clock")
        return next(ticks)

    def forward(image):
        events.append("forward")
        return original_forward(image)

    def resize(logits, **kwargs):
        events.append("resize")
        assert kwargs == {"size": (16, 16), "mode": "bilinear", "align_corners": False}
        return original_resize(logits, **kwargs)

    def sigmoid(logits):
        events.append("sigmoid")
        return original_sigmoid(logits)

    monkeypatch.setattr(demo.time, "perf_counter_ns", clock)
    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(demo.F, "interpolate", resize)
    monkeypatch.setattr(demo.torch, "sigmoid", sigmoid)
    _, elapsed = demo.predict_image(model, torch.zeros(1, 3, 8, 8), 16)
    assert elapsed == 3.25 and model.calls == 1
    assert events == ["clock", "forward", "resize", "sigmoid", "clock"]


def test_summary_uses_component_size_not_total_area_and_includes_threshold_equality():
    probabilities = np.zeros((12, 12), dtype=np.float32)
    probabilities[:2, :5] = 0.5
    probabilities[7:9, :5] = 0.5
    mask, report = demo.prediction_summary(probabilities, 0.5, 20)
    assert mask.sum() == report["predicted_defect_area_px"] == 20
    assert report["max_probability"] == 0.5 and not report["has_defect"]
    probabilities[2:4, :5] = 0.5
    assert demo.prediction_summary(probabilities, 0.5, 20)[1]["has_defect"]
    assert demo.prediction_summary(np.eye(3), 0.5, 3)[1]["has_defect"]  # 8-connected
    assert not demo.prediction_summary(np.zeros((3, 3)), 0.5, 1)[1]["has_defect"]


@pytest.mark.parametrize("threshold", [None, 0.9])
def test_cli_loads_weights_once_and_writes_figures_and_json(demo_cfg, tmp_path,
                                                          monkeypatch, threshold):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(demo_cfg), encoding="utf-8")
    checkpoint = tmp_path / "weights.pth"
    model = torch.nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(2.0)
    torch.save(model.state_dict(), checkpoint)
    builds = []

    def build(cfg):
        assert cfg["model"]["encoder_weights"] is None
        builds.append(cfg)
        return torch.nn.Conv2d(3, 1, 1)

    monkeypatch.setattr("src.model.build_model", build)
    inputs = []
    for folder in ("first", "second"):
        path = tmp_path / folder / "same.jpg"
        path.parent.mkdir()
        Image.new("RGB", (27, 19), (180, 100, 20)).save(path)
        inputs.append(path)
    output = tmp_path / "outputs"
    figures = []
    original_save = plt.Figure.savefig

    def save(fig, *args, **kwargs):
        figures.append(fig)
        return original_save(fig, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", save)
    args = ["--config", str(config), "--checkpoint", str(checkpoint),
            "--images", str(inputs[0]), "--images", str(inputs[1]), "--output-dir", str(output)]
    if threshold is not None:
        args += ["--threshold", str(threshold)]
    previous_threads = torch.get_num_threads()
    demo.main(args)
    assert torch.get_num_threads() == previous_threads and len(builds) == 1
    assert yaml.safe_load(config.read_text()) == demo_cfg
    for index, path in enumerate(inputs, start=1):
        report = json.loads((output / f"{index:03d}_same.json").read_text())
        assert report["image"] == str(path.resolve())
        assert report["original_size"] == {"height": 19, "width": 27}
        assert report["input_size"] == 8 and report["eval_size"] == 16
        assert report["threshold"] == (0.7 if threshold is None else threshold)
        assert report["max_probability"] == pytest.approx(torch.sigmoid(torch.tensor(2.)).item())
        assert report["predicted_defect_area_px"] == (256 if threshold is None else 0)
        assert report["has_defect"] == (threshold is None)
        assert report["inference_time_ms"] >= 0 and report["device"] == "cpu"
        assert report["cpu_threads"] == 2 and "No warm-up" in report["timing_scope"]
        with Image.open(output / f"{index:03d}_same.png") as image:
            assert image.width > image.height and image.width > 1000
        fig = figures[index - 1]
        assert not plt.fignum_exists(fig.number)
        assert len(fig.axes) == 4  # Three panels plus probability colorbar.
        assert fig.axes[1].images[0].get_clim() == (0, 1)
        alpha = fig.axes[2].images[1].get_array()[..., 3]
        assert (alpha > 0).sum() == report["predicted_defect_area_px"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_threshold_fails_before_loading(demo_cfg, tmp_path, value):
    with pytest.raises(ValueError, match="threshold"):
        demo.run_demo(demo_cfg, tmp_path / "missing.pth", [], tmp_path, value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_logits_fail_and_restore_model_mode(value):
    model = PatternModel().train()
    model.pattern.fill_(value)
    with pytest.raises(ValueError, match="finite CPU logits"):
        demo.predict_image(model, torch.zeros(1, 3, 8, 8), 16)
    assert model.training


def test_corrupt_image_reports_error_and_restores_threads(demo_cfg, tmp_path, monkeypatch):
    checkpoint, image = tmp_path / "model.pth", tmp_path / "broken.png"
    checkpoint.touch()
    image.write_text("not an image")
    monkeypatch.setattr(demo, "load_model", lambda *args: PatternModel())
    previous_threads = torch.get_num_threads()
    with pytest.raises((ValueError, cv2.error), match="Cannot read image|imdecode"):
        demo.run_demo(demo_cfg, checkpoint, [image], tmp_path / "output")
    assert torch.get_num_threads() == previous_threads


def test_output_cannot_overwrite_an_input(demo_cfg, tmp_path):
    first, second, checkpoint = [tmp_path / name for name in ("x.png", "001_x.png", "model.pth")]
    for path in (first, second, checkpoint):
        path.touch()
    original = deepcopy(demo_cfg)
    with pytest.raises(ValueError, match="overwrite"):
        demo.run_demo(demo_cfg, checkpoint, [first, second], tmp_path)
    assert demo_cfg == original
