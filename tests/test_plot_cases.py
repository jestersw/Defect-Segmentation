from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from scripts import plot_cases as script
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.metrics import dice_score


class PatternModel(torch.nn.Module):
    """Asymmetric logits make sigmoid-before-resizing produce a different mask."""

    def __init__(self, sign=1):
        super().__init__()
        self.register_buffer("logits", sign * torch.tensor([[[[-4.0, 1.0], [-1.0, 4.0]]]]))
        self.inputs = []

    def forward(self, images):
        assert not self.training and not torch.is_grad_enabled()
        self.inputs.append(images.clone())
        return self.logits.expand(len(images), -1, -1, -1)


def selected_paths(split):
    # Deliberately reverse manifest order.
    return [f"tile/test/crack/{split}.png", f"tile/test/good/{split}.png"]


@pytest.mark.parametrize("split", ["val", "test"])
def test_prepare_reads_only_selected_split_and_paths(training_cfg, split):
    cfg = training_cfg
    cfg["data"]["cache_images"] = True
    original = deepcopy(cfg)
    root, manifests = Path(cfg["data"]["root"]), Path(cfg["data"]["splits_dir"])
    # Other manifests and unselected images must never be read, even with caching on.
    for other in {"train", "val", "test"} - {split}:
        (manifests / f"{other}.txt").unlink()
    (root / f"tile/test/good/{split}.png").unlink()
    path = selected_paths(split)[0]
    loader, rgb = script.prepare_cases(cfg, split, [path])
    images, masks = next(iter(loader))
    assert len(loader.dataset) == 1
    assert images.shape == (1, 3, 64, 64) and masks.shape == (1, 1, 128, 128)
    assert rgb[0].shape == (128, 128, 3)
    assert masks.sum() == 32 * 32
    np.testing.assert_allclose(images[0, :, 0, 0], -IMAGENET_MEAN / IMAGENET_STD)
    assert cfg == original


def test_display_preserves_details_above_input_resolution(training_cfg):
    cfg = training_cfg
    cfg["train"].update(input_size=8, eval_size=64)
    path = selected_paths("val")[0]
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    original[1, 1] = (255, 0, 0)
    Image.fromarray(original).save(Path(cfg["data"]["root"]) / path)
    loader, rgb = script.prepare_cases(cfg, "val", [path])
    assert next(iter(loader))[0].shape[-2:] == (8, 8)
    np.testing.assert_array_equal(rgb[0], original)


@pytest.mark.parametrize("paths,split,message", [
    ([], "val", "at least one"),
    (["tile/test/crack/val.png"] * 2, "val", "Duplicate"),
    (["tile/test/crack/test.png"], "val", "not in the val manifest"),
    (["../tile/test/crack/val.png"], "val", "not in the val manifest"),
    (["tile/test/crack/train.png"], "train", "val or test"),
])
def test_invalid_path_selection_fails_before_image_read(training_cfg, monkeypatch,
                                                       paths, split, message):
    def unexpected_read(*args):
        pytest.fail("Invalid selections must not read images")

    monkeypatch.setattr("src.data.read_image", unexpected_read)
    with pytest.raises(ValueError, match=message):
        script.prepare_cases(training_cfg, split, paths)


@pytest.mark.parametrize("split,model_count", [("val", 2), ("test", 1)])
def test_predictions_order_overlays_and_dice(training_cfg, tmp_path, monkeypatch,
                                           split, model_count):
    cfg = training_cfg
    cfg["eval"]["threshold"] = 0.7
    models = [PatternModel(1), PatternModel(-1)][:model_count]
    checkpoints = [tmp_path / f"model_{i}.pth" for i in range(model_count)]
    for checkpoint in checkpoints:
        checkpoint.touch()
    labels = ["A: BCE", "B: Dice + Focal"][:model_count]
    loaded = []

    def load_model(config, checkpoint):
        assert config == cfg
        loaded.append(checkpoint)
        return models[checkpoints.index(checkpoint)]

    monkeypatch.setattr(script, "load_model", load_model)
    figures = []
    original_figure = script.case_figure

    def capture_figure(*args):
        fig = original_figure(*args)
        figures.append(fig)
        return fig

    monkeypatch.setattr(script, "case_figure", capture_figure)
    paths = selected_paths(split)
    output = script.plot_cases(cfg, checkpoints, labels, split, paths,
                               tmp_path / "nested/cases.png")
    assert loaded == checkpoints
    with Image.open(output) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width > rendered.height and rendered.height > 500
    fig = figures[0]
    assert not plt.fignum_exists(fig.number)
    axes = np.array(fig.axes).reshape(2, 2 + model_count)
    assert "0.7" in fig._suptitle.get_text() and split in fig._suptitle.get_text()
    for row, path in enumerate(paths):
        assert axes[row, 0].get_title().replace("\n", "") == path
        target = np.zeros((128, 128), dtype=bool)
        if row == 0:
            target[8:40, 8:40] = True
        np.testing.assert_array_equal(axes[row, 1].images[1].get_array()[..., 3] > 0, target)
        for index, model in enumerate(models):
            expected = torch.sigmoid(F.interpolate(model.logits, size=(128, 128),
                                                   mode="bilinear", align_corners=False))
            prediction = expected[0, 0].numpy() >= 0.7
            wrong = F.interpolate(torch.sigmoid(model.logits), size=(128, 128),
                                  mode="bilinear", align_corners=False)[0, 0].numpy() >= 0.7
            assert not np.array_equal(prediction, wrong)
            ax = axes[row, index + 2]
            np.testing.assert_array_equal(ax.images[1].get_array()[..., 3] > 0, prediction)
            assert labels[index] in ax.get_title()
            expected_dice = f"{dice_score(prediction, target):.3f}" if row == 0 else "N/A (good)"
            assert f"Dice = {expected_dice}" in ax.get_title()
    for model in models:
        assert model.training  # Evaluation restores mode.
        assert len(model.inputs) == 2
        assert model.inputs[0].shape == (1, 3, 64, 64)
        assert model.inputs[0].max() > model.inputs[1].max()  # Requested order, not manifest order.


@pytest.mark.parametrize("count", [1, 2])
def test_cli_loads_checkpoints_and_single_row(training_cfg, tmp_path, monkeypatch, capsys, count):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(training_cfg), encoding="utf-8")
    models = [torch.nn.Conv2d(3, 1, 1) for _ in range(count)]
    loaded_models = []

    def build_model(cfg):
        assert cfg["model"]["encoder_weights"] is None
        model = torch.nn.Conv2d(3, 1, 1)
        loaded_models.append(model)
        return model

    monkeypatch.setattr("src.model.build_model", build_model)
    output = tmp_path / "single.png"
    args = ["--config", str(config), "--split", "val", "--paths", selected_paths("val")[0],
            "--output", str(output)]
    for index, model in enumerate(models):
        checkpoint = tmp_path / f"model_{index}.pth"
        state = model.state_dict()
        torch.save({"model_state_dict": state} if index else state, checkpoint)
        args += ["--checkpoint", str(checkpoint), "--label", f"Model {index}"]
    script.main(args)
    assert output.is_file() and f"Saved 1 cases with {count} checkpoints" in capsys.readouterr().out
    for expected, actual in zip(models, loaded_models, strict=True):
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[key], value)


@pytest.mark.parametrize("checkpoints,labels,message", [
    ([], [], "matching --label"),
    (["a.pth", "b.pth"], ["A"], "matching --label"),
    (["a.pth"], [" "], "nonempty and unique"),
    (["a.pth", "b.pth"], ["A", "A"], "nonempty and unique"),
])
def test_invalid_checkpoint_labels(training_cfg, tmp_path, checkpoints, labels, message):
    with pytest.raises(ValueError, match=message):
        script.plot_cases(training_cfg, checkpoints, labels, "val", selected_paths("val"),
                          tmp_path / "unused.png")


def test_threshold_includes_equality_and_rejects_nonfinite_predictions(training_cfg,
                                                                     tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    model = torch.nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    monkeypatch.setattr(script, "load_model", lambda *args: model)
    original_figure = script.case_figure

    def check_masks(rgb, targets, predictions, *args):
        assert predictions[0][0].all()  # sigmoid(0) == threshold == 0.5
        return original_figure(rgb, targets, predictions, *args)

    monkeypatch.setattr(script, "case_figure", check_masks)
    args = (training_cfg, [checkpoint], ["A"], "val", [selected_paths("val")[0]],
            tmp_path / "zero.png")
    script.plot_cases(*args)
    with torch.no_grad():
        model.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite predictions"):
        script.plot_cases(*args)
