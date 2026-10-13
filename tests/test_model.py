import pytest
import torch

from src.model import build_model


@pytest.mark.parametrize("weights", [None, "none"])
def test_model_outputs_logits_without_weight_download(training_cfg, weights):
    training_cfg["model"]["encoder_weights"] = weights
    model = build_model(training_cfg)
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 1, 64, 64)
    logits.mean().backward()
    assert model.segmentation_head[0].weight.grad is not None


def test_default_resnet34_contract():
    torch.set_num_threads(2)
    model = build_model({"model": {"encoder": "resnet34", "encoder_weights": None}})
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 1, 64, 64)
