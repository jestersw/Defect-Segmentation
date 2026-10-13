"""Cache must preserve the shared loader's data and augmentation semantics."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.data import MVTecSegDataset


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_cache_matches_uncached_samples(training_cfg, split):
    cached_cfg = deepcopy(training_cfg)
    cached_cfg["data"]["cache_images"] = True
    plain_cfg = deepcopy(training_cfg)
    plain_cfg["data"]["cache_images"] = False
    cached = MVTecSegDataset(split, cached_cfg, train=split == "train")
    plain = MVTecSegDataset(split, plain_cfg, train=split == "train")
    assert (cached.cache is None) == (split == "test")
    originals = None if cached.cache is None else [
        (image.copy(), mask.copy()) for image, mask in cached.cache
    ]
    for _ in range(3):
        for i in range(len(cached)):
            actual, expected = cached[i], plain[i]
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    if originals is not None:
        for before, after in zip(originals, cached.cache):
            assert all(np.array_equal(a, b) for a, b in zip(before, after))
