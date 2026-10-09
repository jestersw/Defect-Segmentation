import numpy as np
import torch

from scripts.plot_predictions import choose_examples, render


def masks(areas, size=32):
    result = []
    for area in areas:
        mask = np.zeros((size, size), dtype=bool)
        mask.flat[:area] = True
        result.append(mask)
    return result


def test_choose_examples_covers_sizes_and_a_good_image():
    targets = masks([0, 5, 400, 50, 0, 900, 10])
    picks = choose_examples(targets, count=4)
    assert len(picks) == 4
    assert picks[0] == 1
    assert 5 in picks
    assert any(targets[i].sum() == 0 for i in picks)


def test_choose_examples_without_defects():
    assert choose_examples(masks([0, 0, 0]), count=2) == [0, 1]


def test_render_writes_png(tmp_path):
    targets = masks([0, 20])
    probabilities = [t.astype(float) for t in targets]
    images = {i: torch.zeros(3, 16, 16) for i in range(2)}
    records = [
        {"category": "tile", "defect_type": "good"},
        {"category": "tile", "defect_type": "crack"},
    ]
    output = render(images, targets, probabilities, records, [1, 0], 0.5, tmp_path / "x.png")
    assert output.exists() and output.stat().st_size > 0
