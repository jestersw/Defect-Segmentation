"""Check projection arithmetic and the CPU evidence contract, without a full benchmark."""

import pytest

from scripts.smoke_test import estimates


@pytest.mark.parametrize("samples, steps", [(582, 73), (550, 69), (8, 1), (9, 2)])
def test_epoch_estimates_include_partial_batch(samples, steps):
    result = estimates(step_seconds=2.5, samples=samples, batch_size=8)
    assert result["steps_per_epoch"] == steps
    assert result["minutes_per_epoch"] == pytest.approx(steps * 2.5 / 60)
    assert result["hours_per_60_epochs"] == pytest.approx(steps * 2.5 * 60 / 3600)
