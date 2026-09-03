"""Tests for the reconstruction metrics."""

from __future__ import annotations

import numpy as np
import pytest

from csgm.metrics import per_pixel_l2, psnr, reconstruction_error


def test_identical_signals_have_zero_error():
    x = np.linspace(0, 1, 784)
    assert reconstruction_error(x, x)[0] == pytest.approx(0.0)
    assert per_pixel_l2(x, x)[0] == pytest.approx(0.0)
    assert np.isinf(psnr(x, x)[0])


def test_per_pixel_error_is_the_mean_squared_error():
    x = np.zeros(100)
    x_hat = np.full(100, 0.25)
    assert per_pixel_l2(x_hat, x)[0] == pytest.approx(0.0625)
    assert reconstruction_error(x_hat, x)[0] == pytest.approx(6.25)


def test_metrics_are_computed_per_signal():
    x = np.zeros((3, 10))
    x_hat = np.stack([np.full(10, v) for v in (0.0, 0.1, 0.2)])
    np.testing.assert_allclose(per_pixel_l2(x_hat, x), [0.0, 0.01, 0.04], atol=1e-12)


def test_image_shaped_inputs_are_flattened():
    """A ``(batch, 28, 28, 1)`` reconstruction must score like its flat version."""
    rng = np.random.default_rng(0)
    x = rng.random((2, 28, 28, 1))
    x_hat = x + 0.1
    np.testing.assert_allclose(
        per_pixel_l2(x_hat, x), per_pixel_l2(x_hat.reshape(2, -1), x.reshape(2, -1))
    )


def test_psnr_matches_its_definition():
    x = np.zeros(100)
    x_hat = np.full(100, 0.1)  # mse = 0.01 -> 20 dB at data_range 1
    assert psnr(x_hat, x)[0] == pytest.approx(20.0)
