"""Tests for the random Gaussian measurement operator."""

from __future__ import annotations

import numpy as np
import pytest

from csgm.measurements import gaussian_measurement_matrix, measure


def test_shape_and_dtype():
    A = gaussian_measurement_matrix(50, 784, seed=0)
    assert A.shape == (50, 784)
    assert A.dtype == np.float32


def test_seed_is_reproducible():
    a = gaussian_measurement_matrix(20, 100, seed=7)
    b = gaussian_measurement_matrix(20, 100, seed=7)
    c = gaussian_measurement_matrix(20, 100, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_entries_have_variance_one_over_m():
    m = 200
    A = gaussian_measurement_matrix(m, 2000, seed=1)
    assert A.std() == pytest.approx(1 / np.sqrt(m), rel=0.05)


def test_operator_is_an_isometry_in_expectation():
    """``E[||A x||^2] = ||x||^2`` is what makes the 1/m variance the right choice.

    A single draw has relative standard deviation ``sqrt(2/m)``, so the average
    is taken over enough matrices for the 6% tolerance to sit several standard
    errors away from the mean.
    """
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500).astype("float32")
    norms = [
        np.sum(measure(x, gaussian_measurement_matrix(100, 500, seed=s)) ** 2) for s in range(400)
    ]
    assert np.mean(norms) == pytest.approx(np.sum(x**2), rel=0.06)


def test_noise_is_added_only_when_requested():
    A = gaussian_measurement_matrix(30, 100, seed=3)
    x = np.ones((2, 100), dtype="float32")
    clean = measure(x, A)
    noisy = measure(x, A, noise_std=0.5, seed=3)
    np.testing.assert_allclose(clean, x @ A.T, rtol=1e-6)
    assert np.abs(noisy - clean).mean() > 0.1


def test_single_signal_is_promoted_to_a_batch():
    A = gaussian_measurement_matrix(10, 100, seed=4)
    assert measure(np.zeros(100, dtype="float32"), A).shape == (1, 10)


def test_dimension_mismatch_raises():
    A = gaussian_measurement_matrix(10, 100, seed=5)
    with pytest.raises(ValueError, match="expects 100"):
        measure(np.zeros(99, dtype="float32"), A)


@pytest.mark.parametrize(("m", "n"), [(0, 10), (10, 0), (-1, 10)])
def test_invalid_dimensions_raise(m, n):
    with pytest.raises(ValueError, match="must be positive"):
        gaussian_measurement_matrix(m, n)
