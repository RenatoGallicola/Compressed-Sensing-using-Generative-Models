"""Tests for the Lasso/DCT sparse-recovery baseline."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import dct

from csgm.baselines import dct_basis, lasso_dct_recover
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2


def test_dct_basis_is_orthonormal():
    psi = dct_basis(32)
    np.testing.assert_allclose(psi.T @ psi, np.eye(32), atol=1e-10)


def test_dct_basis_synthesises_what_the_dct_analyses():
    """``x = Psi (Psi^T x)`` must round-trip, otherwise the design matrix is wrong."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    psi = dct_basis(64)
    np.testing.assert_allclose(psi @ dct(x, norm="ortho"), x, atol=1e-10)


def test_recovers_a_signal_that_is_sparse_in_the_dct_basis():
    n, m, k = 128, 90, 5
    rng = np.random.default_rng(1)
    theta = np.zeros(n)
    theta[rng.choice(n, k, replace=False)] = rng.uniform(0.3, 1.0, k)
    x = np.clip(dct_basis(n) @ theta, 0.0, 1.0)

    A = gaussian_measurement_matrix(m, n, seed=1)
    x_hat = lasso_dct_recover(measure(x, A), A, alpha=1e-6)

    assert per_pixel_l2(x_hat, x)[0] < 1e-3


def test_more_measurements_do_not_hurt():
    n = 128
    rng = np.random.default_rng(2)
    theta = np.zeros(n)
    theta[rng.choice(n, 4, replace=False)] = 1.0
    x = np.clip(dct_basis(n) @ theta, 0.0, 1.0)

    errors = []
    for m in (20, 60, 120):
        A = gaussian_measurement_matrix(m, n, seed=2)
        errors.append(per_pixel_l2(lasso_dct_recover(measure(x, A), A, alpha=1e-6), x)[0])
    assert errors[-1] <= errors[0]


def test_output_is_clipped_to_the_image_range():
    A = gaussian_measurement_matrix(20, 64, seed=3)
    y = np.full((1, 20), 50.0)  # far outside anything a [0, 1] image can produce
    x_hat = lasso_dct_recover(y, A, alpha=1e-6)
    assert x_hat.min() >= 0.0
    assert x_hat.max() <= 1.0


def test_batch_and_single_agree():
    n, m = 64, 40
    rng = np.random.default_rng(4)
    x = rng.random((3, n))
    A = gaussian_measurement_matrix(m, n, seed=4)
    y = measure(x, A)
    batch = lasso_dct_recover(y, A)
    for i in range(3):
        np.testing.assert_allclose(batch[i], lasso_dct_recover(y[i], A)[0], atol=1e-8)


def test_dimension_mismatch_raises():
    A = gaussian_measurement_matrix(10, 64, seed=5)
    with pytest.raises(ValueError, match="10 rows"):
        lasso_dct_recover(np.zeros((1, 9)), A)
