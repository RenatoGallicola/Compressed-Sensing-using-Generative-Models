"""Tests for the sparse-recovery baselines.

Two bases are covered: the pixel basis, which is the one Bora et al. use on
MNIST, and the DCT basis, which they use for natural images.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import dct

from csgm.baselines import dct_basis, lasso_recover
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2


def sparse_in(basis, n, k, seed):
    """Build a signal with ``k`` non-zero coefficients in the given basis."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(n)
    theta[rng.choice(n, k, replace=False)] = rng.uniform(0.3, 1.0, k)
    x = theta if basis == "pixel" else dct_basis(n) @ theta
    return np.clip(x, 0.0, 1.0)


def test_dct_basis_is_orthonormal():
    psi = dct_basis(32)
    np.testing.assert_allclose(psi.T @ psi, np.eye(32), atol=1e-10)


def test_dct_basis_synthesises_what_the_dct_analyses():
    """``x = Psi (Psi^T x)`` must round-trip, otherwise the design matrix is wrong."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    np.testing.assert_allclose(dct_basis(64) @ dct(x, norm="ortho"), x, atol=1e-10)


@pytest.mark.parametrize("basis", ["pixel", "dct"])
def test_recovers_a_signal_that_is_sparse_in_its_own_basis(basis):
    n, m = 128, 90
    x = sparse_in(basis, n, 5, seed=1)
    A = gaussian_measurement_matrix(m, n, seed=1)

    x_hat = lasso_recover(measure(x, A), A, basis=basis, alpha=1e-6)

    assert per_pixel_l2(x_hat, x)[0] < 1e-3


def test_the_basis_matters():
    """A pixel-sparse signal is recovered better in the pixel basis than in the DCT one."""
    n, m = 128, 40
    x = sparse_in("pixel", n, 4, seed=6)
    A = gaussian_measurement_matrix(m, n, seed=6)
    y = measure(x, A)

    in_pixel = per_pixel_l2(lasso_recover(y, A, basis="pixel", alpha=1e-6), x)[0]
    in_dct = per_pixel_l2(lasso_recover(y, A, basis="dct", alpha=1e-6), x)[0]
    assert in_pixel < in_dct


@pytest.mark.parametrize("basis", ["pixel", "dct"])
def test_more_measurements_do_not_hurt(basis):
    n = 128
    x = sparse_in(basis, n, 4, seed=2)

    errors = []
    for m in (20, 60, 120):
        A = gaussian_measurement_matrix(m, n, seed=2)
        x_hat = lasso_recover(measure(x, A), A, basis=basis, alpha=1e-6)
        errors.append(per_pixel_l2(x_hat, x)[0])
    assert errors[-1] <= errors[0]


def test_output_is_clipped_to_the_image_range():
    A = gaussian_measurement_matrix(20, 64, seed=3)
    y = np.full((1, 20), 50.0)  # far outside anything a [0, 1] image can produce
    x_hat = lasso_recover(y, A, basis="dct", alpha=1e-6)
    assert x_hat.min() >= 0.0
    assert x_hat.max() <= 1.0


def test_clipping_can_be_turned_off():
    A = gaussian_measurement_matrix(20, 64, seed=3)
    y = np.full((1, 20), 50.0)
    x_hat = lasso_recover(y, A, basis="dct", alpha=1e-6, clip=False)
    assert x_hat.max() > 1.0


def test_batch_and_single_agree():
    n, m = 64, 40
    rng = np.random.default_rng(4)
    x = rng.random((3, n))
    A = gaussian_measurement_matrix(m, n, seed=4)
    y = measure(x, A)

    batch = lasso_recover(y, A, basis="dct")
    for i in range(3):
        np.testing.assert_allclose(batch[i], lasso_recover(y[i], A, basis="dct")[0], atol=1e-8)


def test_dimension_mismatch_raises():
    A = gaussian_measurement_matrix(10, 64, seed=5)
    with pytest.raises(ValueError, match="10 rows"):
        lasso_recover(np.zeros((1, 9)), A)


def test_unknown_basis_raises():
    A = gaussian_measurement_matrix(10, 64, seed=8)
    with pytest.raises(ValueError, match="unknown basis"):
        lasso_recover(np.zeros((1, 10)), A, basis="wavelet")
