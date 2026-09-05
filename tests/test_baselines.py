"""Tests for the sparse-recovery baselines.

Three bases are covered: the pixel basis, which is the one Bora et al. use on
MNIST; the separable two dimensional DCT, which is the transform image codecs
use; and the one dimensional DCT, which is only appropriate for signals that are
not images.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import dct, dctn

from csgm.baselines import dct2_basis, dct_basis, lasso_recover
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2


def sparse_in(basis, n, k, seed):
    """Build a signal with ``k`` non-zero coefficients in the given basis."""
    rng = np.random.default_rng(seed)
    theta = np.zeros(n)
    theta[rng.choice(n, k, replace=False)] = rng.uniform(0.3, 1.0, k)
    if basis == "pixel":
        x = theta
    elif basis == "dct":
        x = dct2_basis(round(n**0.5)) @ theta
    else:
        x = dct_basis(n) @ theta
    return np.clip(x, 0.0, 1.0)


def test_dct_basis_is_orthonormal():
    psi = dct_basis(32)
    np.testing.assert_allclose(psi.T @ psi, np.eye(32), atol=1e-10)


def test_dct_basis_synthesises_what_the_dct_analyses():
    """``x = Psi (Psi^T x)`` must round-trip, otherwise the design matrix is wrong."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    np.testing.assert_allclose(dct_basis(64) @ dct(x, norm="ortho"), x, atol=1e-10)


def test_two_dimensional_basis_is_the_separable_dct():
    """It must agree with a genuine 2D DCT, not with a DCT of the raster scan."""
    rng = np.random.default_rng(0)
    image = rng.random((12, 12))
    psi = dct2_basis(12)

    np.testing.assert_allclose(psi.T @ psi, np.eye(144), atol=1e-10)
    np.testing.assert_allclose(
        psi.T @ image.reshape(-1), dctn(image, norm="ortho").reshape(-1), atol=1e-10
    )
    np.testing.assert_allclose(psi @ (psi.T @ image.reshape(-1)), image.reshape(-1), atol=1e-10)


def test_the_two_bases_are_not_the_same_transform():
    """A one dimensional DCT of a raster scan is a different, weaker basis."""
    side = 12
    image = sparse_in("dct", side * side, 6, seed=3)
    A = gaussian_measurement_matrix(60, side * side, seed=3)
    y = measure(image, A)

    two_d = per_pixel_l2(lasso_recover(y, A, basis="dct", alpha=1e-6), image)[0]
    one_d = per_pixel_l2(lasso_recover(y, A, basis="dct1", alpha=1e-6), image)[0]
    assert two_d < one_d


@pytest.mark.parametrize("basis", ["pixel", "dct1"])
def test_recovers_a_signal_that_is_sparse_in_its_own_basis(basis):
    n, m = 128, 90
    x = sparse_in(basis, n, 5, seed=1)
    A = gaussian_measurement_matrix(m, n, seed=1)

    x_hat = lasso_recover(measure(x, A), A, basis=basis, alpha=1e-6)

    assert per_pixel_l2(x_hat, x)[0] < 1e-3


def test_recovers_an_image_that_is_sparse_in_the_two_dimensional_basis():
    side, m = 12, 90
    x = sparse_in("dct", side * side, 5, seed=1)
    A = gaussian_measurement_matrix(m, side * side, seed=1)

    x_hat = lasso_recover(measure(x, A), A, basis="dct", alpha=1e-6)

    assert per_pixel_l2(x_hat, x)[0] < 1e-3


def test_the_basis_matters():
    """A pixel-sparse signal is recovered better in the pixel basis than in the DCT one."""
    n, m = 144, 40
    x = sparse_in("pixel", n, 4, seed=6)
    A = gaussian_measurement_matrix(m, n, seed=6)
    y = measure(x, A)

    in_pixel = per_pixel_l2(lasso_recover(y, A, basis="pixel", alpha=1e-6), x)[0]
    in_dct = per_pixel_l2(lasso_recover(y, A, basis="dct", alpha=1e-6), x)[0]
    assert in_pixel < in_dct


@pytest.mark.parametrize("basis", ["pixel", "dct1"])
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


def test_two_dimensional_basis_needs_a_square_signal():
    A = gaussian_measurement_matrix(10, 50, seed=9)
    with pytest.raises(ValueError, match="square image"):
        lasso_recover(np.zeros((1, 10)), A, basis="dct")
