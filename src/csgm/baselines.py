"""Sparsity-based compressed sensing baseline (Lasso in a DCT basis).

The classical alternative to a learned prior is to assume the signal is sparse
in some fixed basis. Writing ``x = Psi theta`` with ``Psi`` the orthonormal DCT
basis, the measurements become ``y = A Psi theta`` and the signal is recovered
by L1-regularised least squares:

    theta_hat = argmin_theta  ||A Psi theta - y||^2 + alpha ||theta||_1 ,
    x_hat     = Psi theta_hat.

This is the reference the generative priors are compared against.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import idct
from sklearn.linear_model import Lasso


@lru_cache(maxsize=4)
def dct_basis(n: int) -> np.ndarray:
    """Return the orthonormal DCT-II synthesis basis ``Psi`` of size ``n``.

    ``Psi`` maps coefficients to the signal domain (``x = Psi theta``) and is
    orthogonal, so the analysis operator is simply ``Psi.T``. Cached because it
    is an ``n x n`` matrix rebuilt on every recovery otherwise.

    Args:
        n: Signal length.

    Returns:
        Array of shape ``(n, n)``.
    """
    return idct(np.eye(n), norm="ortho", axis=0)


def lasso_dct_recover(
    y: np.ndarray,
    A: np.ndarray,
    *,
    alpha: float = 1e-5,
    max_iter: int = 10_000,
    clip: bool = True,
) -> np.ndarray:
    """Recover signals from compressed measurements with a Lasso/DCT prior.

    Args:
        y: Measurements of shape ``(m,)`` or ``(batch, m)``.
        A: Measurement matrix of shape ``(m, n)``.
        alpha: L1 regularisation strength passed to :class:`sklearn.linear_model.Lasso`.
            Its objective is scaled by ``1 / (2 * m)``, so the useful range is
            small; ``1e-5`` is the value used by Bora et al. on MNIST.
        max_iter: Maximum coordinate-descent iterations.
        clip: If ``True``, clip the reconstruction to ``[0, 1]``. MNIST pixels
            live in that range, and the un-clipped Lasso solution overshoots it.

    Returns:
        Reconstructions of shape ``(batch, n)``.

    Raises:
        ValueError: If the trailing dimension of ``y`` does not match ``A``.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    A = np.asarray(A, dtype=np.float64)
    if y.shape[1] != A.shape[0]:
        raise ValueError(f"y has {y.shape[1]} measurements but A has {A.shape[0]} rows")

    psi = dct_basis(A.shape[1])
    design = A @ psi

    # Lasso supports multi-target fits, so the whole batch is solved at once.
    model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    model.fit(design, y.T)
    theta = np.atleast_2d(model.coef_)

    x_hat = theta @ psi.T
    return np.clip(x_hat, 0.0, 1.0) if clip else x_hat
