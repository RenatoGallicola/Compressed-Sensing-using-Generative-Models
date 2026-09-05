"""Sparsity-based compressed sensing baselines.

The classical alternative to a learned prior is to assume the signal is sparse
in some fixed basis. Writing ``x = Psi theta``, the measurements become
``y = A Psi theta`` and the signal is recovered by L1-regularised least squares:

    theta_hat = argmin_theta  ||A Psi theta - y||^2 + alpha ||theta||_1 ,
    x_hat     = Psi theta_hat.

Two choices of ``Psi`` are provided. ``"pixel"`` takes ``Psi = I``, which is
what Bora et al. use on MNIST: digits are mostly background, so they are already
sparse in pixel space. ``"dct"`` takes the orthonormal DCT, which the same
authors use on celebA and which is the usual choice for natural images.
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


def lasso_recover(
    y: np.ndarray,
    A: np.ndarray,
    *,
    basis: str = "pixel",
    alpha: float = 1e-4,
    max_iter: int = 10_000,
    clip: bool = True,
) -> np.ndarray:
    """Recover signals from compressed measurements with a sparsity prior.

    Args:
        y: Measurements of shape ``(m,)`` or ``(batch, m)``.
        A: Measurement matrix of shape ``(m, n)``.
        basis: ``"pixel"`` for sparsity in pixel space, which is what Bora et al.
            use on MNIST, or ``"dct"`` for the orthonormal DCT basis.
        alpha: L1 regularisation strength passed to :class:`sklearn.linear_model.Lasso`.
            Its objective is scaled by ``1 / (2 * m)``, so this is not on the same
            scale as the shrinkage parameter quoted in the paper; the value used
            here was chosen by sweeping it, see ``scripts/tune_lasso.py``.
        max_iter: Maximum coordinate-descent iterations.
        clip: If ``True``, clip the reconstruction to ``[0, 1]``. MNIST pixels
            live in that range, and the un-clipped Lasso solution leaves it.
            Reported results use clipping, which helps the baseline.

    Returns:
        Reconstructions of shape ``(batch, n)``.

    Raises:
        ValueError: If the trailing dimension of ``y`` does not match ``A``, or
            if ``basis`` is not a known one.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    A = np.asarray(A, dtype=np.float64)
    if y.shape[1] != A.shape[0]:
        raise ValueError(f"y has {y.shape[1]} measurements but A has {A.shape[0]} rows")
    if basis not in {"pixel", "dct"}:
        raise ValueError(f"unknown basis {basis!r}, expected 'pixel' or 'dct'")

    psi = None if basis == "pixel" else dct_basis(A.shape[1])
    design = A if psi is None else A @ psi

    # Lasso supports multi-target fits, so the whole batch is solved at once.
    model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    model.fit(design, y.T)
    theta = np.atleast_2d(model.coef_)

    x_hat = theta if psi is None else theta @ psi.T
    return np.clip(x_hat, 0.0, 1.0) if clip else x_hat
