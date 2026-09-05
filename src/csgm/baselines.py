"""Sparsity-based compressed sensing baselines.

The classical alternative to a learned prior is to assume the signal is sparse
in some fixed basis. Writing ``x = Psi theta``, the measurements become
``y = A Psi theta`` and the signal is recovered by L1-regularised least squares:

    theta_hat = argmin_theta  ||A Psi theta - y||^2 + alpha ||theta||_1 ,
    x_hat     = Psi theta_hat.

Three choices of ``Psi`` are provided. ``"pixel"`` takes ``Psi = I``, which is
what Bora et al. use on MNIST: digits are mostly background, so they are already
sparse in pixel space. ``"dct"`` takes the separable two dimensional DCT, which
the same authors apply to natural images and which is the transform image codecs
use. ``"dct1"`` is a one dimensional DCT over the raster-scanned vector, kept for
one dimensional signals; on images it is a substantially weaker basis.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fft import idct
from sklearn.linear_model import Lasso


@lru_cache(maxsize=4)
def dct_basis(n: int) -> np.ndarray:
    """Return the one dimensional orthonormal DCT-II synthesis basis of size ``n``.

    ``Psi`` maps coefficients to the signal domain (``x = Psi theta``) and is
    orthogonal, so the analysis operator is simply ``Psi.T``. Cached because it
    is an ``n x n`` matrix rebuilt on every recovery otherwise.

    Args:
        n: Signal length.

    Returns:
        Array of shape ``(n, n)``.
    """
    return idct(np.eye(n), norm="ortho", axis=0)


@lru_cache(maxsize=4)
def dct2_basis(side: int) -> np.ndarray:
    """Return the separable two dimensional DCT basis for ``side x side`` images.

    Applying a one dimensional DCT to a raster-scanned image is not the same
    transform: it treats the end of one row as adjacent to the start of the
    next, and compresses images far worse. Image codecs, and the reference
    paper, use the two dimensional transform, which is separable, so the basis
    for vectorised images is the Kronecker product of the one dimensional one
    with itself.

    Args:
        side: Image side length; the basis acts on ``side * side`` vectors.

    Returns:
        Array of shape ``(side * side, side * side)``.
    """
    one_d = dct_basis(side)
    return np.kron(one_d, one_d)


def lasso_recover(
    y: np.ndarray,
    A: np.ndarray,
    *,
    basis: str = "pixel",
    alpha: float = 1e-5,
    max_iter: int = 10_000,
    clip: bool = True,
) -> np.ndarray:
    """Recover signals from compressed measurements with a sparsity prior.

    Args:
        y: Measurements of shape ``(m,)`` or ``(batch, m)``.
        A: Measurement matrix of shape ``(m, n)``.
        basis: ``"pixel"`` for sparsity in pixel space, which is what Bora et al.
            use on MNIST; ``"dct"`` for the separable two dimensional DCT, the
            transform image codecs use and the one the same authors apply to
            natural images; ``"dct1"`` for a one dimensional DCT over the
            raster-scanned vector, which is a much weaker basis for images and
            is kept only for one dimensional signals.
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
    if basis not in {"pixel", "dct", "dct1"}:
        raise ValueError(f"unknown basis {basis!r}, expected 'pixel', 'dct' or 'dct1'")

    n = A.shape[1]
    if basis == "pixel":
        psi = None
    elif basis == "dct1":
        psi = dct_basis(n)
    else:
        side = round(n**0.5)
        if side * side != n:
            raise ValueError(f"the 2D DCT needs a square image, got n={n}")
        psi = dct2_basis(side)
    design = A if psi is None else A @ psi

    # Lasso supports multi-target fits, so the whole batch is solved at once.
    model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    model.fit(design, y.T)
    theta = np.atleast_2d(model.coef_)

    x_hat = theta if psi is None else theta @ psi.T
    return np.clip(x_hat, 0.0, 1.0) if clip else x_hat
