"""Reconstruction quality metrics.

Every metric here compares a reconstruction against the *ground-truth signal*,
never against the measurements. Measurement residual ``||A G(z) - y||`` is what
the recovery optimiser minimises, so it is a training diagnostic, not an
evaluation metric -- it keeps falling as ``m`` shrinks simply because there are
fewer constraints to satisfy.
"""

from __future__ import annotations

import numpy as np


def _as_batch(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(1, -1) if a.ndim == 1 else a.reshape(len(a), -1)


def reconstruction_error(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Squared L2 reconstruction error ``||x_hat - x||^2`` per signal.

    Args:
        x_hat: Reconstruction, shape ``(n,)`` or ``(batch, ...)``.
        x: Ground truth, broadcastable to the same shape.

    Returns:
        Array of shape ``(batch,)``.
    """
    diff = _as_batch(x_hat) - _as_batch(x)
    return np.sum(diff**2, axis=1)


def per_pixel_l2(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-pixel reconstruction error ``||x_hat - x||^2 / n``.

    This is the metric reported by Bora et al. and the one used for every curve
    in ``results/``. Dividing by ``n`` makes the value comparable across signal
    sizes; for MNIST in ``[0, 1]``, values below ~0.01 are visually faithful.

    Args:
        x_hat: Reconstruction, shape ``(n,)`` or ``(batch, ...)``.
        x: Ground truth, broadcastable to the same shape.

    Returns:
        Array of shape ``(batch,)``.
    """
    return reconstruction_error(x_hat, x) / _as_batch(x).shape[1]


def psnr(x_hat: np.ndarray, x: np.ndarray, *, data_range: float = 1.0) -> np.ndarray:
    """Peak signal-to-noise ratio in dB, per signal.

    Args:
        x_hat: Reconstruction, shape ``(n,)`` or ``(batch, ...)``.
        x: Ground truth, broadcastable to the same shape.
        data_range: Dynamic range of the data (``1.0`` for images in ``[0, 1]``).

    Returns:
        Array of shape ``(batch,)``. A perfect reconstruction yields ``inf``.
    """
    mse = per_pixel_l2(x_hat, x)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(data_range**2 / mse)
