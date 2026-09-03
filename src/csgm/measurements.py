"""Random Gaussian measurement operators.

Compressed sensing observes a signal ``x* in R^n`` only through

    y = A x* + eta,    A in R^{m x n},  eta in R^m,   m << n.

Following Bora et al. (2017) the entries of ``A`` are drawn i.i.d. from
``N(0, 1/m)``. That variance is what makes ``A`` an approximate isometry in
expectation -- ``E[||A x||^2] = ||x||^2`` -- so the measurement error and the
signal error live on the same scale regardless of ``m``.
"""

from __future__ import annotations

import numpy as np


def gaussian_measurement_matrix(
    m: int, n: int, *, seed: int | None = None, dtype: str = "float32"
) -> np.ndarray:
    """Sample a random Gaussian measurement matrix.

    Args:
        m: Number of measurements (rows).
        n: Ambient dimension (columns).
        seed: Seed of the local random generator; ``None`` draws from OS entropy.
        dtype: Floating dtype of the returned matrix.

    Returns:
        Array of shape ``(m, n)`` with i.i.d. ``N(0, 1/m)`` entries.

    Raises:
        ValueError: If ``m`` or ``n`` is not positive.
    """
    if m <= 0 or n <= 0:
        raise ValueError(f"m and n must be positive, got m={m}, n={n}")
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((m, n)) / np.sqrt(m)).astype(dtype)


def measure(
    x: np.ndarray, A: np.ndarray, *, noise_std: float = 0.0, seed: int | None = None
) -> np.ndarray:
    """Apply the measurement operator to one or more signals.

    Args:
        x: Signal of shape ``(n,)`` or a batch of shape ``(batch, n)``.
        A: Measurement matrix of shape ``(m, n)``.
        noise_std: Standard deviation of the additive white Gaussian noise.
        seed: Seed used for the noise draw.

    Returns:
        Measurements of shape ``(batch, m)`` (a ``(n,)`` input is treated as a
        batch of one and comes back as ``(1, m)``).

    Raises:
        ValueError: If the trailing dimension of ``x`` does not match ``A``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=A.dtype))
    if x.shape[-1] != A.shape[1]:
        raise ValueError(f"x has {x.shape[-1]} features but A expects {A.shape[1]}")

    y = x @ A.T
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.standard_normal(y.shape).astype(A.dtype) * noise_std
    return y
