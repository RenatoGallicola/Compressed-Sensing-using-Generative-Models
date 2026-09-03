"""Tests for the latent-space recovery loop."""

from __future__ import annotations

import numpy as np
import pytest

from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2
from csgm.recovery import RecoveryConfig, recover
from tests.conftest import LATENT_DIM, N_PIXELS

FAST = RecoveryConfig(steps=300, restarts=2, learning_rate=0.05, image_batch_size=4)


def _range_signal(generator, rng, batch=1):
    """Produce signals that lie exactly in the range of the generator."""
    z = rng.standard_normal((batch, LATENT_DIM)).astype("float32")
    return np.asarray(generator(z, training=False)).reshape(batch, N_PIXELS), z


def test_recovers_a_signal_in_the_range_of_the_generator(linear_generator, rng):
    """With m > k and a signal in range, the optimum drives the error to ~0."""
    x, _ = _range_signal(linear_generator, rng)
    A = gaussian_measurement_matrix(100, N_PIXELS, seed=0)
    result = recover(linear_generator, measure(x, A), A, LATENT_DIM, FAST)
    assert per_pixel_l2(result.x_hat, x)[0] < 1e-4


def test_result_shapes(linear_generator, rng):
    x, _ = _range_signal(linear_generator, rng, batch=3)
    A = gaussian_measurement_matrix(60, N_PIXELS, seed=1)
    result = recover(linear_generator, measure(x, A), A, LATENT_DIM, FAST)
    assert result.x_hat.shape == (3, N_PIXELS)
    assert result.z.shape == (3, LATENT_DIM)
    assert result.residual.shape == (3,)


def test_generator_weights_are_not_modified(linear_generator, rng):
    """Only ``z`` is optimised; touching the prior would invalidate every result."""
    before = [w.numpy().copy() for w in linear_generator.weights]
    x, _ = _range_signal(linear_generator, rng)
    A = gaussian_measurement_matrix(60, N_PIXELS, seed=2)
    recover(linear_generator, measure(x, A), A, LATENT_DIM, FAST)
    for w_before, w_after in zip(before, linear_generator.weights, strict=True):
        np.testing.assert_array_equal(w_before, w_after.numpy())


def test_batched_recovery_matches_one_image_at_a_time(linear_generator, rng):
    """Chunking over images must not change the per-image solution."""
    x, _ = _range_signal(linear_generator, rng, batch=4)
    A = gaussian_measurement_matrix(80, N_PIXELS, seed=3)
    y = measure(x, A)

    def run(image_batch_size):
        config = RecoveryConfig(steps=200, restarts=2, image_batch_size=image_batch_size)
        return recover(linear_generator, y, A, LATENT_DIM, config)

    together, apart = run(4), run(1)
    np.testing.assert_allclose(together.residual, apart.residual, rtol=0.15, atol=1e-3)


def test_more_restarts_never_increase_the_residual(linear_generator, rng):
    x, _ = _range_signal(linear_generator, rng)
    A = gaussian_measurement_matrix(40, N_PIXELS, seed=4)
    y = measure(x, A)
    one = recover(linear_generator, y, A, LATENT_DIM, RecoveryConfig(steps=100, restarts=1))
    many = recover(linear_generator, y, A, LATENT_DIM, RecoveryConfig(steps=100, restarts=6))
    assert many.residual[0] <= one.residual[0] * 1.05


def test_history_is_recorded_and_decreasing(linear_generator, rng):
    x, _ = _range_signal(linear_generator, rng)
    A = gaussian_measurement_matrix(60, N_PIXELS, seed=5)
    result = recover(linear_generator, measure(x, A), A, LATENT_DIM, FAST, track_history=True)
    assert result.history.shape == (FAST.steps,)
    assert result.history[-1] < result.history[0]


def test_l2_penalty_shrinks_the_latent_code(linear_generator, rng):
    x, _ = _range_signal(linear_generator, rng)
    A = gaussian_measurement_matrix(30, N_PIXELS, seed=6)
    y = measure(x, A)
    free = recover(linear_generator, y, A, LATENT_DIM, RecoveryConfig(steps=300, restarts=1))
    penalised = recover(
        linear_generator, y, A, LATENT_DIM, RecoveryConfig(steps=300, restarts=1, l2_penalty=1.0)
    )
    assert np.linalg.norm(penalised.z) < np.linalg.norm(free.z)


def test_dimension_mismatch_raises(linear_generator):
    A = gaussian_measurement_matrix(20, N_PIXELS, seed=7)
    with pytest.raises(ValueError, match="19 measurements"):
        recover(linear_generator, np.zeros((1, 19), dtype="float32"), A, LATENT_DIM, FAST)
