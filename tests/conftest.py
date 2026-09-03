"""Shared fixtures.

The tests deliberately avoid the shipped checkpoints: a tiny linear generator
makes the recovery loop exactly solvable, so failures point at the optimiser
rather than at the quality of a trained prior.
"""

from __future__ import annotations

import numpy as np
import pytest

LATENT_DIM = 4
N_PIXELS = 784


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic random generator shared by the test session."""
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def linear_generator():
    """A frozen linear generator ``G(z) = z W`` reshaped to a 28x28x1 image.

    Being linear, ``argmin_z ||A G(z) - y||`` is a least-squares problem with a
    unique solution whenever ``A W`` has full column rank, which is what makes
    it a usable oracle for the recovery tests.
    """
    import keras

    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [
            keras.Input(shape=(LATENT_DIM,)),
            keras.layers.Dense(N_PIXELS, use_bias=False),
            keras.layers.Reshape((28, 28, 1)),
        ]
    )
    model.trainable = False
    return model
