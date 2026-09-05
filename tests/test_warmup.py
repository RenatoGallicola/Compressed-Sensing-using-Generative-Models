"""Tests for the KL warm-up schedule.

The warm-up exists to counter posterior collapse, and it has one property that
must not break: it may scale the training objective, but never the reported
ELBO, or validation losses stop being comparable across epochs and runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from csgm.config import IMAGE_SHAPE
from csgm.models import VAE, KLWarmUp, build_decoder, build_encoder

LATENT_DIM = 4


@pytest.fixture
def vae():
    import keras

    model = VAE(build_encoder(LATENT_DIM), build_decoder(LATENT_DIM))
    model.compile(optimizer=keras.optimizers.Adam())
    return model


def test_weight_starts_at_one_without_a_schedule(vae):
    assert float(vae.kl_weight) == 1.0


def test_ramp_goes_from_zero_to_one(vae):
    warmup = KLWarmUp(epochs=10)
    warmup.set_model(vae)
    seen = []
    for epoch in range(13):
        warmup.on_epoch_begin(epoch)
        seen.append(float(vae.kl_weight))

    assert seen[0] == pytest.approx(0.0)
    assert seen[5] == pytest.approx(0.5)
    assert seen[10] == pytest.approx(1.0)
    # It must stay at one afterwards rather than continuing to grow.
    assert seen[11] == pytest.approx(1.0)
    assert seen[12] == pytest.approx(1.0)
    assert seen == sorted(seen)


def test_zero_length_ramp_is_a_no_op(vae):
    warmup = KLWarmUp(epochs=0)
    warmup.set_model(vae)
    warmup.on_epoch_begin(0)
    assert float(vae.kl_weight) == 1.0


def test_evaluation_ignores_the_warm_up_weight(vae):
    """A partially warmed-up model must still report the full ELBO."""
    import keras

    images = np.random.default_rng(0).random((16, *IMAGE_SHAPE)).astype("float32")

    # An untrained encoder sits almost exactly on the prior, which would leave
    # the KL term too small for this test to detect anything.
    vae.encoder.get_layer("z_mean").bias.assign(np.full(LATENT_DIM, 2.0, "float32"))
    vae.encoder.get_layer("z_log_var").bias.assign(np.full(LATENT_DIM, 1.0, "float32"))

    def evaluate(weight):
        # The encoder samples z, so the seed is reset to make the two
        # evaluations differ only by the KL weight.
        keras.utils.set_random_seed(0)
        for metric in vae.metrics:
            metric.reset_state()
        vae.kl_weight.assign(weight)
        return {k: float(v) for k, v in vae.test_step(images).items()}

    full = evaluate(1.0)
    zeroed = evaluate(0.0)

    # Guard against the test passing because the KL term is negligible: a leak
    # of the weight into evaluation would have to be visible.
    assert full["kl_loss"] > 1e-2 * full["loss"], "the KL term is too small to detect a leak"

    # A leak would drop the reported loss by the whole KL term, so the tolerance
    # is a fraction of that term rather than float precision. Reseeding does not
    # reproduce the same z on every TensorFlow build, and the resulting
    # difference is two orders of magnitude below the effect being tested.
    assert abs(zeroed["loss"] - full["loss"]) < 1e-2 * full["kl_loss"]
    # The divergence itself is computed from the encoder means, with no sampling.
    assert zeroed["kl_loss"] == pytest.approx(full["kl_loss"], rel=1e-3)


def test_reported_loss_is_the_sum_of_its_parts(vae):
    images = np.random.default_rng(1).random((8, *IMAGE_SHAPE)).astype("float32")
    out = {k: float(v) for k, v in vae.test_step(images).items()}
    assert out["loss"] == pytest.approx(out["reconstruction_loss"] + out["kl_loss"], rel=1e-5)
