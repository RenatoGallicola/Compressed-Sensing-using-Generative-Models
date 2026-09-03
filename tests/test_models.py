"""Tests for the VAE / DCGAN architectures and the checkpoint loader."""

from __future__ import annotations

import numpy as np
import pytest

from csgm.config import IMAGE_SHAPE, checkpoint_path
from csgm.models import (
    DCGAN,
    VAE,
    build_decoder,
    build_discriminator,
    build_encoder,
    build_generator,
    load_generator,
)

LATENT_DIM = 8


def test_encoder_outputs_the_posterior_and_a_sample():
    encoder = build_encoder(LATENT_DIM)
    z_mean, z_log_var, z = encoder(np.zeros((2, *IMAGE_SHAPE), dtype="float32"))
    for tensor in (z_mean, z_log_var, z):
        assert tuple(tensor.shape) == (2, LATENT_DIM)


def test_sampling_is_stochastic_but_centred_on_the_mean():
    """The reparameterisation trick must inject noise, not just pass ``mu`` through."""
    from csgm.models import Sampling

    z_mean = np.zeros((4096, 3), dtype="float32")
    z_log_var = np.zeros((4096, 3), dtype="float32")  # sigma = 1
    z = np.asarray(Sampling()([z_mean, z_log_var]))
    assert z.std() == pytest.approx(1.0, rel=0.1)
    assert abs(z.mean()) < 0.1


@pytest.mark.parametrize("builder", [build_decoder, build_generator])
def test_generators_produce_images_in_the_unit_interval(builder):
    generator = builder(LATENT_DIM)
    images = np.asarray(generator(np.zeros((2, LATENT_DIM), dtype="float32")))
    assert images.shape == (2, *IMAGE_SHAPE)
    assert 0.0 <= images.min() and images.max() <= 1.0


def test_discriminator_outputs_a_probability():
    discriminator = build_discriminator()
    p = np.asarray(discriminator(np.zeros((2, *IMAGE_SHAPE), dtype="float32")))
    assert p.shape == (2, 1)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_vae_training_step_reports_its_three_losses():
    import keras

    vae = VAE(build_encoder(LATENT_DIM), build_decoder(LATENT_DIM))
    vae.compile(optimizer=keras.optimizers.Adam())
    images = np.random.default_rng(0).random((8, *IMAGE_SHAPE)).astype("float32")
    history = vae.fit(images, epochs=1, verbose=0)
    assert set(history.history) == {"loss", "reconstruction_loss", "kl_loss"}
    assert np.isfinite(history.history["loss"][0])


def test_dcgan_training_step_reports_both_losses():
    import keras

    gan = DCGAN(build_discriminator(), build_generator(LATENT_DIM), LATENT_DIM)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(),
        g_optimizer=keras.optimizers.Adam(),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    images = np.random.default_rng(0).random((4, *IMAGE_SHAPE)).astype("float32")
    history = gan.fit(images, epochs=1, verbose=0)
    assert set(history.history) == {"d_loss", "g_loss"}


def test_load_generator_rejects_unknown_models():
    with pytest.raises(ValueError, match="unknown model"):
        checkpoint_path("wavelet", 20)


def test_load_generator_reports_a_missing_checkpoint():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_generator("vae", 99999)


@pytest.mark.parametrize(
    ("model", "latent_dim"), [("vae", 20), ("vae", 30), ("dcgan", 20), ("dcgan", 30)]
)
def test_shipped_checkpoints_load_and_generate(model, latent_dim):
    if not checkpoint_path(model, latent_dim).exists():
        pytest.skip(f"checkpoint for {model} k={latent_dim} is not present")
    generator = load_generator(model, latent_dim)
    images = np.asarray(generator(np.zeros((1, latent_dim), dtype="float32")))
    assert images.reshape(1, -1).shape[1] == int(np.prod(IMAGE_SHAPE))
    assert not generator.trainable
