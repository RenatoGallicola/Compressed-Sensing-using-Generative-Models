"""Convolutional variational auto-encoder for MNIST.

The architecture is the one described in ``docs/NAML_project_report.pdf`` (sec.
3.2): a four-layer convolutional encoder producing the parameters of a diagonal
Gaussian posterior, and a two-layer transposed-convolutional decoder. Only the
decoder is needed at recovery time -- it *is* the generator ``G``.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from csgm.config import IMAGE_SHAPE


@keras.saving.register_keras_serializable(package="csgm")
class Sampling(layers.Layer):
    """Reparameterisation trick: ``z = mu + exp(log_var / 2) * eps``.

    Sampling ``z ~ N(mu, sigma^2)`` directly is not differentiable w.r.t. the
    encoder parameters. Pushing the randomness into an auxiliary
    ``eps ~ N(0, I)`` keeps the path from ``z`` back to ``mu``/``log_var``
    deterministic, so gradients flow.
    """

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Draw one sample per row.

        Args:
            inputs: Tuple ``(z_mean, z_log_var)``, each of shape ``(batch, k)``.

        Returns:
            Latent samples of shape ``(batch, k)``.
        """
        z_mean, z_log_var = inputs
        eps = keras.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_encoder(latent_dim: int, input_shape: tuple[int, int, int] = IMAGE_SHAPE) -> keras.Model:
    """Build the recognition network ``q_phi(z | x)``.

    Args:
        latent_dim: Dimensionality ``k`` of the latent space.
        input_shape: Shape of a single input image.

    Returns:
        A model mapping an image to ``[z_mean, z_log_var, z]``.
    """
    inputs = keras.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    return keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(latent_dim: int, input_shape: tuple[int, int, int] = IMAGE_SHAPE) -> keras.Model:
    """Build the generative network ``p_theta(x | z)`` -- the CS generator.

    Args:
        latent_dim: Dimensionality ``k`` of the latent space.
        input_shape: Shape of the image to reconstruct.

    Returns:
        A model mapping ``(batch, latent_dim)`` to ``(batch, *input_shape)``
        with sigmoid outputs in ``[0, 1]``.
    """
    height, width, channels = input_shape
    # Mirror the single stride-2 encoder layer.
    h, w, filters = height // 2, width // 2, 64

    inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(h * w * filters, activation="relu")(inputs)
    x = layers.Reshape((h, w, filters))(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = layers.Conv2DTranspose(
        channels, 3, padding="same", activation="sigmoid", name="decoder_output"
    )(x)
    return keras.Model(inputs, outputs, name="decoder")


@keras.saving.register_keras_serializable(package="csgm")
class VAE(keras.Model):
    """VAE trained by maximising the evidence lower bound.

    The loss is ``-ELBO = reconstruction_loss + kl_loss``, with a Bernoulli
    (binary cross-entropy) likelihood summed over pixels and an analytic KL to
    the ``N(0, I)`` prior.
    """

    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        """Metrics Keras resets between epochs."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Encode and decode a batch of images."""
        _, _, z = self.encoder(inputs, training=training)
        return self.decoder(z, training=training)

    def _compute_losses(self, data: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss

    def _update_trackers(self, total: tf.Tensor, rec: tf.Tensor, kl: tf.Tensor) -> dict:
        self.total_loss_tracker.update_state(total)
        self.reconstruction_loss_tracker.update_state(rec)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data: tf.Tensor) -> dict:
        """Run one gradient step on ``-ELBO``."""
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            total, rec, kl = self._compute_losses(data)
        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights, strict=True))
        return self._update_trackers(total, rec, kl)

    def test_step(self, data: tf.Tensor) -> dict:
        """Evaluate ``-ELBO`` without updating the weights."""
        if isinstance(data, tuple):
            data = data[0]
        return self._update_trackers(*self._compute_losses(data))
