"""Deep convolutional GAN for MNIST.

Architecture follows ``docs/NAML_project_report.pdf`` (sec. 3.1), itself a
scaled-down version of Radford et al. (2015). Only the generator is needed at
recovery time; the discriminator is kept so training can be reproduced.
"""

from __future__ import annotations

from pathlib import Path

import keras
import tensorflow as tf
from keras import layers

from csgm.config import IMAGE_SHAPE


def build_discriminator(input_shape: tuple[int, int, int] = IMAGE_SHAPE) -> keras.Model:
    """Build the discriminator ``D``.

    Three stride-2 convolutions take the 28x28 input down to 4x4 before a
    dropout-regularised sigmoid head outputs ``P(real)``.

    Args:
        input_shape: Shape of a single input image.

    Returns:
        A model mapping ``(batch, *input_shape)`` to ``(batch, 1)``.
    """
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )


def build_generator(latent_dim: int) -> keras.Model:
    """Build the generator ``G : R^k -> [0, 1]^(28x28x1)``.

    The spatial cascade is 3x3 -> 6x6 -> 14x14 -> 28x28; the second transposed
    convolution uses ``padding="valid"`` on purpose, since that is what turns
    6x6 into 14x14 with a stride of 2 and a 4x4 kernel.

    Args:
        latent_dim: Dimensionality ``k`` of the latent space.

    Returns:
        A model mapping ``(batch, latent_dim)`` to ``(batch, 28, 28, 1)``.
    """
    return keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(3 * 3 * 128),
            layers.Reshape((3, 3, 128)),
            layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(256, 4, strides=2, padding="valid"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2DTranspose(512, 4, strides=2, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv2D(1, 5, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )


def smooth_labels(labels: tf.Tensor, strength: float = 0.05) -> tf.Tensor:
    """Move binary targets towards each other by a random amount.

    Smoothing keeps the discriminator from saturating early and starving the
    generator of gradient. It has to move each class towards the interior:
    adding the same positive noise to both, as the widely copied Keras example
    does, sends the positive targets to 1.05, which puts a negative weight on
    ``log(1 - p)`` in the cross-entropy and rewards overshooting instead.

    Args:
        labels: Targets in ``{0, 1}`` of any shape.
        strength: Largest displacement applied to a target.

    Returns:
        Targets in ``[0, 1]``, of the same shape.
    """
    noise = strength * tf.random.uniform(tf.shape(labels))
    return labels + noise * (1.0 - 2.0 * labels)


@keras.saving.register_keras_serializable(package="csgm")
class DCGAN(keras.Model):
    """The adversarial game between ``G`` and ``D``.

    Each step updates ``D`` on a half-real/half-fake batch, then updates ``G``
    against the (frozen) updated ``D``. Bora et al. run two generator updates
    per discriminator update, which is the default here: the discriminator wins
    the game easily on MNIST, and starving the generator of gradient is the
    usual way a DCGAN fails to train. Discriminator labels are perturbed with a
    little uniform noise, a standard trick that keeps ``D`` from saturating
    early.
    """

    def __init__(
        self,
        discriminator: keras.Model,
        generator: keras.Model,
        latent_dim: int,
        generator_updates: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.generator_updates = generator_updates
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self) -> list[keras.metrics.Metric]:
        """Metrics Keras resets between epochs."""
        return [self.d_loss_metric, self.g_loss_metric]

    def compile(self, d_optimizer, g_optimizer, loss_fn=None, **kwargs) -> None:
        """Attach the two optimisers and the adversarial loss."""
        super().compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn or keras.losses.BinaryCrossentropy()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Generate images from latent codes."""
        return self.generator(inputs, training=training)

    def train_step(self, real_images: tf.Tensor) -> dict:
        """Run one discriminator update followed by ``generator_updates`` of ``G``."""
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]

        # Discriminator: fakes are labelled 1 and reals 0. The convention is
        # arbitrary, as long as the generator's target labels are its mirror.
        generated = self.generator(tf.random.normal((batch_size, self.latent_dim)))
        combined = tf.concat([generated, real_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        labels = smooth_labels(labels)

        with tf.GradientTape() as tape:
            d_loss = self.loss_fn(labels, self.discriminator(combined))
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights, strict=True)
        )

        # Generator: push D towards calling its samples real.
        misleading = tf.zeros((batch_size, 1))
        for _ in range(self.generator_updates):
            with tf.GradientTape() as tape:
                fakes = self.generator(tf.random.normal((batch_size, self.latent_dim)))
                g_loss = self.loss_fn(misleading, self.discriminator(fakes))
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(
                zip(grads, self.generator.trainable_weights, strict=True)
            )

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {m.name: m.result() for m in self.metrics}


class GeneratorCheckpoints(keras.callbacks.Callback):
    """Save the generator every few epochs so it can be selected afterwards.

    A GAN has no validation loss, so there is no signal telling training when to
    stop, and sample quality oscillates from epoch to epoch. Keeping whatever
    the final epoch produced is a choice made by the schedule rather than by any
    criterion. Saving intermediate generators costs nothing during training and
    lets ``scripts/select_dcgan.py`` pick one on held-out data afterwards.
    """

    def __init__(self, output_dir: str | Path, latent_dim: int, every: int = 5, start: int = 20):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.latent_dim = latent_dim
        self.every = every
        self.start = start

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Create the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Write the generator if this epoch is on the schedule."""
        done = epoch + 1
        if done < self.start or done % self.every:
            return
        path = self.output_dir / f"gan_gen_dim{self.latent_dim}_epoch{done:03d}.keras"
        self.model.generator.save(path)
        print(f"  checkpoint {path.name}")


class GANMonitor(keras.callbacks.Callback):
    """Save a fixed grid of samples at the end of every epoch.

    Reusing the same latent codes across epochs makes the sample sheets
    directly comparable, which is the only cheap way to eyeball whether a GAN
    is still improving or has collapsed.
    """

    def __init__(self, output_dir: str | Path, num_images: int = 10, seed: int = 42) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.seed = seed
        self._latents: tf.Tensor | None = None

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Create the output directory and freeze the probe latents."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._latents = tf.random.stateless_normal(
            (self.num_images, self.model.latent_dim), seed=(self.seed, 0)
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Write one PNG per probe latent."""
        images = self.model.generator(self._latents, training=False) * 255.0
        for i, image in enumerate(images):
            keras.utils.array_to_img(image).save(
                self.output_dir / f"epoch{epoch:03d}_sample{i:02d}.png"
            )
