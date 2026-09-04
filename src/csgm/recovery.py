"""Compressed sensing recovery with a generative prior.

Given a trained generator ``G : R^k -> R^n`` and measurements ``y = A x* + eta``,
we look for the latent code whose image best explains the measurements:

    z_hat = argmin_z  ||A G(z) - y||^2 + lambda ||z||^2 ,
    x_hat = G(z_hat).

The objective is non-convex, so it is optimised with Adam from several random
restarts and the restart with the lowest measurement residual is kept. All
restarts (and all images of a batch) are optimised *simultaneously* as one
stacked batch of latent codes: the per-item losses are independent, so summing
them gives exactly the same gradients as optimising each one separately, at a
fraction of the wall-clock cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from csgm.config import DEFAULT_SEED


@dataclass(frozen=True)
class RecoveryConfig:
    """Hyper-parameters of the latent-space optimisation.

    Attributes:
        steps: Number of Adam updates per restart.
        restarts: Number of random initialisations; the best one is kept.
        learning_rate: Adam learning rate on ``z``.
        z_init_std: Standard deviation of the initial ``z``. Defaults to ``1.0``
            to match the ``N(0, I)`` prior the generators were trained under --
            initialising much closer to the origin biases the search towards the
            (blurry) centre of the latent space.
        l2_penalty: Weight ``lambda`` of the ``||z||^2`` term that keeps the
            solution inside the high-density region of the prior. ``0`` disables it.
        seed: Seed for the latent initialisation.
        image_batch_size: Images optimised in one pass. ``restarts`` copies of
            each are stacked, so the effective generator batch is
            ``image_batch_size * restarts``; lower it if memory is tight.
        log_every: Print the mean objective every ``log_every`` steps
            (``0`` silences logging).
    """

    steps: int = 1000
    restarts: int = 10
    learning_rate: float = 0.01
    z_init_std: float = 1.0
    l2_penalty: float = 0.0
    seed: int = DEFAULT_SEED
    image_batch_size: int = 16
    log_every: int = 0


@dataclass
class RecoveryResult:
    """Outcome of :func:`recover`.

    Attributes:
        x_hat: Reconstructions, shape ``(batch, n)``.
        z: Winning latent codes, shape ``(batch, k)``.
        residual: Measurement residual ``||A G(z) - y||_2`` per image, shape
            ``(batch,)``, excluding any latent penalty. It is the quantity the
            restarts are ranked on, and an optimisation diagnostic: judge
            quality with :mod:`csgm.metrics` against the ground truth.
        history: Mean objective across the stacked batch at every step, shape
            ``(steps,)``. Empty unless ``track_history=True``.
    """

    x_hat: np.ndarray
    z: np.ndarray
    residual: np.ndarray
    history: np.ndarray = field(default_factory=lambda: np.empty(0))


def _optimise(generator, z, y_rep, A_t, config: RecoveryConfig, n: int, history: list | None):
    """Run Adam on a stacked batch of latent codes and return their final losses.

    Args:
        generator: Frozen generator being inverted.
        z: ``tf.Variable`` of shape ``(items, latent_dim)`` holding every
            (image, restart) pair.
        y_rep: Measurements repeated to match ``z``, shape ``(items, m)``.
        A_t: Transposed measurement matrix, shape ``(n, m)``.
        config: Optimisation hyper-parameters.
        n: Ambient dimension, used to flatten the generator output.
        history: If not ``None``, the mean objective is appended at every step.

    Returns:
        Two arrays of shape ``(items,)``: the squared measurement error, and the
        full objective that was minimised. They differ only when
        ``l2_penalty`` is non-zero.
    """
    import tensorflow as tf

    items = int(z.shape[0])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    @tf.function(reduce_retracing=True)
    def step() -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            generated = tf.reshape(generator(z, training=False), (items, n))
            measurement = tf.reduce_sum((generated @ A_t - y_rep) ** 2, axis=1)
            objective = measurement
            if config.l2_penalty:
                objective = measurement + config.l2_penalty * tf.reduce_sum(z**2, axis=1)
            total = tf.reduce_sum(objective)
        optimizer.apply_gradients([(tape.gradient(total, z), z)])
        return measurement, objective

    measurement = objective = None
    for i in range(config.steps):
        measurement, objective = step()
        if history is not None:
            history.append(float(tf.reduce_mean(objective)))
        if config.log_every and i % config.log_every == 0:
            print(f"  step {i:>5} | mean objective {float(tf.reduce_mean(objective)):.5f}")
    return np.asarray(measurement), np.asarray(objective)


def recover(
    generator,
    y: np.ndarray,
    A: np.ndarray,
    latent_dim: int,
    config: RecoveryConfig | None = None,
    *,
    track_history: bool = False,
) -> RecoveryResult:
    """Recover signals from compressed measurements using a generative prior.

    Args:
        generator: Callable mapping ``(batch, latent_dim)`` to images; any Keras
            model whose output flattens to ``n`` values works (the shipped
            DCGAN generators and VAE decoders both output ``(batch, 28, 28, 1)``).
            Its weights are never updated.
        y: Measurements of shape ``(m,)`` or ``(batch, m)``.
        A: Measurement matrix of shape ``(m, n)``.
        latent_dim: Dimensionality ``k`` of the generator input.
        config: Optimisation hyper-parameters; defaults to ``RecoveryConfig()``.
        track_history: If ``True``, record the mean objective at every step.
            Only meaningful for a single ``image_batch_size`` group.

    Returns:
        A :class:`RecoveryResult` whose arrays are ordered like ``y``.

    Raises:
        ValueError: If the shapes of ``y`` and ``A`` are inconsistent.
    """
    import tensorflow as tf

    config = config or RecoveryConfig()
    y = np.atleast_2d(np.asarray(y, dtype="float32"))
    A = np.asarray(A, dtype="float32")
    if y.shape[1] != A.shape[0]:
        raise ValueError(f"y has {y.shape[1]} measurements but A has {A.shape[0]} rows")

    n_images, n = len(y), A.shape[1]
    A_t = tf.constant(A.T)  # (n, m): applied as G(z) @ A_t

    x_hat = np.empty((n_images, n), dtype="float32")
    z_best = np.empty((n_images, latent_dim), dtype="float32")
    residual = np.empty(n_images, dtype="float32")
    history: list[float] = []

    rng = np.random.default_rng(config.seed)
    r = config.restarts

    for start in range(0, n_images, config.image_batch_size):
        stop = min(start + config.image_batch_size, n_images)
        b = stop - start

        # One latent per (image, restart) pair, laid out as [img0 x r, img1 x r, ...].
        z = tf.Variable(
            rng.standard_normal((b * r, latent_dim)).astype("float32") * config.z_init_std
        )
        y_rep = tf.constant(np.repeat(y[start:stop], r, axis=0))

        measurement, _ = _optimise(
            generator, z, y_rep, A_t, config, n, history if track_history else None
        )
        losses = measurement.reshape(b, r)

        # Keep, for each image, the restart with the smallest measurement error.
        # Bora et al. select this way, and it is the only choice that stays
        # available when the ground truth is unknown. With a non-zero penalty the
        # full objective would also reward a small ||z||, which is not what the
        # reconstruction is judged on.
        winners = losses.argmin(axis=1)
        z_np = np.asarray(z)[np.arange(b) * r + winners]

        z_best[start:stop] = z_np
        reconstructions = np.asarray(generator(z_np, training=False)).reshape(b, n)
        x_hat[start:stop] = reconstructions
        # Measured on the reconstruction being returned, rather than reusing the
        # value from inside the loop, which predates the final Adam update.
        residual[start:stop] = np.linalg.norm(reconstructions @ A.T - y[start:stop], axis=1)

    return RecoveryResult(
        x_hat=x_hat,
        z=z_best,
        residual=residual,
        history=np.asarray(history, dtype="float32"),
    )
