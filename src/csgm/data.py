"""MNIST loading and image sampling helpers."""

from __future__ import annotations

import numpy as np

from csgm.config import DEFAULT_SEED, IMAGE_SHAPE, N_PIXELS


def load_mnist(
    *, flatten: bool = False, normalize: bool = True
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load MNIST through Keras and reshape it for this project.

    Args:
        flatten: If ``True`` return images as ``(N, 784)`` vectors, otherwise as
            ``(N, 28, 28, 1)`` tensors.
        normalize: If ``True`` scale pixels from ``[0, 255]`` to ``[0, 1]``.
            Every model in this repository is trained on ``[0, 1]`` inputs and
            ends in a sigmoid, so this should almost always stay ``True``.

    Returns:
        ``((x_train, y_train), (x_test, y_test))`` with ``float32`` images.
    """
    import keras  # imported lazily: keeps ``import csgm`` cheap

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    def prepare(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        if normalize:
            x = x / 255.0
        return x.reshape(-1, N_PIXELS) if flatten else x.reshape(-1, *IMAGE_SHAPE)

    return (prepare(x_train), y_train), (prepare(x_test), y_test)


def sample_images(
    x: np.ndarray,
    n: int,
    *,
    labels: np.ndarray | None = None,
    stratified: bool = False,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Draw ``n`` images without replacement.

    Args:
        x: Image array, first axis is the sample axis.
        n: Number of images to draw.
        labels: Class labels, required when ``stratified`` is ``True``.
        stratified: If ``True``, spread the draw as evenly as possible over the
            classes present in ``labels``. Benchmarks average over a handful of
            images only, so an unstratified draw can easily miss whole digits.
        seed: Seed of the local random generator.

    Returns:
        Array of shape ``(n, *x.shape[1:])``.

    Raises:
        ValueError: If ``n`` exceeds the number of available images, or if
            ``stratified`` is requested without ``labels``.
    """
    if n > len(x):
        raise ValueError(f"cannot draw {n} images out of {len(x)}")
    rng = np.random.default_rng(seed)

    if not stratified:
        return x[rng.choice(len(x), size=n, replace=False)]

    if labels is None:
        raise ValueError("stratified sampling requires `labels`")
    classes = np.unique(labels)
    per_class = np.full(len(classes), n // len(classes))
    per_class[: n % len(classes)] += 1

    picked: list[int] = []
    for cls, count in zip(classes, per_class, strict=True):
        if count == 0:
            continue
        pool = np.flatnonzero(labels == cls)
        picked.extend(rng.choice(pool, size=min(count, len(pool)), replace=False))
    return x[np.array(sorted(picked))]
