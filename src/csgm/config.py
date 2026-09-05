"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

#: Repository root (this file lives in ``<root>/src/csgm``).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

#: Directory holding the pre-trained ``.keras`` generators/decoders.
MODELS_DIR: Path = ROOT_DIR / "models"

#: Directory for benchmark tables and figures.
RESULTS_DIR: Path = ROOT_DIR / "results"

#: Directory for generated figures.
FIGURES_DIR: Path = RESULTS_DIR / "figures"

#: MNIST image shape (height, width, channels).
IMAGE_SHAPE: tuple[int, int, int] = (28, 28, 1)

#: Ambient dimension n of a flattened MNIST image.
N_PIXELS: int = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]

#: Latent dimensions the shipped checkpoints were trained with.
LATENT_DIMS: tuple[int, ...] = (20, 30)

#: Default seed, used everywhere a stream of randomness must be reproducible.
DEFAULT_SEED: int = 1337

#: Offset separating the noise stream from the measurement-matrix stream.
#: Both are drawn with ``default_rng(base + m)`` from the same kind of standard
#: normal, so without an offset the noise vector is the first ``m`` entries of
#: ``A`` rescaled, and the two are perfectly correlated rather than independent.
#: Every recovery guarantee in Bora et al. assumes ``eta`` is independent of
#: ``A``, so the streams are kept apart by construction.
NOISE_SEED_OFFSET: int = 1_000_000


def checkpoint_path(model: str, latent_dim: int) -> Path:
    """Return the path of a shipped checkpoint.

    Args:
        model: ``"vae"``, ``"fcvae"`` (the fully connected architecture of
            the reference paper) or ``"dcgan"``.
        latent_dim: Latent dimensionality (see :data:`LATENT_DIMS`).

    Returns:
        Path to the ``.keras`` file (existence is not checked).

    Raises:
        ValueError: If ``model`` is not a known model name.
    """
    names = {
        "vae": "vae_decoder_dim{k}.keras",
        "fcvae": "fc_vae_decoder_dim{k}.keras",
        "dcgan": "gan_gen_dim{k}.keras",
    }
    if model not in names:
        raise ValueError(f"unknown model {model!r}, expected one of {sorted(names)}")
    return MODELS_DIR / names[model].format(k=latent_dim)
