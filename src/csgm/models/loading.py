"""Loading the shipped generator checkpoints."""

from __future__ import annotations

from pathlib import Path

from csgm.config import checkpoint_path


def load_generator(model: str, latent_dim: int, *, path: str | Path | None = None):
    """Load a pre-trained generator and freeze it.

    Recovery optimises the latent code only, so the weights are marked
    non-trainable: it makes the intent explicit and keeps a stray gradient
    update from silently modifying the prior.

    Args:
        model: ``"vae"`` or ``"fcvae"`` (loads the decoder) or ``"dcgan"``
            (loads the generator).
        latent_dim: Latent dimensionality of the checkpoint.
        path: Explicit checkpoint path, overriding the ``models/`` lookup.

    Returns:
        A frozen ``keras.Model`` mapping ``(batch, latent_dim)`` to images.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    import keras

    checkpoint = Path(path) if path is not None else checkpoint_path(model, latent_dim)
    if not checkpoint.exists():
        script = "train_vae.py" if "vae" in model else "train_dcgan.py"
        architecture = " --architecture fc" if model == "fcvae" else ""
        raise FileNotFoundError(
            f"checkpoint {checkpoint} not found -- train one with "
            f"'python scripts/{script} --latent-dim {latent_dim}{architecture}'"
        )

    generator = keras.models.load_model(checkpoint, compile=False)
    generator.trainable = False
    return generator
