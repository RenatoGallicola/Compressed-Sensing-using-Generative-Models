"""Choose one DCGAN generator out of the checkpoints a training run produced.

A GAN has no validation loss. Training therefore has no stopping criterion, and
sample quality oscillates from epoch to epoch, so keeping whatever the final
epoch produced is a choice made by the schedule rather than by any measurement.
The VAE side of this project selects its checkpoint on a held-out split; without
something equivalent the two families are not comparable, and any difference
between them could just as well be the epoch the run happened to stop at.

The criterion is representation error: the lowest per-pixel error reachable
inside the range of the generator, obtained by solving the recovery problem with
``A = I``. It is the quantity the reference paper uses to characterise a
generator, it is the floor every recovery result sits above, and it needs no
discriminator. It is measured on images from the tenth of the training split
that training held out, never on the test split the benchmark scores, so the
choice of generator cannot be informed by the numbers it is later judged on.

Example:
    python scripts/select_dcgan.py --latent-dim 20
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from csgm.config import DEFAULT_SEED, MODELS_DIR, N_PIXELS, checkpoint_path
from csgm.data import load_mnist, sample_images
from csgm.metrics import per_pixel_l2
from csgm.recovery import RecoveryConfig, recover


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latent-dim", type=int, default=20, help="latent dimensionality k")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="directory of per-epoch generators (default: models/dcgan_checkpoints)",
    )
    parser.add_argument(
        "--n-images", type=int, default=32, help="held-out images the criterion averages over"
    )
    parser.add_argument("--steps", type=int, default=500, help="Adam steps per restart")
    parser.add_argument("--restarts", type=int, default=3, help="random restarts per image")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="the tail of the training split held out, matching train_dcgan.py",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="score the checkpoints without installing the winner",
    )
    return parser.parse_args()


def held_out_images(fraction: float, n_images: int, seed: int) -> np.ndarray:
    """Return ``n_images`` from the tail of the training split.

    Args:
        fraction: Fraction of the training split held out during training.
        n_images: How many images to draw.
        seed: Seed of the stratified draw.

    Returns:
        Array of shape ``(n_images, N_PIXELS)``.
    """
    (x_train, y_train), _ = load_mnist(flatten=False)
    split = int(len(x_train) * (1 - fraction))
    x_val, y_val = x_train[split:], y_train[split:]
    return sample_images(x_val, n_images, labels=y_val, stratified=True, seed=seed).reshape(
        n_images, N_PIXELS
    )


def main() -> None:
    """Score every checkpoint and install the best generator."""
    args = parse_args()

    from csgm.models import load_generator

    directory = args.checkpoint_dir or MODELS_DIR / "dcgan_checkpoints"
    checkpoints = sorted(directory.glob(f"gan_gen_dim{args.latent_dim}_epoch*.keras"))
    if not checkpoints:
        raise SystemExit(
            f"no checkpoints for k={args.latent_dim} in {directory}; "
            "train one with 'python scripts/train_dcgan.py'"
        )

    images = held_out_images(args.validation_fraction, args.n_images, args.seed)
    print(
        f"scoring {len(checkpoints)} checkpoints for k={args.latent_dim} on "
        f"{args.n_images} held-out training images"
    )

    # A = I turns the recovery problem into a projection onto the range of the
    # generator, which is exactly the representation error.
    identity = np.eye(N_PIXELS, dtype="float32")
    config = RecoveryConfig(
        steps=args.steps,
        restarts=args.restarts,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    scores: dict[Path, float] = {}
    for path in checkpoints:
        generator = load_generator("dcgan", args.latent_dim, path=path)
        result = recover(generator, images, identity, args.latent_dim, config)
        scores[path] = float(per_pixel_l2(result.x_hat, images).mean())
        print(f"  {path.name}  representation error {scores[path]:.5f}")

    best = min(scores, key=scores.get)
    print(f"\nselected {best.name} at {scores[best]:.5f}")

    report = [
        f"k={args.latent_dim}, selected on representation error over {args.n_images} "
        f"held-out training images ({args.restarts} restarts, {args.steps} steps)",
        "",
        *(f"  {p.name}: {s:.5f}" for p, s in scores.items()),
        "",
        f"selected: {best.name}",
    ]
    (MODELS_DIR / f"dcgan_selection_dim{args.latent_dim}.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    if args.dry_run:
        print("dry run, the checkpoint in models/ is unchanged")
        return

    destination = checkpoint_path("dcgan", args.latent_dim)
    shutil.copy(best, destination)
    print(f"installed {destination}")


if __name__ == "__main__":
    main()
