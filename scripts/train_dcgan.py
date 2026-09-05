"""Train the DCGAN on MNIST and save its generator.

Only the training split is used. The recovery benchmark is scored on images
from the test split, and Bora et al. require the generator to have been trained
without seeing them, so folding the test images in would leak the evaluation set
into the prior.

Example:
    python scripts/train_dcgan.py --latent-dim 20 --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from csgm.config import DEFAULT_SEED, MODELS_DIR, RESULTS_DIR
from csgm.data import load_mnist


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latent-dim", type=int, default=20, help="latent dimensionality k")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="mini-batch size, 32 being the value the shipped checkpoints were trained with",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Adam learning rate for G and D"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global random seed")
    parser.add_argument(
        "--include-test-split",
        action="store_true",
        help=(
            "also train on the test images, as the original run of this project did. "
            "It leaks the evaluation set into the prior and is kept only to reproduce "
            "the shipped checkpoints"
        ),
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=None,
        help="directory for per-epoch sample sheets (default: results/samples/dcgan_dim<k>)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODELS_DIR, help="where to write the checkpoints"
    )
    return parser.parse_args()


def main() -> None:
    """Train the DCGAN and write ``gan_{gen,disc}_dim<k>.keras``."""
    args = parse_args()

    import keras

    from csgm.models import DCGAN, GANMonitor, build_discriminator, build_generator

    keras.utils.set_random_seed(args.seed)

    (x_train, _), (x_test, _) = load_mnist()
    dataset = np.concatenate([x_train, x_test]) if args.include_test_split else x_train
    print(f"training on {len(dataset)} images")

    discriminator = build_discriminator()
    generator = build_generator(args.latent_dim)
    discriminator.summary()
    generator.summary()

    gan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=args.latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        g_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    sample_dir = args.sample_dir or RESULTS_DIR / "samples" / f"dcgan_dim{args.latent_dim}"
    history = gan.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[GANMonitor(sample_dir, num_images=10)],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gen_path = args.output_dir / f"gan_gen_dim{args.latent_dim}.keras"
    disc_path = args.output_dir / f"gan_disc_dim{args.latent_dim}.keras"
    generator.save(gen_path)
    discriminator.save(disc_path)
    print(f"saved {gen_path}\nsaved {disc_path}\nsamples in {sample_dir}")

    losses_path = args.output_dir / f"dcgan_dim{args.latent_dim}_history.npz"
    np.savez(losses_path, **{k: np.asarray(v) for k, v in history.history.items()})
    print(f"saved {losses_path}")


if __name__ == "__main__":
    main()
