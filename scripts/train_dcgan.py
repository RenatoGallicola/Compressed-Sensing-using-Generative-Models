"""Train the DCGAN on MNIST and save its generator.

Only the training split is used, and the same tenth of it the VAE holds out is
held out here too, so both families see the same 54,000 images. The recovery
benchmark is scored on the test split, and Bora et al. require the generator not
to have seen it at training time.

The optimiser settings follow the only DCGAN recipe the reference paper gives
(sec. 5.2, on celebA): Adam at a learning rate of 0.0002 with ``beta_1 = 0.5``,
mini-batches of 64, and two generator updates per discriminator update. Those
are also the settings of Radford et al., whose architecture the paper adopts,
and which identify ``beta_1 = 0.9`` as a cause of unstable training.

Checkpoints are written every few epochs. ``scripts/select_dcgan.py`` then picks
one on held-out data, because a GAN has no validation loss and the last epoch is
not in any sense the best one.

Example:
    python scripts/train_dcgan.py --latent-dim 20 --epochs 50
    python scripts/select_dcgan.py --latent-dim 20
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
        "--batch-size", type=int, default=64, help="mini-batch size, 64 in Bora et al. sec. 5.2"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="Adam learning rate for G and D"
    )
    parser.add_argument(
        "--beta-1",
        type=float,
        default=0.5,
        help="Adam beta_1; 0.5 is the DCGAN value, the Keras default of 0.9 destabilises training",
    )
    parser.add_argument(
        "--generator-updates",
        type=int,
        default=2,
        help="generator updates per discriminator update",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="fraction of the training split held out, matching the VAE's data budget",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5, help="epochs between saved generators"
    )
    parser.add_argument(
        "--checkpoint-from", type=int, default=20, help="first epoch eligible to be saved"
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

    from csgm.models import (
        DCGAN,
        GANMonitor,
        GeneratorCheckpoints,
        build_discriminator,
        build_generator,
    )

    keras.utils.set_random_seed(args.seed)

    (x_train, _), (x_test, _) = load_mnist()
    if args.include_test_split:
        dataset = np.concatenate([x_train, x_test])
    else:
        # The same split the VAE uses, so the two families train on the same
        # images and the held-out tenth is available to select a checkpoint on.
        split = int(len(x_train) * (1 - args.validation_fraction))
        dataset = x_train[:split]
    print(f"training on {len(dataset)} images")

    discriminator = build_discriminator()
    generator = build_generator(args.latent_dim)
    discriminator.summary()
    generator.summary()

    gan = DCGAN(
        discriminator=discriminator,
        generator=generator,
        latent_dim=args.latent_dim,
        generator_updates=args.generator_updates,
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1),
        g_optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    sample_dir = args.sample_dir or RESULTS_DIR / "samples" / f"dcgan_dim{args.latent_dim}"
    checkpoint_dir = args.output_dir / "dcgan_checkpoints"
    history = gan.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            GANMonitor(sample_dir, num_images=10),
            GeneratorCheckpoints(
                checkpoint_dir,
                args.latent_dim,
                every=args.checkpoint_every,
                start=args.checkpoint_from,
            ),
        ],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gen_path = args.output_dir / f"gan_gen_dim{args.latent_dim}.keras"
    disc_path = args.output_dir / f"gan_disc_dim{args.latent_dim}.keras"
    generator.save(gen_path)
    discriminator.save(disc_path)
    print(f"saved {gen_path}\nsaved {disc_path}\nsamples in {sample_dir}")
    print(f"per-epoch generators in {checkpoint_dir}; run scripts/select_dcgan.py to choose one")

    losses_path = args.output_dir / f"dcgan_dim{args.latent_dim}_history.npz"
    np.savez(losses_path, **{k: np.asarray(v) for k, v in history.history.items()})
    print(f"saved {losses_path}")


if __name__ == "__main__":
    main()
