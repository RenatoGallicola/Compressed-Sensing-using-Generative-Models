"""Train the convolutional VAE on MNIST and save its decoder.

The decoder is the artefact that matters downstream: it is the generator ``G``
used by ``scripts/run_benchmark.py``. The encoder is saved alongside it so the
latent space can be inspected (see ``notebooks/01_vae_training.ipynb``).

Example:
    python scripts/train_vae.py --latent-dim 20 --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from csgm.config import DEFAULT_SEED, MODELS_DIR
from csgm.data import load_mnist


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latent-dim", type=int, default=20, help="latent dimensionality k")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global random seed")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=10,
        help=(
            "ramp the KL weight from zero to one over this many epochs, which "
            "counters posterior collapse. 0 trains on the plain ELBO"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="early-stopping patience on the validation ELBO (0 disables it)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODELS_DIR, help="where to write the checkpoints"
    )
    return parser.parse_args()


def main() -> None:
    """Train the VAE and write ``vae_{encoder,decoder}_dim<k>.keras``."""
    args = parse_args()

    import keras

    from csgm.models import VAE, KLWarmUp, build_decoder, build_encoder

    keras.utils.set_random_seed(args.seed)

    (x_train, _), (x_test, _) = load_mnist()
    print(f"train {x_train.shape} | test {x_test.shape}")

    encoder = build_encoder(args.latent_dim)
    decoder = build_decoder(args.latent_dim)
    encoder.summary()
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate))

    callbacks = [KLWarmUp(args.kl_warmup_epochs)]
    if args.patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            )
        )

    vae.fit(
        x_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test,),
        callbacks=callbacks,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decoder_path = args.output_dir / f"vae_decoder_dim{args.latent_dim}.keras"
    encoder_path = args.output_dir / f"vae_encoder_dim{args.latent_dim}.keras"
    decoder.save(decoder_path)
    encoder.save(encoder_path)
    print(f"saved {decoder_path}\nsaved {encoder_path}")


if __name__ == "__main__":
    main()
