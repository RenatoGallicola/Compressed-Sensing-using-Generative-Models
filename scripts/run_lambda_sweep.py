"""Measure the effect of the latent regulariser on recovery quality.

Bora et al. minimise ``||A G(z) - y||^2 + lambda ||z||^2``. The second term keeps
the solution inside the region where the prior places most of its mass. The main
benchmark uses the value they recommend, 0.1; this script sweeps it to show how
much that choice matters.

Only the VAE decoders are swept: they invert in about twenty seconds per
configuration, against roughly thirteen minutes for a DCGAN generator, so the
same sweep over both families would cost hours rather than minutes.

Example:
    python scripts/run_lambda_sweep.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from csgm.config import DEFAULT_SEED, N_PIXELS, NOISE_SEED_OFFSET, RESULTS_DIR
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2
from csgm.recovery import RecoveryConfig, recover

DEFAULT_LAMBDAS = (0.0, 0.01, 0.1, 1.0)
DEFAULT_M_VALUES = (10, 25, 50, 75, 100, 200, 300, 400, 500, 750)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["vae-20", "vae-30"])
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--m-values", nargs="+", type=int, default=list(DEFAULT_M_VALUES))
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--noise-norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sweep-seed",
        type=int,
        default=DEFAULT_SEED + 2,
        help=(
            "seed for the whole sweep draw: images, measurement matrices and noise. "
            "Kept different from the benchmark's so that the penalty is not chosen "
            "on the images the benchmark reports"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the sweep and write ``lambda_sweep.csv``."""
    args = parse_args()

    from csgm.models import load_generator

    (_, _), (x_test, y_test) = load_mnist()
    images = sample_images(
        x_test, args.n_images, labels=y_test, stratified=True, seed=args.sweep_seed
    ).reshape(args.n_images, N_PIXELS)

    generators = {}
    for method in args.methods:
        family, dim = method.split("-")
        generators[method] = (load_generator(family, int(dim)), int(dim))

    rows: list[dict] = []
    for m in args.m_values:
        # Same derivation as run_benchmark.py, so the lambda = 0 column is
        # directly comparable with the main benchmark.
        A = gaussian_measurement_matrix(m, N_PIXELS, seed=args.sweep_seed + m)
        y = measure(
            images,
            A,
            noise_std=args.noise_norm / np.sqrt(m),
            seed=args.sweep_seed + m + NOISE_SEED_OFFSET,
        )

        for method, (generator, latent_dim) in generators.items():
            for penalty in args.lambdas:
                start = time.perf_counter()
                result = recover(
                    generator,
                    y,
                    A,
                    latent_dim,
                    RecoveryConfig(
                        steps=args.steps,
                        restarts=args.restarts,
                        learning_rate=args.learning_rate,
                        l2_penalty=penalty,
                        image_batch_size=args.n_images,
                        seed=args.seed,
                    ),
                )
                errors = per_pixel_l2(result.x_hat, images)
                latent_norm = np.linalg.norm(result.z, axis=1)
                for i, err in enumerate(errors):
                    rows.append(
                        {
                            "method": method,
                            "m": m,
                            "l2_penalty": penalty,
                            "image": i,
                            "per_pixel_error": float(err),
                            "latent_norm": float(latent_norm[i]),
                        }
                    )
                print(
                    f"m={m:>4} | {method:<7} | lambda {penalty:<6} | "
                    f"error/pixel {errors.mean():.5f} | ||z|| {latent_norm.mean():5.2f} | "
                    f"{time.perf_counter() - start:5.1f}s"
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "lambda_sweep.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
