"""Benchmark generative priors against Lasso across measurement budgets.

For every number of measurements ``m`` the same measurement matrix ``A`` and the
same test images are shown to every method, so the curves differ only by the
prior. Results are written as a long-format CSV (one row per method/m/image)
plus an ``.npz`` of reconstructions that ``scripts/make_figures.py`` turns into
the figures under ``results/figures/``.

The protocol mirrors Bora et al. (2017), section 6.1.1: Adam at a learning rate
of 0.01, 1000 steps, 10 random restarts, and a noise vector whose expected norm
is held at 0.1 regardless of the budget.

Example:
    python scripts/run_benchmark.py --n-images 10 --steps 1000 --restarts 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from csgm.baselines import lasso_dct_recover
from csgm.config import DEFAULT_SEED, N_PIXELS, RESULTS_DIR
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2, psnr
from csgm.recovery import RecoveryConfig, recover

DEFAULT_M_VALUES = (10, 25, 50, 75, 100, 200, 300, 400, 500, 750)
DEFAULT_METHODS = ("lasso", "vae-20", "vae-30", "dcgan-20", "dcgan-30")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="'lasso' or '<vae|dcgan>-<latent dim>' entries",
    )
    parser.add_argument(
        "--m-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_M_VALUES),
        help="measurement budgets to sweep",
    )
    parser.add_argument("--n-images", type=int, default=10, help="test images averaged over")
    parser.add_argument("--steps", type=int, default=1000, help="Adam steps per restart")
    parser.add_argument("--restarts", type=int, default=10, help="random restarts per image")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam step size on z")
    parser.add_argument(
        "--l2-penalty", type=float, default=0.0, help="weight of the ||z||^2 prior term"
    )
    parser.add_argument(
        "--noise-norm",
        type=float,
        default=0.1,
        help=(
            "expected noise magnitude sqrt(E||eta||^2), held constant across budgets "
            "as in Bora et al.; the per-component sigma is this divided by sqrt(m)"
        ),
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=None,
        help="fixed per-component noise sigma, overriding --noise-norm",
    )
    parser.add_argument("--lasso-alpha", type=float, default=1e-5, help="Lasso L1 strength")
    parser.add_argument(
        "--image-batch-size", type=int, default=10, help="images optimised in one pass"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR, help="where to write csv/npz"
    )
    return parser.parse_args()


def parse_method(method: str) -> tuple[str, int | None]:
    """Split a method string into ``(family, latent_dim)``.

    Args:
        method: ``"lasso"`` or e.g. ``"dcgan-20"``.

    Returns:
        ``("lasso", None)`` or ``("dcgan", 20)``.

    Raises:
        ValueError: If the string is not a recognised method.
    """
    if method == "lasso":
        return "lasso", None
    family, _, dim = method.partition("-")
    if family not in {"vae", "dcgan"} or not dim.isdigit():
        raise ValueError(f"unrecognised method {method!r}")
    return family, int(dim)


def main() -> None:
    """Run the sweep and write ``benchmark.csv`` / ``reconstructions.npz``."""
    args = parse_args()

    from csgm.models import load_generator

    (_, _), (x_test, y_test) = load_mnist(flatten=False)
    images = sample_images(
        x_test, args.n_images, labels=y_test, stratified=True, seed=args.seed
    ).reshape(args.n_images, N_PIXELS)
    print(f"benchmarking {args.methods} on {args.n_images} images, m in {args.m_values}")

    # Load every generator once: reloading per m dominates the runtime otherwise.
    generators = {
        method: load_generator(*parse_method(method))
        for method in args.methods
        if method != "lasso"
    }

    rows: list[dict] = []
    reconstructions: dict[str, np.ndarray] = {"ground_truth": images}

    for m in args.m_values:
        # One matrix per m, shared by every method, derived deterministically
        # from the global seed so the whole sweep is reproducible.
        A = gaussian_measurement_matrix(m, N_PIXELS, seed=args.seed + m)
        # A fixed per-component sigma would make the total noise grow as sqrt(m);
        # scaling by 1/sqrt(m) keeps ||eta|| constant, so budgets stay comparable.
        noise_std = args.noise_std if args.noise_std is not None else args.noise_norm / np.sqrt(m)
        y = measure(images, A, noise_std=noise_std, seed=args.seed + m)

        for method in args.methods:
            family, latent_dim = parse_method(method)
            start = time.perf_counter()

            if family == "lasso":
                x_hat = lasso_dct_recover(y, A, alpha=args.lasso_alpha)
                residual = np.linalg.norm(x_hat @ A.T - y, axis=1)
            else:
                result = recover(
                    generators[method],
                    y,
                    A,
                    latent_dim,
                    RecoveryConfig(
                        steps=args.steps,
                        restarts=args.restarts,
                        learning_rate=args.learning_rate,
                        l2_penalty=args.l2_penalty,
                        image_batch_size=args.image_batch_size,
                        seed=args.seed,
                    ),
                )
                x_hat, residual = result.x_hat, result.residual

            elapsed = time.perf_counter() - start
            errors = per_pixel_l2(x_hat, images)
            reconstructions[f"{method}__m{m}"] = x_hat.astype("float32")

            for i, (err, res) in enumerate(zip(errors, residual, strict=True)):
                rows.append(
                    {
                        "method": method,
                        "family": family,
                        "latent_dim": latent_dim,
                        "m": m,
                        "image": i,
                        "per_pixel_error": float(err),
                        "psnr_db": float(psnr(x_hat[i], images[i])[0]),
                        "measurement_residual": float(res),
                        "seconds_per_batch": elapsed,
                    }
                )
            print(
                f"m={m:>4} | {method:<9} | error/pixel {errors.mean():.5f} "
                f"| PSNR {psnr(x_hat, images).mean():5.2f} dB | {elapsed:6.1f}s"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "benchmark.csv"
    npz_path = args.output_dir / "reconstructions.npz"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **reconstructions)
    print(f"\nwrote {csv_path}\nwrote {npz_path}")


if __name__ == "__main__":
    main()
