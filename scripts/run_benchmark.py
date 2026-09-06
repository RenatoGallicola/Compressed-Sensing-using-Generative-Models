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
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from csgm.baselines import lasso_recover
from csgm.config import (
    DEFAULT_SEED,
    N_PIXELS,
    NOISE_SEED_OFFSET,
    RESULTS_DIR,
    ROOT_DIR,
    checkpoint_path,
)
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2, psnr
from csgm.recovery import RecoveryConfig, recover

DEFAULT_M_VALUES = (10, 25, 50, 75, 100, 200, 300, 400, 500, 750)
DEFAULT_METHODS = ("lasso", "lasso-dct", "vae-20", "vae-30", "fcvae-20", "dcgan-20", "dcgan-30")


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
        "--l2-penalty",
        type=float,
        default=0.1,
        help=(
            "weight of the ||z||^2 prior term. The default is the value Bora et al. "
            "report as best on MNIST and the one the published table uses; pass 0 to "
            "reproduce the unregularised sweep"
        ),
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
    parser.add_argument(
        "--lasso-alpha",
        type=float,
        default=None,
        help=(
            "fixed Lasso L1 strength. By default the per-budget values chosen by "
            "scripts/tune_lasso.py are read from results/lasso_alpha.json, which gives "
            "the baseline its best configuration at every budget"
        ),
    )
    parser.add_argument(
        "--image-batch-size", type=int, default=10, help="images optimised in one pass"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="global random seed")
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR, help="where to write csv/npz"
    )
    parser.add_argument(
        "--allow-stale-checkpoints",
        action="store_true",
        help=(
            "permit --merge to keep rows produced by a different build of a generator. "
            "The result mixes model generations under one label, so it is off by default"
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "update an existing benchmark in place: rows and reconstructions for the "
            "methods being run are replaced, everything else is kept. Refuses to run "
            "if the recorded protocol differs, so merged results stay comparable"
        ),
    )
    return parser.parse_args()


def git_revision() -> str:
    """Return the short revision this run was produced at, or ``"unknown"``."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fingerprint(method: str) -> str:
    """Return a short hash of the checkpoint a method recovers with.

    Two runs can agree on every command-line setting and still be built on
    different generators, which is exactly what merging must not silently pool.
    Baselines have no checkpoint and are identified by their basis instead.
    """
    family, latent_dim = parse_method(method)
    if family.startswith("lasso"):
        return family
    path = checkpoint_path(family, latent_dim)
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def protocol(args) -> dict:
    """The settings that have to match for two runs to be poolable."""
    return {
        "n_images": args.n_images,
        "steps": args.steps,
        "restarts": args.restarts,
        "learning_rate": args.learning_rate,
        "l2_penalty": args.l2_penalty,
        "noise_norm": args.noise_norm,
        "noise_std": args.noise_std,
        "lasso_alpha": args.lasso_alpha if args.lasso_alpha is not None else "per budget",
        # The resolved table, not just the word "per budget": two runs against
        # different tuning outputs would otherwise look identical here.
        "lasso_alpha_table": args.resolved_alpha,
        "seed": args.seed,
        "m_values": sorted(args.m_values),
    }


def parse_method(method: str) -> tuple[str, int | None]:
    """Split a method string into ``(family, latent_dim)``.

    Args:
        method: ``"lasso"``, ``"lasso-dct"``, or e.g. ``"dcgan-20"``.

    Returns:
        ``("lasso", None)`` or ``("dcgan", 20)``.

    Raises:
        ValueError: If the string is not a recognised method.
    """
    if method in {"lasso", "lasso-dct"}:
        return method, None
    family, _, dim = method.partition("-")
    if family not in {"vae", "fcvae", "dcgan"} or not dim.isdigit():
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
        if not method.startswith("lasso")
    }

    alpha_path = args.output_dir / "lasso_alpha.json"
    tuned_alpha = {}
    if args.lasso_alpha is None:
        source = alpha_path if alpha_path.exists() else RESULTS_DIR / "lasso_alpha.json"
        if not source.exists():
            raise SystemExit(
                f"no shrinkage table at {source}; run scripts/tune_lasso.py first "
                "or pass --lasso-alpha"
            )
        tuned_alpha = json.loads(source.read_text(encoding="utf-8"))
        print(f"using the per-budget shrinkage from {source}")
    args.resolved_alpha = tuned_alpha or None

    revision = git_revision()
    fingerprints = {method: fingerprint(method) for method in args.methods}
    print("checkpoints: " + ", ".join(f"{k}={v}" for k, v in fingerprints.items()))

    rows: list[dict] = []
    reconstructions: dict[str, np.ndarray] = {"ground_truth": images}

    for m in args.m_values:
        # One matrix per m, shared by every method, derived deterministically
        # from the global seed so the whole sweep is reproducible.
        A = gaussian_measurement_matrix(m, N_PIXELS, seed=args.seed + m)
        # A fixed per-component sigma would make the total noise grow as sqrt(m);
        # scaling by 1/sqrt(m) keeps ||eta|| constant, so budgets stay comparable.
        noise_std = args.noise_std if args.noise_std is not None else args.noise_norm / np.sqrt(m)
        # The offset matters: drawing the noise from the same seed as A makes it
        # a rescaled copy of A's first rows rather than an independent draw.
        y = measure(images, A, noise_std=noise_std, seed=args.seed + m + NOISE_SEED_OFFSET)

        for method in args.methods:
            family, latent_dim = parse_method(method)
            start = time.perf_counter()

            if family.startswith("lasso"):
                basis = "dct" if family == "lasso-dct" else "pixel"
                alpha = (
                    args.lasso_alpha if args.lasso_alpha is not None else tuned_alpha[basis][str(m)]
                )
                x_hat = lasso_recover(y, A, basis=basis, alpha=alpha)
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
                        "checkpoint": fingerprints[method],
                        "revision": revision,
                    }
                )
            print(
                f"m={m:>4} | {method:<9} | error/pixel {errors.mean():.5f} "
                f"| PSNR {psnr(x_hat, images).mean():5.2f} dB | {elapsed:6.1f}s"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "benchmark.csv"
    npz_path = args.output_dir / "reconstructions.npz"
    meta_path = args.output_dir / "benchmark_meta.json"
    table = pd.DataFrame(rows)

    if args.merge and csv_path.exists():
        if not meta_path.exists():
            raise SystemExit(
                f"refusing to merge, {csv_path.name} has no {meta_path.name} beside it, "
                "so there is nothing to check the protocol against"
            )
        recorded = json.loads(meta_path.read_text(encoding="utf-8"))["protocol"]
        current = protocol(args)
        differing = {k: (recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v}
        if differing:
            raise SystemExit(f"refusing to merge, the protocol differs: {differing}")

        previous = pd.read_csv(csv_path)
        if "checkpoint" in previous.columns:
            # Two runs can agree on every command-line setting and still rest on
            # different generators, which is exactly what a merged table must not
            # hide. Rows that are kept, rather than recomputed, have to have been
            # produced by the checkpoints still on disk.
            kept = previous[~previous["method"].isin(args.methods)]
            current = {method: fingerprint(method) for method in set(kept["method"])}
            stale = {
                method: (sorted(set(group["checkpoint"])), current[method])
                for method, group in kept.groupby("method")
                if set(group["checkpoint"]) != {current[method]}
            }
            if stale and not args.allow_stale_checkpoints:
                raise SystemExit(
                    "refusing to merge: these rows were produced by a different "
                    f"build of their generator, {stale}. Recompute them, or pass "
                    "--allow-stale-checkpoints if mixing them is deliberate"
                )
            if stale:
                print(f"warning: keeping rows from an earlier build of {sorted(stale)}")
        table = pd.concat([previous[~previous["method"].isin(args.methods)], table])
        table = table.sort_values(["method", "m", "image"]).reset_index(drop=True)

        with np.load(npz_path) as archive:
            merged = dict(archive)
        merged.update(reconstructions)
        reconstructions = merged
        print(f"merged {args.methods} into the existing benchmark")

    table.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, **reconstructions)
    meta_path.write_text(
        json.dumps(
            {
                "protocol": protocol(args),
                "methods": sorted(set(table["method"])),
                "git_revision": revision,
                "checkpoints": fingerprints,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {csv_path}\nwrote {npz_path}\nwrote {meta_path}")


if __name__ == "__main__":
    main()
