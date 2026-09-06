"""Choose the shrinkage of the Lasso baseline, in both bases.

The baseline should be given its best configuration: a comparison against a
badly tuned baseline says nothing. Bora et al. quote a shrinkage of 0.1, but
scikit-learn scales its objective by ``1 / (2 m)``, so that number does not
transfer, and the value is swept here instead.

The shrinkage is chosen separately for each basis and each measurement budget.
That is the most favourable configuration the baseline can be given, which is
the point: a comparison that only holds against a badly configured baseline is
worth nothing. A single global value would have been convenient but costs the
baseline up to a factor of two at some budgets.

The sweep runs on its own draw: ten images the benchmark never scores, and its
own measurement matrices and noise. Tuning under the very matrix and noise
realisation the baseline is then evaluated under would select a shrinkage that
suits that particular draw, which is the same kind of leak as tuning on the
evaluation images, only milder. The result is written to ``lasso_alpha.json``,
which ``run_benchmark.py`` reads.

Example:
    python scripts/tune_lasso.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from csgm.baselines import lasso_recover
from csgm.config import DEFAULT_SEED, N_PIXELS, NOISE_SEED_OFFSET, RESULTS_DIR
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2

#: The grid stops at 1e-1 on purpose. Above it the solution collapses to the
#: all-zero image, which scores 0.1178 per pixel on these digits and would be
#: selected as "best" at the budgets where the baseline recovers nothing. That is
#: a degenerate configuration rather than a well tuned one, so the ceiling is kept
#: and the budgets where the baseline sits at that level are reported instead.
DEFAULT_ALPHAS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_M_VALUES = (10, 25, 50, 75, 100, 200, 300, 400, 500, 750)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--bases", nargs="+", default=["pixel", "dct"])
    parser.add_argument("--m-values", nargs="+", type=int, default=list(DEFAULT_M_VALUES))
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--noise-norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tuning-seed",
        type=int,
        default=DEFAULT_SEED + 1,
        help=(
            "seed for the whole tuning draw: images, measurement matrices and noise. "
            "Kept different from the benchmark's"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Sweep the shrinkage and report the best value for each basis."""
    args = parse_args()

    (_, _), (x_test, y_test) = load_mnist()
    images = sample_images(
        x_test, args.n_images, labels=y_test, stratified=True, seed=args.tuning_seed
    ).reshape(args.n_images, N_PIXELS)

    # Disjointness is the whole point of the separate seed, so it is checked
    # rather than assumed: a stratified draw could in principle repeat an image.
    scored = sample_images(
        x_test, args.n_images, labels=y_test, stratified=True, seed=args.seed
    ).reshape(args.n_images, N_PIXELS)
    shared = sum(any(np.array_equal(a, b) for b in scored) for a in images)
    if shared:
        raise SystemExit(
            f"{shared} of the {args.n_images} tuning images are also scored by the "
            f"benchmark; pick a --tuning-seed other than {args.tuning_seed}"
        )
    print(
        f"tuning on {args.n_images} images drawn with seed {args.tuning_seed}, "
        f"disjoint from the {args.n_images} the benchmark scores with seed {args.seed}"
    )

    rows = []
    for m in args.m_values:
        A = gaussian_measurement_matrix(m, N_PIXELS, seed=args.tuning_seed + m)
        y = measure(
            images,
            A,
            noise_std=args.noise_norm / np.sqrt(m),
            seed=args.tuning_seed + m + NOISE_SEED_OFFSET,
        )
        for basis in args.bases:
            for alpha in args.alphas:
                x_hat = lasso_recover(y, A, basis=basis, alpha=alpha)
                errors = per_pixel_l2(x_hat, images)
                rows.append({"basis": basis, "alpha": alpha, "m": m, "error": errors.mean()})
                print(
                    f"m={m:>4} | {basis:<5} | alpha {alpha:<8g} | error/pixel {errors.mean():.5f}"
                )

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "lasso_tuning.csv"
    table.to_csv(path, index=False)

    chosen: dict[str, dict[str, float]] = {}
    print("\nchosen shrinkage per basis and budget:")
    for basis in args.bases:
        per_budget = {}
        for m, group in table[table["basis"] == basis].groupby("m"):
            best = group.loc[group["error"].idxmin()]
            per_budget[str(int(m))] = float(best["alpha"])
        chosen[basis] = per_budget
        print(f"  {basis:<5} " + "  ".join(f"m={m}:{a:g}" for m, a in per_budget.items()))

    alpha_path = args.output_dir / "lasso_alpha.json"
    alpha_path.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}\nwrote {alpha_path}")


if __name__ == "__main__":
    main()
