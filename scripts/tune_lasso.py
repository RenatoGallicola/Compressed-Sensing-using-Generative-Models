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

The sweep runs on ten images drawn with a different seed from the ones the
benchmark scores on, so the baseline's hyper-parameter is not chosen on the
evaluation set. The result is written to ``lasso_alpha.json``, which
``run_benchmark.py`` reads.

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
from csgm.config import DEFAULT_SEED, N_PIXELS, RESULTS_DIR
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2

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
        help="seed for the images used to tune, kept different from the benchmark's",
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
    print(f"tuning on images drawn with seed {args.tuning_seed}, benchmark uses {args.seed}")

    rows = []
    for m in args.m_values:
        A = gaussian_measurement_matrix(m, N_PIXELS, seed=args.seed + m)
        y = measure(images, A, noise_std=args.noise_norm / np.sqrt(m), seed=args.seed + m)
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
