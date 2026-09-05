"""Choose the shrinkage of the Lasso baseline, in both bases.

The baseline should be given its best configuration: a comparison against a
badly tuned baseline says nothing. Bora et al. quote a shrinkage of 0.1, but
scikit-learn scales its objective by ``1 / (2 m)``, so that number does not
transfer, and the value is swept here instead.

Selection is on the mean error over the measurement budgets, per basis, and the
chosen value is then used for every budget. Tuning per budget would give the
baseline an advantage the generative priors do not get, since those use one
setting throughout.

Example:
    python scripts/tune_lasso.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from csgm.baselines import lasso_recover
from csgm.config import DEFAULT_SEED, N_PIXELS, RESULTS_DIR
from csgm.data import load_mnist, sample_images
from csgm.measurements import gaussian_measurement_matrix, measure
from csgm.metrics import per_pixel_l2

DEFAULT_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
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
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    """Sweep the shrinkage and report the best value for each basis."""
    args = parse_args()

    (_, _), (x_test, y_test) = load_mnist()
    images = sample_images(
        x_test, args.n_images, labels=y_test, stratified=True, seed=args.seed
    ).reshape(args.n_images, N_PIXELS)

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

    print("\nmean error per basis and shrinkage:")
    pivot = table.pivot_table(index="alpha", columns="basis", values="error")
    print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))
    print()
    for basis in args.bases:
        best = pivot[basis].idxmin()
        print(
            f"best shrinkage for the {basis} basis: {best:g} (mean error {pivot[basis][best]:.5f})"
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
