"""Test whether the differences between methods are larger than the noise.

Ten images is a small sample, and at several budgets the curves are close enough
that a difference in the means says little on its own. Every method is run on
the same images and the same measurement matrices, so the comparisons are paired
and the pairing should be used: it removes the image-to-image variation, which is
much larger than the differences being tested.

The reported test is the Wilcoxon signed-rank test on the per-image differences.
It is preferred to a paired t-test here because the per-image errors are heavily
skewed, a handful of digits being reconstructed far worse than the rest, and with
ten samples the t-test's normality assumption is doing real work. The t statistic
is reported alongside it so the two can be compared.

Each method pair is tested at all ten budgets, so the p-values are corrected
within that family by the Holm step-down procedure. Uncorrected values are kept
in the table, since the correction depends on the family and a reader may want a
different one.

Example:
    python scripts/run_stats.py
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from csgm.config import RESULTS_DIR

PRIORS = ("fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30")
BASELINES = ("lasso", "lasso-dct")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=RESULTS_DIR / "benchmark.csv",
        help="long-format results table to test",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="significance level used for the verdict column"
    )
    return parser.parse_args()


def holm(p_values: np.ndarray) -> np.ndarray:
    """Adjust p-values by the Holm step-down procedure.

    Args:
        p_values: Uncorrected p-values of one family of tests.

    Returns:
        Adjusted p-values, in the order they were given, each clipped to 1.
    """
    order = np.argsort(p_values)
    n = len(p_values)
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def compare(wide: pd.DataFrame, left: str, right: str) -> list[dict]:
    """Test ``left`` against ``right`` at every budget.

    Args:
        wide: Errors indexed by ``(m, image)`` with one column per method.
        left: Method expected to have the lower error.
        right: Method it is compared against.

    Returns:
        One record per budget, with the paired difference and both tests.
    """
    records = []
    for m in sorted(wide.index.get_level_values("m").unique()):
        a = wide.loc[m, left].to_numpy()
        b = wide.loc[m, right].to_numpy()
        difference = b - a  # positive when the left method is the better one

        # Wilcoxon is undefined when every pair is tied, which happens if two
        # methods degenerate to the same reconstruction.
        if np.allclose(difference, 0.0):
            p_wilcoxon = 1.0
        else:
            p_wilcoxon = float(wilcoxon(a, b).pvalue)

        records.append(
            {
                "better": left,
                "worse": right,
                "m": int(m),
                "mean_error_better": float(a.mean()),
                "mean_error_worse": float(b.mean()),
                "mean_difference": float(difference.mean()),
                "wins": int((difference > 0).sum()),
                "n": len(difference),
                "p_wilcoxon": p_wilcoxon,
                "p_ttest": float(ttest_rel(a, b).pvalue),
            }
        )
    return records


def main() -> None:
    """Run every paired comparison and write the tables."""
    args = parse_args()

    if not args.benchmark.exists():
        raise SystemExit(f"no results at {args.benchmark}; run scripts/run_benchmark.py first")

    benchmark = pd.read_csv(args.benchmark)
    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    available = set(wide.columns)
    priors = [p for p in PRIORS if p in available]
    baselines = [b for b in BASELINES if b in available]

    rows: list[dict] = []
    for prior, baseline in itertools.product(priors, baselines):
        rows.extend(compare(wide, prior, baseline))
    # The two families the write-ups actually compare against each other.
    for left, right in itertools.combinations(priors, 2):
        rows.extend(compare(wide, left, right))

    table = pd.DataFrame(rows)
    # Holm is applied within each pair of methods, across the ten budgets, which
    # is the family a claim like "the difference is significant from m=75 up"
    # implicitly spans.
    table["p_holm"] = (
        table.groupby(["better", "worse"])["p_wilcoxon"]
        .transform(lambda group: holm(group.to_numpy()))
        .astype(float)
    )
    # The test is two-sided, so significance alone does not say who won. The
    # verdict has to carry the direction, otherwise a budget where the first
    # method is significantly *worse* reads as a win for it.
    table["significant"] = table["p_holm"] < args.alpha
    table["verdict"] = np.where(
        ~table["significant"],
        "ns",
        np.where(table["mean_difference"] > 0, "better", "worse"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "significance.csv"
    table.to_csv(csv_path, index=False)

    lines = [
        "# Paired significance tests",
        "",
        f"Generated by `scripts/run_stats.py` from `{args.benchmark.name}`. "
        f"Wilcoxon signed-rank on {int(table['n'].iloc[0])} paired per-image errors, "
        "two-sided, Holm-corrected across the ten budgets within each pair of methods.",
        "",
    ]
    for (better, worse), group in table.groupby(["better", "worse"], sort=False):
        wins = group[group["verdict"] == "better"]["m"]
        losses = group[group["verdict"] == "worse"]["m"]
        verdict = []
        if len(wins):
            verdict.append(f"lower error at m = {', '.join(str(m) for m in wins)}")
        if len(losses):
            verdict.append(f"higher error at m = {', '.join(str(m) for m in losses)}")
        if not verdict:
            verdict.append("no budget reaches significance after correction")

        lines += [f"## {better} against {worse}", "", "; ".join(verdict) + ".", ""]
        lines.append("|   m | mean diff | wins | p (Wilcoxon) | p (Holm) | verdict |")
        lines.append("|----:|----------:|-----:|-------------:|---------:|:--------|")
        for _, row in group.iterrows():
            lines.append(
                f"| {row['m']:>3} | {row['mean_difference']:+.5f} | "
                f"{row['wins']:>2}/{row['n']} | {row['p_wilcoxon']:.4f} | "
                f"{row['p_holm']:.4f} | {row['verdict']} |"
            )
        lines.append("")

    md_path = args.output_dir / "significance.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
