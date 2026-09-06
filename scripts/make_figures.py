"""Turn ``results/benchmark.csv`` into the figures and tables used in the README.

Run ``scripts/run_benchmark.py`` first; this script does no computation of its
own, so figures can be restyled without re-running the sweep.

Example:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from csgm.config import FIGURES_DIR, IMAGE_SHAPE, N_PIXELS, RESULTS_DIR
from csgm.viz import plot_error_curves, save_figure

LABELS = {
    "lasso": "Lasso (pixel basis)",
    "lasso-dct": "Lasso (DCT basis)",
    "fcvae-20": "VAE, paper architecture, k=20",
    "vae-20": "VAE, k=20",
    "vae-30": "VAE, k=30",
    "dcgan-20": "DCGAN, k=20",
    "dcgan-30": "DCGAN, k=30",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="directory holding benchmark.csv"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR, help="where to write the PNGs"
    )
    parser.add_argument(
        "--grid-image", type=int, default=0, help="index of the image shown in the qualitative grid"
    )
    return parser.parse_args()


def error_curves(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot mean per-pixel error against the measurement budget.

    Args:
        df: Long-format benchmark table.
        figures_dir: Output directory.

    Returns:
        Path of the written figure.
    """
    # The reference paper plots 95% confidence intervals, so the bars here are
    # the same quantity. With ten images the normal multiplier is too optimistic,
    # so the interval uses Student's t. The bars describe image-to-image spread
    # at one measurement matrix, not the spread over matrices.
    n_images = df["image"].nunique()
    multiplier = float(student_t.ppf(0.975, n_images - 1))

    curves, errorbars = {}, {}
    for method, group in df.groupby("method", sort=False):
        stats = group.groupby("m")["per_pixel_error"].agg(["mean", "sem"]).sort_index()
        label = LABELS.get(method, method)
        curves[label] = (stats.index.to_numpy(), stats["mean"].to_numpy())
        errorbars[label] = np.nan_to_num(stats["sem"].to_numpy()) * multiplier

    fig = plot_error_curves(
        curves,
        errorbars=errorbars,
        logx=True,
        title=(
            f"MNIST recovery from Gaussian measurements\n"
            f"mean of {n_images} images, bars are 95% confidence intervals"
        ),
    )
    return save_figure(fig, figures_dir / "error_vs_measurements.png")


def metric_comparison(df: pd.DataFrame, figures_dir: Path) -> Path:
    """Plot the two quantities a recovery run produces, side by side.

    The left panel is the reconstruction error against the ground truth, the
    metric every result in this project is reported with. The right panel is the
    measurement residual, the objective the optimiser actually minimises: it
    falls as the budget shrinks because fewer constraints remain to satisfy, so
    it tracks how well the optimisation converged rather than how good the
    reconstruction is. Both come from the same runs.

    Args:
        df: Long-format benchmark table.
        figures_dir: Output directory.

    Returns:
        Path of the written figure.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [
        (
            "per_pixel_error",
            r"$\|\hat{x} - x^*\|^2 / n$",
            "Reconstruction error, against ground truth",
        ),
        (
            "measurement_residual",
            r"$\|A\,G(\hat{z}) - y\| / n$",
            "Measurement residual, the recovery objective",
        ),
    ]
    for ax, (column, ylabel, title) in zip(axes, panels, strict=True):
        for method, group in df.groupby("method", sort=False):
            values = group[column] / (N_PIXELS if column == "measurement_residual" else 1)
            stats = values.groupby(group["m"]).mean().sort_index()
            ax.plot(
                stats.index, stats.to_numpy(), marker="o", ms=4, label=LABELS.get(method, method)
            )
        ax.set_yscale("log")
        ax.set_xlabel("number of measurements $m$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Reconstruction error and optimisation residual", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "metric_comparison.png")


def sample_efficiency(
    df: pd.DataFrame,
    results_dir: Path,
    ground_truth: np.ndarray | None = None,
    reference_m: int = 400,
) -> Path:
    """Report how few measurements each prior needs to match each baseline.

    Bora et al. summarise their MNIST result as "25 measurements match Lasso's
    performance with 400"; this reproduces that statement from our own table.

    The speed-up depends on which baseline it is measured against, so both are
    reported rather than one. Quoting only the more favourable of the two would
    be a choice made after seeing the answer, and the two baselines behave very
    differently at this budget: the pixel basis is midway through its transition
    from failure to near-exact recovery, so its mean is dominated by the few
    digits it has not solved, while the DCT basis has a tight spread. The error
    a blank image achieves is given alongside them, because a baseline scoring at
    that level is not reconstructing anything and no speed-up against it means
    very much.

    Args:
        df: Long-format benchmark table.
        results_dir: Output directory.
        ground_truth: Scored images, used for the trivial-predictor reference.
        reference_m: Baseline budget used as the reference error level.

    Returns:
        Path of the written Markdown table.
    """
    path = results_dir / "sample_efficiency.md"
    wide = df.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    methods = set(df["method"])
    baselines = [b for b in ("lasso-dct", "lasso") if b in methods]
    if not baselines or reference_m not in df["m"].values:
        path.write_text("Reference baseline not available in this benchmark.\n", encoding="utf-8")
        return path

    means = df.groupby(["method", "m"])["per_pixel_error"].mean()
    priors = [m for m in LABELS if not m.startswith("lasso") and m in methods]

    lines = [
        f"# Sample efficiency at m = {reference_m}",
        "",
        "For each prior, the smallest budget whose mean error is at or below the",
        "level a baseline reaches with "
        f"{reference_m} measurements, and how many of the individual test images",
        "it beats there. The comparison is given against both baselines, since the",
        "speed-up depends on which one is used as the reference.",
        "",
    ]

    for reference in baselines:
        per_image = wide.loc[reference_m, reference]
        target = per_image.mean()
        lines += [
            f"## Against {LABELS[reference]}",
            "",
            f"Mean per-pixel error {target:.4f} at {reference_m} measurements "
            f"(median {per_image.median():.4f}, largest {per_image.max():.4f}, "
            f"{int((per_image < 1e-3).sum())} of {len(per_image)} images below 0.001).",
            "",
            "| prior | measurements needed | speed-up | images beaten |",
            "|---|---|---|---|",
        ]
        for method in priors:
            budgets = means[method]
            matching = budgets.index[budgets <= target]
            if len(matching) == 0:
                lines.append(f"| {LABELS[method]} | never, in the sweep | n/a | n/a |")
                continue
            needed = int(matching.min())
            wins = int((wide.loc[needed, method].to_numpy() <= per_image.to_numpy()).sum())
            lines.append(
                f"| {LABELS[method]} | {needed} | {reference_m / needed:.1f}x | "
                f"{wins} of {len(per_image)} |"
            )
        lines.append("")

    if ground_truth is not None:
        trivial = float((np.asarray(ground_truth) ** 2).mean())
        at_or_above = [
            int(m)
            for m in sorted(df["m"].unique())
            if "lasso" in methods and means["lasso"][m] >= trivial
        ]
        lines += [
            "## The level of a blank image",
            "",
            f"Predicting an all-zero image scores {trivial:.4f} per pixel on these digits, "
            "which is the error any method has to beat before it can be said to be "
            "reconstructing anything.",
            "",
        ]
        if at_or_above:
            budgets = ", ".join(str(m) for m in at_or_above)
            lines += [
                f"The pixel-basis baseline is at or above that level at m = {budgets}. "
                "A speed-up quoted against it at those budgets is a comparison against a "
                "blank image, so the DCT baseline is the meaningful reference there.",
                "",
            ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def regularisation_comparison(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Overlay the sweep with and without the latent regulariser.

    The reference paper plots both variants of every prior, so this reproduces
    that comparison: solid lines use the value it recommends, dashed lines drop
    the term entirely.

    Args:
        results_dir: Directory holding ``benchmark.csv`` and ``unregularised/``.
        figures_dir: Output directory.

    Returns:
        The files written, empty if the unregularised sweep is absent.
    """
    import matplotlib.pyplot as plt

    other = results_dir / "unregularised" / "benchmark.csv"
    if not other.exists():
        return []

    main = pd.read_csv(results_dir / "benchmark.csv")
    plain = pd.read_csv(other)
    methods = [m for m in LABELS if not m.startswith("lasso") and m in set(main["method"])]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for colour, method in zip(colours, methods, strict=False):
        for frame, style, suffix in ((main, "-", ""), (plain, "--", r", $\lambda=0$")):
            stats = (
                frame[frame["method"] == method].groupby("m")["per_pixel_error"].mean().sort_index()
            )
            ax.plot(
                stats.index,
                stats.to_numpy(),
                style,
                color=colour,
                marker="o",
                ms=3,
                label=f"{LABELS[method]}{suffix}",
            )
    lasso = main[main["method"] == "lasso"].groupby("m")["per_pixel_error"].mean().sort_index()
    ax.plot(lasso.index, lasso.to_numpy(), color="black", marker="s", ms=3, label=LABELS["lasso"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted(main["m"].unique()))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("number of measurements $m$")
    ax.set_ylabel("reconstruction error per pixel")
    ax.set_title(r"With and without the latent regulariser ($\lambda = 0.1$ against $\lambda = 0$)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return [save_figure(fig, figures_dir / "regularisation_comparison.png")]


def lambda_sweep(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Plot the effect of the latent regulariser, if the sweep has been run.

    Two panels: what the penalty does to the reconstruction error, and what it
    does to the norm of the recovered latent code, which is the quantity it acts
    on directly.

    Args:
        results_dir: Directory holding ``lambda_sweep.csv``.
        figures_dir: Output directory.

    Returns:
        The files written, empty if the sweep has not been run.
    """
    import matplotlib.pyplot as plt

    path = results_dir / "lambda_sweep.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for column, ax, ylabel, logy in (
        ("per_pixel_error", axes[0], "reconstruction error per pixel", True),
        ("latent_norm", axes[1], r"$\|\hat{z}\|_2$", False),
    ):
        for method in methods:
            for penalty, group in df[df["method"] == method].groupby("l2_penalty"):
                stats = group.groupby("m")[column].mean().sort_index()
                ax.plot(
                    stats.index,
                    stats.to_numpy(),
                    marker="o",
                    ms=3,
                    linestyle="-" if method.endswith("30") else "--",
                    label=rf"{LABELS.get(method, method)}, $\lambda$={penalty:g}",
                )
        ax.set_xscale("log")
        ax.set_xticks(sorted(df["m"].unique()))
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.tick_params(axis="x", labelsize=8)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("number of measurements $m$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    axes[0].set_title("Reconstruction error")
    axes[1].set_title("Norm of the recovered latent code")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Effect of the latent regulariser", fontsize=12)
    fig.tight_layout()
    figure = save_figure(fig, figures_dir / "lambda_sweep.png")

    pivot = df.pivot_table(
        index="m", columns=["method", "l2_penalty"], values="per_pixel_error", aggfunc="mean"
    )
    table = results_dir / "lambda_sweep.md"
    table.write_text(
        "# Reconstruction error per pixel by latent penalty\n\n"
        + pivot.to_markdown(floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    return [figure, table]


def reconstruction_grid(
    recons: dict[str, np.ndarray], df: pd.DataFrame, figures_dir: Path, image_index: int
) -> Path:
    """Show one image reconstructed by every method at every budget.

    Args:
        recons: Contents of ``reconstructions.npz``.
        df: Long-format benchmark table.
        figures_dir: Output directory.
        image_index: Which benchmark image to display.

    Returns:
        Path of the written figure.
    """
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(df["method"]))
    # Only budgets present in the archive: a stale .npz beside a fresher csv
    # should degrade to a smaller grid, not crash the whole figure run.
    m_values = [
        m
        for m in sorted(df["m"].unique())
        if all(f"{method}__m{m}" in recons for method in methods)
    ]
    if not m_values:
        raise SystemExit("reconstructions.npz does not match benchmark.csv -- re-run the sweep")
    ground_truth = recons["ground_truth"][image_index].reshape(IMAGE_SHAPE[:2])

    ncols = len(m_values) + 1
    fig, axes = plt.subplots(
        len(methods),
        ncols,
        figsize=(1.05 * ncols + 1.4, 1.05 * len(methods) + 0.7),
        squeeze=False,
    )
    for row, method in enumerate(methods):
        axes[row][0].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_ylabel(
            LABELS.get(method, method), fontsize=8, rotation=0, ha="right", va="center"
        )
        if row == 0:
            axes[row][0].set_title("original", fontsize=8)
        for col, m in enumerate(m_values, start=1):
            image = recons[f"{method}__m{m}"][image_index].reshape(IMAGE_SHAPE[:2])
            axes[row][col].imshow(image, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row][col].set_title(f"m={m}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Reconstruction quality vs. number of measurements", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "reconstruction_grid.png")


def prior_samples(figures_dir: Path) -> Path | None:
    """Draw unconditional samples from each shipped generator.

    Recovery can only ever return something the generator can produce, so the
    quality of these samples upper-bounds the quality of any reconstruction.

    Args:
        figures_dir: Output directory.

    Returns:
        Path of the written figure, or ``None`` if no checkpoint is available.
    """
    import matplotlib.pyplot as plt

    from csgm.config import checkpoint_path
    from csgm.models import load_generator

    n_samples = 8
    rng = np.random.default_rng(0)
    rows = []
    for model, latent_dim in (("vae", 20), ("vae", 30), ("dcgan", 20), ("dcgan", 30)):
        if not checkpoint_path(model, latent_dim).exists():
            continue
        generator = load_generator(model, latent_dim)
        z = rng.standard_normal((n_samples, latent_dim)).astype("float32")
        images = np.asarray(generator(z, training=False)).reshape(n_samples, *IMAGE_SHAPE[:2])
        rows.append((f"{model.upper()}, k={latent_dim}", images))

    if not rows:
        return None

    fig, axes = plt.subplots(
        len(rows),
        n_samples,
        figsize=(1.05 * n_samples + 1.4, 1.05 * len(rows) + 0.7),
        squeeze=False,
    )
    for row, (label, images) in enumerate(rows):
        for col in range(n_samples):
            axes[row][col].imshow(images[col], cmap="gray", vmin=0, vmax=1)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
        axes[row][0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Unconditional samples: the hypothesis space of each prior", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "prior_samples.png")


def summary_table(df: pd.DataFrame, results_dir: Path) -> Path:
    """Write the mean per-pixel error per method and budget as Markdown.

    Args:
        df: Long-format benchmark table.
        results_dir: Output directory.

    Returns:
        Path of the written Markdown table.
    """
    pivot = (
        df.pivot_table(index="m", columns="method", values="per_pixel_error", aggfunc="mean")
        .reindex(columns=[m for m in LABELS if m in set(df["method"])])
        .rename(columns=LABELS)
    )
    path = results_dir / "summary.md"
    path.write_text(
        "# Mean reconstruction error per pixel\n\n" + pivot.to_markdown(floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    pivot.to_csv(results_dir / "summary.csv")
    return path


def main() -> None:
    """Regenerate every figure and table derived from the benchmark."""
    args = parse_args()
    csv_path = args.results_dir / "benchmark.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run scripts/run_benchmark.py first")

    df = pd.read_csv(csv_path)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.results_dir / "reconstructions.npz"
    recons = None
    if npz_path.exists():
        with np.load(npz_path) as archive:
            recons = dict(archive)

    written = [
        error_curves(df, args.figures_dir),
        metric_comparison(df, args.figures_dir),
        summary_table(df, args.results_dir),
        sample_efficiency(
            df, args.results_dir, None if recons is None else recons.get("ground_truth")
        ),
    ]

    if recons is not None:
        written.append(reconstruction_grid(recons, df, args.figures_dir, args.grid_image))

    samples = prior_samples(args.figures_dir)
    if samples is not None:
        written.append(samples)

    written.extend(lambda_sweep(args.results_dir, args.figures_dir))
    written.extend(regularisation_comparison(args.results_dir, args.figures_dir))

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
