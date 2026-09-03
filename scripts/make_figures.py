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

from csgm.config import FIGURES_DIR, IMAGE_SHAPE, RESULTS_DIR
from csgm.viz import plot_error_curves, save_figure

LABELS = {
    "lasso": "Lasso (DCT basis)",
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
    curves, errorbars = {}, {}
    for method, group in df.groupby("method", sort=False):
        stats = group.groupby("m")["per_pixel_error"].agg(["mean", "sem"]).sort_index()
        label = LABELS.get(method, method)
        curves[label] = (stats.index.to_numpy(), stats["mean"].to_numpy())
        errorbars[label] = np.nan_to_num(stats["sem"].to_numpy())

    fig = plot_error_curves(
        curves,
        errorbars=errorbars,
        title=f"MNIST recovery from Gaussian measurements (mean of {df['image'].nunique()} images)",
    )
    return save_figure(fig, figures_dir / "error_vs_measurements.png")


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
    m_values = sorted(df["m"].unique())
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

    written = [error_curves(df, args.figures_dir), summary_table(df, args.results_dir)]

    npz_path = args.results_dir / "reconstructions.npz"
    if npz_path.exists():
        with np.load(npz_path) as recons:
            written.append(reconstruction_grid(dict(recons), df, args.figures_dir, args.grid_image))

    samples = prior_samples(args.figures_dir)
    if samples is not None:
        written.append(samples)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
