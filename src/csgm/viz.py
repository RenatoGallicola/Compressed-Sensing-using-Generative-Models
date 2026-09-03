"""Plotting helpers shared by the notebooks and the figure scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from csgm.config import IMAGE_SHAPE

_GRID = {"alpha": 0.3, "linewidth": 0.5}


def show_images(
    images: np.ndarray,
    titles: Sequence[str] | None = None,
    *,
    ncols: int = 8,
    scale: float = 1.4,
    suptitle: str | None = None,
) -> plt.Figure:
    """Draw a grid of MNIST images.

    Args:
        images: Array of shape ``(k, 784)`` or ``(k, 28, 28, ...)``.
        titles: Optional per-image titles.
        ncols: Images per row.
        scale: Size in inches of a single tile.
        suptitle: Optional figure title.

    Returns:
        The created figure.
    """
    images = np.asarray(images).reshape(-1, *IMAGE_SHAPE[:2])
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * scale * 1.15))
    for i, (ax, img) in enumerate(zip(np.atleast_1d(axes).ravel(), images, strict=False)):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)
    for ax in np.atleast_1d(axes).ravel()[len(images) :]:
        ax.set_visible(False)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def plot_error_curves(
    curves: dict[str, tuple[Sequence[float], Sequence[float]]],
    *,
    ylabel: str = "reconstruction error per pixel",
    xlabel: str = "number of measurements $m$",
    title: str | None = None,
    logy: bool = True,
    errorbars: dict[str, Sequence[float]] | None = None,
) -> plt.Figure:
    """Plot one error-vs-measurements curve per method.

    Args:
        curves: Mapping ``label -> (m_values, errors)``.
        ylabel: Y axis label.
        xlabel: X axis label.
        title: Optional axes title.
        logy: Use a logarithmic Y axis; errors span more than an order of
            magnitude between the sparse and the learned priors.
        errorbars: Optional mapping ``label -> half-width`` per point.

    Returns:
        The created figure.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, (xs, ys) in curves.items():
        if errorbars and label in errorbars:
            ax.errorbar(xs, ys, yerr=errorbars[label], marker="o", ms=4, capsize=3, label=label)
        else:
            ax.plot(xs, ys, marker="o", ms=4, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", **_GRID)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = 150) -> Path:
    """Save a figure, creating parent directories as needed.

    Args:
        fig: Figure to save.
        path: Destination path.
        dpi: Output resolution.

    Returns:
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
