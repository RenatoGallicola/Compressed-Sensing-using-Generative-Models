"""Check that the numbers in the write-ups still match the benchmark.

The report, the README, the slides and the notebooks all quote figures that come
from ``results/benchmark.csv``. Nothing stops those from drifting apart when the
experiment is re-run, and a stale number in a report is worse than no number, so
they are verified here rather than by proofreading.

These tests read the committed results; they do not run the experiment.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from csgm.config import RESULTS_DIR, ROOT_DIR

REPORT = ROOT_DIR / "docs" / "report"
PRIORS = ["fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]
BASELINES = ["lasso", "lasso-dct"]
REFERENCE, REFERENCE_M = "lasso-dct", 400
#: The names scripts/make_figures.py prints for each method.
LABELS = {
    "fcvae-20": "VAE, paper architecture, k=20",
    "vae-20": "VAE, k=20",
    "vae-30": "VAE, k=30",
    "dcgan-20": "DCGAN, k=20",
    "dcgan-30": "DCGAN, k=30",
}


@pytest.fixture(scope="module")
def benchmark():
    path = RESULTS_DIR / "benchmark.csv"
    if not path.exists():
        pytest.skip("benchmark.csv is not present")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def means(benchmark):
    return benchmark.pivot_table(index="m", columns="method", values="per_pixel_error")


def numbers_in(text):
    """Every decimal number in a piece of prose, as strings, in order."""
    return re.findall(r"\d+\.\d+", text)


def test_readme_table_matches_the_benchmark(means):
    """Every cell of the README results table is the mean it claims to be."""
    text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    body = text[text.index("|   m | Lasso (pixel)") :].splitlines()
    header = [c.strip(" *") for c in body[0].strip("|").split("|")]
    columns = dict(
        zip(
            header[1:],
            ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"],
            strict=True,
        )
    )

    checked = 0
    for line in body[2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        m = int(cells[0])
        for name, cell in zip(header[1:], cells[1:], strict=True):
            claimed = float(cell.strip("* "))
            assert claimed == pytest.approx(means[columns[name]][m], abs=5e-5), (
                f"README row m={m}, column {name}"
            )
            checked += 1
    assert checked == 70


def test_report_table_matches_the_benchmark(means):
    """Same for the LaTeX table in the report."""
    text = (REPORT / "results.tex").read_text(encoding="utf-8")
    body = text[text.index(r"\label{tab:results}") - 3000 : text.index(r"\label{tab:results}")]
    order = ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]

    checked = 0
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        m = int(cells[0])
        values = [
            c.replace(r"\textbf{", "").replace("}", "").replace(r"\\", "").strip()
            for c in cells[1:]
        ]
        for method, value in zip(order, values, strict=True):
            assert float(value) == pytest.approx(means[method][m], abs=5e-5), (
                f"report row m={m}, column {method}"
            )
            checked += 1
    assert checked == 70


def test_bold_marks_the_best_method_in_the_report(means):
    """A bold entry must be the row minimum, and every row must have one."""
    text = (REPORT / "results.tex").read_text(encoding="utf-8")
    body = text[text.index(r"\hline") : text.index(r"\label{tab:results}")]
    order = ["lasso", "lasso-dct", "fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]

    rows = 0
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        m = int(cells[0])
        bolded = [order[i] for i, c in enumerate(cells[1:]) if r"\textbf{" in c]
        best = means.loc[m, order].idxmin()
        assert bolded == [best], f"row m={m}: bold on {bolded}, best is {best}"
        rows += 1
    assert rows == 10


def test_sample_efficiency_table_is_reproducible(benchmark, means):
    """The generated table matches a fresh computation, per-image counts included."""
    path = RESULTS_DIR / "sample_efficiency.md"
    if not path.exists():
        pytest.skip("sample_efficiency.md is not present")
    text = path.read_text(encoding="utf-8")

    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")

    # The table is generated for both baselines, so both sections are checked.
    sections = text.split("## Against ")[1:]
    assert len(sections) == 2, "expected one section per baseline"

    checked = 0
    for baseline, section in zip(["lasso-dct", "lasso"], sections, strict=True):
        reference = wide.loc[REFERENCE_M, baseline]
        target = reference.mean()
        assert f"{target:.4f}" in section, f"{baseline}: reference level missing"

        for method in PRIORS:
            budgets = means[method]
            matching = budgets.index[budgets <= target]
            label = LABELS[method]
            row = next(line for line in section.splitlines() if line.startswith(f"| {label} |"))
            if len(matching) == 0:
                assert "never" in row, f"{method}: expected no matching budget in {row!r}"
                checked += 1
                continue
            needed = int(matching.min())
            wins = int((wide.loc[needed, method].to_numpy() <= reference.to_numpy()).sum())
            assert f"| {needed} |" in row, f"{method}: expected budget {needed} in {row!r}"
            assert f"{wins} of 10" in row, f"{method}: expected {wins} wins in {row!r}"
            checked += 1
    assert checked == 2 * len(PRIORS)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("vae-30", 0.0070),
        ("fcvae-20", 0.0072),
        ("vae-20", 0.0076),
        ("dcgan-20", 0.0098),
        ("dcgan-30", 0.0235),
    ],
)
def test_quoted_error_floors(means, method, expected):
    """The floors quoted in the prose are the means from 300 measurements up."""
    floor = means[method][means.index >= 300].mean()
    assert round(floor, 4) == expected

    for path in (REPORT / "results.tex", ROOT_DIR / "README.md"):
        assert f"{expected:.4f}" in path.read_text(encoding="utf-8"), (
            f"{expected} missing from {path.name}"
        )


def test_the_reference_baseline_is_the_well_behaved_one(benchmark):
    """The stated reason for preferring the DCT reference has to hold."""
    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    pixel, dct = wide.loc[REFERENCE_M, "lasso"], wide.loc[REFERENCE_M, "lasso-dct"]

    # The pixel baseline is mid-transition here: mean far above median.
    assert pixel.mean() > 10 * pixel.median()
    assert int((pixel < 1e-3).sum()) == 6
    # The DCT baseline is not.
    assert dct.mean() == pytest.approx(dct.median(), rel=0.15)
    assert int((dct < 1e-3).sum()) == 0


def test_quoted_low_budget_factors(means):
    """The factors quoted at 25 measurements, against both baselines."""
    best = means.loc[25, PRIORS].min()
    assert means["lasso"][25] / best == pytest.approx(6.1, abs=0.05)
    assert means["lasso-dct"][25] / best == pytest.approx(5.1, abs=0.05)


def test_the_trivial_predictor_level_is_quoted_where_it_matters(benchmark, means):
    """A baseline at or above the blank-image error is not reconstructing anything.

    The write-ups quote that level and name the budgets where the pixel baseline
    sits at or above it, so both have to keep matching the data.
    """
    path = RESULTS_DIR / "reconstructions.npz"
    if not path.exists():
        pytest.skip("reconstructions.npz is not present")
    with np.load(path) as archive:
        trivial = float((archive["ground_truth"] ** 2).mean())

    degenerate = [int(m) for m in means.index if means["lasso"][m] >= trivial]
    assert degenerate == [10, 25]

    for doc in (ROOT_DIR / "README.md", REPORT / "results.tex"):
        text = doc.read_text(encoding="utf-8")
        assert f"{trivial:.4f}" in text, f"the blank-image level is missing from {doc.name}"


def test_crossover_budget(means):
    """Both baselines overtake every prior from 500 measurements up."""
    for m in (500, 750):
        assert means.loc[m, BASELINES].max() < means.loc[m, PRIORS].min()
    assert means.loc[400, PRIORS].min() < means.loc[400, BASELINES].min()


def test_quoted_recovery_costs(benchmark):
    """The per-configuration timings quoted in the cost sections."""
    seconds = benchmark.groupby("method")["seconds_per_batch"].mean()
    assert seconds["lasso"] == pytest.approx(1.5, abs=0.3)
    assert seconds["lasso-dct"] == pytest.approx(1.5, abs=0.3)
    assert seconds["fcvae-20"] == pytest.approx(6.6, abs=0.5)
    assert seconds[["dcgan-20", "dcgan-30"]].mean() / seconds["fcvae-20"] == pytest.approx(
        102, abs=6
    )


def test_lambda_sweep_endpoints():
    """The claim that the best penalty is 1 at the smallest budgets and 0 at the largest."""
    path = RESULTS_DIR / "lambda_sweep.csv"
    if not path.exists():
        pytest.skip("lambda_sweep.csv is not present")
    sweep = pd.read_csv(path)
    table = sweep.pivot_table(index="m", columns=["method", "l2_penalty"], values="per_pixel_error")

    for method in table.columns.get_level_values(0).unique():
        best = table[method].idxmin(axis=1)
        assert best[10] == 1.0, f"{method} at the smallest budget"
        assert all(best[m] == 0.0 for m in (200, 300, 400, 500, 750)), f"{method} at the largest"
        # In between the three priors disagree, which is the point the write-ups
        # make about a single recommended value being a compromise.
    disagree = {
        float(table[method].idxmin(axis=1)[25])
        for method in table.columns.get_level_values(0).unique()
    }
    assert len(disagree) > 1, "the priors are expected to disagree at 25 measurements"


def test_no_superseded_figures_survive_in_the_prose():
    """Values from earlier versions of the experiment must not linger."""
    stale = {
        "an 8x saving": "the headline is 5.3x, against either baseline",
        "5.6x more accurate": "the m=25 factors are 6.1x and 5.1x",
        "factor of three across seeds": "the seed spread is 3.5",
        "The paper's simpler architecture beats ours": (
            "the three VAE priors are statistically indistinguishable"
        ),
        "factor 73": "the cost ratio is 102",
        "swept over six values": "the shrinkage sweep covers eight values, per budget",
    }
    files = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "docs" / "presentation_outline.md",
        ROOT_DIR / "docs" / "model_selection.md",
        *REPORT.glob("*.tex"),
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for phrase, why in stale.items():
            assert phrase not in text, f"{path.name} still contains {phrase!r}: {why}"
