"""Check the committed artefacts against each other and against the models.

``test_reported_numbers.py`` checks the prose against ``benchmark.csv``. That
leaves a table which is internally consistent but no longer describes the
repository: the generators can be replaced without the numbers moving, the
derived tables can drift from the raw one, and the protocol the write-ups
describe can stop being the protocol that produced the results. Those are the
cases here.

The first of them is not hypothetical. Retraining a generator changes a file in
``models/`` and nothing else, so without this the published table would go on
describing models that are no longer shipped, and every other test would still
pass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from itertools import product

import numpy as np
import pandas as pd
import pytest

from csgm.config import (
    N_PIXELS,
    NOISE_SEED_OFFSET,
    RESULTS_DIR,
    ROOT_DIR,
    checkpoint_path,
)
from csgm.measurements import gaussian_measurement_matrix, measure

PRIORS = ["fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]


def _load_script(name):
    """Import a file from ``scripts/``, which is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, ROOT_DIR / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _defaults(name):
    """Return the arguments a script gets when run with no flags."""
    module = _load_script(name)
    argv = sys.argv
    sys.argv = [f"{name}.py"]
    try:
        return module.parse_args()
    finally:
        sys.argv = argv


def _load_run_stats():
    """Import ``scripts/run_stats.py``, which is not part of the package."""
    return _load_script("run_stats")


@pytest.fixture(scope="module")
def benchmark():
    path = RESULTS_DIR / "benchmark.csv"
    if not path.exists():
        pytest.skip("benchmark.csv is not present")
    return pd.read_csv(path)


def test_every_row_names_the_checkpoint_that_is_shipped(benchmark):
    """The table has to describe the generators currently in ``models/``.

    This is the guard that fires when a generator is retrained and the benchmark
    is not re-run. Nothing else in the suite would notice.
    """
    if "checkpoint" not in benchmark.columns:
        pytest.skip("this benchmark predates checkpoint fingerprinting")

    for method in PRIORS:
        rows = benchmark[benchmark["method"] == method]
        if rows.empty:
            continue
        family, latent_dim = method.rsplit("-", 1)
        path = checkpoint_path(family, int(latent_dim))
        if not path.exists():
            pytest.skip(f"{path.name} is not present")
        shipped = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        recorded = set(rows["checkpoint"])
        assert recorded == {shipped}, (
            f"{method} rows were produced by checkpoint {sorted(recorded)}, but "
            f"{path.name} now hashes to {shipped}. Re-run the benchmark for this "
            "method, or restore the checkpoint the table describes"
        )


def test_the_recorded_protocol_is_the_one_the_write_ups_describe(benchmark):
    """A re-run at other settings would rewrite the table and pass every test."""
    meta_path = RESULTS_DIR / "benchmark_meta.json"
    if not meta_path.exists():
        pytest.skip("benchmark_meta.json is not present")
    protocol = json.loads(meta_path.read_text(encoding="utf-8"))["protocol"]

    assert protocol["l2_penalty"] == 0.1, "the write-ups describe the paper's penalty"
    assert protocol["steps"] == 1000
    assert protocol["restarts"] == 10
    assert protocol["learning_rate"] == 0.01
    assert protocol["noise_norm"] == 0.1
    assert protocol["n_images"] == 10
    assert protocol["lasso_alpha"] == "per budget"
    assert benchmark["image"].nunique() == protocol["n_images"]
    assert sorted(benchmark["m"].unique()) == protocol["m_values"]


def test_the_unregularised_sweep_shares_the_protocol_it_is_compared_against():
    """The two sweeps are compared, so they must differ only in the penalty."""
    paths = [
        RESULTS_DIR / "benchmark_meta.json",
        RESULTS_DIR / "unregularised" / "benchmark_meta.json",
    ]
    if not all(p.exists() for p in paths):
        pytest.skip("both sweeps are needed")
    main, unregularised = (json.loads(p.read_text(encoding="utf-8")) for p in paths)

    differing = {
        key
        for key in main["protocol"]
        if main["protocol"][key] != unregularised["protocol"].get(key)
    }
    unexpected = differing - {"l2_penalty"}
    assert not unexpected, f"the two sweeps also differ in {unexpected}"
    assert unregularised["protocol"]["l2_penalty"] == 0.0
    assert main["checkpoints"] == unregularised["checkpoints"], (
        "the two sweeps were run on different generators"
    )


def test_the_significance_table_reproduces(benchmark):
    """``significance.csv`` must be a function of ``benchmark.csv``.

    The write-ups quote which comparisons survive correction. Without this the
    committed table could say anything and the prose would be checked against it
    rather than against the data.
    """
    path = RESULTS_DIR / "significance.csv"
    if not path.exists():
        pytest.skip("significance.csv is not present")
    committed = pd.read_csv(path)
    run_stats = _load_run_stats()

    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")
    for (better, worse), group in committed.groupby(["better", "worse"], sort=False):
        fresh = pd.DataFrame(run_stats.compare(wide, better, worse)).set_index("m")
        recorded = group.set_index("m")
        for m in recorded.index:
            assert fresh.loc[m, "p_wilcoxon"] == pytest.approx(
                recorded.loc[m, "p_wilcoxon"], abs=1e-12
            ), f"{better} vs {worse} at m={m}"
            assert fresh.loc[m, "wins"] == recorded.loc[m, "wins"]
            assert fresh.loc[m, "mean_difference"] == pytest.approx(
                recorded.loc[m, "mean_difference"], abs=1e-12
            )


def test_the_verdict_column_carries_the_direction():
    """A two-sided test says nothing about who won until the sign is read."""
    path = RESULTS_DIR / "significance.csv"
    if not path.exists():
        pytest.skip("significance.csv is not present")
    table = pd.read_csv(path)

    better = table[table["verdict"] == "better"]
    worse = table[table["verdict"] == "worse"]
    assert (better["mean_difference"] > 0).all()
    assert (worse["mean_difference"] < 0).all()
    assert (table[table["verdict"] == "ns"]["p_holm"] >= 0.05).all()


def test_the_recorded_protocol_has_every_field_the_merge_check_compares():
    """A field added to ``protocol()`` after a run would block merging into it.

    ``run_benchmark.py --merge`` refuses when any recorded field differs from the
    current one, and a field absent from an older record reads as differing.
    Adding one is therefore enough to lock the committed table against the very
    merge the retraining plan depends on, with no other test failing.
    """
    spec = importlib.util.spec_from_file_location(
        "run_benchmark", ROOT_DIR / "scripts" / "run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_benchmark"] = module
    spec.loader.exec_module(module)

    argv = sys.argv
    sys.argv = ["run_benchmark.py"]
    try:
        args = module.parse_args()
    finally:
        sys.argv = argv
    args.resolved_alpha = None
    expected = set(module.protocol(args))

    for name in ("benchmark_meta.json", "unregularised/benchmark_meta.json"):
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        recorded = set(json.loads(path.read_text(encoding="utf-8"))["protocol"])
        assert recorded == expected, (
            f"{name} records {sorted(recorded)}, but protocol() compares "
            f"{sorted(expected)}; --merge would refuse on {sorted(expected ^ recorded)}"
        )


def test_the_sweep_defaults_produce_the_committed_sweep():
    """The documented command has to reproduce the table it is documented for.

    A default that drifts from the artefact leaves the reader with a command
    that silently produces something else. This has happened twice: once with
    the benchmark's latent penalty, once with the list of priors swept here.
    """
    path = RESULTS_DIR / "lambda_sweep.csv"
    if not path.exists():
        pytest.skip("lambda_sweep.csv is not present")

    args = _defaults("run_lambda_sweep")
    committed = sorted(pd.read_csv(path)["method"].unique())
    assert sorted(args.methods) == committed, (
        f"run_lambda_sweep.py defaults to {sorted(args.methods)} but "
        f"lambda_sweep.csv holds {committed}; the documented command would "
        "overwrite the table with a different set of priors"
    )
    assert sorted(pd.read_csv(path)["l2_penalty"].unique()) == sorted(args.lambdas)
    assert sorted(pd.read_csv(path)["m"].unique()) == sorted(args.m_values)


def test_the_tuning_defaults_produce_the_committed_shrinkage_table():
    """Same for the baseline's shrinkage sweep."""
    path = RESULTS_DIR / "lasso_tuning.csv"
    if not path.exists():
        pytest.skip("lasso_tuning.csv is not present")

    args = _defaults("tune_lasso")
    table = pd.read_csv(path)
    assert sorted(table["basis"].unique()) == sorted(args.bases)
    assert sorted(table["alpha"].unique()) == sorted(args.alphas)
    assert sorted(table["m"].unique()) == sorted(args.m_values)


def test_the_benchmark_defaults_produce_the_committed_protocol():
    """Running the benchmark bare has to give the protocol that is published."""
    meta_path = RESULTS_DIR / "benchmark_meta.json"
    if not meta_path.exists():
        pytest.skip("benchmark_meta.json is not present")
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))["protocol"]

    args = _defaults("run_benchmark")
    for field in ("n_images", "steps", "restarts", "learning_rate", "l2_penalty", "noise_norm"):
        assert getattr(args, field) == recorded[field], (
            f"run_benchmark.py defaults {field}={getattr(args, field)}, but the "
            f"committed sweep was run with {recorded[field]}"
        )
    assert sorted(args.m_values) == recorded["m_values"]


def test_the_report_figures_are_the_generated_ones():
    """The report embeds copies of results/figures, which have to be current.

    Regenerating the figures leaves the copies under docs/report/assets untouched,
    so the document can end up plotting one experiment beside a table describing
    another, with nothing else in the suite noticing.
    """
    generated = ROOT_DIR / "results" / "figures"
    embedded = ROOT_DIR / "docs" / "report" / "assets"
    if not generated.exists() or not embedded.exists():
        pytest.skip("figures are not present")

    stale = []
    for figure in sorted(generated.glob("*.png")):
        copy = embedded / figure.name
        if not copy.exists():
            continue
        if (
            hashlib.sha256(figure.read_bytes()).digest()
            != hashlib.sha256(copy.read_bytes()).digest()
        ):
            stale.append(figure.name)
    assert not stale, (
        f"docs/report/assets is behind results/figures for {stale}; "
        "copy the regenerated figures across"
    )


def test_the_significance_table_matches_an_exact_computation(benchmark):
    """The table must be right, not merely reproducible.

    ``test_the_significance_table_reproduces`` re-runs ``run_stats.py`` and
    compares, so a mistake inside that script reproduces perfectly and passes.
    Here the signed-rank p-value is computed by enumerating all 2**n sign
    assignments, which is the definition of the exact test, and Holm is applied
    independently. Nothing from ``scripts/`` is imported.
    """
    path = RESULTS_DIR / "significance.csv"
    if not path.exists():
        pytest.skip("significance.csv is not present")
    committed = pd.read_csv(path)
    wide = benchmark.pivot_table(index=["m", "image"], columns="method", values="per_pixel_error")

    def exact(differences):
        assert (differences != 0).all(), "a tie would need a rule of its own"
        ranks = pd.Series(abs(differences)).rank().to_numpy()
        assert len(set(abs(differences))) == len(differences), "tied magnitudes need average ranks"
        centre = ranks.sum() / 2
        observed = abs(ranks[differences > 0].sum() - centre)
        sums = np.array(
            [ranks[list(signs)].sum() for signs in product([False, True], repeat=len(ranks))]
        )
        return float((abs(sums - centre) >= observed - 1e-9).mean())

    def holm(values):
        adjusted, running = [0.0] * len(values), 0.0
        for rank, i in enumerate(sorted(range(len(values)), key=lambda j: values[j])):
            running = max(running, (len(values) - rank) * values[i])
            adjusted[i] = min(1.0, running)
        return adjusted

    for (better, worse), group in committed.groupby(["better", "worse"], sort=False):
        budgets = sorted(group["m"])
        raw = [
            exact(wide.loc[m, better].to_numpy() - wide.loc[m, worse].to_numpy())
            for m in budgets
        ]
        adjusted = holm(raw)
        recorded = group.set_index("m")
        for i, m in enumerate(budgets):
            assert raw[i] == pytest.approx(recorded.loc[m, "p_wilcoxon"], abs=1e-12), (
                f"{better} vs {worse} at m={m}"
            )
            assert adjusted[i] == pytest.approx(recorded.loc[m, "p_holm"], abs=1e-12), (
                f"{better} vs {worse} at m={m}, after Holm"
            )


def test_the_columns_of_the_benchmark_agree_with_each_other(benchmark):
    """PSNR is a function of the error, and the residual of the reconstruction.

    Every check on the results reads ``per_pixel_error`` and takes the rest of
    the row on trust. These two columns are quoted in the write-ups as well, so
    they are derived again here: PSNR from the error, and the residual from the
    measurement matrix and the noise the recorded seeds produce.
    """
    archive_path = RESULTS_DIR / "reconstructions.npz"
    if not archive_path.exists():
        pytest.skip("reconstructions.npz is not present")

    expected = -10 * np.log10(benchmark["per_pixel_error"].clip(lower=1e-12))
    assert (benchmark["psnr_db"] - expected).abs().max() < 1e-3

    protocol = _defaults("run_benchmark")
    with np.load(archive_path) as archive:
        truth = archive["ground_truth"].reshape(-1, N_PIXELS)
        for m in sorted(benchmark["m"].unique()):
            A = gaussian_measurement_matrix(m, N_PIXELS, seed=protocol.seed + m)
            y = measure(
                truth,
                A,
                noise_std=protocol.noise_norm / np.sqrt(m),
                seed=protocol.seed + m + NOISE_SEED_OFFSET,
            )
            for method in sorted(benchmark["method"].unique()):
                key = f"{method}__m{m}"
                if key not in archive:
                    continue
                x_hat = archive[key].reshape(-1, N_PIXELS)
                rows = benchmark[(benchmark["m"] == m) & (benchmark["method"] == method)]
                rows = rows.sort_values("image")
                residual = np.linalg.norm(x_hat @ A.T - y, axis=1)
                assert residual == pytest.approx(
                    rows["measurement_residual"].to_numpy(), abs=1e-4
                ), f"{method} at m={m}: the recorded residual is not the one these seeds give"


def test_the_error_bars_stay_inside_the_range_of_a_squared_error():
    """A bar reaching the axis is an interval leaving the domain of the quantity.

    The curve figure plots an interval for the mean per-pixel error, which cannot
    be negative. A symmetric normal-theory interval does go negative here, at
    three of the seventy points, because ten per-image errors are strongly
    skewed; the bootstrap interval the figure uses does not. Without this the
    statistic could be swapped back and the only symptom would be three bars
    clipped by the logarithmic axis.
    """
    path = RESULTS_DIR / "benchmark.csv"
    if not path.exists():
        pytest.skip("benchmark.csv is not present")
    make_figures = _load_script("make_figures")
    table = pd.read_csv(path)

    for (method, m), group in table.groupby(["method", "m"]):
        values = group["per_pixel_error"].to_numpy()
        low, high = make_figures.bootstrap_interval(values, seed=int(m))
        assert low > 0, f"{method} at m={m}: the interval reaches zero"
        assert low <= values.mean() <= high, f"{method} at m={m}: the interval misses the mean"
