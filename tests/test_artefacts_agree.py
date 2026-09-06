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

import pandas as pd
import pytest

from csgm.config import RESULTS_DIR, ROOT_DIR, checkpoint_path

PRIORS = ["fcvae-20", "vae-20", "vae-30", "dcgan-20", "dcgan-30"]


def _load_run_stats():
    """Import ``scripts/run_stats.py``, which is not part of the package."""
    path = ROOT_DIR / "scripts" / "run_stats.py"
    spec = importlib.util.spec_from_file_location("run_stats", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_stats"] = module
    spec.loader.exec_module(module)
    return module


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
