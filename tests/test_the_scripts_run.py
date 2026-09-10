"""Run the commands the write-ups tell a reader to run.

The suite exercises the library and reads the scripts' parsers for their
defaults, but until now nothing executed a script. The logic inside them is
covered elsewhere; what was not covered is the scaffolding a reader meets first:
the parser, the paths, the files written at the end. That is also the part that
breaks quietly, since a broken command still leaves every committed artefact
exactly as it was.

The two cheap commands are run in full, into a temporary directory so the
committed results are never touched. Recovery itself is not run here: it needs
TensorFlow, the dataset and a generator, and the reproduction it performs is
already checked against the committed table.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from csgm.config import RESULTS_DIR, ROOT_DIR

SCRIPTS = sorted(path.stem for path in (ROOT_DIR / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_script_builds_its_parser(script):
    """``--help`` imports the module and builds the parser, which is most of the risk.

    An import that no longer resolves, a flag declared twice, a default that
    raises: all of them surface here, for every script, including the two
    training ones that are far too slow to run.
    """
    spec = importlib.util.spec_from_file_location(script, ROOT_DIR / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[script] = module
    spec.loader.exec_module(module)

    argv = sys.argv
    sys.argv = [f"{script}.py", "--help"]
    try:
        with pytest.raises(SystemExit) as exit_code:
            module.parse_args()
    finally:
        sys.argv = argv
    assert exit_code.value.code == 0, f"{script} --help exits with {exit_code.value.code}"


def test_the_statistics_script_runs_and_reproduces_its_table(tmp_path):
    """``run_stats.py`` with no flags has to write the table that is committed."""
    benchmark = RESULTS_DIR / "benchmark.csv"
    committed = RESULTS_DIR / "significance.csv"
    if not benchmark.exists() or not committed.exists():
        pytest.skip("the committed results are not present")

    finished = subprocess.run(
        [sys.executable, str(ROOT_DIR / "scripts" / "run_stats.py"), "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]

    written = tmp_path / "significance.csv"
    assert written.exists(), f"nothing was written; the script said: {finished.stdout[-500:]}"
    assert (tmp_path / "significance.md").exists()

    fresh = pd.read_csv(written)
    stored = pd.read_csv(committed)
    pd.testing.assert_frame_equal(fresh, stored)


def test_the_figure_script_runs_end_to_end(tmp_path):
    """``make_figures.py`` has to read the results and write every output it names.

    The bytes are not compared: a different matplotlib renders the same numbers
    differently, and this is a check that the command works, not that two
    machines draw identically.
    """
    inputs, figures = tmp_path / "results", tmp_path / "figures"
    inputs.mkdir()
    needed = ["benchmark.csv", "reconstructions.npz", "lambda_sweep.csv"]
    if not all((RESULTS_DIR / name).exists() for name in needed):
        pytest.skip("the committed results are not present")
    for name in needed:
        shutil.copy(RESULTS_DIR / name, inputs / name)
    # The regulariser comparison is drawn only when the second sweep is there.
    unregularised = RESULTS_DIR / "unregularised" / "benchmark.csv"
    if unregularised.exists():
        (inputs / "unregularised").mkdir()
        shutil.copy(unregularised, inputs / "unregularised" / "benchmark.csv")

    finished = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "make_figures.py"),
            "--results-dir",
            str(inputs),
            "--figures-dir",
            str(figures),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]

    expected_figures = {
        "error_vs_measurements.png",
        "metric_comparison.png",
        "reconstruction_grid.png",
        "regularisation_comparison.png",
        "lambda_sweep.png",
    }
    written = {path.name for path in figures.glob("*.png")}
    assert expected_figures <= written, f"missing {sorted(expected_figures - written)}"

    for table in ("summary.md", "summary.csv", "sample_efficiency.md", "lambda_sweep.md"):
        assert (inputs / table).exists(), f"{table} was not written"

    # The committed tables are the output of this command, so a fresh run of it
    # must agree with them cell for cell.
    assert (inputs / "summary.md").read_text(encoding="utf-8") == (
        RESULTS_DIR / "summary.md"
    ).read_text(encoding="utf-8")
