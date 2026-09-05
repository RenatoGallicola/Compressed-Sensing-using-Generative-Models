"""Tests for the paired significance tests.

The p-values quoted in the report come from ``scripts/run_stats.py``. A
correction applied wrongly is worse than none at all, since it produces
confident-looking numbers, so the adjustment is checked against cases whose
answer is known by hand.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from csgm.config import ROOT_DIR


def _load_run_stats():
    """Import ``scripts/run_stats.py``, which is not part of the package."""
    path = ROOT_DIR / "scripts" / "run_stats.py"
    spec = importlib.util.spec_from_file_location("run_stats", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_stats"] = module
    spec.loader.exec_module(module)
    return module


run_stats = _load_run_stats()


def test_holm_matches_a_hand_computed_example():
    """Sorted p times the number of remaining tests, made monotone."""
    adjusted = run_stats.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    # 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 lifted to 0.06.
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_holm_is_monotone_in_the_sorted_order():
    """A step-down procedure may never rank a larger raw p as more significant."""
    rng = np.random.default_rng(0)
    p = rng.random(20)
    adjusted = run_stats.holm(p)[np.argsort(p)]
    assert np.all(np.diff(adjusted) >= -1e-12)


def test_holm_never_reports_more_significance_than_the_raw_value():
    rng = np.random.default_rng(1)
    p = rng.random(15)
    assert np.all(run_stats.holm(p) >= p - 1e-12)
    assert np.all(run_stats.holm(p) <= 1.0)


def test_a_single_test_is_left_alone():
    np.testing.assert_allclose(run_stats.holm(np.array([0.03])), [0.03])


def test_comparison_orients_the_difference_towards_the_first_method():
    """A positive mean difference has to mean the first method is the better one."""
    index = pd.MultiIndex.from_product([[10], range(6)], names=["m", "image"])
    wide = pd.DataFrame({"good": np.full(6, 0.01), "bad": np.linspace(0.02, 0.07, 6)}, index=index)

    (record,) = run_stats.compare(wide, "good", "bad")
    assert record["mean_difference"] > 0
    assert record["wins"] == 6
    assert record["mean_error_better"] < record["mean_error_worse"]

    (flipped,) = run_stats.compare(wide, "bad", "good")
    assert flipped["mean_difference"] < 0
    assert flipped["wins"] == 0


def test_identical_methods_are_not_reported_as_different():
    """Wilcoxon is undefined on all-zero differences and must not raise."""
    index = pd.MultiIndex.from_product([[10], range(8)], names=["m", "image"])
    values = np.linspace(0.01, 0.05, 8)
    wide = pd.DataFrame({"a": values, "b": values}, index=index)

    (record,) = run_stats.compare(wide, "a", "b")
    assert record["p_wilcoxon"] == pytest.approx(1.0)
    assert record["mean_difference"] == pytest.approx(0.0)
