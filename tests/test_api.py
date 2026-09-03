"""Tests for the public surface of the package.

The README documents a short example against ``csgm``'s top-level names. These
tests keep that example from rotting silently when the internals move.
"""

from __future__ import annotations

import inspect

import pytest

import csgm


def test_everything_in_all_is_importable():
    missing = [name for name in csgm.__all__ if not hasattr(csgm, name)]
    assert missing == []


def test_all_is_sorted_and_unique():
    assert csgm.__all__ == sorted(set(csgm.__all__))


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("load_mnist", ["flatten", "normalize"]),
        ("gaussian_measurement_matrix", ["m", "n", "seed", "dtype"]),
        ("measure", ["x", "A", "noise_std", "seed"]),
        ("recover", ["generator", "y", "A", "latent_dim", "config", "track_history"]),
        ("per_pixel_l2", ["x_hat", "x"]),
        ("lasso_dct_recover", ["y", "A", "alpha", "max_iter", "clip"]),
    ],
)
def test_documented_call_shapes(name, parameters):
    """The keyword names the README and the notebooks use must keep working."""
    signature = inspect.signature(getattr(csgm, name))
    assert [p for p in parameters if p not in signature.parameters] == []


def test_load_generator_is_reachable_from_the_models_subpackage():
    from csgm.models import load_generator

    assert "latent_dim" in inspect.signature(load_generator).parameters
