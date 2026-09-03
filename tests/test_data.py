"""Tests for the image-sampling helpers.

``load_mnist`` itself is not tested: it downloads a dataset, which does not
belong in a unit-test run.
"""

from __future__ import annotations

import numpy as np
import pytest

from csgm.data import sample_images


@pytest.fixture
def labelled():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(10), 20)
    return rng.random((200, 28, 28, 1)), labels


def test_returns_the_requested_number_without_repetition(labelled):
    x, _ = labelled
    picked = sample_images(x, 12, seed=1)
    assert picked.shape == (12, 28, 28, 1)
    assert len({row.tobytes() for row in picked}) == 12


def test_is_reproducible(labelled):
    x, _ = labelled
    np.testing.assert_array_equal(sample_images(x, 8, seed=3), sample_images(x, 8, seed=3))
    assert not np.array_equal(sample_images(x, 8, seed=3), sample_images(x, 8, seed=4))


def test_stratified_sampling_covers_every_class(labelled):
    """Averaging a benchmark over 10 images is only fair if all digits appear."""
    x, labels = labelled
    index = {row.tobytes(): i for i, row in enumerate(x)}
    picked = sample_images(x, 10, labels=labels, stratified=True, seed=5)
    drawn = sorted(labels[index[row.tobytes()]] for row in picked)
    assert drawn == list(range(10))


def test_stratified_sampling_spreads_a_non_multiple_count(labelled):
    x, labels = labelled
    index = {row.tobytes(): i for i, row in enumerate(x)}
    picked = sample_images(x, 13, labels=labels, stratified=True, seed=6)
    counts = np.bincount([labels[index[row.tobytes()]] for row in picked], minlength=10)
    assert counts.sum() == 13
    assert counts.max() - counts.min() <= 1


def test_stratified_sampling_requires_labels(labelled):
    x, _ = labelled
    with pytest.raises(ValueError, match="requires `labels`"):
        sample_images(x, 4, stratified=True)


def test_asking_for_too_many_images_raises(labelled):
    x, _ = labelled
    with pytest.raises(ValueError, match="cannot draw"):
        sample_images(x, 500)
