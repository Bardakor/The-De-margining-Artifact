"""Tests for effect-size translation and correlation intervals."""

from __future__ import annotations

import numpy as np
import pytest

from footy.eval.effect import (
    bootstrap_correlation,
    bootstrap_mean_difference,
    cluster_bootstrap_correlation,
    equivalent_sample_size,
    spearman,
    spread_as_share_of_gap,
)


def test_spearman_matches_a_known_ranking() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 0.7 * x + rng.normal(size=60)
    rho, low, high = bootstrap_correlation(x, y, rng=rng, n_resamples=2000)
    assert low < rho < high
    assert low >= -1.0 and high <= 1.0


def test_a_small_sample_gives_a_wide_interval() -> None:
    """The point of reporting it: rho on 14 books is not a precise quantity."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=14)
    y = 0.6 * x + rng.normal(size=14)
    _, low, high = bootstrap_correlation(x, y, rng=rng, n_resamples=2000)
    assert high - low > 0.4


def test_cluster_bootstrap_is_wider_when_dependence_is_within_cluster() -> None:
    """Fourteen independent books, each duplicated into several cells, must not
    be treated as 70 independent observations."""
    rng = np.random.default_rng(2)
    books = np.arange(14)
    x_book = rng.normal(size=14)
    y_book = 0.5 * x_book + rng.normal(size=14)
    x = np.repeat(x_book, 5)
    y = np.repeat(y_book, 5)
    clusters = np.repeat(books, 5)
    _, pair_low, pair_high = bootstrap_correlation(x, y, rng=rng, n_resamples=1500)
    _, cl_low, cl_high = cluster_bootstrap_correlation(x, y, clusters, rng=rng, n_resamples=1500)
    assert (cl_high - cl_low) > (pair_high - pair_low)


def test_paired_mean_difference_brackets_the_point() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.02, 0.01, size=15)
    b = rng.normal(0.01, 0.01, size=15)
    point, low, high = bootstrap_mean_difference(a, b, rng=rng, n_resamples=2000)
    assert low < point < high


def test_spread_expressed_against_the_model_gap() -> None:
    """An RPS spread of 1e-4 against a model-market gap of 6.5e-3 is 1.5% of
    the quantity a paper would be claiming."""
    assert spread_as_share_of_gap(1e-4, 6.5e-3) == pytest.approx(0.01538, abs=1e-5)


def test_equivalent_sample_size_grows_as_the_spread_shrinks() -> None:
    big = equivalent_sample_size(1e-3, per_match_sd=0.2)
    small = equivalent_sample_size(1e-4, per_match_sd=0.2)
    assert small > big
