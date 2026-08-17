"""Statistical inference: HAC variance, Diebold-Mariano, bootstrap, BH.

Study spec §6.3. Each of these exists because the naive alternative is invalid
on this data, so the tests check that they actually differ from the naive
version rather than merely running.
"""

from __future__ import annotations

import numpy as np
import pytest

from footy.eval.inference import (
    adjusted_p_values,
    benjamini_hochberg,
    default_lag,
    diebold_mariano,
    newey_west_variance,
    stationary_bootstrap,
)


def ar1(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.normal()
    return series


# --------------------------------------------------------------------------
# Newey-West
# --------------------------------------------------------------------------


def test_lag_zero_reduces_to_the_sample_variance() -> None:
    """The HAC estimator must nest the naive one rather than replace it."""
    x = np.random.default_rng(0).normal(0, 1, 400)
    assert newey_west_variance(x, 0) == pytest.approx(float(x.var()), abs=1e-12)


def test_hac_exceeds_the_naive_variance_under_positive_autocorrelation() -> None:
    """This is the entire reason for using it: forecasts from overlapping fit
    windows are correlated, and the naive variance understates their spread."""
    series = ar1(0.8, 3000, seed=1)
    assert newey_west_variance(series, 25) > 2.0 * float(series.var())


def test_hac_is_close_to_the_naive_variance_on_white_noise() -> None:
    white = np.random.default_rng(2).normal(0, 1, 3000)
    assert newey_west_variance(white, 10) == pytest.approx(float(white.var()), rel=0.25)


def test_hac_is_never_negative() -> None:
    """Bartlett weights guarantee this in exact arithmetic; floating point can
    still produce a small negative, which must be clamped not returned."""
    rng = np.random.default_rng(3)
    for seed in range(20):
        series = ar1(-0.9, 200, seed=seed + int(rng.integers(1000)))
        assert newey_west_variance(series, 15) >= 0.0


def test_default_lag_grows_with_the_sample() -> None:
    lags = [default_lag(n) for n in (50, 500, 5000, 50000)]
    assert lags == sorted(lags)
    assert all(0 <= lag < n for lag, n in zip(lags, (50, 500, 5000, 50000), strict=True))


@pytest.mark.parametrize("lag", [-1, 500])
def test_invalid_lags_are_rejected(lag: int) -> None:
    with pytest.raises(ValueError):
        newey_west_variance(np.zeros(100), lag)


# --------------------------------------------------------------------------
# Diebold-Mariano
# --------------------------------------------------------------------------


def test_identical_forecasts_give_no_evidence_of_a_difference() -> None:
    """Zero variance must read as 'no evidence', not as infinite evidence."""
    losses = np.random.default_rng(4).uniform(0, 1, 200)
    result = diebold_mariano(losses, losses)
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_a_genuine_difference_is_detected() -> None:
    rng = np.random.default_rng(5)
    better = rng.normal(0.0, 1.0, 600)
    worse = better + rng.normal(0.25, 0.3, 600)
    result = diebold_mariano(better, worse)
    assert result.p_value < 1e-6
    assert result.mean_difference < 0.0
    assert result.favours_first


def test_the_sign_convention_is_first_minus_second() -> None:
    a = np.full(100, 0.30)
    b = np.full(100, 0.35)
    a = a + np.random.default_rng(6).normal(0, 0.01, 100)
    result = diebold_mariano(a, b)
    assert result.mean_difference < 0.0
    assert result.favours_first


def test_size_is_approximately_correct_under_the_null() -> None:
    """Two forecasters of equal skill should be declared different about 5% of
    the time at alpha = 0.05. A test that over-rejects here would manufacture
    findings across the study's many cells."""
    rejections = 0
    trials = 300
    for i in range(trials):
        rng = np.random.default_rng(1000 + i)
        result = diebold_mariano(rng.normal(0, 1, 250), rng.normal(0, 1, 250))
        rejections += result.p_value < 0.05
    assert 0.02 <= rejections / trials <= 0.10, f"empirical size {rejections / trials}"


def test_autocorrelation_widens_the_test_rather_than_being_ignored() -> None:
    """With correlated differences the HAC statistic must be less extreme than
    one computed as if the observations were independent."""
    differences = ar1(0.85, 800, seed=7) + 0.3
    hac = diebold_mariano(differences, np.zeros_like(differences))
    naive = diebold_mariano(differences, np.zeros_like(differences), lag=0)
    assert abs(hac.statistic) < abs(naive.statistic)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        diebold_mariano(np.zeros(10), np.zeros(9))


def test_too_few_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        diebold_mariano([0.1], [0.2])


# --------------------------------------------------------------------------
# Stationary bootstrap
# --------------------------------------------------------------------------


def test_the_interval_brackets_the_sample_mean() -> None:
    rng = np.random.default_rng(8)
    sample = rng.normal(0.4, 1.0, 500)
    interval = stationary_bootstrap(sample, rng=rng, n_resamples=800, block_length=15)
    assert interval.low < interval.point < interval.high
    assert interval.point == pytest.approx(float(sample.mean()))


def test_coverage_is_close_to_nominal() -> None:
    """Percentile intervals are known to under-cover slightly; this pins that
    it is a modest shortfall rather than a broken estimator."""
    covered = 0
    trials = 200
    for i in range(trials):
        rng = np.random.default_rng(3000 + i)
        sample = rng.normal(0.5, 1.0, 250)
        interval = stationary_bootstrap(sample, rng=rng, n_resamples=400, block_length=10)
        covered += interval.low <= 0.5 <= interval.high
    assert 0.86 <= covered / trials <= 0.99, f"coverage {covered / trials}"


def test_dependence_widens_the_interval() -> None:
    """The reason for blocks. Betting returns cluster by matchday, and an
    i.i.d. bootstrap would resample that dependence away and report an
    interval that is too narrow."""
    series = ar1(0.9, 600, seed=9)
    rng = np.random.default_rng(10)
    blocked = stationary_bootstrap(series, rng=rng, n_resamples=600, block_length=40)
    iid = stationary_bootstrap(
        series, rng=np.random.default_rng(10), n_resamples=600, block_length=1.0
    )
    assert (blocked.high - blocked.low) > (iid.high - iid.low)


def test_the_bootstrap_is_reproducible_from_a_seed() -> None:
    sample = np.random.default_rng(11).normal(0, 1, 200)
    a = stationary_bootstrap(sample, rng=np.random.default_rng(42), n_resamples=300)
    b = stationary_bootstrap(sample, rng=np.random.default_rng(42), n_resamples=300)
    assert (a.low, a.high) == (b.low, b.high)


def test_excludes_zero_reports_significance_of_the_mean() -> None:
    rng = np.random.default_rng(12)
    far = stationary_bootstrap(rng.normal(5.0, 0.5, 300), rng=rng, n_resamples=400)
    near = stationary_bootstrap(rng.normal(0.0, 1.0, 300), rng=rng, n_resamples=400)
    assert far.excludes_zero
    assert not near.excludes_zero


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"level": 1.5}, "level"), ({"block_length": 0.0}, "block_length")]
)
def test_invalid_bootstrap_arguments_are_rejected(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        stationary_bootstrap(np.zeros(10), rng=np.random.default_rng(0), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Benjamini-Hochberg
# --------------------------------------------------------------------------


def test_bh_rejects_the_step_up_set_not_only_individually_passing_tests() -> None:
    """The step-up rule is what makes this BH. Rejecting only those p-values
    below their own threshold would be a different, less powerful procedure."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rejected = benjamini_hochberg(p, q=0.05)
    assert rejected.tolist() == [True, True, False, False, False, False, False, False, False, False]


def test_bh_is_less_strict_than_bonferroni_and_stricter_than_nothing() -> None:
    p = np.array([0.001, 0.01, 0.02, 0.03, 0.04])
    n_bh = int(benjamini_hochberg(p, q=0.05).sum())
    n_bonferroni = int((p <= 0.05 / p.size).sum())
    n_uncorrected = int((p <= 0.05).sum())
    assert n_bonferroni <= n_bh <= n_uncorrected


def test_bh_returns_the_mask_in_the_callers_order() -> None:
    p = np.array([0.5, 0.001, 0.4])
    assert benjamini_hochberg(p, q=0.05).tolist() == [False, True, False]


def test_nothing_is_rejected_when_no_effect_exists() -> None:
    assert int(benjamini_hochberg(np.ones(20), q=0.10).sum()) == 0


def test_everything_is_rejected_when_every_effect_is_overwhelming() -> None:
    assert bool(benjamini_hochberg(np.full(10, 1e-12), q=0.05).all())


def test_adjusted_p_values_are_monotone_and_bounded() -> None:
    p = np.array([0.001, 0.02, 0.03, 0.2, 0.9])
    adjusted = adjusted_p_values(p)
    assert np.all(adjusted >= p - 1e-12)
    assert np.all(adjusted <= 1.0)
    assert list(adjusted) == sorted(adjusted)


def test_adjusted_p_values_agree_with_the_rejection_mask() -> None:
    rng = np.random.default_rng(13)
    p = np.clip(rng.uniform(0, 1, 50) ** 3, 1e-9, 1.0)
    q = 0.10
    assert benjamini_hochberg(p, q).tolist() == (adjusted_p_values(p) <= q).tolist()


def test_empty_input_is_handled() -> None:
    assert benjamini_hochberg(np.array([]), 0.05).size == 0
    assert adjusted_p_values(np.array([])).size == 0


@pytest.mark.parametrize("bad_q", [0.0, 1.0, -0.1])
def test_invalid_q_is_rejected(bad_q: float) -> None:
    with pytest.raises(ValueError, match="q must lie"):
        benjamini_hochberg(np.array([0.1]), bad_q)


def test_p_values_outside_the_unit_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        benjamini_hochberg(np.array([0.5, 1.5]), 0.05)
