"""MODEL.md §11 — evaluating a forecast."""

import math

import pytest

from footy.eval.scoring import brier_score, log_loss, ranked_probability_score


def test_perfect_forecast_scores_zero() -> None:
    assert ranked_probability_score([1.0, 0.0, 0.0], 0) == pytest.approx(0.0, abs=1e-12)
    assert brier_score([1.0, 0.0, 0.0], 0) == pytest.approx(0.0, abs=1e-12)


def test_rps_matches_the_closed_form() -> None:
    """MODEL.md §11: RPS = 1/(r-1) * sum_i (cumsum(p) - cumsum(o))**2 over i < r."""
    p, outcome = [0.5, 0.3, 0.2], 1
    o = [0.0, 1.0, 0.0]
    cp = cq = 0.0
    total = 0.0
    for i in range(len(p) - 1):
        cp += p[i]
        cq += o[i]
        total += (cp - cq) ** 2
    assert ranked_probability_score(p, outcome) == pytest.approx(total / (len(p) - 1), abs=1e-12)


def test_rps_is_distance_sensitive_where_brier_is_not() -> None:
    """MODEL.md §11: forecasting a home win scores the same under Brier whether
    the match was drawn or lost. RPS must distinguish them.

    The forecast gives the draw and the away win equal probability, which is what
    isolates ordering from likelihood. Brier then cannot tell the two apart at
    all, because it only sees squared error per category. RPS can, because a draw
    is the nearer miss when you forecast a home win.
    """
    forecast = [0.7, 0.15, 0.15]
    assert brier_score(forecast, 1) == pytest.approx(brier_score(forecast, 2), abs=1e-12)
    assert ranked_probability_score(forecast, 1) < ranked_probability_score(forecast, 2)


def test_brier_sums_over_categories() -> None:
    p, outcome = [0.5, 0.3, 0.2], 1
    o = [0.0, 1.0, 0.0]
    assert brier_score(p, outcome) == pytest.approx(
        sum((a - b) ** 2 for a, b in zip(p, o, strict=True)), abs=1e-12
    )


def test_log_loss_matches_the_closed_form() -> None:
    assert log_loss([0.5, 0.3, 0.2], 1) == pytest.approx(-math.log(0.3), abs=1e-12)


def test_log_loss_is_finite_for_a_zero_probability_outcome() -> None:
    assert math.isfinite(log_loss([1.0, 0.0, 0.0], 1))


def test_rejects_a_forecast_that_does_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ranked_probability_score([0.5, 0.3, 0.3], 0)


def test_rejects_an_out_of_range_outcome() -> None:
    with pytest.raises(IndexError):
        brier_score([0.5, 0.3, 0.2], 3)
