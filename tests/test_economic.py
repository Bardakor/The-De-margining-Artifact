"""Economic evaluation: staking, ROI, closing-line value.

Study spec §6.2. Prediction P3 — that measured edge changes sign between
de-margining transforms — is an economic statement, so these are the functions
that decide whether it holds. A bug here would not produce an obviously wrong
number; it would produce a plausible one.
"""

from __future__ import annotations

import numpy as np
import pytest

from footy.eval.economic import (
    closing_line_value,
    outcome_won,
    settle,
    value_selections,
)


def test_only_selections_the_model_rates_above_the_benchmark_are_backed() -> None:
    model = np.array([0.55, 0.20, 0.25])
    benchmark = np.array([0.50, 0.25, 0.25])
    assert value_selections(model, benchmark).tolist() == [True, False, False]


def test_a_minimum_edge_threshold_filters_marginal_selections() -> None:
    model = np.array([0.51, 0.60])
    benchmark = np.array([0.50, 0.50])
    assert value_selections(model, benchmark, minimum_edge=0.05).tolist() == [False, True]


def test_a_winning_flat_bet_returns_the_net_price() -> None:
    result = settle(
        np.array([0.60]), np.array([0.50]), np.array([2.50]), np.array([True]), staking="flat"
    )
    assert result.stake[0] == 1.0
    assert result.profit[0] == pytest.approx(1.50)
    assert result.roi == pytest.approx(1.50)


def test_a_losing_flat_bet_loses_the_stake() -> None:
    result = settle(np.array([0.60]), np.array([0.50]), np.array([2.50]), np.array([False]))
    assert result.profit[0] == pytest.approx(-1.0)
    assert result.roi == pytest.approx(-1.0)


def test_unselected_rows_stake_and_return_nothing() -> None:
    """A row the model does not rate must contribute neither stake nor profit,
    even when it won — otherwise ROI would credit bets never placed."""
    result = settle(
        np.array([0.40, 0.60]),
        np.array([0.50, 0.50]),
        np.array([2.50, 2.50]),
        np.array([True, True]),
    )
    assert result.stake[0] == 0.0
    assert result.profit[0] == 0.0
    assert result.n_bets == 1


def test_roi_is_profit_over_turnover() -> None:
    result = settle(
        np.array([0.6, 0.6, 0.6]),
        np.array([0.5, 0.5, 0.5]),
        np.array([2.0, 2.0, 2.0]),
        np.array([True, False, False]),
    )
    assert result.turnover == pytest.approx(3.0)
    assert result.total_profit == pytest.approx(1.0 - 1.0 - 1.0)
    assert result.roi == pytest.approx(-1.0 / 3.0)


def test_a_fair_priced_book_returns_roughly_zero_over_many_bets() -> None:
    """Sanity anchor. Betting true probabilities at fair odds is a martingale,
    so ROI must sit near zero. A systematic bias here would masquerade as edge."""
    rng = np.random.default_rng(0)
    n = 20_000
    p = rng.uniform(0.15, 0.85, n)
    odds = 1.0 / p
    won = rng.uniform(size=n) < p
    result = settle(p, np.full(n, 0.0), odds, won)  # benchmark 0 forces every row selected
    assert result.n_bets == n
    assert abs(result.roi) < 0.05


def test_a_genuine_edge_shows_as_positive_roi() -> None:
    """The complement: if the model really is better than the price, ROI must
    be positive, or the settlement logic is inverted."""
    rng = np.random.default_rng(1)
    n = 20_000
    true_p = rng.uniform(0.2, 0.8, n)
    odds = 1.0 / (true_p * 0.9)  # priced as if less likely than they are
    won = rng.uniform(size=n) < true_p
    result = settle(true_p, np.full(n, 0.0), odds, won)
    assert result.roi > 0.05


def test_kelly_staking_scales_with_the_edge() -> None:
    result = settle(
        np.array([0.55, 0.75]),
        np.array([0.50, 0.50]),
        np.array([2.0, 2.0]),
        np.array([True, True]),
        staking="kelly",
    )
    assert result.stake[1] > result.stake[0] > 0.0


def test_kelly_never_stakes_more_than_flat() -> None:
    """Quarter Kelly on a modest edge must be a fraction of a unit."""
    result = settle(
        np.array([0.55]), np.array([0.50]), np.array([2.0]), np.array([True]), staking="kelly"
    )
    assert 0.0 < result.stake[0] < 1.0


def test_no_selections_gives_nan_roi_not_zero() -> None:
    """'No bets placed' and 'bets that broke even' are different findings, and
    collapsing the first into the second would hide empty cells in the study."""
    result = settle(np.array([0.40]), np.array([0.50]), np.array([2.50]), np.array([True]))
    assert result.n_bets == 0
    assert np.isnan(result.roi)


def test_unpayable_odds_are_rejected() -> None:
    with pytest.raises(ValueError, match="must exceed 1"):
        settle(np.array([0.6]), np.array([0.5]), np.array([1.0]), np.array([True]))


def test_an_unknown_staking_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="flat"):
        settle(
            np.array([0.6]),
            np.array([0.5]),
            np.array([2.0]),
            np.array([True]),
            staking="martingale",
        )


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        settle(np.array([0.6, 0.5]), np.array([0.5]), np.array([2.0]), np.array([True]))


# --------------------------------------------------------------------------
# Closing-line value
# --------------------------------------------------------------------------


def test_beating_the_close_is_positive_clv() -> None:
    """Taking 2.10 on a selection that closes at 2.00 means the market moved
    toward the bet."""
    assert float(closing_line_value(np.array([2.10]), np.array([2.00]))[0]) > 0.0


def test_being_beaten_by_the_close_is_negative_clv() -> None:
    assert float(closing_line_value(np.array([2.00]), np.array([2.10]))[0]) < 0.0


def test_clv_is_symmetric_in_the_log() -> None:
    """A ratio is not symmetric; the log is. Taking 2.10 against a 2.00 close
    must be exactly the opposite of taking 2.00 against a 2.10 close."""
    forward = float(closing_line_value(np.array([2.10]), np.array([2.00]))[0])
    reverse = float(closing_line_value(np.array([2.00]), np.array([2.10]))[0])
    assert forward == pytest.approx(-reverse, abs=1e-15)


def test_no_movement_is_zero_clv() -> None:
    assert float(closing_line_value(np.array([2.00]), np.array([2.00]))[0]) == pytest.approx(0.0)


def test_clv_is_zeroed_on_rows_that_were_not_backed() -> None:
    clv = closing_line_value(
        np.array([2.10, 2.10]), np.array([2.00, 2.00]), selected=np.array([True, False])
    )
    assert clv[0] > 0.0
    assert clv[1] == 0.0


def test_clv_rejects_unpayable_odds() -> None:
    with pytest.raises(ValueError, match="must exceed 1"):
        closing_line_value(np.array([1.0]), np.array([2.0]))


# --------------------------------------------------------------------------
# Outcome flattening
# --------------------------------------------------------------------------


def test_exactly_one_selection_per_match_wins() -> None:
    won = outcome_won(np.array([0, 1, 2]), n_outcomes=3)
    assert won.tolist() == [True, False, False, False, True, False, False, False, True]
    assert int(won.reshape(-1, 3).sum(axis=1).min()) == 1
    assert int(won.reshape(-1, 3).sum(axis=1).max()) == 1


def test_two_outcome_markets_flatten_correctly() -> None:
    assert outcome_won(np.array([0, 1]), n_outcomes=2).tolist() == [True, False, False, True]


def test_an_out_of_range_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        outcome_won(np.array([3]), n_outcomes=3)
