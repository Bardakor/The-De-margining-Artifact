"""Spec §8 Layer 2 — invariants that must hold for any admissible parameters.

Hypothesis searches the admissible region rather than trusting hand-picked
fixtures, which is what catches the boundary cases a fixed example misses.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from footy.core.dixon_coles import rho_bounds
from footy.core.markets import (
    asian_fair_odds,
    asian_handicap,
    both_teams_to_score,
    correct_score,
    double_chance,
    match_odds,
    totals,
)
from footy.core.matrix import scoreline_matrix
from footy.market.demargin import shin
from footy.market.overround import apply_overround

rates = st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False)


@st.composite
def admissible_parameters(draw: st.DrawFn) -> tuple[float, float, float]:
    """(lam, mu, rho) with rho strictly inside its admissible region."""
    lam = draw(rates)
    mu = draw(rates)
    lo, hi = rho_bounds(lam, mu)
    inset = (hi - lo) * 0.02
    rho = draw(st.floats(min_value=lo + inset, max_value=hi - inset, allow_nan=False))
    return lam, mu, rho


@given(admissible_parameters())
@settings(max_examples=200, deadline=None)
def test_matrix_is_a_probability_distribution(params: tuple[float, float, float]) -> None:
    m = scoreline_matrix(*params)
    assert np.all(m >= 0.0)
    assert m.sum() == pytest.approx(1.0, abs=1e-9)


@given(admissible_parameters())
@settings(max_examples=200, deadline=None)
def test_every_market_is_a_valid_probability(params: tuple[float, float, float]) -> None:
    m = scoreline_matrix(*params)
    values = [*match_odds(m), *totals(m, 2.5), *both_teams_to_score(m), correct_score(m, 1, 1)]
    for value in values:
        assert 0.0 <= value <= 1.0 + 1e-12


@given(admissible_parameters())
@settings(max_examples=200, deadline=None)
def test_markets_cannot_contradict_one_another(params: tuple[float, float, float]) -> None:
    """Spec §8 Layer 2: they are marginals of one distribution."""
    m = scoreline_matrix(*params)
    home, draw, away = match_odds(m)
    dc_1x, _, dc_x2 = double_chance(m)

    assert home + draw + away == pytest.approx(1.0, abs=1e-9)
    assert dc_1x == pytest.approx(home + draw, abs=1e-9)
    assert dc_x2 == pytest.approx(draw + away, abs=1e-9)

    over, under = totals(m, 2.5)
    assert over + under == pytest.approx(1.0, abs=1e-9)

    triangle = sum(
        correct_score(m, x, y) for x in range(m.shape[0]) for y in range(m.shape[1]) if x > y
    )
    assert triangle == pytest.approx(home, abs=1e-9)


@given(admissible_parameters())
@settings(max_examples=100, deadline=None)
def test_level_ball_handicap_equals_draw_no_bet(params: tuple[float, float, float]) -> None:
    m = scoreline_matrix(*params)
    home, _, away = match_odds(m)
    side, _ = asian_handicap(m, 0.0)
    assert asian_fair_odds(side) == pytest.approx((home + away) / home, abs=1e-9)


@given(
    admissible_parameters(),
    st.floats(min_value=1.001, max_value=1.30, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_overround_hits_its_target_and_shortens_prices(
    params: tuple[float, float, float], target: float
) -> None:
    fair = np.array(match_odds(scoreline_matrix(*params)))
    if np.any(fair <= 1e-6):
        return  # a degenerate simplex is not a book
    odds = apply_overround(fair, target)
    assert float(np.sum(1.0 / odds)) == pytest.approx(target, abs=1e-9)
    assert np.all(odds < 1.0 / fair)


@given(
    st.lists(st.floats(min_value=0.02, max_value=0.9), min_size=2, max_size=4),
    st.floats(min_value=1.001, max_value=1.20, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_shin_recovers_a_simplex(raw: list[float], margin: float) -> None:
    implied = np.array(raw) / sum(raw) * margin
    # A real book never offers decimal odds at or below 1, so no single implied
    # probability reaches 1. Generated books that breach this are not books.
    assume(np.all(implied < 1.0))
    recovered, z = shin(implied)
    assert recovered.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(recovered > 0.0)
    assert 0.0 <= z < 1.0


@given(st.lists(st.floats(min_value=0.05, max_value=0.9), min_size=2, max_size=4))
@settings(max_examples=100, deadline=None)
def test_shin_approaches_the_identity_as_margin_vanishes(raw: list[float]) -> None:
    """Spec §4: all transforms must converge to p as B -> 1."""
    fair = np.array(raw) / sum(raw)
    assume(np.all(fair * 1.0000001 < 1.0))
    recovered, _ = shin(fair * 1.0000001)
    assert recovered == pytest.approx(fair, abs=1e-5)
