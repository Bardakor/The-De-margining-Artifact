"""MODEL.md §9 — recovering true probabilities from offered odds."""

from typing import Any

import numpy as np
import pytest

from footy.market.demargin import (
    METHODS,
    _shin_probabilities,
    demargin,
    demargin_all,
    shin,
)


def test_reproduces_the_model_md_round_trip(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §9."""
    spec = model_md["shin_round_trip"]
    recovered, z = shin(spec["implied"])
    assert recovered == pytest.approx(spec["recovered"], abs=spec["tolerance"])
    assert z == pytest.approx(spec["z"], abs=spec["tolerance"])


def test_recovered_probabilities_sum_to_one() -> None:
    recovered, _ = shin([0.5, 0.35, 0.25])
    assert float(recovered.sum()) == pytest.approx(1.0, abs=1e-12)


def test_a_fair_book_is_returned_unchanged() -> None:
    """With no margin there is no insider mass to remove."""
    fair = np.array([0.5, 0.3, 0.2])
    recovered, z = shin(fair)
    assert recovered == pytest.approx(fair, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_shortens_the_longshot_more_than_proportional_normalisation() -> None:
    """Shin attributes margin to insider trading, which concentrates on longshots."""
    implied = np.array([0.5, 0.35, 0.25])
    recovered, _ = shin(implied)
    proportional = implied / implied.sum()
    longshot = int(np.argmin(implied))
    assert recovered[longshot] < proportional[longshot]


def test_rejects_a_book_summing_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        shin([0.3, 0.3, 0.3])


# --------------------------------------------------------------------------
# The four transforms — the study's instrument (spec §4)
# --------------------------------------------------------------------------


def _book(odds: list[float]) -> np.ndarray:
    return 1.0 / np.array(odds, dtype=float)


MARGINED = _book([1.85, 3.50, 4.00])
TWO_WAY = _book([1.90, 2.00])


@pytest.mark.parametrize("method", METHODS)
def test_every_transform_returns_a_simplex(method: str) -> None:
    result = demargin(MARGINED, method)
    assert float(result.probabilities.sum()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(result.probabilities > 0.0)
    assert result.method == method


@pytest.mark.parametrize("method", METHODS)
def test_every_transform_preserves_the_ordering_of_outcomes(method: str) -> None:
    """A transform that reordered favourite and longshot would not be removing
    margin, it would be changing the forecast."""
    result = demargin(MARGINED, method)
    assert list(np.argsort(result.probabilities)) == list(np.argsort(MARGINED))


@pytest.mark.parametrize("method", METHODS)
def test_every_transform_is_the_identity_on_a_fair_book(method: str) -> None:
    """Spec §4: all four must converge to p as B -> 1. With no margin there is
    nothing to remove, so any change is an artefact of the solver."""
    fair = np.array([0.5, 0.3, 0.2])
    assert demargin(fair, method).probabilities == pytest.approx(fair, abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_every_transform_approaches_the_identity_as_margin_vanishes(method: str) -> None:
    fair = np.array([0.5, 0.3, 0.2])
    recovered = demargin(fair * 1.0000001, method).probabilities
    assert recovered == pytest.approx(fair, abs=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_every_transform_shortens_every_outcome(method: str) -> None:
    """Removing margin must reduce total mass from B to 1, and no individual
    probability may rise above its raw implied value."""
    recovered = demargin(MARGINED, method).probabilities
    assert np.all(recovered <= MARGINED + 1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_every_transform_handles_a_two_outcome_book(method: str) -> None:
    """Over/Under 2.5 is the P2 contrast case."""
    result = demargin(TWO_WAY, method)
    assert float(result.probabilities.sum()) == pytest.approx(1.0, abs=1e-12)


def test_proportional_divides_by_the_book_sum() -> None:
    result = demargin(MARGINED, "proportional")
    assert result.probabilities == pytest.approx(MARGINED / MARGINED.sum(), abs=1e-15)


def test_power_exponent_solves_its_defining_equation() -> None:
    result = demargin(MARGINED, "power")
    assert float(np.sum(MARGINED**result.parameter)) == pytest.approx(1.0, abs=1e-9)
    assert result.parameter > 1.0


def test_odds_ratio_holds_the_odds_ratio_constant() -> None:
    """The transform's defining property: p/(1-p) / [pi/(1-pi)] is the same c
    for every outcome."""
    result = demargin(MARGINED, "odds_ratio")
    pi = result.probabilities
    ratios = (MARGINED / (1 - MARGINED)) / (pi / (1 - pi))
    assert ratios == pytest.approx(np.full_like(ratios, result.parameter), rel=1e-6)


def test_shin_z_lies_in_its_modelled_range() -> None:
    """z is a proportion of insider money, so it must be a proportion."""
    result = demargin(MARGINED, "shin")
    assert 0.0 <= result.parameter < 1.0


def test_shin_limit_as_z_vanishes_is_p_over_sqrt_b() -> None:
    """Regression. The z<=0 branch returned p/B, but the true limit of Shin's
    closed form as z -> 0 is p/sqrt(B). The wrong limit sums to exactly 1 and
    so looks plausible, while breaking the monotonicity the bisection assumes."""
    p, book_sum = MARGINED, float(MARGINED.sum())
    assert _shin_probabilities(p, book_sum, 0.0) == pytest.approx(p / np.sqrt(book_sum))
    assert float(_shin_probabilities(p, book_sum, 0.0).sum()) == pytest.approx(np.sqrt(book_sum))


def test_transforms_disagree_on_the_longshot_of_a_margined_book() -> None:
    """If they agreed there would be no study. This is the premise, asserted."""
    longshot = int(np.argmin(MARGINED))
    values = [demargin(MARGINED, m).probabilities[longshot] for m in METHODS]
    assert max(values) - min(values) > 1e-3


def test_disagreement_grows_with_the_book_margin() -> None:
    """Prediction P1, on synthetic books: the transforms coincide as B -> 1,
    so their spread can only grow with the mass being removed."""

    def longshot_spread(scale: float) -> float:
        fair = np.array([0.5, 0.3, 0.2])
        book = fair * scale
        values = [demargin(book, m).probabilities[2] for m in METHODS]
        return float(max(values) - min(values))

    spreads = [longshot_spread(s) for s in (1.01, 1.03, 1.05, 1.08, 1.12)]
    assert spreads == sorted(spreads), f"spread not monotone in margin: {spreads}"


def test_three_outcome_disagreement_exceeds_two_outcome() -> None:
    """Prediction P2: the transforms differ chiefly in longshot treatment, and
    a near-even two-way book has no longshot."""

    def spread(fair: np.ndarray, scale: float) -> float:
        book = fair * scale
        per_outcome = [
            max(demargin(book, m).probabilities[i] for m in METHODS)
            - min(demargin(book, m).probabilities[i] for m in METHODS)
            for i in range(len(fair))
        ]
        return float(max(per_outcome))

    three_way = spread(np.array([0.55, 0.25, 0.20]), 1.07)
    two_way = spread(np.array([0.52, 0.48]), 1.07)
    assert three_way > two_way


def test_demargin_all_returns_every_method() -> None:
    results = demargin_all(MARGINED)
    assert set(results) == set(METHODS)
    assert all(r.book_sum == pytest.approx(float(MARGINED.sum())) for r in results.values())


def test_unknown_method_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="unknown transform"):
        demargin(MARGINED, "wishful_thinking")


@pytest.mark.parametrize("method", METHODS)
def test_a_book_summing_below_one_is_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        demargin(np.array([0.3, 0.3, 0.3]), method)


@pytest.mark.parametrize("method", METHODS)
def test_degenerate_books_are_rejected(method: str) -> None:
    with pytest.raises(ValueError):
        demargin(np.array([1.5, 0.2]), method)
    with pytest.raises(ValueError):
        demargin(np.array([0.9]), method)
