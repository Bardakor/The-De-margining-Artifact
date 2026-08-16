"""MODEL.md §8 — margin, in the correct direction."""

from typing import Any

import numpy as np
import pytest

from footy.core.markets import double_chance, match_odds
from footy.core.matrix import scoreline_matrix
from footy.market.overround import apply_overround, margin_double_chance, solve_power_exponent

FAIR = np.array(match_odds(scoreline_matrix(1.6, 1.1, -0.10)))


def test_book_sum_lands_exactly_on_target() -> None:
    for target in (1.02, 1.05, 1.08, 1.15):
        odds = apply_overround(FAIR, target)
        assert float(np.sum(1.0 / odds)) == pytest.approx(target, abs=1e-12)


def test_margin_shortens_every_price() -> None:
    """MODEL.md §8.1: the old implementation had this backwards."""
    odds = apply_overround(FAIR, 1.05)
    assert np.all(odds < 1.0 / FAIR)


def test_exponent_is_one_when_no_margin_is_taken() -> None:
    assert solve_power_exponent(FAIR, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_reproduces_the_favourite_longshot_bias(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §8.2."""
    spec = model_md["favourite_longshot_bias"]
    fair = np.array(match_odds(scoreline_matrix(spec["lam"], spec["mu"], -0.10)))
    offered = apply_overround(fair, spec["book_sum"])
    ratio = (1.0 / fair) / offered

    longshot, favourite = int(np.argmin(fair)), int(np.argmax(fair))
    assert ratio[longshot] == pytest.approx(
        spec["fair_to_offered_ratio_longshot"], abs=spec["tolerance"]
    )
    assert ratio[favourite] == pytest.approx(
        spec["fair_to_offered_ratio_favourite"], abs=spec["tolerance"]
    )
    assert ratio[longshot] > ratio[favourite]


def test_rejects_an_unreachable_target() -> None:
    """MODEL.md §8.2: throw rather than return a plausible-looking wrong book."""
    with pytest.raises(ValueError, match="unreachable"):
        apply_overround(FAIR, float(len(FAIR)) + 0.1)
    with pytest.raises(ValueError, match="at least 1"):
        apply_overround(FAIR, 0.95)


def test_double_chance_is_margined_against_twice_the_book(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §8.3."""
    spec = model_md["double_chance_validity"]
    m = scoreline_matrix(spec["lam"], spec["mu"], -0.10)
    odds = margin_double_chance(m, spec["book_sum"])
    assert float(np.sum(1.0 / odds)) == pytest.approx(2.0 * spec["book_sum"], abs=1e-12)
    assert np.min(odds) == pytest.approx(spec["dc_1x_price"], abs=spec["tolerance"])


def test_double_chance_stays_payable_for_a_heavy_favourite(model_md: dict[str, Any]) -> None:
    """MODEL.md §8.3: summing the already-margined 1X2 legs yields odds below 1."""
    spec = model_md["double_chance_validity"]
    break_point = spec["naive_method_breaks_at"]
    m = scoreline_matrix(break_point["lam"], break_point["mu"], -0.10)

    assert np.all(margin_double_chance(m, 1.05) > 1.0)

    legs = 1.0 / apply_overround(np.array(match_odds(m)), 1.05)
    home, draw, away = legs
    naive_1x = 1.0 / (home + draw)
    assert naive_1x < 1.0, "the naive method should be demonstrably broken here"


def test_double_chance_agrees_with_1x2_on_fair_probabilities() -> None:
    """MODEL.md §8.3: fair probabilities agree; only the margined ones differ."""
    m = scoreline_matrix(1.6, 1.1, -0.10)
    home, draw, away = match_odds(m)
    dc_1x, _, _ = double_chance(m)
    assert dc_1x == pytest.approx(home + draw, abs=1e-12)
