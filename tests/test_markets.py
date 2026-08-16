"""MODEL.md §6 — every market is a sum over regions of one matrix."""

import pytest

from footy.core.markets import (
    both_teams_to_score,
    correct_score,
    double_chance,
    match_odds,
    totals,
)
from footy.core.matrix import scoreline_matrix

MATRIX = scoreline_matrix(1.6, 1.1, -0.10)


def test_match_odds_partition_the_matrix() -> None:
    home, draw, away = match_odds(MATRIX)
    assert home + draw + away == pytest.approx(1.0, abs=1e-12)
    assert home > away, "lam > mu should favour the home side"


def test_totals_are_complementary() -> None:
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        over, under = totals(MATRIX, line)
        assert over + under == pytest.approx(1.0, abs=1e-12)
    assert totals(MATRIX, 0.5)[0] > totals(MATRIX, 5.5)[0]


def test_btts_is_complementary() -> None:
    yes, no = both_teams_to_score(MATRIX)
    assert yes + no == pytest.approx(1.0, abs=1e-12)


def test_correct_score_cells_sum_to_the_whole() -> None:
    total = sum(
        correct_score(MATRIX, x, y) for x in range(MATRIX.shape[0]) for y in range(MATRIX.shape[1])
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_correct_score_triangle_equals_home_win() -> None:
    """MODEL.md §6: markets are marginals of one matrix, so they cannot disagree."""
    home, _, _ = match_odds(MATRIX)
    triangle = sum(
        correct_score(MATRIX, x, y)
        for x in range(MATRIX.shape[0])
        for y in range(MATRIX.shape[1])
        if x > y
    )
    assert triangle == pytest.approx(home, abs=1e-12)


def test_double_chance_is_the_union_of_1x2_regions() -> None:
    home, draw, away = match_odds(MATRIX)
    dc_1x, dc_12, dc_x2 = double_chance(MATRIX)
    assert dc_1x == pytest.approx(home + draw, abs=1e-12)
    assert dc_12 == pytest.approx(home + away, abs=1e-12)
    assert dc_x2 == pytest.approx(draw + away, abs=1e-12)
    assert dc_1x + dc_12 + dc_x2 == pytest.approx(2.0, abs=1e-12)


def test_btts_agrees_with_the_correct_score_cells() -> None:
    yes, _ = both_teams_to_score(MATRIX)
    cells = sum(
        correct_score(MATRIX, x, y)
        for x in range(1, MATRIX.shape[0])
        for y in range(1, MATRIX.shape[1])
    )
    assert cells == pytest.approx(yes, abs=1e-12)


def test_rejects_a_whole_number_total_line() -> None:
    """A whole line can push, so a two-way over/under cannot describe it."""
    with pytest.raises(ValueError, match="half-integer"):
        totals(MATRIX, 2.0)


def test_correct_score_rejects_out_of_grid() -> None:
    with pytest.raises(IndexError):
        correct_score(MATRIX, 11, 0)
