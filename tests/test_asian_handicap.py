"""MODEL.md §6.1 — a handicap bet can push, so one probability cannot describe it."""

import pytest

from footy.core.markets import asian_fair_odds, asian_handicap, match_odds
from footy.core.matrix import scoreline_matrix

MATRIX = scoreline_matrix(1.6, 1.1, -0.10)


def test_each_side_partitions_into_win_push_lose() -> None:
    home, away = asian_handicap(MATRIX, 0.0)
    for side in (home, away):
        assert side.win + side.push + side.lose == pytest.approx(1.0, abs=1e-12)


def test_the_two_sides_mirror_each_other() -> None:
    home, away = asian_handicap(MATRIX, -0.5)
    assert home.win == pytest.approx(away.lose, abs=1e-12)
    assert home.lose == pytest.approx(away.win, abs=1e-12)
    assert home.push == pytest.approx(away.push, abs=1e-12)


def test_level_ball_push_is_the_draw() -> None:
    _, draw, _ = match_odds(MATRIX)
    home, _ = asian_handicap(MATRIX, 0.0)
    assert home.push == pytest.approx(draw, abs=1e-12)


def test_level_ball_equals_draw_no_bet() -> None:
    """MODEL.md §6: the level-ball handicap must equal the draw-no-bet price."""
    p_home, draw, p_away = match_odds(MATRIX)
    dnb_home = p_home / (p_home + p_away)
    home, _ = asian_handicap(MATRIX, 0.0)
    assert asian_fair_odds(home) == pytest.approx(1.0 / dnb_home, abs=1e-12)


def test_fair_odds_account_for_the_returned_stake() -> None:
    """MODEL.md §6.1: d_fair = 1 + P(lose)/P(win)."""
    home, _ = asian_handicap(MATRIX, -0.5)
    assert asian_fair_odds(home) == pytest.approx(1.0 + home.lose / home.win, abs=1e-12)


def test_half_line_cannot_push() -> None:
    home, away = asian_handicap(MATRIX, -0.5)
    assert home.push == 0.0
    assert away.push == 0.0


def test_quarter_line_splits_the_two_adjacent_lines() -> None:
    """MODEL.md §6.1: quarter lines split the stake evenly, which is how they settle."""
    quarter, _ = asian_handicap(MATRIX, -0.25)
    level, _ = asian_handicap(MATRIX, 0.0)
    half, _ = asian_handicap(MATRIX, -0.5)
    assert quarter.win == pytest.approx((level.win + half.win) / 2.0, abs=1e-12)
    assert quarter.push == pytest.approx((level.push + half.push) / 2.0, abs=1e-12)
    assert quarter.lose == pytest.approx((level.lose + half.lose) / 2.0, abs=1e-12)


def test_a_bigger_handicap_against_the_home_side_lowers_its_win_probability() -> None:
    assert asian_handicap(MATRIX, -1.5)[0].win < asian_handicap(MATRIX, -0.5)[0].win


def test_rejects_an_eighth_line() -> None:
    with pytest.raises(ValueError, match="quarter"):
        asian_handicap(MATRIX, -0.125)
