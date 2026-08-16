"""Markets as marginals of the scoreline matrix.

MODEL.md §6. Every market is a sum over a region of one distribution. No
market has a model of its own, so no two markets can disagree about the same
event.

Indexing convention throughout: m[x, y] is P(home scores x, away scores y).
A home win is therefore x > y, which is strictly below the main diagonal.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]


def match_odds(m: Matrix) -> tuple[float, float, float]:
    """1X2 probabilities: (home, draw, away)."""
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return home, draw, away


def totals(m: Matrix, line: float) -> tuple[float, float]:
    """Over/under a half-integer goal line: (over, under).

    Whole lines are rejected: they can push, so two probabilities cannot
    describe the market.
    """
    if float(line).is_integer():
        raise ValueError(f"line must be a half-integer, got {line}")
    x = np.arange(m.shape[0])[:, None]
    y = np.arange(m.shape[1])[None, :]
    over = float(m[(x + y) > line].sum())
    return over, 1.0 - over


def both_teams_to_score(m: Matrix) -> tuple[float, float]:
    """(yes, no) — yes is every cell with x > 0 and y > 0."""
    yes = float(m[1:, 1:].sum())
    return yes, 1.0 - yes


def correct_score(m: Matrix, x: int, y: int) -> float:
    """The single cell (x, y)."""
    if not (0 <= x < m.shape[0] and 0 <= y < m.shape[1]):
        raise IndexError(f"scoreline ({x}, {y}) outside the {m.shape} grid")
    return float(m[x, y])


def double_chance(m: Matrix) -> tuple[float, float, float]:
    """(1X, 12, X2) — unions of the 1X2 regions. Sums to 2 by construction."""
    home, draw, away = match_odds(m)
    return home + draw, home + away, draw + away


class AsianOutcome(NamedTuple):
    """One side of a handicap bet. Sums to 1."""

    win: float
    push: float
    lose: float


def _whole_or_half_handicap(m: Matrix, handicap: float) -> tuple[AsianOutcome, AsianOutcome]:
    """Handicap applied to the home side: win when x + h > y, push when equal."""
    x = np.arange(m.shape[0])[:, None]
    y = np.arange(m.shape[1])[None, :]
    adjusted = x + handicap - y

    home_win = float(m[adjusted > 0].sum())
    home_push = float(m[adjusted == 0].sum())
    home_lose = float(m[adjusted < 0].sum())

    home = AsianOutcome(home_win, home_push, home_lose)
    away = AsianOutcome(home_lose, home_push, home_win)
    return home, away


def asian_handicap(m: Matrix, handicap: float) -> tuple[AsianOutcome, AsianOutcome]:
    """Asian handicap applied to the home side: (home, away).

    Quarter lines split the stake evenly across the two adjacent lines, which
    is how they settle in practice (MODEL.md §6.1).

    Raises:
        ValueError: If the handicap is not a multiple of 0.25.
    """
    quarters = handicap * 4.0
    if not float(quarters).is_integer():
        raise ValueError(f"handicap must be a multiple of a quarter goal, got {handicap}")

    is_quarter_line = int(quarters) % 2 != 0
    if not is_quarter_line:
        return _whole_or_half_handicap(m, handicap)

    lower_home, lower_away = _whole_or_half_handicap(m, handicap - 0.25)
    upper_home, upper_away = _whole_or_half_handicap(m, handicap + 0.25)

    def blend(a: AsianOutcome, b: AsianOutcome) -> AsianOutcome:
        return AsianOutcome((a.win + b.win) / 2.0, (a.push + b.push) / 2.0, (a.lose + b.lose) / 2.0)

    return blend(lower_home, upper_home), blend(lower_away, upper_away)


def asian_fair_odds(outcome: AsianOutcome) -> float:
    """Fair decimal price, accounting for the stake being returned on a push.

    MODEL.md §6.1:  d_fair = 1 + P(lose) / P(win)
    """
    if not outcome.win > 0.0:
        raise ValueError("cannot price a handicap the side can never win")
    return 1.0 + outcome.lose / outcome.win
