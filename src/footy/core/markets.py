"""Markets as marginals of the scoreline matrix.

MODEL.md §6. Every market is a sum over a region of one distribution. No
market has a model of its own, so no two markets can disagree about the same
event.

Indexing convention throughout: m[x, y] is P(home scores x, away scores y).
A home win is therefore x > y, which is strictly below the main diagonal.
"""

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
