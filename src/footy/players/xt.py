"""Expected threat (xT) on a 16 x 12 pitch grid, Singh (2019) formulation. Pure.

For every cell z,

    xT(z) = s(z) g(z) + m(z) * sum_z' T(z -> z') xT(z')

    s(z) = shots / (shots + moves)        chance the action taken in z is a shot
    m(z) = moves / (shots + moves)        chance it is a move (pass or carry)
    g(z) = goals / shots                  chance a shot from z scores
    T(z -> z') = successful moves z -> z' / all moves from z

Failed moves stay in T's denominator and lead nowhere, so losing the ball is worth
0. A cell with no action has s = m = 0; a cell with moves but no shot has g = 0.
xT is the fixed point of the equation, reached by iterating from xT = 0 until the
largest change in one sweep is below ``tol``.

Coordinates are StatsBomb's: x in [0, 120] along the pitch, y in [0, 80] across it,
the acting team attacking towards x = 120. Cell (i, j) covers
x in [7.5 i, 7.5 (i + 1)) and y in [80 j / 12, 80 (j + 1) / 12); points on or
beyond the far edges fall in the last cell.

``xt_added`` of a move is xT(end cell) - xT(start cell). Which actions get it is the
caller's choice; over an unbroken chain of moves it telescopes to
xT(last end) - xT(first start).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

N_X = 16
N_Y = 12
LENGTH = 120.0
WIDTH = 80.0

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class XTGrid:
    values: FloatArray  # shape (N_X, N_Y): [i, j] = i-th cell along the pitch, j-th across
    iterations: int
    max_change: float  # largest change in the last sweep


def cell(x: npt.ArrayLike, y: npt.ArrayLike) -> IntArray:
    """Flat cell index i * N_Y + j of each point."""
    ix = np.clip(np.floor(np.asarray(x, dtype=float) / LENGTH * N_X), 0, N_X - 1)
    iy = np.clip(np.floor(np.asarray(y, dtype=float) / WIDTH * N_Y), 0, N_Y - 1)
    return (ix * N_Y + iy).astype(np.int64)


def fit(
    shot_x: npt.ArrayLike,
    shot_y: npt.ArrayLike,
    shot_goal: npt.ArrayLike,
    move_x: npt.ArrayLike,
    move_y: npt.ArrayLike,
    move_end_x: npt.ArrayLike,
    move_end_y: npt.ArrayLike,
    move_success: npt.ArrayLike,
    *,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> XTGrid:
    """Fit the grid from shots (with outcome) and move attempts (with success flag)."""
    n = N_X * N_Y
    shot_cell = cell(shot_x, shot_y)
    shots = np.bincount(shot_cell, minlength=n).astype(float)
    goals = np.bincount(shot_cell, weights=np.asarray(shot_goal, dtype=float), minlength=n)
    start = cell(move_x, move_y)
    ok = np.asarray(move_success, dtype=bool)
    moves = np.bincount(start, minlength=n).astype(float)
    counts = np.zeros((n, n))
    np.add.at(counts, (start[ok], cell(move_end_x, move_end_y)[ok]), 1.0)

    actions = shots + moves
    s = np.divide(shots, actions, out=np.zeros(n), where=actions > 0)
    m = np.divide(moves, actions, out=np.zeros(n), where=actions > 0)
    g = np.divide(goals, shots, out=np.zeros(n), where=shots > 0)
    transition = np.divide(counts, moves[:, None], out=np.zeros((n, n)), where=moves[:, None] > 0)

    reward = s * g
    step = m[:, None] * transition
    xt = np.zeros(n)
    change = np.inf
    for iteration in range(1, max_iter + 1):
        new = reward + step @ xt
        change = float(np.max(np.abs(new - xt)))
        xt = new
        if change < tol:
            return XTGrid(xt.reshape(N_X, N_Y), iteration, change)
    raise ValueError(f"xT did not converge in {max_iter} sweeps (last change {change:.2e})")


def value(grid: XTGrid, x: npt.ArrayLike, y: npt.ArrayLike) -> FloatArray:
    """xT of the cell containing each point."""
    out: FloatArray = grid.values.ravel()[cell(x, y)]
    return out


def xt_added(
    grid: XTGrid,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    end_x: npt.ArrayLike,
    end_y: npt.ArrayLike,
) -> FloatArray:
    """xT(end cell) - xT(start cell) for each move."""
    return value(grid, end_x, end_y) - value(grid, x, y)
