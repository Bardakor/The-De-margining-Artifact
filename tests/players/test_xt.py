import numpy as np
import pytest

from footy.players import xt

# Cell centres on one row (j = 6) of the 16 x 12 grid.
A = (116.0, 43.0)  # i = 15
B = (101.0, 43.0)  # i = 13
C = (86.0, 43.0)  # i = 11


def _toy() -> xt.XTGrid:
    """Planted truth, solved by hand.

    A: 10 shots, 4 goals, no move            -> xT = 0.4
    B: 5 shots, 1 goal; 15 moves, 10 reach A -> xT = 0.25 * 0.2 + 0.75 * (10/15) * 0.4 = 0.25
    C: no shot; 4 moves: 2 reach B, 1 stays in C, 1 fails
                                             -> xT = 0.5 * 0.25 + 0.25 * xT(C) = 1/6
    """
    shots = [A] * 10 + [B] * 5
    goals = [1] * 4 + [0] * 6 + [1] + [0] * 4
    starts = [B] * 15 + [C] * 4
    ends = [A] * 10 + [C] * 5 + [B, B, C, A]
    success = [True] * 10 + [False] * 5 + [True, True, True, False]
    sx, sy = np.array(shots).T
    mx, my = np.array(starts).T
    ex, ey = np.array(ends).T
    return xt.fit(sx, sy, goals, mx, my, ex, ey, success, tol=1e-12)


def test_cells_cover_the_pitch_edges() -> None:
    assert xt.cell(0.0, 0.0) == 0
    assert xt.cell(7.5, 0.0) == xt.N_Y
    assert xt.cell(120.0, 80.0) == xt.N_X * xt.N_Y - 1
    assert xt.cell(-1.0, 81.0) == xt.N_Y - 1


def test_planted_toy_grid_is_recovered() -> None:
    grid = _toy()
    got = xt.value(grid, *np.array([A, B, C]).T)
    np.testing.assert_allclose(got, [0.4, 0.25, 1 / 6], atol=1e-10)
    assert grid.values.sum() == pytest.approx(0.4 + 0.25 + 1 / 6)
    assert grid.iterations > 2
    assert grid.max_change < 1e-12


def test_xt_added_telescopes_over_an_unbroken_chain() -> None:
    grid = _toy()
    path = np.array([C, (90.0, 10.0), B, C, B, A])
    added = xt.xt_added(grid, path[:-1, 0], path[:-1, 1], path[1:, 0], path[1:, 1])
    end, start = xt.value(grid, *path[[-1, 0]].T)
    assert added.sum() == pytest.approx(end - start)
    assert added.sum() == pytest.approx(0.4 - 1 / 6)


def test_non_convergence_raises() -> None:
    with pytest.raises(ValueError, match="did not converge"):
        xt.fit([A[0]], [A[1]], [1], [B[0]], [B[1]], [A[0]], [A[1]], [True], max_iter=1)
