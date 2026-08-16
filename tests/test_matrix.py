"""MODEL.md §5 — the scoreline matrix, and §3.1 as a Layer-1 target."""

from typing import Any

import numpy as np
import pytest

from footy.core.dixon_coles import GRID
from footy.core.matrix import scoreline_matrix


def test_sums_to_one() -> None:
    assert scoreline_matrix(1.6, 1.1, -0.10).sum() == pytest.approx(1.0, abs=1e-12)


def test_shape_and_non_negativity() -> None:
    m = scoreline_matrix(1.6, 1.1, -0.10)
    assert m.shape == (GRID, GRID)
    assert np.all(m >= 0.0)


def test_reproduces_the_model_md_rho_table(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §3.1."""
    spec = model_md["dixon_coles_rho_table"]
    lam, mu, tol = spec["lam"], spec["mu"], spec["tolerance"]

    for row in spec["rows"]:
        m = scoreline_matrix(lam, mu, row["rho"])
        draw = float(np.trace(m))
        assert m[0, 0] == pytest.approx(row["p_0_0"], abs=tol), f"P(0-0) at rho={row['rho']}"
        assert m[1, 1] == pytest.approx(row["p_1_1"], abs=tol), f"P(1-1) at rho={row['rho']}"
        assert draw == pytest.approx(row["p_draw"], abs=tol), f"P(draw) at rho={row['rho']}"
        assert 1.0 / draw == pytest.approx(row["fair_draw_odds"], abs=1e-3)


def test_draw_probability_falls_as_rho_rises(model_md: dict[str, Any]) -> None:
    """MODEL.md §3.1: a positive rho suppresses the draws the correction exists to raise."""
    spec = model_md["dixon_coles_rho_table"]
    draws = [
        float(np.trace(scoreline_matrix(spec["lam"], spec["mu"], row["rho"])))
        for row in spec["rows"]
    ]
    assert draws == sorted(draws, reverse=True)


def test_independence_when_rho_is_zero() -> None:
    """At rho = 0 the matrix is exactly the outer product of two Poisson margins."""
    m = scoreline_matrix(2.0, 1.0, 0.0)
    row_margin = m.sum(axis=1)
    col_margin = m.sum(axis=0)
    assert m == pytest.approx(np.outer(row_margin, col_margin), abs=1e-9)
