"""MODEL.md §7 — corroborating the matrix from outside itself."""

import numpy as np
import pytest

from footy.core.matrix import scoreline_matrix
from footy.core.skellam import skellam_pmf


def matrix_goal_difference(m: np.ndarray, k: int) -> float:
    """Sum the anti-diagonal x - y == k."""
    x = np.arange(m.shape[0])[:, None]
    y = np.arange(m.shape[1])[None, :]
    return float(m[(x - y) == k].sum())


def test_sums_to_one_over_a_wide_support() -> None:
    assert skellam_pmf(np.arange(-40, 41), 1.6, 1.1).sum() == pytest.approx(1.0, abs=1e-10)


def test_agrees_with_the_matrix_when_rho_is_zero() -> None:
    """MODEL.md §7 consequence 1: at rho = 0 the two derivations must agree.

    This is the only test in the suite that corroborates the matrix by a route
    that does not use the matrix.
    """
    lam, mu = 1.6, 1.1
    m = scoreline_matrix(lam, mu, 0.0)
    for k in range(-5, 6):
        assert skellam_pmf(k, lam, mu) == pytest.approx(matrix_goal_difference(m, k), abs=1e-6), (
            f"goal difference {k}"
        )


def test_diverges_from_the_matrix_when_rho_is_non_zero() -> None:
    """MODEL.md §7 consequence 2: if they agreed, tau would be doing nothing."""
    lam, mu = 1.6, 1.1
    m = scoreline_matrix(lam, mu, -0.10)
    assert abs(float(skellam_pmf(0, lam, mu)) - matrix_goal_difference(m, 0)) > 1e-3


def test_stays_finite_for_large_rates() -> None:
    """The Bessel series must be evaluated stably, not by naive summation."""
    result = skellam_pmf(np.arange(-60, 61), 40.0, 35.0)
    assert np.all(np.isfinite(result))
    assert result.sum() == pytest.approx(1.0, abs=1e-8)


def test_symmetric_rates_give_a_symmetric_distribution() -> None:
    values = skellam_pmf(np.arange(-8, 9), 1.5, 1.5)
    assert values == pytest.approx(values[::-1], abs=1e-12)
