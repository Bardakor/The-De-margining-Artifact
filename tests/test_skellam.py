"""MODEL.md §7 — corroborating the matrix from outside itself."""

import numpy as np
import pytest

from footy.core.markets import asian_handicap
from footy.core.matrix import scoreline_matrix
from footy.core.skellam import skellam_pmf, skellam_supremacy


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


def test_supremacy_agrees_with_matrix_handicap_at_rho_zero() -> None:
    """Second external corroboration: Skellam supremacy vs matrix asian_handicap.

    At rho = 0 the two routes must agree, since Skellam assumes independence
    and the matrix at rho = 0 enforces independence. Covers whole, half, and
    quarter handicaps.
    """
    lam, mu = 1.6, 1.1
    m = scoreline_matrix(lam, mu, 0.0)

    # Test handicaps spanning whole, half, and quarter lines
    handicaps = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.5, 1.0]

    for h in handicaps:
        # skellam_supremacy returns (home, push, away) as a flat tuple
        skellam_home, skellam_push, skellam_away = skellam_supremacy(lam, mu, h)

        # asian_handicap returns (home, away) as AsianOutcome(win, push, lose)
        matrix_home, matrix_away = asian_handicap(m, h)

        assert skellam_home == pytest.approx(matrix_home.win, abs=1e-6), (
            f"handicap {h}: home win probability mismatch"
        )
        assert skellam_push == pytest.approx(matrix_home.push, abs=1e-6), (
            f"handicap {h}: push probability mismatch"
        )
        assert skellam_away == pytest.approx(matrix_home.lose, abs=1e-6), (
            f"handicap {h}: away win probability mismatch"
        )
