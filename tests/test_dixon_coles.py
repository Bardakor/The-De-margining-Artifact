"""MODEL.md §3 — the dependence correction."""

import numpy as np
import pytest

from footy.core.dixon_coles import GRID, rho_bounds, tau_matrix, validate_rho


def test_only_the_four_low_scoring_cells_are_corrected() -> None:
    tau = tau_matrix(1.6, 1.1, -0.10)
    assert tau.shape == (GRID, GRID)
    corrected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for x in range(GRID):
        for y in range(GRID):
            if (x, y) not in corrected:
                assert tau[x, y] == 1.0


def test_the_four_cells_take_their_specified_values() -> None:
    lam, mu, rho = 1.6, 1.1, -0.10
    tau = tau_matrix(lam, mu, rho)
    assert tau[0, 0] == pytest.approx(1.0 - lam * mu * rho, abs=1e-12)
    assert tau[0, 1] == pytest.approx(1.0 + lam * rho, abs=1e-12)
    assert tau[1, 0] == pytest.approx(1.0 + mu * rho, abs=1e-12)
    assert tau[1, 1] == pytest.approx(1.0 - rho, abs=1e-12)


def test_negative_rho_inflates_the_low_draws() -> None:
    """MODEL.md §3.1: tau(0,0) and tau(1,1) exceed 1 only when rho < 0."""
    negative = tau_matrix(1.6, 1.1, -0.10)
    positive = tau_matrix(1.6, 1.1, 0.06)
    assert negative[0, 0] > 1.0 and negative[1, 1] > 1.0
    assert positive[0, 0] < 1.0 and positive[1, 1] < 1.0


def test_rho_zero_is_the_identity() -> None:
    assert np.all(tau_matrix(1.6, 1.1, 0.0) == 1.0)


def test_bounds_match_the_specified_region() -> None:
    """MODEL.md §3.2."""
    lam, mu = 1.6, 1.1
    lo, hi = rho_bounds(lam, mu)
    assert lo == pytest.approx(max(-1.0 / lam, -1.0 / mu), abs=1e-12)
    assert hi == pytest.approx(min(1.0 / (lam * mu), 1.0), abs=1e-12)


def test_validate_rejects_rho_outside_the_region() -> None:
    lam, mu = 1.6, 1.1
    lo, hi = rho_bounds(lam, mu)
    validate_rho(lam, mu, (lo + hi) / 2.0)  # must not raise
    with pytest.raises(ValueError, match="admissible"):
        validate_rho(lam, mu, lo - 0.01)
    with pytest.raises(ValueError, match="admissible"):
        validate_rho(lam, mu, hi + 0.01)


def test_tau_is_non_negative_throughout_the_admissible_region() -> None:
    lam, mu = 1.6, 1.1
    lo, hi = rho_bounds(lam, mu)
    for rho in np.linspace(lo, hi, 50):
        assert np.all(tau_matrix(lam, mu, float(rho)) >= 0.0)
