"""The Dixon-Coles dependence correction.

MODEL.md §3. Independent Poisson under-counts low-scoring draws. Dixon and
Coles (1997) correct exactly four cells and leave the rest alone:

    tau(0,0) = 1 - lam*mu*rho
    tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1               otherwise

Sign matters. tau(0,0) and tau(1,1) exceed 1 -- inflating the low draws, which
is the whole point of the correction -- only when rho < 0. A positive rho does
the opposite of what the correction exists for.
"""

import numpy as np
import numpy.typing as npt

GRID = 11
"""Scoreline grid is 0..10 goals per side (MODEL.md §5)."""


def rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Admissible range for rho, keeping all four corrected cells non-negative.

    MODEL.md §3.2:

        max(-1/lam, -1/mu) <= rho <= min(1/(lam*mu), 1)
    """
    if not lam > 0.0 or not mu > 0.0:
        raise ValueError(f"rates must be strictly positive, got lam={lam}, mu={mu}")
    return max(-1.0 / lam, -1.0 / mu), min(1.0 / (lam * mu), 1.0)


def validate_rho(lam: float, mu: float, rho: float) -> None:
    """Raise if rho would drive a corrected cell negative."""
    lo, hi = rho_bounds(lam, mu)
    if not lo <= rho <= hi:
        raise ValueError(
            f"rho={rho} outside the admissible region [{lo}, {hi}] for lam={lam}, mu={mu}"
        )


def tau_matrix(lam: float, mu: float, rho: float) -> npt.NDArray[np.float64]:
    """The correction factor for every cell of the scoreline grid."""
    validate_rho(lam, mu, rho)
    tau = np.ones((GRID, GRID), dtype=np.float64)
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    return tau
