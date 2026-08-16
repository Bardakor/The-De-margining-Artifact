"""The joint distribution over scorelines.

MODEL.md §5. One matrix is the single source of truth for every price:

    P(x,y) = tau(x,y) * Pois(x; lam) * Pois(y; mu) / Z

over 0 <= x, y <= 10. Renormalisation absorbs both the truncated tail beyond
10 goals and the mass shifted by tau.
"""

import numpy as np
import numpy.typing as npt

from footy.core.dixon_coles import GRID, tau_matrix
from footy.core.poisson import poisson_pmf

DEFAULT_RHO = -0.10
"""MODEL.md §3.1. Dixon and Coles' own fitted value is near -0.13."""


def scoreline_matrix(lam: float, mu: float, rho: float = DEFAULT_RHO) -> npt.NDArray[np.float64]:
    """Joint distribution over scorelines.

    Args:
        lam: Expected goals, home.
        mu: Expected goals, away.
        rho: Dependence parameter; must lie in the admissible region.

    Returns:
        An 11x11 array where [x, y] is P(home scores x, away scores y),
        summing to 1.
    """
    goals = np.arange(GRID)
    joint = np.outer(poisson_pmf(goals, lam), poisson_pmf(goals, mu))
    joint = joint * tau_matrix(lam, mu, rho)
    total = joint.sum()
    if not total > 0.0:
        raise ValueError(f"degenerate matrix for lam={lam}, mu={mu}, rho={rho}")
    return joint / total
