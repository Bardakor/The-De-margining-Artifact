"""Poisson probability mass, evaluated in log space.

MODEL.md §2:

    P(X = k) = exp(k ln lam - lam - ln k!),  ln k! = ln Gamma(k + 1)

The naive form lam**k / k! overflows to inf/inf = NaN for large k. Working in
logs keeps every intermediate finite.

MODEL.md specifies a Lanczos approximation for ln Gamma. We use
scipy.special.gammaln instead: same function, an implementation we did not
write, and one whose accuracy is independently maintained.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln


def poisson_pmf(k: npt.ArrayLike, lam: float) -> npt.NDArray[np.float64]:
    """Poisson pmf at each k for rate lam.

    Args:
        k: Non-negative integer counts.
        lam: Rate, strictly positive.

    Returns:
        Probability mass at each k, same shape as k.

    Raises:
        ValueError: If lam is not strictly positive, or any k is negative.
    """
    if not lam > 0.0:
        raise ValueError(f"rate must be strictly positive, got {lam}")
    counts = np.asarray(k, dtype=np.float64)
    if np.any(counts < 0.0):
        raise ValueError("counts must be non-negative")
    log_pmf = counts * np.log(lam) - lam - gammaln(counts + 1.0)
    return np.asarray(np.exp(log_pmf), dtype=np.float64)
