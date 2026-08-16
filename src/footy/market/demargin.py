"""Recovering true probabilities from bookmaker odds.

MODEL.md §9. Shin (1993) models a bookmaker's prices as containing a
proportion z of insider money, and inverts:

    pi_i = [ sqrt(z**2 + 4(1-z) * p_i**2 / B) - z ] / (2(1-z))

with z solved by bisection so that sum(pi) == 1.

This runs in the opposite direction to overround.py. Applying margin and
removing it are different operations, and conflating them is an easy mistake:
an inverse method must never be used to price a market.

Plan 3 adds the proportional, power and odds-ratio transforms alongside this
one. They are the instrument of the study.
"""

import numpy as np
import numpy.typing as npt

_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200


def _shin_probabilities(
    p: npt.NDArray[np.float64], book_sum: float, z: float
) -> npt.NDArray[np.float64]:
    if z <= 0.0:
        return p / book_sum
    inner = z**2 + 4.0 * (1.0 - z) * p**2 / book_sum
    result: npt.NDArray[np.float64] = (np.sqrt(inner) - z) / (2.0 * (1.0 - z))
    return result


def shin(implied: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], float]:
    """Recover true probabilities and the insider proportion z.

    Args:
        implied: Raw implied probabilities 1/odds, summing to at least 1.

    Returns:
        (true probabilities summing to 1, the fitted z).
    """
    p = np.asarray(implied, dtype=np.float64)
    if np.any(p <= 0.0):
        raise ValueError("every implied probability must be strictly positive")
    book_sum = float(p.sum())
    if book_sum < 1.0 - 1e-12:
        raise ValueError(f"book sum must be at least 1, got {book_sum}")

    if abs(book_sum - 1.0) < 1e-12:
        return p.copy(), 0.0

    # sum(pi) is decreasing in z over [0, 1); at z = 0 it is book_sum > 1.
    lo, hi = 0.0, 1.0 - 1e-12
    z = 0.0
    for _ in range(_MAX_ITERATIONS):
        z = (lo + hi) / 2.0
        total = float(_shin_probabilities(p, book_sum, z).sum())
        if abs(total - 1.0) < _TOLERANCE:
            break
        if total > 1.0:
            lo = z
        else:
            hi = z

    recovered = _shin_probabilities(p, book_sum, z)
    return recovered / recovered.sum(), z
