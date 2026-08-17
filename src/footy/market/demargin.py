"""Recovering true probabilities from bookmaker odds — the study's instrument.

Study spec §4. Given offered decimal odds ``d_i``, the raw implied
probabilities are ``p_i = 1/d_i`` with book sum ``B = sum(p_i) > 1``. Each
transform maps ``p -> pi`` with ``sum(pi) = 1``:

    proportional   pi = p / B
    power          solve k with sum(p**k) = 1,  pi = p**k
    shin           pi = [sqrt(z^2 + 4(1-z) p^2 / B) - z] / (2(1-z)),  solve z
    odds_ratio     pi = p / (c(1-p) + p),                             solve c

These run in the OPPOSITE direction to ``overround.py``, which applies a
margin. Conflating the two — using an inverse method to price a market — is a
real and easy mistake, and the modules stay separate for that reason.

The four are not equivalent. They differ most in how they treat longshots,
which is where a model's claimed edge tends to concentrate, and whether that
difference is large enough to change a published conclusion is the question
this repository exists to answer. So they share one bisection driver at one
tolerance: no transform may be advantaged by a sloppier solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

TOLERANCE = 1e-12
MAX_ITERATIONS = 200
FAIR_BOOK_TOLERANCE = 1e-12

METHODS: tuple[str, ...] = ("proportional", "power", "shin", "odds_ratio")


@dataclass(frozen=True)
class DemarginResult:
    """Recovered probabilities plus the transform's fitted free parameter.

    ``parameter`` is z for Shin, the exponent k for power, c for odds-ratio,
    and the book sum for proportional (which fits nothing). It is carried so
    the study can report how hard each transform had to work on a given book.
    """

    probabilities: FloatArray
    parameter: float
    method: str
    book_sum: float


def _validated(implied: npt.ArrayLike) -> tuple[FloatArray, float]:
    p = np.asarray(implied, dtype=np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("need a one-dimensional book of at least two outcomes")
    if np.any(p <= 0.0):
        raise ValueError("every implied probability must be strictly positive")
    if np.any(p >= 1.0):
        raise ValueError("every implied probability must be below 1")
    book_sum = float(p.sum())
    if book_sum < 1.0 - FAIR_BOOK_TOLERANCE:
        raise ValueError(f"book sum must be at least 1, got {book_sum}")
    return p, book_sum


def _bisect(objective: Callable[[float], float], lo: float, hi: float) -> float:
    """Shared root-finder. ``objective`` must be decreasing on ``[lo, hi]``.

    One driver for all four transforms so none is advantaged by a different
    tolerance or iteration budget.
    """
    low, high = lo, hi
    mid = 0.5 * (low + high)
    for _ in range(MAX_ITERATIONS):
        mid = 0.5 * (low + high)
        value = objective(mid)
        if abs(value) < TOLERANCE:
            return mid
        if value > 0.0:
            low = mid
        else:
            high = mid
    return mid


def proportional(implied: npt.ArrayLike) -> DemarginResult:
    """Normalise. The field's default, and the transform most papers assume."""
    p, book_sum = _validated(implied)
    return DemarginResult(p / book_sum, book_sum, "proportional", book_sum)


def power(implied: npt.ArrayLike) -> DemarginResult:
    """Solve for the exponent k with ``sum(p**k) = 1``.

    Every ``p_i`` lies in (0,1), so ``sum(p**k)`` is strictly decreasing in k:
    it is the outcome count at k=0 and the book sum at k=1. The root therefore
    lies in (1, infinity) for a margined book, since the sum must fall to 1.
    """
    p, book_sum = _validated(implied)
    if abs(book_sum - 1.0) < FAIR_BOOK_TOLERANCE:
        return DemarginResult(p.copy(), 1.0, "power", book_sum)

    # sum(p**k) falls below 1 for k large enough; grow the bracket until it does.
    high = 2.0
    while float(np.sum(p**high)) > 1.0 and high < 1e6:
        high *= 2.0
    k = _bisect(lambda k: float(np.sum(p**k)) - 1.0, 1.0, high)
    recovered = p**k
    return DemarginResult(recovered / recovered.sum(), k, "power", book_sum)


def _shin_probabilities(p: FloatArray, book_sum: float, z: float) -> FloatArray:
    """Shin's closed form. At z -> 0 this tends to ``p / sqrt(B)``, not p / B."""
    if z <= 0.0:
        return np.asarray(p / np.sqrt(book_sum), dtype=np.float64)
    inner = z**2 + 4.0 * (1.0 - z) * p**2 / book_sum
    return np.asarray((np.sqrt(inner) - z) / (2.0 * (1.0 - z)), dtype=np.float64)


def shin(implied: npt.ArrayLike) -> tuple[FloatArray, float]:
    """Shin (1993): recover probabilities and the insider proportion z.

    Kept returning a bare tuple because it predates the other three and the
    core tests are written against that shape; :func:`demargin` gives the
    uniform interface.
    """
    result = shin_result(implied)
    return result.probabilities, result.parameter


def shin_result(implied: npt.ArrayLike) -> DemarginResult:
    """Shin (1993), modelling a proportion z of insider money in the book."""
    p, book_sum = _validated(implied)
    if abs(book_sum - 1.0) < FAIR_BOOK_TOLERANCE:
        return DemarginResult(p.copy(), 0.0, "shin", book_sum)

    z = _bisect(
        lambda z: float(_shin_probabilities(p, book_sum, z).sum()) - 1.0,
        0.0,
        1.0 - 1e-12,
    )
    recovered = _shin_probabilities(p, book_sum, z)
    return DemarginResult(recovered / recovered.sum(), z, "shin", book_sum)


def odds_ratio(implied: npt.ArrayLike) -> DemarginResult:
    """Constant odds ratio c between offered and true probabilities.

    ``p/(1-p) = c * pi/(1-pi)`` rearranges to ``pi = p / (c(1-p) + p)``. At
    c = 1 this is the identity, and pi decreases in c, so a margined book has
    its root at c > 1. Surveyed against the others by Strumbelj (2014).
    """
    p, book_sum = _validated(implied)
    if abs(book_sum - 1.0) < FAIR_BOOK_TOLERANCE:
        return DemarginResult(p.copy(), 1.0, "odds_ratio", book_sum)

    def recovered_at(c: float) -> FloatArray:
        return np.asarray(p / (c * (1.0 - p) + p), dtype=np.float64)

    high = 2.0
    while float(recovered_at(high).sum()) > 1.0 and high < 1e9:
        high *= 2.0
    c = _bisect(lambda c: float(recovered_at(c).sum()) - 1.0, 1.0, high)
    recovered = recovered_at(c)
    return DemarginResult(recovered / recovered.sum(), c, "odds_ratio", book_sum)


_DISPATCH: dict[str, Callable[[npt.ArrayLike], DemarginResult]] = {
    "proportional": proportional,
    "power": power,
    "shin": shin_result,
    "odds_ratio": odds_ratio,
}


def demargin(implied: npt.ArrayLike, method: str) -> DemarginResult:
    """Apply one named transform. The uniform entry point for the study."""
    try:
        transform = _DISPATCH[method]
    except KeyError as exc:
        raise KeyError(f"unknown transform {method!r}; expected one of {METHODS}") from exc
    return transform(implied)


def demargin_all(implied: npt.ArrayLike) -> dict[str, DemarginResult]:
    """Every transform applied to one book — the study's per-market unit."""
    return {method: demargin(implied, method) for method in METHODS}
