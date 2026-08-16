"""Applying a bookmaker margin by the power method.

MODEL.md §8. Fair odds are 1/p. To apply a margin, solve for the exponent k
such that

    sum_i p_i**k = B

where B is the target book sum. Every p_i lies in (0, 1), so the sum is
strictly decreasing in k and bisection converges. The offered price is
d_i = p_i**(-k).

An exponent rather than a constant multiplier reproduces favourite-longshot
bias: it takes proportionally more from the longshot.

Direction matters. The implementation this replaced computed p * (1 - margin),
which *lengthens* every price -- the book paid out roughly 105% of fair value.

Note on the reference longshot figure: MODEL.md §8.2 prints a favourite-
longshot ratio pair (1.105 longshot, 1.055 favourite) that is internally
inconsistent -- the two printed numbers imply two different exponents, and no
value of rho reconciles them. This implementation follows the exponent
(solve_power_exponent) rather than either printed figure; see
tests/fixtures/model_md_values.json's "_longshot_provenance" note for the
derivation.
"""

import numpy as np
import numpy.typing as npt

from footy.core.markets import Matrix, double_chance

_TOLERANCE = 1e-13
_MAX_ITERATIONS = 200


def solve_power_exponent(probabilities: npt.ArrayLike, target_sum: float) -> float:
    """Find k such that sum(p**k) == target_sum, by bisection.

    Raises:
        ValueError: If target_sum is below 1 or at/above the number of
            outcomes, where no exponent exists.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("every probability must lie strictly in (0, 1)")
    if target_sum < 1.0:
        raise ValueError(f"target book sum must be at least 1, got {target_sum}")
    if target_sum >= p.size:
        raise ValueError(
            f"target book sum {target_sum} is unreachable: sum(p**k) approaches "
            f"{p.size} as k approaches 0"
        )

    # sum(p**k) is strictly decreasing in k: it is p.size at k=0 and sum(p) at k=1.
    lo, hi = 0.0, 1.0
    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        value = float(np.sum(p**mid))
        if abs(value - target_sum) < _TOLERANCE:
            return mid
        if value > target_sum:
            lo = mid
        else:
            hi = mid
    raise ValueError(f"bisection failed to reach book sum {target_sum} in {_MAX_ITERATIONS} steps")


def apply_overround(probabilities: npt.ArrayLike, target_sum: float) -> npt.NDArray[np.float64]:
    """Offered decimal odds whose implied probabilities sum to target_sum."""
    p = np.asarray(probabilities, dtype=np.float64)
    k = solve_power_exponent(p, target_sum)
    odds: npt.NDArray[np.float64] = p ** (-k)

    achieved = float(np.sum(1.0 / odds))
    if abs(achieved - target_sum) > 1e-9:
        raise ValueError(f"postcondition failed: book sum {achieved}, target {target_sum}")
    return odds


def margin_double_chance(m: Matrix, target_sum: float) -> npt.NDArray[np.float64]:
    """Double-chance odds, margined independently against 2 * target_sum.

    MODEL.md §8.3. A fair double-chance book already sums to 2, since each
    selection covers two of three outcomes.

    Summing the already-margined 1X2 legs looks tidier and is wrong: for a
    heavy favourite it yields decimal odds below 1, which is not a payable
    price. Margining independently is structurally safe because p**k < 1 for
    any p < 1 and k > 0, so d = p**(-k) always exceeds 1.
    """
    return apply_overround(np.array(double_chance(m)), 2.0 * target_sum)
