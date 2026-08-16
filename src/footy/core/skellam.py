"""The goal difference, derived without the matrix.

MODEL.md §7. Everything else in this package derives from one 11x11 matrix,
which is internally consistent by construction -- meaning a systematic error
in the matrix would be invisible to every consistency test.

The difference of two independent Poisson variables is Skellam-distributed
(Karlis and Ntzoufras, 2009), with a closed form in the modified Bessel
function of the first kind:

    P(K = k) = exp(-(lam + mu)) * (lam/mu)^(k/2) * I_|k|(2*sqrt(lam*mu))

No matrix is involved. Where this agrees with the matrix anti-diagonals, the
matrix is corroborated from outside itself.

Numerically we use scipy.special.ive, the exponentially scaled Bessel
function ive(v, z) = I_v(z) * exp(-|z|). Folding exp(z) back in analytically
keeps every intermediate finite for large rates, where I_v(z) alone overflows.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import ive


def skellam_pmf(k: npt.ArrayLike, lam: float, mu: float) -> npt.NDArray[np.float64]:
    """Probability that the goal difference (home minus away) equals k."""
    if not lam > 0.0 or not mu > 0.0:
        raise ValueError(f"rates must be strictly positive, got lam={lam}, mu={mu}")
    diff = np.asarray(k, dtype=np.float64)
    z = 2.0 * np.sqrt(lam * mu)
    # exp(-(lam+mu)) * I_v(z) == exp(-(lam+mu) + z) * ive(v, z)
    scale = np.exp(-(lam + mu) + z)
    result = scale * (lam / mu) ** (diff / 2.0) * ive(np.abs(diff), z)
    return np.asarray(result, dtype=np.float64)


def skellam_supremacy(lam: float, mu: float, handicap: float) -> tuple[float, float, float]:
    """(home covers, push, away covers) for a supremacy line, via Skellam.

    A second, independent route to the Asian handicap. The handicap is applied
    to the home side, so the home side covers when K + handicap > 0.

    Quarter lines split the stake across the two adjacent whole/half lines,
    which is how they settle in practice.
    """
    quarters = handicap * 4.0
    if not float(quarters).is_integer():
        raise ValueError(f"handicap must be a multiple of a quarter goal, got {handicap}")

    is_quarter_line = int(quarters) % 2 != 0
    if not is_quarter_line:
        return _whole_or_half_supremacy(lam, mu, handicap)

    # For quarter lines, blend the two adjacent whole/half lines.
    lower_home, lower_push, lower_away = _whole_or_half_supremacy(lam, mu, handicap - 0.25)
    upper_home, upper_push, upper_away = _whole_or_half_supremacy(lam, mu, handicap + 0.25)

    return (
        (lower_home + upper_home) / 2.0,
        (lower_push + upper_push) / 2.0,
        (lower_away + upper_away) / 2.0,
    )


def _whole_or_half_supremacy(lam: float, mu: float, handicap: float) -> tuple[float, float, float]:
    """Compute (home, push, away) for a whole or half line.

    Support window: [-60, 61]. This is sufficient because for typical
    rates (lam, mu ~ 1-2 goals), the tail mass beyond ±60 is negligible
    and does not affect markets at the 1e-6 level (six decimal places).
    """
    support = np.arange(-60, 61)
    mass = skellam_pmf(support, lam, mu)
    adjusted = support + handicap

    home = float(mass[adjusted > 0].sum())
    push = float(mass[adjusted == 0].sum())
    away = float(mass[adjusted < 0].sum())
    return home, push, away
