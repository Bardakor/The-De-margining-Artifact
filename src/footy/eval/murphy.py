"""Murphy's vector partition of the Brier score.

MODEL.md §11. Turns "the model scored 0.58" into a statement about why:

    BS = REL - RES + UNC + WBV

REL (reliability) asks whether stated probabilities match observed
frequencies; zero is perfect calibration. RES (resolution) asks whether the
model says anything beyond the base rate; zero means it does not. UNC belongs
to the sport, not the model.

WBV is the part that is easy to get wrong, and this engine got it wrong first
time. The classical three-way identity is exact only when each bin holds a
single distinct forecast -- the discrete case Murphy (1973) was writing about.
Bin *continuous* forecasts and a residual appears, because REL compares each
bin's mean forecast against its observed frequency while BS uses each
individual forecast. That residual is exactly the within-bin variance, so the
identity that holds for arbitrary forecasts carries four terms, not three.

All four are reported. Reducing this to three terms is a bug, not a
simplification.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class MurphyTerms(NamedTuple):
    """BS == reliability - resolution + uncertainty + within_bin_variance."""

    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    brier: float


def murphy_decomposition(
    forecasts: npt.ArrayLike, outcomes: npt.ArrayLike, n_bins: int = 10
) -> MurphyTerms:
    """Decompose the binary Brier score into its four exact components.

    Args:
        forecasts: Probability of the event, one per observation, in [0, 1].
        outcomes: Realised outcomes, each 0 or 1.
        n_bins: Equal-width bins over [0, 1].
    """
    p = np.asarray(forecasts, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)

    if p.shape != o.shape:
        raise ValueError(
            f"forecasts and outcomes must be the same length, got {p.shape} and {o.shape}"
        )
    if p.size == 0:
        raise ValueError("need at least one observation")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("forecasts must lie in [0, 1]")
    if not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("outcomes must each be 0 or 1")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    n = p.size
    base_rate = float(o.mean())
    brier = float(np.mean((p - o) ** 2))

    # Equal-width bins; a forecast of exactly 1.0 belongs in the top bin.
    bin_index = np.clip((p * n_bins).astype(int), 0, n_bins - 1)

    reliability = resolution = within_bin_variance = 0.0
    for k in range(n_bins):
        members = bin_index == k
        count = int(members.sum())
        if count == 0:
            continue
        mean_forecast = float(p[members].mean())
        observed_rate = float(o[members].mean())

        reliability += count * (mean_forecast - observed_rate) ** 2
        resolution += count * (observed_rate - base_rate) ** 2
        within_bin_variance += float(np.sum((p[members] - mean_forecast) ** 2))

    return MurphyTerms(
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=base_rate * (1.0 - base_rate),
        within_bin_variance=within_bin_variance / n,
        brier=brier,
    )
