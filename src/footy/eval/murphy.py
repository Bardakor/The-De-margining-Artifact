"""Murphy's vector partition of the Brier score.

MODEL.md §11. Turns "the model scored 0.58" into a statement about why:

    BS = REL - RES + UNC + WBV - 2*COV

REL (reliability) asks whether stated probabilities match observed
frequencies; zero is perfect calibration. RES (resolution) asks whether the
model says anything beyond the base rate; zero means it does not. UNC belongs
to the sport, not the model.

**The fifth term, and why the specification is wrong without it.**

MODEL.md §11.1 argues -- correctly -- that the classical three-term identity
is exact only when each bin holds a single distinct forecast, and that binning
continuous forecasts leaves a residual. It identifies that residual as the
within-bin forecast variance and stops there. That is one term short.

Expanding sum (p_i - o_i)^2 within a bin around the bin means gives

    (p_i - o_i) = (p_i - pbar) + (pbar - obar) + (obar - o_i)

and squaring leaves three squared terms plus one surviving cross term,
-2 (p_i - pbar)(o_i - obar). Summed, that is minus twice the within-bin
COVARIANCE between forecast and outcome. It vanishes only when, inside every
bin, higher forecasts carry no information about which events occurred.

MODEL.md's own worked example satisfies that by accident -- each of its bins
holds either one point, or two points sharing an outcome -- so the term is
identically zero there and the omission is invisible. On real continuous
forecasts it is not: measured on 4,000 simulated forecasts it accounted for
1.2% of the Brier score, which the four-term form silently misattributes.

Including COV drops the residual from order 1e-3 to exactly zero. All five
terms are reported. Reducing this to four -- or to three -- is a bug, not a
simplification, and :func:`residual` exists so nobody has to take that on
trust.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class MurphyTerms(NamedTuple):
    """The exact partition.

    ``brier == reliability - resolution + uncertainty + within_bin_variance
    - 2 * within_bin_covariance``
    """

    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    brier: float
    within_bin_covariance: float = 0.0

    @property
    def three_term(self) -> float:
        """The classical form. Exact only for single-valued bins."""
        return self.reliability - self.resolution + self.uncertainty

    @property
    def four_term(self) -> float:
        """MODEL.md §11.1's form. Exact only when within-bin covariance is zero."""
        return self.three_term + self.within_bin_variance

    @property
    def reconstructed(self) -> float:
        """The exact five-term reconstruction of the Brier score."""
        return self.four_term - 2.0 * self.within_bin_covariance

    @property
    def residual(self) -> float:
        """How far the exact reconstruction misses. Should be at machine zero."""
        return self.brier - self.reconstructed


def murphy_decomposition(
    forecasts: npt.ArrayLike, outcomes: npt.ArrayLike, n_bins: int = 10
) -> MurphyTerms:
    """Decompose the binary Brier score into its five exact components.

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

    reliability = resolution = within_bin_variance = within_bin_covariance = 0.0
    for k in range(n_bins):
        members = bin_index == k
        count = int(members.sum())
        if count == 0:
            continue
        mean_forecast = float(p[members].mean())
        observed_rate = float(o[members].mean())
        forecast_deviation = p[members] - mean_forecast
        outcome_deviation = o[members] - observed_rate

        reliability += count * (mean_forecast - observed_rate) ** 2
        resolution += count * (observed_rate - base_rate) ** 2
        within_bin_variance += float(np.sum(forecast_deviation**2))
        within_bin_covariance += float(np.sum(forecast_deviation * outcome_deviation))

    return MurphyTerms(
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=base_rate * (1.0 - base_rate),
        within_bin_variance=within_bin_variance / n,
        brier=brier,
        within_bin_covariance=within_bin_covariance / n,
    )
