"""Scoring rules for probabilistic forecasts.

MODEL.md §11. Football outcomes are *ordered* -- home, draw, away is not an
arbitrary set of three labels -- and the choice of metric turns on that.

Brier is blind to ordering: forecasting a home win scores the same whether the
match was drawn or lost. RPS (Epstein, 1969) is distance-sensitive and
punishes a near miss less than a distant one.

This is contested. Wheatcroft (2021) argues distance sensitivity is not in
fact desirable here. Both are implemented so the choice stays explicit.
"""

import numpy as np
import numpy.typing as npt

_EPSILON = 1e-15
"""Floor for log loss, so a zero-probability realised outcome is finite."""


def _prepare(
    forecast: npt.ArrayLike, outcome: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    p = np.asarray(forecast, dtype=np.float64)
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError(f"forecast must sum to 1, got {float(p.sum())}")
    if not 0 <= outcome < p.size:
        raise IndexError(f"outcome {outcome} outside the {p.size}-category forecast")
    observed = np.zeros_like(p)
    observed[outcome] = 1.0
    return p, observed


def ranked_probability_score(forecast: npt.ArrayLike, outcome: int) -> float:
    """Distance-sensitive score for ordered categories. Lower is better."""
    p, observed = _prepare(forecast, outcome)
    cumulative = np.cumsum(p) - np.cumsum(observed)
    return float(np.sum(cumulative[:-1] ** 2) / (p.size - 1))


def brier_score(forecast: npt.ArrayLike, outcome: int) -> float:
    """Squared error summed over categories. Lower is better. Ignores ordering."""
    p, observed = _prepare(forecast, outcome)
    return float(np.sum((p - observed) ** 2))


def log_loss(forecast: npt.ArrayLike, outcome: int) -> float:
    """Negative log probability of the realised outcome. Lower is better."""
    p, _ = _prepare(forecast, outcome)
    return float(-np.log(max(float(p[outcome]), _EPSILON)))
