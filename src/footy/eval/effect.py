"""Effect-size translation and correlation intervals.

Answers cheap reviewer questions: how large is 1e-4 RPS? How precise is ρ on 14
observations? These functions report an RPS spread as a fraction of the
model-market gap, and bootstrap the Spearman interval to show uncertainty in a
small sample.

Bootstrap resamples pairs (x_i, y_i) together with replacement, preserving
exchangeability — ideal for iid observations. Contrast with serially dependent
data, which would need :func:`footy.eval.inference.stationary_bootstrap`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

FloatArray = npt.NDArray[np.float64]


def spearman(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Spearman rank correlation coefficient.

    A non-parametric measure of monotonic association. Returns a value in
    [-1, 1], where 1 is perfect positive rank correlation and -1 is perfect
    negative rank correlation.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same shape, got {x_arr.shape} and {y_arr.shape}")
    if x_arr.size < 2:
        raise ValueError("need at least two observations")

    rho, _ = spearmanr(x_arr, y_arr)
    return float(rho)


def bootstrap_correlation(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile interval for Spearman correlation, resampling pairs.

    Resamples (x_i, y_i) pairs together with replacement to preserve their
    association, then computes Spearman rho on each resample. Returns the point
    estimate (rho on the full sample) and the lower and upper percentile bounds.

    Args:
        x: First variable.
        y: Second variable, same length as x.
        rng: Numpy random generator for reproducibility.
        n_resamples: Number of bootstrap resamples.
        level: Confidence level in (0, 1), e.g., 0.95 for a 95% interval.

    Returns:
        (rho, low, high) where rho is the point estimate on the full sample,
        and low, high are the lower and upper percentile bounds.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same shape, got {x_arr.shape} and {y_arr.shape}")
    if x_arr.size < 2:
        raise ValueError("need at least two observations")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if n_resamples < 1:
        raise ValueError("need at least one resample")

    # Point estimate on the full sample.
    rho = spearman(x_arr, y_arr)

    # Resample pairs with replacement.
    n = x_arr.size
    indices = rng.choice(n, size=(n_resamples, n), replace=True)
    rhos = np.array([spearman(x_arr[idx], y_arr[idx]) for idx in indices], dtype=np.float64)

    # Percentile interval.
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(rhos, [tail, 1.0 - tail])

    return (rho, float(low), float(high))


def spread_as_share_of_gap(spread: float, model_gap: float) -> float:
    """Express a spread (e.g. RPS difference) as a share of the model-market gap.

    Answers: how much of the claimed effect does this measurement uncertainty
    represent? An RPS spread of 1e-4 against a model-market gap of 6.5e-3 is
    about 1.5% of the quantity the paper would be claiming.

    Args:
        spread: The measurement value (e.g., RPS spread).
        model_gap: The reference gap (e.g., model-market RPS gap).

    Returns:
        spread / model_gap.

    Raises:
        ValueError: If model_gap is zero.
    """
    if model_gap == 0.0:
        raise ValueError(f"model_gap must be non-zero, got {model_gap}")
    return spread / model_gap


def equivalent_sample_size(spread: float, per_match_sd: float) -> float:
    """Sample size at which the spread equals one standard error of the mean.

    A spread of per_match_sd / sqrt(n) becomes one standard error when
    n = (per_match_sd / spread) ** 2. This answers: how many matches would
    we need to collect before this measurement precision represents one
    standard error of the mean?

    Args:
        spread: The measurement precision (e.g., RPS spread).
        per_match_sd: The per-match standard deviation.

    Returns:
        The equivalent sample size n = (per_match_sd / spread) ** 2.

    Raises:
        ValueError: If spread is zero.
    """
    if spread == 0.0:
        raise ValueError(f"spread must be non-zero, got {spread}")
    return (per_match_sd / spread) ** 2
