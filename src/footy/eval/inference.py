"""Statistical inference for comparing forecasts and returns.

Study spec §6.3. Three tools, each chosen because the naive alternative is
invalid on this data rather than merely less powerful:

**Diebold-Mariano** with Newey-West standard errors. Forecasts produced by
overlapping fit windows are serially correlated, so treating per-match score
differences as independent would understate the variance and overstate
significance.

**Stationary block bootstrap** (Politis and Romano, 1994). Betting returns are
heavy-tailed and clustered by matchday. An i.i.d. bootstrap resamples away
exactly the dependence that makes the interval wide.

**Benjamini-Hochberg.** The study runs a family of tests across leagues,
books, markets and transform pairs. Reporting uncorrected p-values from that
many comparisons would manufacture significance.

Everything here is deterministic given a seed; the bootstrap takes an explicit
``numpy.random.Generator`` and never touches global state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class DieboldMariano:
    """Test of equal predictive accuracy between two forecasters."""

    statistic: float
    p_value: float
    mean_difference: float
    n: int
    lag: int

    @property
    def favours_first(self) -> bool:
        """True when the first forecaster had the lower mean loss."""
        return self.mean_difference < 0.0


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval from a stationary block bootstrap."""

    point: float
    low: float
    high: float
    level: float
    n_resamples: int
    block_length: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def newey_west_variance(x: FloatArray, lag: int) -> float:
    """Long-run variance of the mean of ``x``, Bartlett-kernel HAC.

    At ``lag = 0`` this is the ordinary sample variance, so the estimator
    nests the naive one rather than replacing it with something unrelated.
    """
    values = np.asarray(x, dtype=np.float64)
    n = values.size
    if n < 2:
        raise ValueError("need at least two observations")
    if lag < 0:
        raise ValueError(f"lag must be non-negative, got {lag}")
    if lag >= n:
        raise ValueError(f"lag {lag} must be below the sample size {n}")

    centred = values - values.mean()
    gamma0 = float(np.dot(centred, centred) / n)
    total = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.dot(centred[k:], centred[:-k]) / n)
        total += 2.0 * weight * gamma_k
    # Bartlett weights guarantee a non-negative estimate in exact arithmetic;
    # clamp the floating-point residue rather than returning a negative variance.
    return max(total, 0.0)


def default_lag(n: int) -> int:
    """Newey-West truncation lag, the standard ``floor(4 (n/100)^(2/9))``."""
    if n < 2:
        raise ValueError("need at least two observations")
    return max(0, min(n - 1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))))


def diebold_mariano(
    loss_a: npt.ArrayLike,
    loss_b: npt.ArrayLike,
    *,
    lag: int | None = None,
) -> DieboldMariano:
    """Test whether two forecasters have equal expected loss.

    Args:
        loss_a: Per-observation loss of the first forecaster (e.g. RPS).
        loss_b: Per-observation loss of the second, same observations, same order.
        lag: Newey-West truncation lag. ``None`` uses :func:`default_lag`.

    Returns:
        The statistic, a two-sided normal p-value, and the mean loss
        difference ``mean(loss_a - loss_b)``. Negative means the first
        forecaster did better.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"loss arrays must be the same shape, got {a.shape} and {b.shape}")
    if a.size < 2:
        raise ValueError("need at least two observations")

    differences = a - b
    n = differences.size
    truncation = default_lag(n) if lag is None else lag
    mean_difference = float(differences.mean())

    variance = newey_west_variance(differences, truncation)
    if variance <= 0.0:
        # Identical forecasts: no evidence of a difference, not infinite evidence.
        statistic = 0.0 if abs(mean_difference) < 1e-15 else float(np.inf)
        p_value = 1.0 if abs(mean_difference) < 1e-15 else 0.0
        return DieboldMariano(statistic, p_value, mean_difference, n, truncation)

    statistic = mean_difference / np.sqrt(variance / n)
    p_value = float(2.0 * (1.0 - _standard_normal_cdf(abs(statistic))))
    return DieboldMariano(float(statistic), p_value, mean_difference, n, truncation)


def _standard_normal_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def stationary_bootstrap(
    x: npt.ArrayLike,
    *,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    block_length: float = 20.0,
    level: float = 0.95,
) -> BootstrapInterval:
    """Percentile interval for the mean, preserving serial dependence.

    Blocks have geometric length with mean ``block_length`` and wrap around the
    end of the series, which is what makes the resampled series stationary
    (Politis and Romano, 1994). An i.i.d. bootstrap would destroy the matchday
    clustering that widens the true interval.
    """
    values = np.asarray(x, dtype=np.float64)
    n = values.size
    if n < 2:
        raise ValueError("need at least two observations")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if block_length <= 0.0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if n_resamples < 1:
        raise ValueError("need at least one resample")

    p_restart = min(1.0, 1.0 / block_length)
    starts = rng.integers(0, n, size=(n_resamples, n))
    restart = rng.random((n_resamples, n)) < p_restart

    # Walk each resample forward: continue the current block, or jump to a new
    # random start. Vectorised over resamples, one pass over the series length.
    index = np.empty((n_resamples, n), dtype=np.int64)
    index[:, 0] = starts[:, 0]
    for t in range(1, n):
        advanced = (index[:, t - 1] + 1) % n
        index[:, t] = np.where(restart[:, t], starts[:, t], advanced)

    means = values[index].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return BootstrapInterval(
        point=float(values.mean()),
        low=float(low),
        high=float(high),
        level=level,
        n_resamples=n_resamples,
        block_length=block_length,
    )


def benjamini_hochberg(p_values: npt.ArrayLike, q: float = 0.10) -> BoolArray:
    """Which hypotheses are rejected controlling the false discovery rate at q.

    Returns a boolean mask in the caller's original order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie in (0, 1), got {q}")

    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    thresholds = q * ranks / n
    passed = p[order] <= thresholds

    rejected = np.zeros(n, dtype=bool)
    if np.any(passed):
        # Reject everything up to the LARGEST rank that passes, not only those
        # that pass individually — that step is what makes this BH rather than
        # an uncorrected sweep.
        cutoff = int(np.max(np.nonzero(passed)[0]))
        rejected[order[: cutoff + 1]] = True
    return rejected


def adjusted_p_values(p_values: npt.ArrayLike) -> FloatArray:
    """Benjamini-Hochberg adjusted p-values (q-values), original order."""
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return np.zeros(0, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    scaled = p[order] * n / ranks
    # Enforce monotonicity from the largest p downward.
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
