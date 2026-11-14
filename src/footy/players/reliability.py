"""M1 — sampling noise in per-90 metrics (design §5).

How much of a player's per-90 number is the player, and how much is the luck of
which matches he played. Pure: no I/O, randomness only through the
``numpy.random.Generator`` argument. Input is a player-match table with
``player_id``, ``season``, ``minutes`` and the count column named by ``metric``.

Assumptions:

- A per-90 rate is minutes-weighted, ``90 * sum(count) / sum(minutes)`` over the
  matches in question, never a mean of per-match rates.
- Split-half: each player-season's matches are put in random order and dealt
  alternately into two halves (match counts differ by at most one). A
  player-season enters a split only if both halves reach ``min_half_minutes``,
  so the eligible set can change slightly between splits. The default 450 per
  half matches the pre-registered 900-minute floor. Each split's Pearson
  correlation across player-seasons (unweighted) is stepped up with
  Spearman–Brown, ``2r / (1 + r)``; the estimate is the mean of the stepped-up
  values over splits, the spread their standard deviation. Player-seasons
  differ in minutes, so the estimate is the reliability at about the harmonic
  mean of their minutes.
- Stabilisation: in each random split, each half is cut to the shortest prefix
  (in the random order) that reaches the level ``m``, so a half holds ``m`` to
  ``m`` plus one match. The raw correlation of the two halves is the
  reliability at that length (no step-up); its x-coordinate is the harmonic
  mean of the actual half minutes. Only player-seasons whose halves both reach
  ``m`` enter a level, so the population shifts towards regulars as ``m``
  grows; the per-level counts are returned. ``k`` in ``r(m) = m / (m + k)`` is
  fitted by least squares weighted by the number of player-seasons per level;
  reliability then reaches 0.5 at ``k`` and 0.7 at ``7k / 3`` minutes.
- Bootstrap: matches are resampled with replacement within each unit;
  percentile intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit

UNIT = ("player_id", "season")
MIN_UNITS = 3


@dataclass(frozen=True)
class SplitHalf:
    estimate: float  # mean stepped-up reliability over splits
    sd: float  # its standard deviation over splits
    n_units: float  # mean number of player-seasons per split


@dataclass(frozen=True)
class Stabilisation:
    k: float
    minutes_05: float
    minutes_07: float
    curve: pd.DataFrame  # level, minutes (harmonic mean per half), reliability, n_units


def split_half(
    pm: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    *,
    n_splits: int = 200,
    min_half_minutes: float = 450.0,
) -> SplitHalf:
    codes, minutes, counts, n = _arrays(pm, metric, UNIT)
    stepped = np.empty(n_splits)
    used = np.empty(n_splits)
    for s in range(n_splits):
        key, _ = _random_halves(codes, minutes, rng)
        rate, mins = _half_rates(key, minutes, counts, n, np.ones(len(key), dtype=bool))
        ok = (mins >= min_half_minutes).all(axis=1)
        r = _corr(rate[ok, 0], rate[ok, 1])
        stepped[s] = 2 * r / (1 + r)
        used[s] = ok.sum()
    return SplitHalf(float(stepped.mean()), float(stepped.std(ddof=1)), float(used.mean()))


def stabilisation(
    pm: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    *,
    levels: Sequence[float] = (180.0, 360.0, 540.0, 720.0, 900.0),
    n_splits: int = 200,
) -> Stabilisation:
    codes, minutes, counts, n = _arrays(pm, metric, UNIT)
    r = np.full((n_splits, len(levels)), np.nan)
    inv_minutes = np.full_like(r, np.nan)
    used = np.zeros_like(r)
    for s in range(n_splits):
        key, before = _random_halves(codes, minutes, rng)
        for j, level in enumerate(levels):
            rate, mins = _half_rates(key, minutes, counts, n, before < level)
            ok = (mins >= level).all(axis=1)
            used[s, j] = ok.sum()
            if ok.sum() >= MIN_UNITS:
                r[s, j] = _corr(rate[ok, 0], rate[ok, 1])
                inv_minutes[s, j] = (1 / mins[ok]).mean()
    curve = pd.DataFrame(
        {
            "level": np.asarray(levels, dtype=np.float64),
            "minutes": 1 / np.nanmean(inv_minutes, axis=0),
            "reliability": np.nanmean(r, axis=0),
            "n_units": used.mean(axis=0),
        }
    )
    fit = curve.dropna()
    (k,), _ = curve_fit(
        lambda m, k: m / (m + k),
        fit["minutes"].to_numpy(),
        fit["reliability"].to_numpy(),
        p0=[float(fit["minutes"].median())],
        sigma=1 / np.sqrt(fit["n_units"].to_numpy()),
        bounds=(0.0, np.inf),
    )
    return Stabilisation(float(k), float(k), float(k * 7 / 3), curve)


def bootstrap_rates(
    pm: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    *,
    keys: Sequence[str] = UNIT,
    n_boot: int = 200,
    level: float = 0.9,
) -> pd.DataFrame:
    """Per unit: matches, minutes, per-90 rate, bootstrap SE and percentile interval."""
    codes, minutes, counts, n = _arrays(pm, metric, keys)
    order = np.argsort(codes, kind="stable")
    unit, m, c = codes[order], minutes[order], counts[order]
    size = np.bincount(unit, minlength=n)
    start = np.cumsum(size) - size
    reps = np.empty((n_boot, n))
    for b in range(n_boot):
        draw = start[unit] + (rng.random(len(unit)) * size[unit]).astype(np.int64)
        reps[b] = 90 * np.bincount(unit, c[draw], n) / np.bincount(unit, m[draw], n)
    out = pm.groupby(list(keys)).size().rename("matches").reset_index()
    out["minutes"] = np.bincount(codes, minutes, n)
    out["rate"] = 90 * np.bincount(codes, counts, n) / out["minutes"]
    out["se"] = reps.std(axis=0, ddof=1)
    out["lo"], out["hi"] = np.quantile(reps, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return out


def _arrays(
    pm: pd.DataFrame, metric: str, keys: Sequence[str]
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], int]:
    codes = np.asarray(pm.groupby(list(keys)).ngroup(), dtype=np.int64)
    minutes = np.asarray(pm["minutes"], dtype=np.float64)
    counts = np.asarray(pm[metric], dtype=np.float64)
    return codes, minutes, counts, int(codes.max()) + 1


def _random_halves(
    codes: NDArray[np.int64], minutes: NDArray[np.float64], rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per row: its half key ``2 * unit + half`` and the minutes dealt to that half before it."""
    n_rows = len(codes)
    first = np.lexsort((rng.random(n_rows), codes))
    sorted_codes = codes[first]
    pos = np.arange(n_rows) - np.searchsorted(sorted_codes, sorted_codes)
    key = np.empty(n_rows, dtype=np.int64)
    key[first] = 2 * sorted_codes + pos % 2
    rank = np.empty(n_rows, dtype=np.int64)
    rank[first] = pos
    order = np.lexsort((rank, key))
    m, k = minutes[order], key[order]
    excl = np.cumsum(m) - m
    before = np.empty(n_rows)
    before[order] = excl - excl[np.searchsorted(k, k)]
    return key, before


def _half_rates(
    key: NDArray[np.int64],
    minutes: NDArray[np.float64],
    counts: NDArray[np.float64],
    n: int,
    include: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-90 rate and minutes of each unit's two halves, shape (n, 2)."""
    mins = np.bincount(key, minutes * include, 2 * n).reshape(n, 2)
    cnt = np.bincount(key, counts * include, 2 * n).reshape(n, 2)
    return 90 * cnt / np.where(mins > 0, mins, 1.0), mins


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.corrcoef(a, b)[0, 1])
