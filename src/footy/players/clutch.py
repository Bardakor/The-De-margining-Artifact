"""M7(b) — is "big-game player" a stable trait? (design §5, pre-registered H7)

If some players rise to knockout games, their knockout over-performance should
repeat: high in 2020–2022 and high again in 2023–2026. Pure: no I/O,
randomness only through the ``numpy.random.Generator`` argument. Input is a
player-match table with ``player_id``, ``game_id``, ``date``, ``minutes``,
boolean ``knockout`` and the count column named by ``metric``.

Assumptions:

- Per-90 rates are minutes-weighted, ``90 * sum(count) / sum(minutes)``.
- Over-performance of a player in a period is his knockout rate minus his own
  non-knockout rate in the same period, divided by the Poisson standard error
  of that difference under the null of one common rate:
  ``sqrt(90 * pooled_rate * (1 / knockout_minutes + 1 / other_minutes))``.
  For a non-count metric (zone-xG) this is a quasi-Poisson approximation; a
  dispersion factor shared by all players rescales every z alike and leaves
  the correlation unchanged.
- Periods are closed date intervals on ``date``. A player enters if, in both
  periods, he has at least ``min_knockout_minutes`` of knockout football, some
  non-knockout minutes and a non-zero pooled rate.
- Persistence is the Pearson correlation of z between the periods across
  players. The interval is a percentile bootstrap over players. The p-value is
  a one-sided permutation test, ``(1 + #{r_perm >= r}) / (n_perm + 1)``: a trait
  can only show up as a positive correlation.
- Placebo: the real knockout rows are dropped; among the remaining games a
  random set, the same share of them as knockout games are of all games, is
  labelled knockout. A sound test gives a correlation near 0 there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PERIODS = (("2020-01-01", "2022-12-31"), ("2023-01-01", "2026-08-31"))


@dataclass(frozen=True)
class Persistence:
    r: float
    ci_low: float
    ci_high: float
    p_value: float
    n_players: int
    players: pd.DataFrame  # index player_id: knockout_minutes_a/_b, z_a, z_b


def knockout_persistence(
    pm: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    *,
    periods: tuple[tuple[str, str], tuple[str, str]] = PERIODS,
    min_knockout_minutes: float = 360.0,
    n_boot: int = 2000,
    n_perm: int = 2000,
    level: float = 0.9,
) -> Persistence:
    (a_start, a_end), (b_start, b_end) = periods
    players = _overperformance(pm, metric, a_start, a_end).join(
        _overperformance(pm, metric, b_start, b_end), lsuffix="_a", rsuffix="_b", how="inner"
    )
    players = players[
        (players["knockout_minutes_a"] >= min_knockout_minutes)
        & (players["knockout_minutes_b"] >= min_knockout_minutes)
    ]
    za = players["z_a"].to_numpy(dtype=np.float64)
    zb = players["z_b"].to_numpy(dtype=np.float64)
    n = len(za)
    r = float(np.corrcoef(za, zb)[0, 1])
    draw = rng.integers(n, size=(n_boot, n))
    lo, hi = np.quantile(_rowcorr(za[draw], zb[draw]), [(1 - level) / 2, (1 + level) / 2])
    perm = np.argsort(rng.random((n_perm, n)), axis=1)
    r_perm = _rowcorr(np.broadcast_to(za, perm.shape), zb[perm])
    p = (1 + int((r_perm >= r).sum())) / (n_perm + 1)
    return Persistence(r, float(lo), float(hi), p, n, players)


def placebo_knockout(pm: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Non-knockout rows only, with a random set of their games relabelled knockout."""
    share = pm.loc[pm["knockout"], "game_id"].nunique() / pm["game_id"].nunique()
    rest = pm[~pm["knockout"]].copy()
    games = rest["game_id"].unique()
    chosen = rng.choice(games, size=round(share * len(games)), replace=False)
    rest["knockout"] = rest["game_id"].isin(chosen)
    return rest


def _overperformance(pm: pd.DataFrame, metric: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(pm["date"])
    rows = pm[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    sums = rows.groupby(["player_id", "knockout"])[["minutes", metric]].sum().unstack(fill_value=0)
    m_ko, m_other = sums[("minutes", True)], sums[("minutes", False)]
    c_ko, c_other = sums[(metric, True)], sums[(metric, False)]
    pooled = 90 * (c_ko + c_other) / (m_ko + m_other)
    se = np.sqrt(90 * pooled * (1 / m_ko + 1 / m_other))
    z = (90 * c_ko / m_ko - 90 * c_other / m_other) / se
    out = pd.DataFrame({"knockout_minutes": m_ko, "z": z})
    return out[(m_ko > 0) & (m_other > 0) & (pooled > 0)]


def _rowcorr(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pearson correlation of each row of ``a`` with the same row of ``b``."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    out: NDArray[np.float64] = (a * b).sum(axis=1) / np.sqrt(
        (a * a).sum(axis=1) * (b * b).sum(axis=1)
    )
    return out
