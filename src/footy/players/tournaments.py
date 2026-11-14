"""M7(a) — how much event-visible on-ball value the public box score sees (design §5).

Exploratory: this analysis is not in the pre-registration. Pure: no I/O, randomness only
through the ``numpy.random.Generator`` argument.

Question. For a player-competition (a unit: ``player_id`` x ``season``, where the caller
puts the competition-season label in ``season``), how well do box-score per-90 rates
(goals, assists, shots, ...) predict an event metric's per-90 rate (xT added,
progressive passes, ...)? Off-ball value is in no event data; this only measures what
events can see.

Estimators, per target and per sample of units:

- ``r2_cv``: cross-validated R² of a ridge regression of the target rate on the box
  rates. Folds are by player (a player's units are never split between training and
  test). Out-of-fold predictions are averaged over ``n_cv`` random fold draws, each
  still out of fold, and R² = 1 - SSE / SST with SST around the full-sample mean.
- ``rel_target``, ``rel_box``: split-half reliability (M1's definition, reusing its
  helpers): each unit's matches are dealt at random into two halves, per-90 rates are
  computed per half, the Pearson correlation across units is stepped up with
  Spearman–Brown and averaged over ``n_splits`` splits. ``rel_box`` is the reliability
  of the box-score prediction itself: the composite sum_k w_k x_k with each unit's
  out-of-fold weights (linear in per-90 rates, so it is the rate of the per-match count
  sum_k w_k count_k). A unit enters a split when both halves reach ``min_half_minutes``.
- ``r2_disatt = r2_cv / (rel_target * rel_box)``: the requested correction. It assumes
  that the sampling errors of target and box score are independent. They are not when
  one match moves both (a dominant team-match raises shots and progressive passes; an
  assist is also a key pass and carries sb_xa), and then it is biased upwards and can
  exceed 1.
- ``r2_cross``: the cross-half version, free of that bias. In each split,
  rho_PY = mean(corr(P_A, Y_B), corr(P_B, Y_A)) uses box score and target from
  disjoint matches, so shared match noise cancels; the true-score correlation is
  mean(rho_PY) / sqrt(mean(r_PP) * mean(r_YY)) with the half-length (not stepped-up)
  correlations, and ``r2_cross`` is its signed square. Same estimand as ``r2_disatt``
  when the errors are independent.

Both are the visibility of the target to the *fitted* composite: weights fitted on
noisy rates need not be the best weights for true rates, so both are lower bounds on
what the best linear box-score summary could see, up to sampling error. Neither is
clipped to [0, 1]; NaN when a reliability is not positive.

Assumptions:

- Unit rates are minutes-weighted (``90 * sum(count) / sum(minutes)``); units are
  unweighted in regressions and correlations.
- Ridge: features standardised on the training fold (a constant feature gets weight
  0), intercept unpenalised, one penalty per target from ``ALPHAS`` by exact
  leave-one-unit-out error within the training fold (not grouped by player).
- Bootstrap: players are resampled with replacement and keep all their units; each
  drawn copy is a separate unit for reliability, but copies of a player share a fold,
  so no copy is ever predicted from itself. Inside a replicate the CV and split counts
  are ``boot_cv`` and ``boot_splits`` (Monte Carlo noise widens the interval slightly).
  Percentile intervals over the finite replicates; ``boot_nan`` is the share of
  replicates in which either estimator is NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from footy.players.reliability import UNIT, _half_rates, _random_halves

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BOX_FEATURES = (
    "goals", "assists", "shots", "shots_on_target", "fouls_won", "fouls_committed",
    "offsides", "yellow_cards", "red_cards",
)  # fmt: skip
TARGETS = (
    "xt_added", "progressive_passes", "progressive_carries", "passes_into_box", "key_passes",
    "sb_xa", "pressures", "tackles_interceptions",
)  # fmt: skip
ALPHAS = 10.0 ** np.arange(-2.0, 4.25, 0.25)
ESTIMATES = ("r2_cv", "rel_target", "rel_box", "r2_disatt", "r2_cross")


def unit_groups(pm: pd.DataFrame) -> pd.DataFrame:
    """Per unit: total minutes and the position group with the most minutes.

    Ties go to the alphabetically first group.
    """
    by = pm.groupby([*UNIT, "position_group"])["minutes"].sum().reset_index()
    modal = by.sort_values("minutes", ascending=False, kind="stable").drop_duplicates(list(UNIT))
    out = modal.set_index(list(UNIT))[["position_group"]].sort_index()
    out["minutes"] = pm.groupby(list(UNIT))["minutes"].sum()
    return out


def _by_unit(unit: IntArray, values: FloatArray, n: int) -> FloatArray:
    return np.column_stack([np.bincount(unit, v, n) for v in values.T])


def _colcorr(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pearson correlation of each column of a with the same column of b."""
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = (a * b).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return out


def _ridge(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Intercepts (T,) and raw-scale weights (p, T) of a ridge fit of each column of y."""
    n = len(x)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    keep = sd > 0
    z = (x[:, keep] - mu[keep]) / sd[keep]
    ym = y.mean(axis=0)
    yc = y - ym
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    uty = u.T @ yc
    shrink = s**2 / (s**2 + ALPHAS[:, None])  # (A, k)
    fitted = np.einsum("nk,ak,kt->ant", u, shrink, uty)
    leverage = 1.0 / n + shrink @ (u**2).T  # (A, n)
    loo = (((yc - fitted) / (1.0 - leverage)[:, :, None]) ** 2).sum(axis=1)  # (A, T)
    alpha = ALPHAS[loo.argmin(axis=0)]
    coef = np.zeros((x.shape[1], y.shape[1]))
    coef[keep] = vt.T @ (s[:, None] / (s[:, None] ** 2 + alpha) * uty) / sd[keep, None]
    return ym - mu @ coef, coef


def _estimate(
    unit: IntArray,
    player: IntArray,
    minutes: FloatArray,
    x: FloatArray,
    y: FloatArray,
    rng: np.random.Generator,
    n_folds: int,
    n_cv: int,
    n_splits: int,
    min_half_minutes: float,
) -> FloatArray:
    """Rows of ``ESTIMATES`` x targets, for rows grouped by ``unit`` (0..n-1);
    ``player`` (per unit) defines the folds."""
    n = int(unit.max()) + 1
    mins = np.bincount(unit, minutes, n)
    xu = 90.0 * _by_unit(unit, x, n) / mins[:, None]
    yu = 90.0 * _by_unit(unit, y, n) / mins[:, None]
    pred = np.zeros_like(yu)
    weights = np.zeros((n, x.shape[1], y.shape[1]))
    n_players = int(player.max()) + 1
    for _ in range(n_cv):
        fold = rng.permutation(n_players)[player] % n_folds
        for f in range(n_folds):
            test = fold == f
            intercept, coef = _ridge(xu[~test], yu[~test])
            pred[test] += (intercept + xu[test] @ coef) / n_cv
            weights[test] += coef / n_cv
    r2_cv = 1.0 - ((yu - pred) ** 2).sum(axis=0) / ((yu - yu.mean(axis=0)) ** 2).sum(axis=0)

    composite = np.einsum("rp,rpt->rt", x, weights[unit])
    every = np.ones(len(unit), dtype=bool)
    r = np.empty((n_splits, 3, y.shape[1]))  # r_YY, r_PP, rho_PY per split, half length
    for s in range(n_splits):
        key, _ = _random_halves(unit, minutes, rng)
        half_minutes = _half_rates(key, minutes, y[:, 0], n, every)[1]
        ok = (half_minutes >= min_half_minutes).all(axis=1)
        yh = np.stack([_half_rates(key, minutes, c, n, every)[0][ok] for c in y.T], axis=2)
        ph = np.stack([_half_rates(key, minutes, c, n, every)[0][ok] for c in composite.T], 2)
        r[s, 0] = _colcorr(yh[:, 0], yh[:, 1])
        r[s, 1] = _colcorr(ph[:, 0], ph[:, 1])
        r[s, 2] = (_colcorr(ph[:, 0], yh[:, 1]) + _colcorr(ph[:, 1], yh[:, 0])) / 2.0
    rel_target, rel_box = (2.0 * r[:, :2] / (1.0 + r[:, :2])).mean(axis=0)  # Spearman–Brown
    mean_r = r.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (rel_target > 0) & (rel_box > 0)
        r2_disatt = np.where(valid, r2_cv / (rel_target * rel_box), np.nan)
        rho = mean_r[2] / np.sqrt(mean_r[0] * mean_r[1])
    r2_cross = np.where((mean_r[0] > 0) & (mean_r[1] > 0), np.sign(rho) * rho**2, np.nan)
    return np.vstack([r2_cv, rel_target, rel_box, r2_disatt, r2_cross])


def visibility(
    pm: pd.DataFrame,
    targets: Sequence[str],
    features: Sequence[str],
    rng: np.random.Generator,
    *,
    n_folds: int = 5,
    n_cv: int = 20,
    n_splits: int = 200,
    n_boot: int = 500,
    boot_cv: int = 2,
    boot_splits: int = 20,
    min_half_minutes: float = 90.0,
    level: float = 0.95,
) -> pd.DataFrame:
    """One row per target: sample sizes, ``ESTIMATES`` and player-bootstrap intervals.

    ``pm`` has one row per player-match with ``player_id``, ``season``, ``minutes`` and
    the count columns; every row's unit is analysed (filter units first).
    """
    unit = np.asarray(pm.groupby(list(UNIT)).ngroup(), dtype=np.int64)
    n = int(unit.max()) + 1
    player = np.zeros(n, dtype=np.int64)
    player[unit] = pd.factorize(pm["player_id"])[0]
    minutes = pm["minutes"].to_numpy(dtype=np.float64)
    x = pm[list(features)].to_numpy(dtype=np.float64)
    y = pm[list(targets)].to_numpy(dtype=np.float64)
    args = (n_folds, n_cv, n_splits, min_half_minutes)
    point = _estimate(unit, player, minutes, x, y, rng, *args)

    n_players = int(player.max()) + 1
    units_of = [np.flatnonzero(player == p) for p in range(n_players)]
    size = np.bincount(unit, minlength=n)
    rows_of = np.split(np.argsort(unit, kind="stable"), np.cumsum(size)[:-1])
    boot = np.empty((n_boot, 2, len(targets)))
    for b in range(n_boot):
        units = np.concatenate([units_of[p] for p in rng.integers(n_players, size=n_players)])
        rows = np.concatenate([rows_of[u] for u in units])
        new_unit = np.repeat(np.arange(len(units)), size[units]).astype(np.int64)
        est = _estimate(
            new_unit, player[units], minutes[rows], x[rows], y[rows], rng,
            n_folds, boot_cv, boot_splits, min_half_minutes,
        )  # fmt: skip
        boot[b] = est[[3, 4]]
    tails = [(1 - level) / 2, (1 + level) / 2]
    lo, hi = np.nanquantile(boot, tails, axis=0)
    out = pd.DataFrame(point.T, index=pd.Index(list(targets), name="target"), columns=ESTIMATES)
    out.insert(0, "n_players", n_players)
    out.insert(0, "n_units", n)
    out["r2_disatt_lo"], out["r2_disatt_hi"] = lo[0], hi[0]
    out["r2_cross_lo"], out["r2_cross_hi"] = lo[1], hi[1]
    out["boot_nan"] = np.isnan(boot).any(axis=1).mean(axis=0)
    return out
