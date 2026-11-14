"""The unmeasured share of on-pitch impact (design §5, M4).

For the players of one position group:

1. Out-of-fold predictions of RAPM from the per-90 statistics by ridge
   regression, K folds over players. Statistics are z-scored on the training
   folds, the intercept is unpenalised and the ridge strength
   ``alpha * n_train`` is the ``alpha`` in ``ALPHAS`` with the smallest
   out-of-fold squared error. ``R2_obs = max(corr(prediction, RAPM), 0)**2``:
   out-of-sample predictions anti-correlated with RAPM explain nothing.
2. ``rel_pred``: the ridge at that ``alpha`` refitted on all players of the
   group and applied to each half's statistics; the correlation of the two
   half-predictions, stepped up by Spearman-Brown. (Using the out-of-fold
   models instead would let the per-fold intercepts pass for reliable signal.)
3. ``rel_RAPM``: the correlation of the RAPM split-half estimates, averaged
   over splits and stepped up by Spearman-Brown, times the group's
   ``rapm.attribution_factor`` (split-half alone counts teammate leakage as
   signal and biases the share upward).
4. ``R2_true = R2_obs / (rel_RAPM * rel_pred)``; ``share = 1 - R2_true``; both
   reported raw and clipped to [0, 1]. ``share_naive = 1 - R2_obs`` is the
   share without disattenuation.
5. Joint bootstrap (pre-registration amendment A8): each draw resamples players
   with replacement within each group (copies of one player stay in one fold)
   and re-estimates everything above on them, both reliabilities included.
   When RAPM bootstrap replicates are passed, draw ``b`` uses replicate ``b``
   and the split of replicate ``b``'s own games, so match sampling enters the
   share and its reliability together. Percentile intervals.

Assumptions: only players present in every input enter; the attribution factor
is held fixed (a property of the schedule, with a sampling SD near 0.005 at a
few hundred players); a replicate's reliability comes from one split of its
games, against the average of many splits for the point estimate, which makes
the intervals slightly conservative; the selection of ``alpha`` on the same
folds makes ``R2_obs`` very slightly optimistic (seven candidate values).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ALPHAS = 10.0 ** np.arange(-4.0, 3.0)
H3_HIGH = ("CB", "DM")
H3_LOW = ("CF", "W")

FloatArray = NDArray[np.float64]
Frames3 = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def unmeasured_share(
    rapm: pd.Series,
    rapm_halves: tuple[pd.DataFrame, pd.DataFrame],
    attribution: Mapping[str, float],
    stats: pd.DataFrame,
    stats_a: pd.DataFrame,
    stats_b: pd.DataFrame,
    groups: pd.Series,
    rng: np.random.Generator,
    *,
    n_boot: int = 200,
    replicates: Frames3 | None = None,
    n_folds: int = 5,
    level: float = 0.9,
) -> pd.DataFrame:
    """One row per group label of ``groups`` (player id -> label).

    ``rapm`` is one RAPM component (index player id); ``rapm_halves`` its
    split-half estimates (one row per split, one column per player, e.g.
    ``halves.a["offence"], halves.b["offence"]`` from ``rapm.split_halves``);
    ``attribution`` maps group -> ``rapm.attribution_factor``. ``stats``,
    ``stats_a`` and ``stats_b`` are per-90 rates indexed by player id, from all
    matches and from two disjoint halves of them. ``replicates`` is
    (values, half_a, half_b) of the same component from ``rapm.bootstrap_rapm``,
    one row per replicate.
    """
    point, draws = _estimate(
        rapm, rapm_halves, attribution, stats, stats_a, stats_b, groups, rng,
        n_boot, replicates, n_folds,
    )  # fmt: skip
    rows = {}
    for g, est in point.items():
        lo, hi = _interval(draws[g]["share"], level)
        rows[g] = {
            **est,
            "r2_true_clipped": _clip(est["r2_true"]),
            "share_clipped": _clip(est["share"]),
            "ci_low": lo,
            "ci_high": hi,
            "ci_low_clipped": _clip(lo),
            "ci_high_clipped": _clip(hi),
        }
        for rel in ("rel_rapm", "rel_pred"):
            rows[g][f"{rel}_ci_low"], rows[g][f"{rel}_ci_high"] = _interval(draws[g][rel], level)
    return pd.DataFrame.from_dict(rows, orient="index")


def position_contrast(
    rapm: pd.Series,
    rapm_halves: tuple[pd.DataFrame, pd.DataFrame],
    attribution: Mapping[str, float],
    stats: pd.DataFrame,
    stats_a: pd.DataFrame,
    stats_b: pd.DataFrame,
    groups: pd.Series,
    rng: np.random.Generator,
    *,
    high: Sequence[str] = H3_HIGH,
    low: Sequence[str] = H3_LOW,
    n_boot: int = 200,
    replicates: Frames3 | None = None,
    n_folds: int = 5,
    level: float = 0.9,
) -> pd.Series:
    """H3: unmeasured share of the pooled ``high`` groups minus that of the
    pooled ``low`` groups, raw and clipped, with joint-bootstrap percentile
    intervals (both pools use the same replicate in a draw).

    The pools are labelled ``"+".join(high)`` and ``"+".join(low)`` (default
    ``"CB+DM"`` and ``"CF+W"``); ``attribution`` is keyed by these labels.
    """
    hi_label, lo_label = "+".join(high), "+".join(low)
    pool = {**{g: hi_label for g in high}, **{g: lo_label for g in low}}
    point, draws = _estimate(
        rapm, rapm_halves, attribution, stats, stats_a, stats_b, groups.map(pool).dropna(),
        rng, n_boot, replicates, n_folds,
    )  # fmt: skip
    hi_s, lo_s = draws[hi_label]["share"], draws[lo_label]["share"]
    lo, hi = _interval(hi_s - lo_s, level)
    lo_c, hi_c = _interval(_clip(hi_s) - _clip(lo_s), level)
    return pd.Series(
        {
            "diff": point[hi_label]["share"] - point[lo_label]["share"],
            "ci_low": lo,
            "ci_high": hi,
            "diff_clipped": _clip(point[hi_label]["share"]) - _clip(point[lo_label]["share"]),
            "ci_low_clipped": lo_c,
            "ci_high_clipped": hi_c,
        }
    )


def _estimate(
    rapm: pd.Series,
    rapm_halves: tuple[pd.DataFrame, pd.DataFrame],
    attribution: Mapping[str, float],
    stats: pd.DataFrame,
    stats_a: pd.DataFrame,
    stats_b: pd.DataFrame,
    groups: pd.Series,
    rng: np.random.Generator,
    n_boot: int,
    replicates: Frames3 | None,
    n_folds: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, FloatArray]]]:
    frames = (stats, stats_a, stats_b)
    ids = rapm.index.intersection(groups.dropna().index)
    for idx in (*(s.index for s in frames), *(h.columns for h in rapm_halves)):
        ids = ids.intersection(idx)
    if replicates is not None:
        if min(len(r) for r in replicates) < n_boot:
            raise ValueError(f"{len(replicates[0])} RAPM replicates for n_boot={n_boot}")
        for r in replicates:
            ids = ids.intersection(r.columns)
    y = rapm.loc[ids].to_numpy(dtype=np.float64)
    ha, hb = (h.loc[:, ids].to_numpy(dtype=np.float64) for h in rapm_halves)
    z, za, zb = (s.loc[ids, stats.columns].to_numpy(dtype=np.float64) for s in frames)
    labels = groups.loc[ids].to_numpy()
    members = {str(g): np.flatnonzero(labels == g) for g in sorted(set(labels))}

    def one(
        g: str, idx: NDArray[np.int64], yv: FloatArray, a: FloatArray, b: FloatArray
    ) -> dict[str, float]:
        rel = _spearman_brown(a[:, idx], b[:, idx]) * attribution[g]
        return _share(yv[idx], rel, z[idx], za[idx], zb[idx], idx, rng, n_folds)

    point = {g: one(g, m, y, ha, hb) for g, m in members.items()}
    keys = ("share", "rel_rapm", "rel_pred")
    draws = {g: {k: np.empty(n_boot) for k in keys} for g in members}
    for i in range(n_boot):
        yv, a, b = y, ha, hb
        if replicates is not None:
            rows, a, b = (r.iloc[i : i + 1][ids].to_numpy(dtype=np.float64) for r in replicates)
            yv = rows[0]
        for g, m in members.items():
            est = one(g, m[rng.integers(len(m), size=len(m))], yv, a, b)
            for k in keys:
                draws[g][k][i] = est[k]
    return point, draws


def _share(
    y: FloatArray,
    rel_rapm: float,
    z: FloatArray,
    za: FloatArray,
    zb: FloatArray,
    units: NDArray[np.int64],
    rng: np.random.Generator,
    n_folds: int,
) -> dict[str, float]:
    _, unit = np.unique(units, return_inverse=True)
    fold = (rng.permutation(unit.max() + 1) % n_folds)[unit]
    pred = np.empty((len(ALPHAS), len(y)))
    for k in range(n_folds):
        train, test = fold != k, fold == k
        for i, alpha in enumerate(ALPHAS):
            pred[i, test] = _ridge(z[train], y[train], alpha)(z[test])
    best = int(np.argmin(((pred - y) ** 2).sum(axis=1)))
    r2_obs = max(float(np.corrcoef(pred[best], y)[0, 1]), 0.0) ** 2
    model = _ridge(z, y, ALPHAS[best])
    rel_pred = _spearman_brown(model(za)[None, :], model(zb)[None, :])
    r2_true = r2_obs / (rel_rapm * rel_pred)
    return {
        "n": float(unit.max() + 1),
        "r2_obs": r2_obs,
        "rel_rapm": rel_rapm,
        "rel_pred": rel_pred,
        "r2_true": r2_true,
        "share_naive": 1 - r2_obs,
        "share": 1 - r2_true,
    }


def _ridge(z: FloatArray, y: FloatArray, alpha: float) -> Callable[[FloatArray], FloatArray]:
    """Ridge on z-scored columns, unpenalised intercept, strength alpha * n."""
    mu, sd = z.mean(axis=0), z.std(axis=0)
    sd[sd == 0] = 1.0
    s = (z - mu) / sd
    beta = np.linalg.solve(s.T @ s + alpha * len(y) * np.eye(z.shape[1]), s.T @ (y - y.mean()))
    y_mean = float(y.mean())

    def predict(new: FloatArray) -> FloatArray:
        return np.asarray(y_mean + ((new - mu) / sd) @ beta, dtype=np.float64)

    return predict


def _spearman_brown(a: FloatArray, b: FloatArray) -> float:
    """Step-up of the row-wise correlation of a and b (splits x players),
    averaged over rows."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    r = float(np.mean((a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))))
    return 2 * r / (1 + r)


def _interval(x: FloatArray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def _clip(x: float | FloatArray) -> FloatArray:
    return np.clip(x, 0.0, 1.0)
