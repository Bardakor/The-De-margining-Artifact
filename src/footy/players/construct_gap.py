"""The unmeasured share of on-pitch impact (design §5, M4).

For the players of one position group:

1. Prediction model. Ridge regression of RAPM on the per-90 statistics
   (z-scored, unpenalised intercept, strength ``alpha * n``); ``alpha`` is the
   value in ``ALPHAS`` with the smallest out-of-fold squared error over K folds
   of players. ``R2_full = max(corr(out-of-fold prediction, RAPM), 0)**2`` and
   ``share_naive = 1 - R2_full`` is the share without any correction. The model
   refitted on all players at that ``alpha`` makes the half predictions below.
2. Game splits. For each split, RAPM is fitted on each half of the games and
   the statistics are computed on the same halves (``split_stats``), so half A
   of one side shares no match with half B of the other.
3. ``estimator="cross"`` (default, pre-registration amendment A10):
   ``r = mean(corr(pred(stats_A), RAPM_B), corr(pred(stats_B), RAPM_A))`` over
   splits, ``R2_obs = max(r, 0)**2``, divided by the half-length reliabilities
   ``rel_pred = corr(pred(stats_A), pred(stats_B))`` and
   ``rel_RAPM = corr(RAPM_A, RAPM_B) * attribution`` (half-data attribution,
   ``rapm.attribution_factor(..., halves=True)``). A match that moves both a
   player's shots and his stints' goals sits in one half only, so it cannot
   inflate ``R2_obs``.
   ``estimator="within"``: ``R2_obs = R2_full``, divided by the Spearman-Brown
   stepped-up versions of the same reliabilities (full-data attribution).
   Shared match noise inflates ``R2_obs`` and biases the share down.
4. ``R2_true = R2_obs / (rel_RAPM * rel_pred)``; ``share = 1 - R2_true``; both
   reported raw and clipped to [0, 1].
5. Joint bootstrap (amendment A8): each draw resamples players with
   replacement within each group (copies of one player stay in one fold) and
   re-estimates everything above, both reliabilities included. With RAPM
   bootstrap replicates, draw ``b`` uses replicate ``b`` and the split of its
   own games, for RAPM and statistics alike. Percentile intervals.

``attribution`` corrects split-half reliability for teammate leakage (A9; see
``rapm.attribution_factor``). Assumptions: only players present in every input
enter; the attribution factor is held fixed in the bootstrap (a property of the
schedule, sampling SD near 0.005 at a few hundred players); a replicate has one
split, against several for the point estimate; ``alpha`` is selected on the
same folds that give ``R2_full`` (seven candidates, very slightly optimistic).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from footy.players.rapm import Halves

ALPHAS = 10.0 ** np.arange(-4.0, 3.0)
H3_HIGH = ("CB", "DM")
H3_LOW = ("CF", "W")

FloatArray = NDArray[np.float64]
Estimator = Literal["cross", "within"]
SplitArrays = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


@dataclass(frozen=True)
class Splits:
    """Per game split ``r``: one RAPM component fitted on each half (row ``r``
    of ``rapm_a`` / ``rapm_b``, one column per player id) and the per-90
    statistics on the same halves (item ``r`` of ``stats``)."""

    rapm_a: pd.DataFrame
    rapm_b: pd.DataFrame
    stats: list[tuple[pd.DataFrame, pd.DataFrame]]


def split_stats(
    player_match: pd.DataFrame, halves: Halves, columns: Sequence[str]
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Per-90 rates of the count ``columns`` per player on both halves of every
    split in ``halves``, games weighted as in the RAPM fits (so bootstrap counts
    carry over). ``player_match`` has game_id, player_id, minutes and
    ``columns``; players without minutes in a half are left out of it."""
    cols = list(columns)
    out = []
    for r in range(len(halves.games_a)):
        pair = []
        for games in (halves.games_a.iloc[r], halves.games_b.iloc[r]):
            w = player_match["game_id"].map(games).fillna(0.0).to_numpy()
            sums = (
                player_match[["minutes", *cols]].mul(w, axis=0).groupby(player_match["player_id"])
            ).sum()
            sums = sums[sums["minutes"] > 0]
            pair.append(sums[cols].div(sums["minutes"], axis=0) * 90)
        out.append((pair[0], pair[1]))
    return out


def unmeasured_share(
    rapm: pd.Series,
    stats: pd.DataFrame,
    splits: Splits,
    attribution: Mapping[str, float],
    groups: pd.Series,
    rng: np.random.Generator,
    *,
    estimator: Estimator = "cross",
    n_boot: int = 200,
    replicates: tuple[pd.DataFrame, Splits] | None = None,
    n_folds: int = 5,
    level: float = 0.9,
) -> pd.DataFrame:
    """One row per group label of ``groups`` (player id -> label).

    ``rapm`` is one RAPM component and ``stats`` the per-90 statistics from all
    matches, both indexed by player id; ``splits`` holds the same component and
    statistics on game halves; ``attribution`` maps group -> attribution factor
    (half-data for "cross", full-data for "within"). ``replicates`` is
    (values with one row per replicate, their ``Splits``) from
    ``rapm.bootstrap_rapm``.
    """
    point, draws = _estimate(
        rapm, stats, splits, attribution, groups, rng, estimator, n_boot, replicates, n_folds
    )
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
    stats: pd.DataFrame,
    splits: Splits,
    attribution: Mapping[str, float],
    groups: pd.Series,
    rng: np.random.Generator,
    *,
    high: Sequence[str] = H3_HIGH,
    low: Sequence[str] = H3_LOW,
    estimator: Estimator = "cross",
    n_boot: int = 200,
    replicates: tuple[pd.DataFrame, Splits] | None = None,
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
        rapm, stats, splits, attribution, groups.map(pool).dropna(), rng, estimator,
        n_boot, replicates, n_folds,
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
    stats: pd.DataFrame,
    splits: Splits,
    attribution: Mapping[str, float],
    groups: pd.Series,
    rng: np.random.Generator,
    estimator: Estimator,
    n_boot: int,
    replicates: tuple[pd.DataFrame, Splits] | None,
    n_folds: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, FloatArray]]]:
    all_splits = [splits] if replicates is None else [splits, replicates[1]]
    ids = rapm.index.intersection(groups.dropna().index).intersection(stats.index)
    for s in all_splits:
        ids = ids.intersection(s.rapm_a.columns).intersection(s.rapm_b.columns)
        for a, b in s.stats:
            ids = ids.intersection(a.index).intersection(b.index)
    cols = stats.columns
    y = rapm.loc[ids].to_numpy(dtype=np.float64)
    z = stats.loc[ids, cols].to_numpy(dtype=np.float64)
    labels = groups.loc[ids].to_numpy()
    members = {str(g): np.flatnonzero(labels == g) for g in sorted(set(labels))}

    def arrays(s: Splits) -> SplitArrays:
        ra, rb = (h.loc[:, ids].to_numpy(dtype=np.float64) for h in (s.rapm_a, s.rapm_b))
        sa, sb = (
            np.stack([pair[k].loc[ids, cols].to_numpy(dtype=np.float64) for pair in s.stats])
            for k in (0, 1)
        )
        return ra, rb, sa, sb

    def one(g: str, idx: NDArray[np.int64], yv: FloatArray, sp: SplitArrays) -> dict[str, float]:
        ra, rb, sa, sb = (x[:, idx] for x in sp)
        return _share(yv[idx], z[idx], ra, rb, sa, sb, attribution[g], idx, rng, n_folds, estimator)

    base = arrays(splits)
    point = {g: one(g, m, y, base) for g, m in members.items()}
    reps = None
    if replicates is not None:
        if min(len(replicates[0]), len(replicates[1].stats)) < n_boot:
            raise ValueError(f"{len(replicates[0])} RAPM replicates for n_boot={n_boot}")
        reps = (replicates[0].loc[:, ids].to_numpy(dtype=np.float64), arrays(replicates[1]))
    keys = ("share", "rel_rapm", "rel_pred")
    draws = {g: {k: np.empty(n_boot) for k in keys} for g in members}
    for i in range(n_boot):
        yv, sp = y, base
        if reps is not None:
            ra, rb, sa, sb = reps[1]
            yv, sp = reps[0][i], (ra[i : i + 1], rb[i : i + 1], sa[i : i + 1], sb[i : i + 1])
        for g, m in members.items():
            est = one(g, m[rng.integers(len(m), size=len(m))], yv, sp)
            for k in keys:
                draws[g][k][i] = est[k]
    return point, draws


def _share(
    y: FloatArray,
    z: FloatArray,
    ra: FloatArray,
    rb: FloatArray,
    sa: FloatArray,
    sb: FloatArray,
    attribution: float,
    units: NDArray[np.int64],
    rng: np.random.Generator,
    n_folds: int,
    estimator: Estimator,
) -> dict[str, float]:
    """y, z: full-data RAPM and statistics; ra, rb: (splits, players) RAPM on
    each half; sa, sb: (splits, players, stats) statistics on the same halves."""
    _, unit = np.unique(units, return_inverse=True)
    fold = (rng.permutation(unit.max() + 1) % n_folds)[unit]
    pred = np.empty((len(ALPHAS), len(y)))
    for k in range(n_folds):
        train, test = fold != k, fold == k
        for i, alpha in enumerate(ALPHAS):
            pred[i, test] = _ridge(z[train], y[train], alpha)(z[test])
    best = int(np.argmin(((pred - y) ** 2).sum(axis=1)))
    r2_full = max(float(np.corrcoef(pred[best], y)[0, 1]), 0.0) ** 2
    model = _ridge(z, y, ALPHAS[best])
    pa, pb = model(sa), model(sb)
    rel_pred, rel_rapm = _mean_corr(pa, pb), _mean_corr(ra, rb)
    if estimator == "cross":
        r2_obs = max((_mean_corr(pa, rb) + _mean_corr(pb, ra)) / 2, 0.0) ** 2
    else:
        r2_obs, rel_pred, rel_rapm = r2_full, _step_up(rel_pred), _step_up(rel_rapm)
    rel_rapm *= attribution
    r2_true = r2_obs / (rel_rapm * rel_pred)
    return {
        "n": float(unit.max() + 1),
        "r2_obs": r2_obs,
        "rel_rapm": rel_rapm,
        "rel_pred": rel_pred,
        "r2_true": r2_true,
        "share_naive": 1 - r2_full,
        "share": 1 - r2_true,
    }


def _ridge(z: FloatArray, y: FloatArray, alpha: float) -> Callable[[FloatArray], FloatArray]:
    """Ridge on z-scored columns, unpenalised intercept, strength alpha * n.
    The returned predictor works on the last axis (players x stats, or
    splits x players x stats)."""
    mu, sd = z.mean(axis=0), z.std(axis=0)
    sd[sd == 0] = 1.0
    s = (z - mu) / sd
    beta = np.linalg.solve(s.T @ s + alpha * len(y) * np.eye(z.shape[1]), s.T @ (y - y.mean()))
    y_mean = float(y.mean())

    def predict(new: FloatArray) -> FloatArray:
        return np.asarray(y_mean + ((new - mu) / sd) @ beta, dtype=np.float64)

    return predict


def _mean_corr(a: FloatArray, b: FloatArray) -> float:
    """Row-wise correlation of a and b (splits x players), averaged over rows."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return float(np.mean((a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))))


def _step_up(r: float) -> float:
    """Spearman-Brown: reliability of the full-length measure from a half's."""
    return 2 * r / (1 + r)


def _interval(x: FloatArray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def _clip(x: float | FloatArray) -> FloatArray:
    return np.clip(x, 0.0, 1.0)
