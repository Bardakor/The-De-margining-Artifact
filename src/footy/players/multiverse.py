"""Specification uncertainty: every defensible "best player" ranking (design §5, M5).

The paper refuses to pick one definition of "best". This module runs the full
factorial of definitions in ``DIMENSIONS`` (4·4·3·2·5·3·2·2 = 5,760 specifications),
ranks the pool under each, and measures how much each choice moves each rank.

One specification, in order. Every population statistic (means, SDs, quantiles,
principal components, ridge fits) is computed on the qualifying player-seasons
of the segment, pool and non-pool players alike; only the ranking is restricted
to the pool.

1. competition: the scope rows (league, ucl, intl) of a player-season are merged.
   Rates are averaged with weights ``competition weight × minutes``; ``minutes``
   becomes that weighted sum and ``raw_minutes`` the plain sum over included scopes.
2. min_minutes: a player-season qualifies if ``raw_minutes`` >= the threshold.
3. shrinkage: empirical Bayes, ``mean_g + r (x - mean_g)`` with ``r = m / (m + k)``,
   ``m = raw_minutes``, ``mean_g`` the minutes-weighted mean of the player's position
   group (all seasons pooled), ``k`` the metric's stabilisation constant from M1: one
   per metric, or one per (metric, position group) (A16 (i)).
4. context: within each position group, each metric minus its OLS slope on team Elo
   times the centred Elo (the group mean is kept, so only the within-group Elo
   effect is removed).
5. position: ``none`` = z over the segment; ``group_z`` = z within position group;
   ``replacement`` = (x - group 20th percentile) / segment SD (a per-group baseline
   on a common scale). A zero SD is replaced by 1.
6. family × weights: season score = sum of weight × normalised metric, weights
   scaled so their absolute values sum to 1 (``Multiverse.weights`` exposes them).
   Families (A11, A13 (i)): ``box`` goals, non-penalty goals, assists; ``box_gd`` box
   plus raw on-pitch goal difference; ``impact`` RAPM offence and defence; ``hybrid``
   all six.
   ``equal_z`` 1/p each; ``pc1`` first principal component of the normalised
   metrics, sign set so goals load positively; ``rapm_ridge`` ridge regression of
   pooled RAPM on the normalised metrics, lambda by 5-fold CV with folds grouped by
   player; ``v1_like`` the ``expert_weights`` dict.
7. aggregation over the player's qualifying seasons (``aggregate_seasons``).
8. rank within the pool, outfield and goalkeepers separately; 1 = best, ties share
   the best rank ("min"); a pool player with no qualifying season ranks last.

Input contract and assumptions (Claude's calls):

- ``player_season``: player_id, season ("2021/22"), scope, position_group, minutes,
  team_elo and the per-90 rates in ``STAT_METRICS``, all oriented higher = better
  (``on_pitch_gd``: raw on-pitch goal difference per 90 over the minutes in matches
  with stints).
  Rows with another scope (e.g. "all") are ignored. Values must be finite.
  position_group must be constant within a player (design §4.4). club_id is not used.
- ``impact_season``: player_id, season, rapm_offence, rapm_defence (per-season RAPM,
  goals per 90; defence positive = fewer goals conceded) and rapm_total_pooled
  (the player's whole-window RAPM, repeated on his rows). Every qualifying
  player-season must have a row.
- RAPM is estimated once per season over all competitions, so the competition
  dimension acts on impact metrics only through minutes (threshold and the
  weights of mean, volume and recency).
- The regression target is pooled RAPM total rather than per-season RAPM, which is
  much noisier. Ridge folds are the only randomness (``rng``), drawn once per segment.
- Goalkeepers and outfield players are separate populations throughout.
- Recency weights are ``0.5 ** ((last - year) / 2)`` times minutes, anchored on the
  last season present in ``player_season``, not on the player's last season.
- Volume's replacement level is the 20th percentile of season scores over the
  segment (position ``none``) or within the player's group (otherwise).
- ``V1_LIKE_WEIGHTS`` are A13 (i)'s: a stylised imitation of the V1 ranking's emphasis
  on goals and assists, not V1's actual weights. On-pitch goal difference stands in
  for V1's team-results component.

Fallbacks for combinations that are undefined or circular (counted in
``Multiverse.fallbacks``):

- rapm_ridge in ``impact``: regressing RAPM on its own components is circular, so the
  score is RAPM total (offence + defence): weights proportional to each component's
  raw SD, which under position ``none`` gives exactly the total up to a constant.
- rapm_ridge in ``hybrid``: the ridge uses the stat metrics (box and on_pitch_gd)
  only; the impact block gets the RAPM-total weights; each block carries half the
  weight.
- shrinkage ``eb`` and context ``elo_partialled`` are not applied to RAPM metrics:
  RAPM is already ridge-shrunk and adjusted for team-mates and opponents (M3; A12
  fits it without an Elo covariate).
  ``impact × eb`` and ``impact × elo_partialled`` are therefore duplicates.
- pc1 sign without a usable goals loading (impact family, goalkeepers): the sign is set
  so the loadings sum to a positive number (counted per spec × segment).
- best3 with fewer than three qualifying seasons: mean of those available (counted per
  spec × pool player).
- no qualifying season: ranked last (counted per spec × pool player).

Degenerate duplicates are kept so the factorial stays balanced; ``distinct_rankings``
reports how many different rankings exist. Besides the two above, ``v1_like`` in
``impact`` equals ``equal_z`` (equal RAPM weights), and with a single scope the three
competition levels coincide.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
# metric -> k in minutes, or metric -> {position group -> k}
Stabilisation = Mapping[str, float | Mapping[str, float]]

SCOPES = ("league", "ucl", "intl")
GOALKEEPER = "GK"
BOX = ("goals", "np_goals", "assists")
ON_PITCH_GD = ("on_pitch_gd",)
IMPACT_METRICS = ("rapm_offence", "rapm_defence")
STAT_METRICS = BOX + ON_PITCH_GD
METRICS = STAT_METRICS + IMPACT_METRICS
FAMILIES: dict[str, tuple[str, ...]] = {
    "box": BOX,
    "box_gd": STAT_METRICS,
    "impact": IMPACT_METRICS,
    "hybrid": METRICS,
}
DIMENSIONS: dict[str, tuple[str | int, ...]] = {
    "family": tuple(FAMILIES),
    "weights": ("equal_z", "pc1", "rapm_ridge", "v1_like"),
    "position": ("none", "group_z", "replacement"),
    "context": ("none", "elo_partialled"),
    "aggregation": ("peak", "best3", "mean", "volume", "recency"),
    "competition": ("equal", "ucl_intl_1.5", "league_only"),
    "min_minutes": (900, 1800),
    "shrinkage": ("raw", "eb"),
}
COMPETITION_WEIGHTS: dict[str, dict[str, float]] = {
    "equal": {"league": 1.0, "ucl": 1.0, "intl": 1.0},
    "ucl_intl_1.5": {"league": 1.0, "ucl": 1.5, "intl": 1.5},
    "league_only": {"league": 1.0},
}
# A13 (i): a stylised imitation of V1's emphasis on goals and assists, not V1's weights.
V1_LIKE_WEIGHTS: dict[str, float] = {
    "goals": 0.30,
    "np_goals": 0.15,
    "assists": 0.25,
    "on_pitch_gd": 0.20,
    "rapm_offence": 0.05,
    "rapm_defence": 0.05,
}
REPLACEMENT_QUANTILE = 0.2
RECENCY_HALF_LIFE = 2.0
CV_FOLDS = 5
RIDGE_GRID = np.logspace(-4, 2, 13)  # times the number of rows

_SHAPE = tuple(len(levels) for levels in DIMENSIONS.values())
_N_FAMILIES = len(DIMENSIONS["family"])
_N_WEIGHTS = len(DIMENSIONS["weights"])
_N_AGGREGATIONS = len(DIMENSIONS["aggregation"])
_STAT_IDX = [METRICS.index(m) for m in STAT_METRICS]


@dataclass(frozen=True)
class Multiverse:
    """``ranks``: one row per (spec, pool player): spec_id, the eight dimension levels,
    player_id, segment, score, rank. ``weights``: one row per (spec, segment with pool
    players, metric in ``METRICS``): spec_id, segment, metric, weight, the signed weight
    applied to the normalised metric (absolute values sum to 1; 0 outside the family).
    ``fallbacks``: how often each fallback applied."""

    ranks: pd.DataFrame
    weights: pd.DataFrame
    fallbacks: dict[str, int]


def spec_grid() -> pd.DataFrame:
    """The full factorial, spec_id in C order over ``DIMENSIONS``."""
    grid = pd.DataFrame(list(itertools.product(*DIMENSIONS.values())), columns=list(DIMENSIONS))
    for dim, levels in DIMENSIONS.items():
        grid[dim] = pd.Categorical(grid[dim], categories=list(levels))
    grid.insert(0, "spec_id", np.arange(len(grid)))
    return grid


def run_multiverse(
    player_season: pd.DataFrame,
    impact_season: pd.DataFrame,
    stabilisation: Stabilisation,
    pool: Sequence[int],
    rng: np.random.Generator,
    *,
    expert_weights: Mapping[str, float] = V1_LIKE_WEIGHTS,
) -> Multiverse:
    """Score and rank the pool under every specification (contract in the module doc)."""
    ps = player_season[player_season["scope"].isin(SCOPES)]
    if ps[[*STAT_METRICS, "minutes", "team_elo"]].isna().to_numpy().any():
        raise ValueError("player_season has missing metrics, minutes or team_elo")
    by_player = ps.groupby("player_id")["position_group"]
    if (by_player.nunique() > 1).any():
        raise ValueError("position_group must be constant within a player")
    group_of = by_player.first()
    pool_ids = np.asarray(pool, dtype=np.int64)
    absent = np.setdiff1d(pool_ids, group_of.index.to_numpy())
    if absent.size:
        raise ValueError(f"pool players absent from player_season: {absent.tolist()}")
    impact = impact_season.set_index(["player_id", "season"])[
        [*IMPACT_METRICS, "rapm_total_pooled"]
    ]
    seasons = sorted(ps["season"].unique())
    expert = np.array([expert_weights.get(m, 0.0) for m in METRICS])
    is_gk = group_of.loc[pd.Index(pool_ids)].to_numpy() == GOALKEEPER
    counts = dict.fromkeys(
        (
            "pc1_sign_by_loading_sum",
            "best3_fewer_than_3_seasons",
            "no_qualifying_season_ranked_last",
        ),
        0,
    )
    grid = spec_grid()
    scores = np.full((len(grid), pool_ids.size), np.nan)
    ranks = np.empty(scores.shape, dtype=np.int64)
    weights = []
    for gk in (False, True):
        members = is_gk == gk
        if members.any():
            rows = ps[(ps["position_group"] == GOALKEEPER) == gk]
            part, applied = _segment_scores(
                rows, impact, stabilisation, pool_ids[members], seasons, rng, expert, counts
            )
            part = part.reshape(len(grid), -1)
            scores[:, members] = part
            ranks[:, members] = rankdata(
                -np.where(np.isnan(part), -np.inf, part), method="min", axis=1
            )
            weights.append(
                pd.DataFrame(
                    {
                        "spec_id": np.repeat(grid["spec_id"].to_numpy(), len(METRICS)),
                        "segment": "goalkeeper" if gk else "outfield",
                        "metric": np.tile(METRICS, len(grid)),
                        "weight": applied.ravel(),
                    }
                )
            )
    long = grid.loc[grid.index.repeat(pool_ids.size)].reset_index(drop=True)
    long["player_id"] = np.tile(pool_ids, len(grid))
    long["segment"] = pd.Categorical(
        np.tile(np.where(is_gk, "goalkeeper", "outfield"), len(grid)),
        categories=["outfield", "goalkeeper"],
    )
    long["score"] = scores.ravel()
    long["rank"] = ranks.ravel()
    fallbacks = {
        "impact_rapm_ridge_is_rapm_total": _n_specs(family="impact", weights="rapm_ridge"),
        "hybrid_rapm_ridge_stat_ridge_plus_rapm_total": _n_specs(
            family="hybrid", weights="rapm_ridge"
        ),
        "impact_eb_not_applied": _n_specs(family="impact", shrinkage="eb"),
        "impact_elo_partialling_not_applied": _n_specs(family="impact", context="elo_partialled"),
        **counts,
    }
    return Multiverse(
        ranks=long, weights=pd.concat(weights, ignore_index=True), fallbacks=fallbacks
    )


def shrink(rates: FloatArray, minutes: FloatArray, k: FloatArray, group: IntArray) -> FloatArray:
    """Empirical-Bayes rates: ``mean_g + r (x - mean_g)`` with ``r = m / (m + k)``.

    ``rates`` is (rows × metrics), ``k`` the metrics' stabilisation constants in
    minutes, (metrics,) or (rows × metrics), ``mean_g`` the minutes-weighted mean rate
    of the row's group.
    """
    _, codes = np.unique(group, return_inverse=True)
    n_groups = int(codes.max()) + 1
    total = np.zeros((n_groups, rates.shape[1]))
    np.add.at(total, codes, rates * minutes[:, None])
    mean = (total / np.bincount(codes, weights=minutes, minlength=n_groups)[:, None])[codes]
    r = minutes[:, None] / (minutes[:, None] + k)
    return mean + r * (rates - mean)


def aggregate_seasons(
    scores: FloatArray, minutes: FloatArray, years: FloatArray, baseline: FloatArray
) -> dict[str, FloatArray]:
    """The five time aggregations of season scores.

    ``scores`` is (players × seasons × columns), NaN where the season does not qualify;
    ``minutes`` (players × seasons) weights the means and the volume; ``years``
    (seasons,) are start years; ``baseline`` (players × columns) is the replacement
    level of the volume rule. Each result is (players × columns), NaN for a player
    with no qualifying season.
    """
    valid = ~np.isnan(scores)
    n = valid.sum(axis=1)
    filled = np.where(valid, scores, -np.inf)
    top = -np.sort(-filled, axis=1)[:, :3]
    s0 = np.where(valid, scores, 0.0)
    m = np.where(valid, minutes[:, :, None], 0.0)
    mr = m * (0.5 ** ((years.max() - years) / RECENCY_HALF_LIFE))[None, :, None]
    above = np.maximum(s0 - baseline[:, None, :], 0.0) * m
    with np.errstate(invalid="ignore", divide="ignore"):
        result = {
            "peak": filled.max(axis=1),
            "best3": np.where(np.isfinite(top), top, 0.0).sum(axis=1) / np.minimum(n, 3),
            "mean": (m * s0).sum(axis=1) / m.sum(axis=1),
            "volume": above.sum(axis=1) / 90.0,
            "recency": (mr * s0).sum(axis=1) / mr.sum(axis=1),
        }
    return {name: np.where(n > 0, value, np.nan) for name, value in result.items()}


def rank_probabilities(ranks: pd.DataFrame) -> pd.DataFrame:
    """P(#1), P(top-3), P(top-10) per pool player across specifications."""
    flags = ranks[["player_id", "segment"]].assign(
        p_first=ranks["rank"] == 1, p_top3=ranks["rank"] <= 3, p_top10=ranks["rank"] <= 10
    )
    grouped = flags.groupby(["player_id", "segment"], observed=True)
    return grouped[["p_first", "p_top3", "p_top10"]].mean().reset_index()


def specification_curve(ranks: pd.DataFrame) -> pd.DataFrame:
    """Each player's specifications sorted from best to worst rank (ties by spec_id),
    with ``curve_position`` 0, 1, ... for plotting against the dimension levels."""
    curve = ranks.sort_values(["player_id", "rank", "spec_id"]).reset_index(drop=True)
    curve["curve_position"] = curve.groupby("player_id").cumcount()
    return curve


def distinct_rankings(ranks: pd.DataFrame) -> dict[str, int]:
    """Number of different pool rankings across specifications, per segment."""
    out: dict[str, int] = {}
    for segment, part in ranks.groupby("segment", observed=True):
        table = part.pivot(index="spec_id", columns="player_id", values="rank").to_numpy()
        out[str(segment)] = len(np.unique(table, axis=0))
    return out


def variance_indices(ranks: pd.DataFrame) -> pd.DataFrame:
    """Exact first-order and total-order indices of each player's rank per dimension.

    On the full factorial with every specification weighted equally:
    first = Var(E[rank | dim]) / Var(rank), total = 1 - Var(E[rank | other dims]) /
    Var(rank). NaN when the player's rank never changes.
    """
    codes = [
        pd.Categorical(ranks[dim], categories=list(levels)).codes
        for dim, levels in DIMENSIONS.items()
    ]
    cell = np.ravel_multi_index(codes, _SHAPE)
    player_code, players = pd.factorize(ranks["player_id"])
    table = np.full((len(players), int(np.prod(_SHAPE))), np.nan)
    table[player_code, cell] = ranks["rank"].to_numpy(np.float64)
    if np.isnan(table).any():
        raise ValueError("ranks is not the full factorial for every player")
    y = table.reshape((len(players), *_SHAPE))
    total_var = table.var(axis=1)
    rows = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for axis, dim in enumerate(DIMENSIONS, start=1):
            others = tuple(a for a in range(1, y.ndim) if a != axis)
            first = y.mean(axis=others).var(axis=1) / total_var
            total = 1.0 - y.mean(axis=axis).reshape(len(players), -1).var(axis=1) / total_var
            rows.append(
                pd.DataFrame(
                    {
                        "player_id": np.asarray(players),
                        "dimension": dim,
                        "first_order": first,
                        "total_order": total,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def _segment_scores(
    rows: pd.DataFrame,
    impact: pd.DataFrame,
    stabilisation: Stabilisation,
    pool_ids: NDArray[np.int64],
    seasons: list[str],
    rng: np.random.Generator,
    expert: FloatArray,
    counts: dict[str, int],
) -> tuple[FloatArray, FloatArray]:
    """Scores (*DIMENSIONS shape, pool players) and applied weights (*DIMENSIONS shape,
    metrics) for one segment."""
    out = np.full((*_SHAPE, pool_ids.size), np.nan)
    applied = np.zeros((*_SHAPE, len(METRICS)))
    group_of = rows.groupby("player_id")["position_group"].first()
    codes, labels = pd.factorize(group_of)
    group_code = pd.Series(codes, index=group_of.index)
    n_groups = int(group_code.max()) + 1
    pool_index, season_index = pd.Index(pool_ids), pd.Index(seasons)
    pool_group = group_code.loc[pool_index].to_numpy()
    fold_of = pd.Series(
        rng.permutation(len(group_of)) % min(CV_FOLDS, len(group_of)), index=group_of.index
    )
    k = _group_constants(stabilisation, list(labels))
    years = np.array([int(s[:4]) for s in seasons], dtype=np.float64)
    # specifications sharing one (competition, threshold, aggregation) triple
    inner = _n_specs(competition="", min_minutes="", aggregation="")
    for ic, competition in enumerate(DIMENSIONS["competition"]):
        combined = _combine_scopes(rows, COMPETITION_WEIGHTS[str(competition)])
        combined = combined.join(impact, on=["player_id", "season"])
        for it, threshold in enumerate(DIMENSIONS["min_minutes"]):
            q = combined[combined["raw_minutes"] >= int(threshold)]
            if q[[*IMPACT_METRICS, "rapm_total_pooled"]].isna().to_numpy().any():
                raise ValueError("impact_season lacks a qualifying player-season")
            pid = q["player_id"].to_numpy()
            g = group_code.loc[pid].to_numpy()
            x0 = q[list(METRICS)].to_numpy(np.float64)
            y = q["rapm_total_pooled"].to_numpy(np.float64)
            folds = fold_of.loc[pid].to_numpy()
            raw_minutes = q["raw_minutes"].to_numpy(np.float64)
            elo = q["team_elo"].to_numpy(np.float64)
            in_pool = np.isin(pid, pool_ids)
            pi = pool_index.get_indexer(pd.Index(pid[in_pool]))
            si = season_index.get_indexer(pd.Index(q["season"].to_numpy()[in_pool]))
            minutes = np.zeros((pool_ids.size, len(seasons)))
            minutes[pi, si] = q["minutes"].to_numpy(np.float64)[in_pool]
            n_seasons = (minutes > 0).sum(axis=1)
            counts["best3_fewer_than_3_seasons"] += (
                int(((n_seasons > 0) & (n_seasons < 3)).sum()) * inner
            )
            counts["no_qualifying_season_ranked_last"] += (
                int((n_seasons == 0).sum()) * inner * _N_AGGREGATIONS
            )
            for ish, shrinkage in enumerate(DIMENSIONS["shrinkage"]):
                x1 = x0.copy()
                if shrinkage == "eb":
                    x1[:, _STAT_IDX] = shrink(x0[:, _STAT_IDX], raw_minutes, k[g], g)
                for ix, context in enumerate(DIMENSIONS["context"]):
                    x2 = x1 if context == "none" else _partial_elo(x1, elo, g)
                    sd = x2.std(axis=0)
                    for ip, position in enumerate(DIMENSIONS["position"]):
                        z = _normalise(x2, g, str(position))
                        w, by_sum = _weights(z, y, folds, sd, expert)
                        applied[:, :, ip, ix, :, ic, it, ish] = w.reshape(
                            len(METRICS), _N_FAMILIES, _N_WEIGHTS, 1
                        ).transpose(1, 2, 3, 0)
                        counts["pc1_sign_by_loading_sum"] += by_sum * _N_AGGREGATIONS
                        s = z @ w
                        pooled = position == "none"
                        table = _quantile_by_group(s, np.zeros_like(g) if pooled else g, n_groups)
                        baseline = table[np.zeros_like(pool_group) if pooled else pool_group]
                        padded = np.full((pool_ids.size, len(seasons), s.shape[1]), np.nan)
                        padded[pi, si] = s[in_pool]
                        agg = aggregate_seasons(padded, minutes, years, baseline)
                        stacked = np.stack([agg[str(a)] for a in DIMENSIONS["aggregation"]])
                        out[:, :, ip, ix, :, ic, it, ish, :] = stacked.reshape(
                            _N_AGGREGATIONS, pool_ids.size, _N_FAMILIES, _N_WEIGHTS
                        ).transpose(2, 3, 0, 1)
    return out, applied


def _group_constants(stabilisation: Stabilisation, labels: list[str]) -> FloatArray:
    """(groups × STAT_METRICS) constants: the metric's k, or its k for each group."""
    table = np.empty((len(labels), len(STAT_METRICS)))
    for j, metric in enumerate(STAT_METRICS):
        k = stabilisation[metric]
        table[:, j] = [k[g] for g in labels] if isinstance(k, Mapping) else float(k)
    return table


def _combine_scopes(rows: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """One row per (player_id, season): raw_minutes, weighted minutes, weighted rates."""
    w = rows["scope"].map(weights).fillna(0.0).to_numpy(np.float64)
    kept = rows[w > 0]
    wm = w[w > 0] * kept["minutes"].to_numpy(np.float64)
    rated = [*STAT_METRICS, "team_elo"]
    parts = pd.DataFrame(
        kept[rated].to_numpy(np.float64) * wm[:, None], columns=rated, index=kept.index
    )
    parts["player_id"] = kept["player_id"]
    parts["season"] = kept["season"]
    parts["raw_minutes"] = kept["minutes"]
    parts["minutes"] = wm
    sums = parts.groupby(["player_id", "season"], as_index=False).sum()
    sums[rated] = sums[rated].div(sums["minutes"], axis=0)
    return sums


def _partial_elo(x: FloatArray, elo: FloatArray, g: IntArray) -> FloatArray:
    """Remove each stat metric's within-group OLS slope on team Elo, keeping group means."""
    out = x.copy()
    for code in np.unique(g):
        rows = np.flatnonzero(g == code)
        e = elo[rows] - elo[rows].mean()
        ss = float(e @ e)
        if ss > 0:
            block = x[np.ix_(rows, _STAT_IDX)]
            out[np.ix_(rows, _STAT_IDX)] = block - np.outer(e, e @ block / ss)
    return out


def _normalise(x: FloatArray, g: IntArray, position: str) -> FloatArray:
    if position == "none":
        centre, scale = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True)
    else:
        centre, scale = np.empty_like(x), np.empty_like(x)
        for code in np.unique(g):
            rows = g == code
            if position == "group_z":
                centre[rows], scale[rows] = x[rows].mean(axis=0), x[rows].std(axis=0)
            else:
                centre[rows] = np.quantile(x[rows], REPLACEMENT_QUANTILE, axis=0)
                scale[rows] = x.std(axis=0)
    return (x - centre) / np.where(scale > 0, scale, 1.0)


def _quantile_by_group(x: FloatArray, g: IntArray, n_groups: int) -> FloatArray:
    table = np.full((n_groups, x.shape[1]), np.nan)
    for code in np.unique(g):
        table[code] = np.quantile(x[g == code], REPLACEMENT_QUANTILE, axis=0)
    return table


def _weights(
    z: FloatArray, y: FloatArray, folds: IntArray, sd: FloatArray, expert: FloatArray
) -> tuple[FloatArray, int]:
    """(metrics × families·weights) weight matrix, and how many pc1 signs fell back."""
    out = np.zeros((len(METRICS), _N_FAMILIES, _N_WEIGHTS))
    by_sum = 0
    for f, family in enumerate(FAMILIES.values()):
        cols = [METRICS.index(m) for m in family]
        stat = [c for c in cols if METRICS[c] not in IMPACT_METRICS]
        impact = [c for c in cols if METRICS[c] in IMPACT_METRICS]
        out[cols, f, 0] = 1.0 / len(cols)
        pc, fell_back = _pc1(z[:, cols], family.index("goals") if "goals" in family else None)
        out[cols, f, 1] = _unit(pc)
        by_sum += fell_back
        ridge = np.zeros(len(METRICS))
        if stat:
            ridge[stat] = _unit(_ridge_cv(z[:, stat], y, folds))
        if impact:
            ridge[impact] = _unit(sd[impact])
        out[:, f, 2] = _unit(ridge)
        out[cols, f, 3] = _unit(expert[cols])
    return out.reshape(len(METRICS), -1), by_sum


def _pc1(z: FloatArray, anchor: int | None) -> tuple[FloatArray, bool]:
    """First principal component; sign by the anchor's loading, else by the loading sum."""
    zc = z - z.mean(axis=0)
    w = np.linalg.eigh(zc.T @ zc)[1][:, -1]
    fell_back = anchor is None or bool(abs(w[anchor]) < 1e-12)
    ref = w.sum() if fell_back or anchor is None else w[anchor]
    return (w if ref >= 0 else -w), fell_back


def _ridge_cv(x: FloatArray, y: FloatArray, folds: IntArray) -> FloatArray:
    grid = RIDGE_GRID * len(y)
    error = np.zeros(grid.size)
    for fold in np.unique(folds):
        test = folds == fold
        if test.all():
            continue
        beta = _ridge(x[~test], y[~test], grid)
        pred = (x[test] - x[~test].mean(axis=0)) @ beta.T + y[~test].mean()
        error += ((pred - y[test][:, None]) ** 2).sum(axis=0)
    return _ridge(x, y, grid[[int(np.argmin(error))]]).reshape(-1)


def _ridge(x: FloatArray, y: FloatArray, lambdas: FloatArray) -> FloatArray:
    """Ridge coefficients on centred data, one row per lambda."""
    xc = x - x.mean(axis=0)
    values, vectors = np.linalg.eigh(xc.T @ xc)
    proj = vectors.T @ (xc.T @ (y - y.mean()))
    return (vectors @ (proj[:, None] / (values[:, None] + lambdas[None, :]))).T


def _unit(w: FloatArray) -> FloatArray:
    total = float(np.abs(w).sum())
    return w / total if total > 0 else w


def _n_specs(**fixed: str) -> int:
    return int(np.prod([1 if d in fixed else len(v) for d, v in DIMENSIONS.items()]))
