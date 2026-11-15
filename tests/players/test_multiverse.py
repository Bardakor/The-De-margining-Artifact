"""Multiverse engine (design §5, M5) on synthetic frames with a planted answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from footy.players.multiverse import (
    DIMENSIONS,
    METRICS,
    STAT_METRICS,
    aggregate_seasons,
    distinct_rankings,
    rank_probabilities,
    run_multiverse,
    shrink,
    spec_grid,
    specification_curve,
    variance_indices,
)

SEASONS = ("2020/21", "2021/22", "2022/23", "2023/24")
GROUPS = ("CF", "W", "AM", "CM", "DM", "FB", "CB")
K = dict.fromkeys(STAT_METRICS, 600.0)
SHAPE = tuple(len(levels) for levels in DIMENSIONS.values())
N_SPECS = int(np.prod(SHAPE))


def _population(rng: np.random.Generator, n_players: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three scopes per player-season; every rate and RAPM follows one latent quality.

    Players 2-4 are goalkeepers and never score.
    """
    n_ps = n_players * len(SEASONS)
    quality = rng.standard_normal(n_players)
    form = np.repeat(quality, len(SEASONS)) + 0.5 * rng.standard_normal(n_ps)
    group = rng.choice(GROUPS, n_players).astype(object)
    group[1:4] = "GK"
    seasons = pd.DataFrame(
        {
            "player_id": np.repeat(np.arange(1, n_players + 1), len(SEASONS)),
            "season": np.tile(SEASONS, n_players),
            "form": form,
        }
    )
    rows = seasons.loc[seasons.index.repeat(3)].reset_index(drop=True)
    rows["scope"] = np.tile(["league", "ucl", "intl"], n_ps)
    rows["position_group"] = group[rows["player_id"].to_numpy() - 1]
    rows["minutes"] = np.where(
        rows["scope"] == "league",
        rng.integers(300, 3300, len(rows)),
        rng.integers(0, 700, len(rows)),
    ).astype(float)
    rows["team_elo"] = 1600 + 50 * rows["form"] + 80 * rng.standard_normal(len(rows))
    for j, metric in enumerate(STAT_METRICS):
        noise = 0.2 * rng.standard_normal(len(rows))
        rows[metric] = (0.1 + 0.05 * j) * np.exp(0.3 * rows["form"] + noise)
    rows.loc[rows["position_group"] == "GK", "goals"] = 0.0
    pooled = 0.08 * quality + 0.01 * rng.standard_normal(n_players)
    impact = seasons[["player_id", "season"]].assign(
        rapm_offence=0.05 * form + 0.03 * rng.standard_normal(n_ps),
        rapm_defence=0.03 * form + 0.03 * rng.standard_normal(n_ps),
        rapm_total_pooled=np.repeat(pooled, len(SEASONS)),
    )
    return rows.drop(columns="form"), impact


def test_planted_dominant_player_is_first_in_every_specification() -> None:
    rng = np.random.default_rng(20261114)
    ps, impact = _population(rng, 120)
    star = ps["player_id"] == 1
    ps.loc[star, "position_group"] = "CF"
    ps.loc[star, "team_elo"] = 1600.0
    ps.loc[star, "minutes"] = ps.loc[star, "scope"].map(
        {"league": 3000.0, "ucl": 900.0, "intl": 600.0}
    )
    for metric in STAT_METRICS:
        ps.loc[star, metric] = 3 * ps[metric].max()
    for column in ("rapm_offence", "rapm_defence", "rapm_total_pooled"):
        impact.loc[impact["player_id"] == 1, column] = 3 * impact[column].max()
    pool = list(range(1, 41))

    result = run_multiverse(ps, impact, K, pool, np.random.default_rng(0))
    ranks = result.ranks

    assert len(ranks) == N_SPECS * len(pool)
    assert (ranks.loc[ranks["player_id"] == 1, "rank"] == 1).all()
    keepers = ranks[ranks["segment"] == "goalkeeper"]
    assert set(keepers["player_id"]) == {2, 3, 4}
    assert (keepers.groupby("spec_id")["rank"].min() == 1).all()
    assert keepers["rank"].max() <= 3
    probs = rank_probabilities(ranks).set_index("player_id")
    assert probs.loc[1, "p_first"] == 1.0
    assert result.fallbacks["impact_rapm_ridge_is_rapm_total"] == N_SPECS // 16
    assert result.fallbacks["impact_eb_not_applied"] == N_SPECS // 8
    assert result.fallbacks["pc1_sign_by_loading_sum"] > 0
    assert 1 < distinct_rankings(ranks)["outfield"] < N_SPECS
    curve = specification_curve(ranks)
    first = curve[curve["player_id"] == 5]
    assert first["curve_position"].tolist() == list(range(N_SPECS))
    assert first["rank"].is_monotonic_increasing


def test_only_the_minutes_threshold_can_move_ranks() -> None:
    """One season, one scope, one group, constant Elo, every metric affine in one quality.

    Player 1 is far ahead but plays 1,000 minutes, so only the 1,800 threshold
    removes him; every other choice is a monotone transform of the same order.
    """
    quality = np.array([30.0, 5.0, 4.0, 3.0, 2.0, *np.linspace(1.0, -1.0, 10)])
    ps = pd.DataFrame(
        {
            "player_id": np.arange(1, quality.size + 1),
            "season": "2023/24",
            "scope": "league",
            "position_group": "CF",
            "minutes": 2000.0,
            "team_elo": 1600.0,
        }
    )
    ps.loc[0, "minutes"] = 1000.0
    for j, metric in enumerate(STAT_METRICS):
        ps[metric] = 1.0 + 0.1 * (j + 1) * quality
    impact = ps[["player_id", "season"]].assign(
        rapm_offence=0.02 * quality, rapm_defence=0.01 * quality, rapm_total_pooled=0.03 * quality
    )

    result = run_multiverse(ps, impact, K, [1, 2, 3, 4, 5], np.random.default_rng(3))
    indices = variance_indices(result.ranks)

    ranks = result.ranks
    assert (ranks.loc[ranks["min_minutes"] == 900].groupby("player_id")["rank"].max()).tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    threshold = indices["dimension"] == "min_minutes"
    values = ["first_order", "total_order"]
    np.testing.assert_allclose(indices.loc[threshold, values], 1.0, atol=1e-9)
    np.testing.assert_allclose(indices.loc[~threshold, values], 0.0, atol=1e-9)


def test_exposed_weights_times_normalised_metrics_reproduce_the_scores() -> None:
    """One season, one scope, constant Elo: a peak score is the season score."""
    rng = np.random.default_rng(5)
    n = 30
    ps = pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            "season": "2023/24",
            "scope": "league",
            "position_group": rng.choice(["CF", "W", "CB"], n),
            "minutes": rng.uniform(1800.0, 3300.0, n),
            "team_elo": 1600.0,
        }
    )
    for metric in STAT_METRICS:
        ps[metric] = rng.gamma(2.0, 0.2, n)
    impact = ps[["player_id", "season"]].assign(
        rapm_offence=0.05 * rng.standard_normal(n),
        rapm_defence=0.03 * rng.standard_normal(n),
        rapm_total_pooled=0.1 * ps["goals"] + 0.02 * rng.standard_normal(n),
    )
    pool = list(range(1, 11))

    result = run_multiverse(ps, impact, K, pool, np.random.default_rng(6))

    weights = result.weights.pivot(index="spec_id", columns="metric", values="weight")
    np.testing.assert_allclose(weights.abs().sum(axis=1), 1.0, rtol=1e-12)
    scores = result.ranks.pivot(index="spec_id", columns="player_id", values="score")
    x = ps.merge(impact)[list(METRICS)].to_numpy(np.float64)
    group = pd.factorize(ps["position_group"])[0]
    grid = spec_grid()
    chosen = grid[
        (grid["aggregation"] == "peak")
        & (grid["competition"] == "equal")
        & (grid["min_minutes"] == 900)
        & (grid["context"] == "none")
        & (grid["position"] != "replacement")
    ]
    for spec in chosen.itertuples():
        x1 = x.copy()
        if spec.shrinkage == "eb":
            stat = slice(0, len(STAT_METRICS))
            k = np.array([K[m] for m in STAT_METRICS])
            x1[:, stat] = shrink(x[:, stat], ps["minutes"].to_numpy(), k, group)
        by = group if spec.position == "group_z" else np.zeros(n, dtype=np.intp)
        z = np.empty_like(x1)
        for code in np.unique(by):
            rows = by == code
            z[rows] = (x1[rows] - x1[rows].mean(axis=0)) / x1[rows].std(axis=0)
        expected = z[: len(pool)] @ weights.loc[spec.spec_id, list(METRICS)].to_numpy()
        np.testing.assert_allclose(scores.loc[spec.spec_id, pool], expected, rtol=1e-9)
    assert len(chosen) == 64


def _over(dim: str, levels: list[float]) -> NDArray[np.float64]:
    shape = [1] * len(SHAPE)
    shape[list(DIMENSIONS).index(dim)] = -1
    return np.reshape(np.asarray(levels, dtype=np.float64), shape)


def _indices_of(values: NDArray[np.float64]) -> pd.DataFrame:
    ranks = spec_grid().assign(player_id=7, rank=np.broadcast_to(values, SHAPE).ravel())
    return variance_indices(ranks).set_index("dimension")


def test_additive_dimensions_split_rank_variance_exactly() -> None:
    family = _over("family", [0.0, 1.0, 3.0, 4.0])
    aggregation = _over("aggregation", [0.0, 2.0, 1.0, 5.0, 3.0])
    indices = _indices_of(family + aggregation)

    first, total = indices["first_order"], indices["total_order"]
    assert first[["family", "aggregation"]].sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(first, total, atol=1e-12)
    np.testing.assert_allclose(first.drop(["family", "aggregation"]), 0.0, atol=1e-12)


def test_planted_interaction_lifts_total_above_first_order() -> None:
    family = _over("family", [0.0, 1.0, 3.0, 4.0])
    aggregation = _over("aggregation", [0.0, 2.0, 1.0, 5.0, 3.0])
    interaction = (family - family.mean()) * (aggregation - aggregation.mean())
    indices = _indices_of(family + aggregation + interaction)

    first, total = indices["first_order"], indices["total_order"]
    assert first[["family", "aggregation"]].sum() < 0.99
    assert (total[["family", "aggregation"]] > first[["family", "aggregation"]] + 0.01).all()
    np.testing.assert_allclose(total.drop(["family", "aggregation"]), 0.0, atol=1e-12)


def test_aggregation_rules_on_a_hand_computed_example() -> None:
    nan = np.nan
    scores = np.array([[1.0, 2.0, 3.0], [4.0, nan, 0.0], [nan, nan, -1.0], [nan, nan, nan]])[
        :, :, None
    ]
    minutes = np.array(
        [[900.0, 1800.0, 2700.0], [1000.0, 0.0, 2000.0], [0.0, 0.0, 1500.0], [0.0, 0.0, 0.0]]
    )
    years = np.array([2021.0, 2022.0, 2023.0])
    baseline = np.full((4, 1), 0.5)

    out = {k: v[:, 0] for k, v in aggregate_seasons(scores, minutes, years, baseline).items()}

    root_half = 0.5**0.5  # recency weight one season before the anchor (half-life two)
    expected = {
        "peak": [3.0, 4.0, -1.0, nan],
        "best3": [2.0, 2.0, -1.0, nan],
        "mean": [12600 / 5400, 4000 / 3000, -1.0, nan],
        "volume": [(0.5 * 900 + 1.5 * 1800 + 2.5 * 2700) / 90, 3.5 * 1000 / 90, 0.0, nan],
        "recency": [
            (0.5 * 900 * 1 + root_half * 1800 * 2 + 2700 * 3)
            / (0.5 * 900 + root_half * 1800 + 2700),
            (0.5 * 1000 * 4) / (0.5 * 1000 + 2000),
            -1.0,
            nan,
        ],
    }
    for rule, values in expected.items():
        np.testing.assert_allclose(out[rule], values, rtol=1e-12, err_msg=rule)


def test_empirical_bayes_moves_rate_toward_group_mean_by_exact_factor() -> None:
    rates = np.array([[0.2], [0.5], [0.8], [1.1], [3.0], [2.0]])
    minutes = np.array([300.0, 2000.0, 2500.0, 3000.0, 400.0, 1000.0])
    group = np.array([0, 0, 0, 0, 1, 1])
    k = np.array([600.0])

    out = shrink(rates, minutes, k, group)

    for row, members in ((0, slice(0, 4)), (4, slice(4, 6))):
        mean = (rates[members, 0] * minutes[members]).sum() / minutes[members].sum()
        factor = 1 - minutes[row] / (minutes[row] + k[0])
        moved = out[row, 0] - rates[row, 0]
        assert moved == pytest.approx(factor * (mean - rates[row, 0]), rel=1e-12)
