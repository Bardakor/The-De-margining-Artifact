"""Synthesis (design §5, M8) on planted mixtures with a known answer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.stats import rankdata

from footy.players.multiverse import STAT_METRICS, run_multiverse, spec_grid, variance_indices
from footy.players.synthesis import (
    Synthesis,
    distinct_ranking_weights,
    kendalls_w,
    median_specification,
    multiverse_runs,
    rank_confidence_sets,
    stats_weight_share,
    synthesise,
    verdicts,
    view_agreement,
)

FloatArray = NDArray[np.float64]
COMPONENTS = ["var_specification", "var_sampling", "var_unmeasured"]
SHARES = ["share_specification", "share_sampling", "share_unmeasured"]
EQUAL_Z_OMEGA = {"box": 1.0, "box_gd": 3 / 4, "impact": 0.0, "hybrid": 3 / 6}


def _omega(family: Sequence[str], values: Sequence[float] | None = None) -> pd.Series:
    """omega per (spec_id, segment), the same in both segments; equal_z's by default."""
    om = [EQUAL_Z_OMEGA[f] for f in family] if values is None else list(values)
    index = pd.MultiIndex.from_product(
        [range(len(om)), ["outfield", "goalkeeper"]], names=["spec_id", "segment"]
    )
    return pd.Series(np.repeat(om, 2), index=index)


def _weights(values: Sequence[float]) -> pd.Series:
    """Specification weights per (spec_id, segment), the same in both segments."""
    index = pd.MultiIndex.from_product(
        [range(len(values)), ["outfield", "goalkeeper"]], names=["spec_id", "segment"]
    )
    return pd.Series(np.repeat(np.asarray(values, dtype=np.float64), 2), index=index)


def _frames(
    scores: FloatArray, family: Sequence[str], segment: Sequence[str]
) -> list[pd.DataFrame]:
    """M5-like ranks frames, one per row of ``scores`` (frames x specs x players)."""
    _, n_specs, n = scores.shape
    base = pd.DataFrame(
        {
            "spec_id": np.repeat(np.arange(n_specs), n),
            "family": np.repeat(family, n),
            "player_id": np.tile(np.arange(1, n + 1), n_specs),
            "segment": np.tile(segment, n_specs),
        }
    )
    out = []
    for x in scores:
        frame = base.assign(score=x.ravel())
        frame["rank"] = (
            frame.groupby(["spec_id", "segment"])["score"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype(int)
        )
        out.append(frame)
    return out


def _run(
    scores: FloatArray,
    family: Sequence[str],
    groups: Sequence[str],
    share: dict[str, float],
    *,
    omega: Sequence[float] | None = None,
    n_unmeasured: int = 6,
    seed: int = 0,
    onpitch_ratio: dict[str, float] | None = None,
    spec_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Synthesis of ``scores`` (replicates x specs x players); frame 0 is the point."""
    segment = ["goalkeeper" if g == "GK" else "outfield" for g in groups]
    frames = _frames(scores, family, segment)
    group_of = pd.Series(list(groups), index=np.arange(1, len(groups) + 1))
    result = synthesise(
        frames[0],
        frames[1:],
        share,
        group_of,
        np.random.default_rng(seed),
        omega=_omega(family, omega),
        n_unmeasured=n_unmeasured,
        onpitch_ratio=onpitch_ratio,
        spec_weights=spec_weights,
    )
    return result.players.set_index("player_id")


def test_components_sum_to_the_total_rank_variance() -> None:
    rng = np.random.default_rng(1)
    family = ["box", "box_gd", "impact", "hybrid", "hybrid", "box", "impact", "box_gd"]
    omega = [1.0, 0.6, 0.0, 0.7, 0.4, 1.0, 0.0, 0.75]
    groups = ["CF"] * 5 + ["CB"] * 4 + ["W"] * 3 + ["GK"] * 4
    quality = rng.standard_normal(len(groups))
    scores = quality + 0.5 * rng.standard_normal((1, len(family), len(groups)))
    scores = scores + 0.5 * rng.standard_normal((7, 1, len(groups)))  # 6 replicates
    scores[:, 2:4, 3] = np.nan  # no qualifying season in two specifications
    share = {"CF": 0.3, "CB": 0.7, "W": 0.4, "GK": 0.5}

    players = _run(scores, family, groups, share, omega=omega, n_unmeasured=200)

    np.testing.assert_allclose(players[COMPONENTS].sum(axis=1), players["var_total"], atol=1e-9)
    np.testing.assert_allclose(players[SHARES].sum(axis=1), 1.0, atol=1e-9)
    assert (players[COMPONENTS] > 0).all().all()
    assert set(players.loc[players["segment"] == "goalkeeper"].index) == {13, 14, 15, 16}
    assert players.loc[players["segment"] == "goalkeeper", "rank_hi"].max() <= 4


def test_planted_sources_take_the_whole_rank_variance() -> None:
    rng = np.random.default_rng(2)
    n, n_specs, n_rep = 10, 6, 41
    groups = ["CF"] * n
    by_spec = np.array([rng.permutation(n) for _ in range(n_specs)], dtype=np.float64)
    by_rep = np.array([rng.permutation(n) for _ in range(n_rep)], dtype=np.float64)

    only_spec = _run(np.broadcast_to(by_spec, (n_rep, n_specs, n)), ["impact"] * n_specs, groups,
                     {"CF": 0.5})  # fmt: skip
    only_sampling = _run(np.broadcast_to(by_rep[:, None, :], (n_rep, n_specs, n)),
                         ["impact"] * n_specs, groups, {"CF": 0.5})  # fmt: skip
    only_u = _run(np.broadcast_to(np.arange(n, dtype=np.float64), (n_rep, n_specs, n)),
                  ["box"] * n_specs, groups, {"CF": 0.5})  # fmt: skip

    np.testing.assert_allclose(only_spec[SHARES], [[1.0, 0.0, 0.0]] * n, atol=1e-12)
    np.testing.assert_allclose(only_sampling[SHARES], [[0.0, 1.0, 0.0]] * n, atol=1e-12)
    np.testing.assert_allclose(only_u[SHARES], [[0.0, 0.0, 1.0]] * n, atol=1e-12)


def test_decomposition_matches_a_brute_force_nested_simulation() -> None:
    """Nested Monte Carlo with 4,000 draws per (s, b) and the scale computed by hand."""
    rng = np.random.default_rng(9)
    family, omega = ["box", "hybrid", "impact"], [1.0, 0.7, 0.0]  # 0.7: hybrid v1_like
    n, n_rep, sigma, n_u = 5, 4, 0.5, 4000
    scores = 0.4 * np.arange(n)[::-1] + rng.standard_normal((n_rep + 1, len(family), n))
    players = _run(scores, family, ["CF"] * n, {"CF": sigma}, omega=omega, n_unmeasured=n_u)

    x = scores[1:]
    mad = np.median(np.abs(x - np.median(x, axis=2, keepdims=True)), axis=2, keepdims=True)
    eff = np.array(omega)[None, :, None] * sigma
    tau = np.sqrt(eff / (1 - eff)) * 1.4826 * mad
    y = x[:, :, None, :] + tau[:, :, None, :] * rng.standard_normal((n_rep, len(family), n_u, n))
    r = rankdata(-y, axis=-1, method="min")
    e_sb = r.mean(axis=2)
    expected = {
        "var_total": r.reshape(-1, n).var(axis=0),
        "var_specification": e_sb.mean(axis=0).var(axis=0),
        "var_sampling": e_sb.var(axis=0).mean(axis=0),
        "var_unmeasured": r.var(axis=2).mean(axis=(0, 1)),
    }
    for column, value in expected.items():
        np.testing.assert_allclose(players[column], value, rtol=0.03, atol=0.005)


def test_onpitch_ratio_matches_a_brute_force_nested_simulation() -> None:
    """With omega = 0 the leakage term alone sets tau^2 = lambda Var_g(x), whatever sigma.

    E_u[r^2] is a plain Monte Carlo mean on both sides (relative SE about 4% at 4,000
    draws), so this test uses 100,000.
    """
    rng = np.random.default_rng(11)
    family, omega = ["impact", "impact", "hybrid"], [0.0, 0.0, 0.0]
    n, n_rep, lam, n_u = 5, 4, 0.6, 100_000
    scores = 0.4 * np.arange(n)[::-1] + rng.standard_normal((n_rep + 1, len(family), n))
    groups, share = ["CF"] * n, {"CF": 0.9}
    without = _run(scores, family, groups, share, omega=omega, n_unmeasured=50)
    players = _run(scores, family, groups, share, omega=omega, n_unmeasured=n_u,
                   onpitch_ratio={"CF": lam})  # fmt: skip

    np.testing.assert_allclose(without["var_unmeasured"], 0.0, atol=1e-12)
    x = scores[1:]
    mad = np.median(np.abs(x - np.median(x, axis=2, keepdims=True)), axis=2, keepdims=True)
    tau = np.sqrt(lam) * 1.4826 * mad
    y = x[:, :, None, :] + tau[:, :, None, :] * rng.standard_normal((n_rep, len(family), n_u, n))
    r = rankdata(-y, axis=-1, method="min")
    e_sb = r.mean(axis=2)
    expected = {
        "var_total": r.reshape(-1, n).var(axis=0),
        "var_specification": e_sb.mean(axis=0).var(axis=0),
        "var_sampling": e_sb.var(axis=0).mean(axis=0),
        "var_unmeasured": r.var(axis=2).mean(axis=(0, 1)),
    }
    for column, value in expected.items():
        np.testing.assert_allclose(players[column], value, rtol=0.03, atol=0.005)


def test_weight_k_equals_k_equally_weighted_copies_of_a_specification() -> None:
    """A25 (iii) read backwards: k copies of a specification are that specification with
    weight k.

    Equal up to rounding wherever u is integrated analytically: the pairwise
    probabilities, the views, the specification and sampling variances. The rank
    histogram and E_u[r^2] are Monte Carlo and every copy draws its own u, so the
    probabilities, the unmeasured and the total variance agree up to Monte Carlo error
    only. With 20,000 draws in each of 6 replicates a specification's P(rank <= k) has a
    standard error of at most 0.0015: atol = 0.006. The mean of r^2 over the same draws
    differs between the two runs with a standard error near 0.02 for seven players (the
    largest difference over the 20 values was 0.05 on seven other seeds): atol = 0.12,
    on variances between 0.2 and 4.
    """
    rng = np.random.default_rng(21)
    family, omega, copies = (
        ["box", "impact", "hybrid", "box_gd"],
        [1.0, 0.0, 0.5, 0.75],
        [1, 3, 2, 5],
    )
    groups = ["CF"] * 4 + ["CB"] * 3 + ["GK"] * 3
    segment = ["goalkeeper" if g == "GK" else "outfield" for g in groups]
    group_of = pd.Series(groups, index=np.arange(1, len(groups) + 1))
    share, lam = {"CF": 0.4, "CB": 0.7, "GK": 0.5}, {"CF": 0.5, "CB": 0.8, "GK": 1.0}
    scores = rng.standard_normal(len(groups)) + rng.standard_normal((7, 4, len(groups)))
    which = np.repeat(np.arange(4), copies)

    def run(x: FloatArray, fam: list[str], om: list[float], w: pd.Series | None) -> Synthesis:
        frames = _frames(x, fam, segment)
        return synthesise(
            frames[0], frames[1:], share, group_of, np.random.default_rng(3),
            omega=_omega(fam, om), n_unmeasured=20_000, onpitch_ratio=lam, spec_weights=w,
        )  # fmt: skip

    weighted = run(scores, family, omega, _weights(copies))
    copied = run(scores[:, which], [family[i] for i in which], [omega[i] for i in which], None)

    for seg in ("outfield", "goalkeeper"):
        np.testing.assert_allclose(weighted.pairwise[seg], copied.pairwise[seg], rtol=1e-12)
    a, b = weighted.players, copied.players
    exact = ["var_specification", "var_sampling", "stats_view", "impact_view"]
    np.testing.assert_allclose(a[exact], b[exact], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a[["p_first", "p_top3"]], b[["p_first", "p_top3"]], atol=0.006)
    monte_carlo = ["var_unmeasured", "var_total"]
    np.testing.assert_allclose(a[monte_carlo], b[monte_carlo], rtol=0, atol=0.12)
    assert (np.abs(a["median_rank"] - b["median_rank"]) <= 1).all()


def test_distinct_rankings_share_one_unit_of_weight() -> None:
    """Outfield: specifications 0, 1 and 3 rank alike, 2 differs. Goalkeepers: all alike."""
    outfield = {0: [1, 2, 3], 1: [1, 2, 3], 2: [2, 1, 3], 3: [1, 2, 3]}
    rows = [
        {"spec_id": s, "player_id": p, "segment": seg, "rank": r}
        for s, ranking in outfield.items()
        for p, seg, r in [
            *zip((1, 2, 3), ["outfield"] * 3, ranking, strict=True),
            (8, "goalkeeper", 1),
            (9, "goalkeeper", 2),
        ]
    ]

    weights = distinct_ranking_weights(pd.DataFrame(rows))

    np.testing.assert_allclose(weights.xs("outfield", level="segment").sort_index(),
                               [1 / 3, 1 / 3, 1.0, 1 / 3])  # fmt: skip
    np.testing.assert_allclose(weights.xs("goalkeeper", level="segment"), 0.25)
    assert weights.groupby(level="segment").sum().round(12).to_dict() == {
        "goalkeeper": 1.0,
        "outfield": 2.0,
    }


def test_components_sum_to_the_total_with_unequal_weights() -> None:
    rng = np.random.default_rng(1)
    family = ["box", "box_gd", "impact", "hybrid", "hybrid", "box", "impact", "box_gd"]
    omega = [1.0, 0.6, 0.0, 0.7, 0.4, 1.0, 0.0, 0.75]
    groups = ["CF"] * 5 + ["CB"] * 4 + ["W"] * 3 + ["GK"] * 4
    scores = rng.standard_normal(len(groups)) + rng.standard_normal((7, len(family), len(groups)))
    scores[:, 2:4, 3] = np.nan
    share = {"CF": 0.3, "CB": 0.7, "W": 0.4, "GK": 0.5}

    weights = _weights([5.0, 1.0, 0.25, 2.0, 1.0, 0.5, 3.0, 1.0])
    players = _run(scores, family, groups, share, omega=omega, spec_weights=weights)
    uniform = _run(scores, family, groups, share, omega=omega, spec_weights=_weights([2.0] * 8))
    default = _run(scores, family, groups, share, omega=omega)

    np.testing.assert_allclose(players[COMPONENTS].sum(axis=1), players["var_total"], atol=1e-9)
    np.testing.assert_allclose(players[SHARES].sum(axis=1), 1.0, atol=1e-9)
    assert not np.allclose(players["var_specification"], default["var_specification"])
    numeric = default.columns.drop("segment")
    np.testing.assert_allclose(uniform[numeric], default[numeric], rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError, match="weights"):
        _run(scores, family, groups, share, omega=omega, spec_weights=_weights([-1.0] + [1.0] * 7))


def test_dominant_player_is_first_in_every_draw_and_decides_every_pair() -> None:
    rng = np.random.default_rng(3)
    family = ["box", "box_gd", "impact", "hybrid", "hybrid", "box"]
    groups = ["CF"] * 6 + ["CB"] * 6 + ["DM"] * 3 + ["GK"] * 3
    scores = rng.standard_normal((11, len(family), len(groups)))
    scores[:, :, 0] = 100.0
    share = {"CF": 0.5, "CB": 0.9, "DM": 0.99, "GK": 0.5}
    segment = ["goalkeeper" if g == "GK" else "outfield" for g in groups]
    frames = _frames(scores, family, segment)
    group_of = pd.Series(groups, index=np.arange(1, len(groups) + 1))

    result = synthesise(
        frames[0], frames[1:], share, group_of, np.random.default_rng(0), omega=_omega(family)
    )
    players = result.players.set_index("player_id")
    pairwise = result.pairwise["outfield"]

    assert players.loc[1, "p_first"] == 1.0
    assert list(players.loc[1, ["median_rank", "rank_lo", "rank_hi"]]) == [1, 1, 1]
    assert (pairwise.loc[1].drop(1) >= 0.9).all()
    assert (pairwise.drop(index=1)[1] <= 0.1).all()
    np.testing.assert_allclose(pairwise + pairwise.T, 1.0 + np.diag([np.nan] * 15))
    assert result.pairwise["goalkeeper"].shape == (3, 3)
    assert players.loc[players["segment"] == "outfield", "p_first"].sum() == pytest.approx(1.0)
    assert (
        result.rank_sets.set_index("player_id").loc[1, ["marginal_lo", "marginal_hi"]] == 1
    ).all()

    indices = pd.DataFrame(
        {"player_id": 1, "dimension": ["family", "aggregation"], "total_order": [0.2, 0.6]}
    )
    table = verdicts(result, indices).set_index("hypothesis")
    assert table.loc["H2", "statistic"] == 1.0
    assert not table.loc["H2", "supported"]
    assert table.loc["H5", "supported"]
    assert 0.0 < table["statistic"]["H1"] < 1.0


def test_rank_sets_cover_under_near_ties_where_the_percentile_interval_does_not() -> None:
    """Ten players 0.05 SE apart, parametric bootstrap with B = 200, 300 simulations."""
    rng = np.random.default_rng(4)
    theta = 0.05 * np.arange(10)[::-1]
    true = np.arange(1, 11)
    marginal = naive = simultaneous = 0.0
    n_sims = 300
    for _ in range(n_sims):
        point = theta + rng.standard_normal(10)
        boot = point + rng.standard_normal((200, 10))
        sets = rank_confidence_sets(point, boot)
        marginal += ((sets["marginal_lo"] <= true) & (true <= sets["marginal_hi"])).mean()
        naive += ((sets["naive_lo"] <= true) & (true <= sets["naive_hi"])).mean()
        simultaneous += bool(
            ((sets["simultaneous_lo"] <= true) & (true <= sets["simultaneous_hi"])).all()
        )
    marginal, naive, simultaneous = marginal / n_sims, naive / n_sims, simultaneous / n_sims
    print(f"coverage: marginal {marginal:.3f}, simultaneous {simultaneous:.3f}, naive {naive:.3f}")

    assert marginal >= 0.9
    assert simultaneous >= 0.9
    assert naive < 0.85


def test_rank_sets_leave_out_players_without_a_score() -> None:
    point = np.array([10.0, 5.0, 0.0, np.nan])
    boot = point + np.zeros((20, 4))
    boot[:, :3] += 0.01 * np.random.default_rng(5).standard_normal((20, 3))

    sets = rank_confidence_sets(point, boot)

    assert sets["point_rank"].tolist() == [1, 2, 3, 4]
    assert sets["marginal_lo"].tolist()[:3] == [1, 2, 3]
    assert sets["marginal_hi"].tolist()[:3] == [2, 3, 4]  # the unscored player may be anywhere
    assert sets["marginal_lo"].isna().tolist() == [False, False, False, True]


def test_kendalls_w_is_one_for_identical_views_and_one_over_m_for_random_ones() -> None:
    rng = np.random.default_rng(6)
    ranks = pd.Series([1, 2, 2, 4, 5, 5, 5, 8], dtype=float)
    assert kendalls_w(pd.concat([ranks] * 3, axis=1)) == pytest.approx(1.0)

    m, n = 3, 30
    w = np.array(
        [kendalls_w(pd.DataFrame(rng.random((n, m)).argsort(axis=0))) for _ in range(2000)]
    )
    assert w.mean() == pytest.approx(1 / m, abs=0.01)
    assert ((m * w - 1) / (m - 1)).mean() == pytest.approx(0.0, abs=0.01)


def test_median_specification_is_the_one_nearest_the_median_ranking() -> None:
    table = np.array([[3, 1, 2, 4], [1, 2, 3, 4], [1, 3, 2, 4], [4, 3, 2, 1], [1, 2, 4, 3]])
    ranks = pd.DataFrame(
        {
            "spec_id": np.repeat(np.arange(5), 4),
            "player_id": np.tile([7, 8, 9, 10], 5),
            "rank": table.ravel(),
        }
    )
    assert median_specification(ranks) == 1


def _population(rng: np.random.Generator, n_players: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = ("2021/22", "2022/23", "2023/24")
    quality = rng.standard_normal(n_players)
    rows = pd.DataFrame(
        {
            "player_id": np.repeat(np.arange(1, n_players + 1), len(seasons)),
            "season": np.tile(seasons, n_players),
            "scope": "league",
            "position_group": np.repeat(
                np.where(
                    np.arange(n_players) < 4, "GK", np.where(np.arange(n_players) % 2, "CF", "CB")
                ),
                len(seasons),
            ),
            "minutes": rng.integers(1000, 3300, n_players * len(seasons)).astype(float),
            "team_elo": 1600 + 80 * rng.standard_normal(n_players * len(seasons)),
        }
    )
    form = np.repeat(quality, len(seasons)) + 0.5 * rng.standard_normal(len(rows))
    for j, metric in enumerate(STAT_METRICS):
        rows[metric] = (0.1 + 0.05 * j) * np.exp(0.3 * form + 0.2 * rng.standard_normal(len(rows)))
    impact = rows[["player_id", "season"]].assign(
        rapm_offence=0.05 * form,
        rapm_defence=0.03 * form,
        rapm_total_pooled=0.08 * np.repeat(quality, 3),
    )
    return rows, impact


def test_omega_is_the_box_share_of_m5s_weights_on_any_data() -> None:
    """Box 1, impact 0, equal_z hybrid 3/6 and box_gd 3/4 whatever the data; v1_like
    from A13's weights; the hybrid ridge's box share is at most its stats half."""
    k = dict.fromkeys(STAT_METRICS, 600.0)
    for seed in (10, 11):
        ps, impact = _population(np.random.default_rng(seed), 30)
        run = run_multiverse(ps, impact, k, list(range(1, 21)), np.random.default_rng(seed))
        omega = stats_weight_share(run.weights).reset_index()
        omega = omega.merge(spec_grid()[["spec_id", "family", "weights"]], on="spec_id")
        assert set(omega["segment"]) == {"outfield", "goalkeeper"}
        cell = omega.groupby(["family", "weights"], observed=True)["omega"]
        lo, hi = cell.min(), cell.max()
        for family, value in EQUAL_Z_OMEGA.items():
            assert hi[family, "equal_z"] == pytest.approx(lo[family, "equal_z"]) == value
        np.testing.assert_allclose(omega.loc[omega["family"] == "box", "omega"], 1.0)
        np.testing.assert_allclose(omega.loc[omega["family"] == "impact", "omega"], 0.0)
        assert lo["hybrid", "v1_like"] == pytest.approx(hi["hybrid", "v1_like"]) == 0.7
        assert lo["box_gd", "v1_like"] == pytest.approx(hi["box_gd", "v1_like"]) == 0.7 / 0.9
        assert lo["hybrid", "rapm_ridge"] > 0.0 and hi["hybrid", "rapm_ridge"] <= 0.5 + 1e-12
        assert lo["box_gd", "pc1"] > 0.0 and hi["box_gd", "pc1"] < 1.0


def test_end_to_end_with_the_multiverse() -> None:
    rng = np.random.default_rng(7)
    ps, impact = _population(rng, 40)
    replicates = []
    for _ in range(3):
        noisy = ps.copy()
        noisy[list(STAT_METRICS)] *= np.exp(0.1 * rng.standard_normal((len(ps), len(STAT_METRICS))))
        replicates.append((noisy, impact))
    pool = list(range(1, 21))
    groups = ps.groupby("player_id")["position_group"].first()

    runs = multiverse_runs([(ps, impact), *replicates], dict.fromkeys(STAT_METRICS, 600.0),
                           pool, np.random.default_rng(8))  # fmt: skip
    first = next(runs)
    point, omega = first.ranks, stats_weight_share(first.weights)
    result = synthesise(point, (r.ranks for r in runs), {"CF": 0.4, "CB": 0.7, "GK": 0.5},
                        groups, rng, omega=omega, n_unmeasured=4)  # fmt: skip
    players = result.players.set_index("player_id")

    assert result.n_replicates == 3
    assert result.fixed_spec == median_specification(point[point["segment"] == "outfield"])
    np.testing.assert_allclose(players[COMPONENTS].sum(axis=1), players["var_total"], atol=1e-9)
    assert players["stats_view"].notna().all() and players["impact_view"].notna().all()
    table = verdicts(result, variance_indices(point))
    assert table["hypothesis"].tolist() == ["H1", "H2", "H4", "H5"]
    recognition = pd.Series(np.arange(20, 0, -1.0), index=pool)
    assert 0.0 <= view_agreement(result.players, recognition) <= 1.0
