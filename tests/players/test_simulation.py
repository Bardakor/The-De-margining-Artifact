import numpy as np
import pandas as pd
import pytest

from footy.players.construct_gap import Splits, position_contrast, split_stats, unmeasured_share
from footy.players.rapm import attribution_factor, build_design, fit_rapm, split_halves
from footy.players.simulation import (
    STATS,
    SimParams,
    Simulated,
    parametric_bootstrap_contrast,
    rank_recovery,
    simulate_on_schedule,
)
from footy.players.synthetic import League, simulate_league

LAM = 1e4  # what cv_lambda picks on leagues like these (tests/players/test_rapm.py)
MIN_MINUTES = 500.0
N_REPEATS = 2
POOLS = {"CB": "CB+DM", "DM": "CB+DM", "CF": "CF+W", "W": "CF+W"}


@pytest.fixture(scope="module")
def league() -> League:
    """16 clubs x 20 players x 3 seasons: ~4.9k stints, 320 players."""
    return simulate_league(np.random.default_rng(0), n_teams=16, squad_size=20, n_seasons=3)


@pytest.fixture(scope="module")
def positions(league: League) -> pd.Series:
    ids = np.sort(league.stint_players["player_id"].unique())
    return pd.Series(np.random.default_rng(1).choice(list(POOLS), size=len(ids)), index=ids)


def test_schedule_is_preserved_exactly(league: League, positions: pd.Series) -> None:
    sim = simulate_on_schedule(
        league.stints, league.stint_players, positions, np.random.default_rng(2), SimParams()
    )
    goals = ["home_goals", "away_goals"]
    pd.testing.assert_frame_equal(sim.stints.drop(columns=goals), league.stints.drop(columns=goals))
    pd.testing.assert_frame_equal(sim.stint_players, league.stint_players)
    # Minutes per player-match equal those the league generator recorded.
    got = sim.player_match.set_index(["game_id", "player_id"]).sort_index()
    ref = league.player_match.set_index(["game_id", "player_id"]).sort_index()
    assert got.index.equals(ref.index)
    np.testing.assert_array_equal(got["minutes"].to_numpy(), ref["minutes"].to_numpy())
    assert set(sim.truth["player_id"]) == set(league.stint_players["player_id"])


def test_given_effects_are_planted_as_they_are(league: League, positions: pd.Series) -> None:
    ids = np.sort(league.stint_players["player_id"].unique())
    effects = pd.DataFrame(
        {"player_id": ids, "offence": 0.01 * (ids % 7), "defence": -0.02 * (ids % 3)}
    )
    sim = simulate_on_schedule(
        league.stints, league.stint_players, positions, np.random.default_rng(2), SimParams(),
        effects=effects.sample(frac=1.0, random_state=0),
    )  # fmt: skip
    truth = sim.truth.set_index("player_id").loc[ids]
    np.testing.assert_array_equal(truth["offence"].to_numpy(), effects["offence"].to_numpy())
    np.testing.assert_array_equal(truth["defence"].to_numpy(), effects["defence"].to_numpy())
    goals = ["home_goals", "away_goals"]
    pd.testing.assert_frame_equal(sim.stints.drop(columns=goals), league.stints.drop(columns=goals))
    pd.testing.assert_frame_equal(sim.stint_players, league.stint_players)
    got = sim.player_match.set_index(["game_id", "player_id"]).sort_index()
    ref = league.player_match.set_index(["game_id", "player_id"]).sort_index()
    np.testing.assert_array_equal(got["minutes"].to_numpy(), ref["minutes"].to_numpy())
    with pytest.raises(ValueError, match="every player"):
        simulate_on_schedule(
            league.stints, league.stint_players, positions, np.random.default_rng(2), SimParams(),
            effects=effects.iloc[1:],
        )  # fmt: skip


def test_planted_measurable_share_holds_in_a_large_population() -> None:
    lg = simulate_league(np.random.default_rng(3), n_teams=40, squad_size=25, n_seasons=1)
    ids = np.sort(lg.stint_players["player_id"].unique())
    pos = pd.Series(np.where(ids % 2 == 0, "A", "B"), index=ids)
    defence_sd, share = {"A": 0.05, "B": 0.1}, {"A": 0.3, "B": 0.8}
    params = SimParams(offence_sd=0.1, defence_sd=defence_sd, measurable_share=share)
    truth = simulate_on_schedule(
        lg.stints, lg.stint_players, pos, np.random.default_rng(4), params
    ).truth
    for g, t in truth.groupby("position_group"):
        # 500 players a group: over 40 seeds the squared correlation with seen
        # was 0.299 +- 0.037 (m = 0.3) and 0.795 +- 0.016 (m = 0.8).
        m = share[str(g)]
        assert abs(np.corrcoef(t["total"], t["seen"])[0, 1] ** 2 - m) < 0.12
        assert abs(np.corrcoef(t["total"], t["shot_rate"])[0, 1] ** 2 - m) < 0.12  # rare floor
        assert t["total"].std() == pytest.approx(np.hypot(0.1, defence_sd[str(g)]), rel=0.1)


def test_goal_rates_follow_base_rate_and_home_term(league: League, positions: pd.Series) -> None:
    st = league.stints.merge(league.games[["game_id", "season"]], on="game_id")
    base = {"2020/21": 1.0, "2021/22": 1.5, "2022/23": 2.0}
    params = SimParams(offence_sd=0.0, defence_sd=0.0, base_rate=base, home_advantage=0.4)
    sim = simulate_on_schedule(
        st, league.stint_players, positions, np.random.default_rng(5), params, group="season"
    )
    per = sim.stints.groupby("season")[["home_goals", "away_goals", "minutes"]].sum()
    home, away = (per[c] * 90 / per["minutes"] for c in ("home_goals", "away_goals"))
    # 240 games a season: SE ~0.06 of the mean rate, ~0.065 of the mean home edge.
    np.testing.assert_allclose(((home + away) / 2).to_numpy(), list(base.values()), atol=0.15)
    assert abs((home - away).mean() - 0.4) < 0.15


def _m4_inputs(sim: Simulated, rng: np.random.Generator) -> tuple[pd.Series, pd.DataFrame, Splits]:
    """M3 and M4's inputs on one league, as a caller builds them on real data."""
    d = build_design(sim.stints, sim.stint_players, min_minutes=MIN_MINUTES)
    halves = split_halves(d, LAM, rng, n_repeats=N_REPEATS)
    pm = sim.player_match
    g = pm.groupby("player_id")[["minutes", *STATS]].sum()
    stats = g[list(STATS)].div(g["minutes"], axis=0) * 90
    splits = Splits(halves.a["total"], halves.b["total"], split_stats(pm, halves, STATS))
    return fit_rapm(d, LAM).players["total"], stats, splits


def test_parametric_interval_covers_the_planted_contrast(
    league: League, positions: pd.Series
) -> None:
    """CB+DM carry defence, which no statistic sees (share 1); CF+W carry
    offence with measurable share 0.6 (share 0.4): planted contrast 0.6.

    Each of 20 test leagues on the same schedule is analysed as real data: M4's
    contrast, measurable shares from M4's per-pool shares (clipped), effect
    SDs taken as known, then the parametric bootstrap (30 simulations).
    Observed here: coverage raw 0.95, clipped 0.90; the raw interval excludes 0
    in 0.70 of leagues. Over nine batches of 20 leagues (three seeds; one or
    two splits; CB+DM share 1 or 0.8) raw covered 0.85-1.0 and clipped only
    0.5-0.9: clipping at share 1 makes the clipped error non-pivotal.
    """
    sd, c = 0.25, 0.6

    def params(m_high: float, m_low: float) -> SimParams:
        return SimParams(
            offence_sd={"CB": 0.0, "DM": 0.0, "CF": sd, "W": sd},
            defence_sd={"CB": sd, "DM": sd, "CF": 0.0, "W": 0.0},
            measurable_share={"CB": m_high, "DM": m_high, "CF": m_low, "W": m_low},
        )

    d = build_design(league.stints, league.stint_players, min_minutes=MIN_MINUTES)
    att = attribution_factor(
        d, LAM, np.random.default_rng(3), min_minutes=MIN_MINUTES, n_sims=3,
        groups=positions.map(POOLS), halves=True,
    )  # fmt: skip
    attribution = {g: float(att["total"].to_dict()[g]) for g in ("CB+DM", "CF+W")}
    rng = np.random.default_rng(10)
    covered, excludes_zero = [], []
    for _ in range(20):
        sim = simulate_on_schedule(
            league.stints, league.stint_players, positions, rng, params(0.0, c)
        )
        rapm, stats, splits = _m4_inputs(sim, rng)
        real = position_contrast(rapm, stats, splits, attribution, positions, rng, n_boot=1)
        shares = unmeasured_share(
            rapm, stats, splits, attribution, positions.map(POOLS), rng, n_boot=1
        )["share_clipped"].to_dict()
        out = parametric_bootstrap_contrast(
            league.stints, league.stint_players, positions, real, attribution,
            params(1 - shares["CB+DM"], 1 - shares["CF+W"]), rng, lam=LAM,
            min_minutes=MIN_MINUTES, n_sims=30, n_repeats=N_REPEATS, n_boot=1,
        ).summary  # fmt: skip
        assert out["planted"].to_numpy() == pytest.approx(shares["CB+DM"] - shares["CF+W"])
        lo, hi = out["ci_low"].to_numpy(), out["ci_high"].to_numpy()
        covered.append((lo <= c) & (c <= hi))
        excludes_zero.append(lo > 0)
    coverage = dict(zip(out.index, np.mean(covered, axis=0), strict=True))
    assert coverage["raw"] >= 0.8
    assert np.mean(excludes_zero, axis=0)[0] >= 0.5  # raw: not covering by being uninformative


@pytest.fixture(scope="module")
def pool(league: League) -> list[int]:
    minutes = league.player_match.groupby("player_id")["minutes"].sum()
    return [int(p) for p in minutes.nlargest(8).index]


def test_recovery_rises_with_the_planted_gap(
    league: League, positions: pd.Series, pool: list[int]
) -> None:
    params = SimParams(offence_sd=0.25, defence_sd=0.25, measurable_share=0.8)
    out = rank_recovery(
        league.stints, league.stint_players, positions, pool, params, np.random.default_rng(6),
        lam=LAM, min_minutes=MIN_MINUTES, n_sims=80, n_bins=2,
    )  # fmt: skip
    low, high = out.by_gap.iloc[0], out.by_gap.iloc[-1]
    assert high["gap_low"] >= low["gap_high"]
    # Observed: RAPM 0.375 -> 0.65, box 0.55 -> 0.825 (40 simulations a bin, SE ~0.077).
    for name in ("rapm", "box"):
        assert high[f"p_top_{name}"] > low[f"p_top_{name}"] + 0.15


def test_recovery_is_chance_when_planted_effects_are_equal(
    league: League, positions: pd.Series, pool: list[int]
) -> None:
    params = SimParams(offence_sd=0.0, defence_sd=0.0)
    out = rank_recovery(
        league.stints, league.stint_players, positions, pool, params, np.random.default_rng(7),
        lam=LAM, min_minutes=MIN_MINUTES, n_sims=100,
    )  # fmt: skip
    n = len(pool)
    assert (out.sims["gap"] == 0).all()
    for name in ("rapm", "box"):
        # 100 simulations: SE of 1/8 is 0.033; a random permutation's mean
        # absolute rank error is (n^2 - 1) / (3n) = 2.625.
        assert abs(out.sims[f"top_{name}"].mean() - 1 / n) < 0.1
        assert abs(out.sims[f"mae_{name}"].mean() - (n * n - 1) / (3 * n)) < 0.3
    with pytest.raises(ValueError, match="min_minutes"):
        rank_recovery(
            league.stints, league.stint_players, positions, pool, params,
            np.random.default_rng(7), lam=LAM, min_minutes=1e9, n_sims=1,
        )  # fmt: skip
