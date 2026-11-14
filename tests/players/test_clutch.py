"""M7(b) on synthetic leagues with knockout games labelled here.

A random 30% of games are knockouts. Season 2020/21 is dated in period A and
2021/22 in period B. With a trait, 30% of players get extra knockout shots at
``boost`` per 90 in both periods; without one, knockout shots follow the same
Poisson rate as all others.
"""

from functools import cache

import numpy as np
import pandas as pd

from footy.players.clutch import knockout_persistence, placebo_knockout
from footy.players.synthetic import simulate_league

DATES = {"2020/21": "2021-03-01", "2021/22": "2024-03-01"}


@cache
def _league(seed: int, boost: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pm = simulate_league(rng, n_teams=20, n_seasons=2).player_match.copy()
    games = pm["game_id"].unique()
    knockouts = rng.choice(games, size=int(0.3 * len(games)), replace=False)
    pm["knockout"] = pm["game_id"].isin(knockouts)
    pm["date"] = pd.to_datetime(pm["season"].map(DATES))
    players = pm["player_id"].unique()
    clutch = pm["player_id"].isin(rng.choice(players, size=int(0.3 * len(players)), replace=False))
    rows = pm["knockout"] & clutch
    pm.loc[rows, "shots"] += rng.poisson(boost * pm.loc[rows, "minutes"].to_numpy() / 90)
    return pm


def test_planted_trait_gives_clearly_positive_persistence() -> None:
    res = knockout_persistence(_league(1, 1.0), "shots", np.random.default_rng(0))
    assert res.n_players > 250
    assert res.r > 0.25
    assert res.ci_low > 0.1
    assert res.ci_low < res.r < res.ci_high
    assert res.p_value < 0.01


def test_no_trait_gives_persistence_near_zero() -> None:
    res = knockout_persistence(_league(2, 0.0), "shots", np.random.default_rng(0))
    assert res.n_players > 250
    assert abs(res.r) < 0.15
    assert res.ci_low < 0 < res.ci_high
    assert res.p_value > 0.05
    # Without a trait z is pure Poisson noise, so the standard error is right if sd(z) is ~1.
    assert abs(res.players["z_a"].std() - 1) < 0.15


def test_placebo_split_is_null_even_when_a_trait_exists() -> None:
    pm = _league(1, 1.0)
    placebo = placebo_knockout(pm, np.random.default_rng(3))
    assert not placebo["game_id"].isin(pm.loc[pm["knockout"], "game_id"]).any()
    share = placebo.groupby("game_id")["knockout"].first().mean()
    assert abs(share - pm.groupby("game_id")["knockout"].first().mean()) < 0.01
    res = knockout_persistence(placebo, "shots", np.random.default_rng(0))
    assert abs(res.r) < 0.15
    assert res.p_value > 0.05


def test_minimum_knockout_minutes_filters_players() -> None:
    pm = _league(2, 0.0)
    res = knockout_persistence(
        pm, "shots", np.random.default_rng(0), min_knockout_minutes=900, n_boot=10, n_perm=10
    )
    assert (res.players[["knockout_minutes_a", "knockout_minutes_b"]] >= 900).all().all()
    assert res.n_players == len(res.players)
