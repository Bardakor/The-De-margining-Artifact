from typing import Any

import numpy as np
import pandas as pd
import pytest

from footy.players.synthetic import (
    AGE_MAX,
    AGE_MIN,
    CUP,
    CUP_TEAMS,
    STARTERS,
    League,
    simulate_league,
)


def test_same_seed_same_league() -> None:
    a = simulate_league(np.random.default_rng(7))
    b = simulate_league(np.random.default_rng(7))
    pd.testing.assert_frame_equal(a.stints, b.stints)
    pd.testing.assert_frame_equal(a.player_match, b.player_match)


def test_stints_reconcile_with_games() -> None:
    lg = simulate_league(np.random.default_rng(1))
    per_game = lg.stints.groupby("game_id")[["home_goals", "away_goals", "minutes"]].sum()
    games = lg.games.set_index("game_id")
    assert (per_game["home_goals"] == games["home_goals"]).all()
    assert (per_game["away_goals"] == games["away_goals"]).all()
    assert (per_game["minutes"] == 90).all()
    counts = lg.stint_players.groupby(["stint_id", "side"]).size()
    assert (counts == STARTERS).all()


def test_player_minutes_and_on_pitch_goals_match_stints() -> None:
    lg = simulate_league(np.random.default_rng(2))
    sp = lg.stint_players.merge(lg.stints, on="stint_id")
    sp["gf"] = np.where(sp["side"] == "home", sp["home_goals"], sp["away_goals"])
    sp["ga"] = np.where(sp["side"] == "home", sp["away_goals"], sp["home_goals"])
    derived = sp.groupby(["game_id", "player_id"])[["minutes", "gf", "ga"]].sum()
    pm = lg.player_match.set_index(["game_id", "player_id"]).sort_index()
    derived = derived.loc[pm.index]
    assert (derived["minutes"] == pm["minutes"]).all()
    assert (derived["gf"] == pm["on_pitch_gf"]).all()
    assert (derived["ga"] == pm["on_pitch_ga"]).all()
    starters = lg.player_match.groupby(["game_id", "club_id"])["started"].sum()
    assert (starters == STARTERS).all()


def test_planted_measurable_share_holds_in_a_large_population() -> None:
    lg = simulate_league(
        np.random.default_rng(3), n_teams=40, squad_size=25, n_seasons=1, measurable_share=0.3
    )
    t = lg.truth
    r2 = np.corrcoef(t["offence"], t["seen"])[0, 1] ** 2
    assert abs(r2 - 0.3) < 0.08


def test_transfers_keep_squad_sizes_and_move_players() -> None:
    lg = simulate_league(np.random.default_rng(4), n_seasons=3, transfer_rate=0.3)
    clubs = lg.player_match.groupby(["season", "player_id"])["club_id"].first().unstack(0)
    moved = (clubs.nunique(axis=1) > 1).mean()
    assert moved > 0.2


def test_match_shocks_move_team_shots_and_goals_together() -> None:
    def shots_goals_corr(match_shock_sd: float) -> float:
        lg = simulate_league(np.random.default_rng(12), n_seasons=4, match_shock_sd=match_shock_sd)
        g = lg.games
        goals = pd.concat(
            [
                g[["game_id", "home_club_id", "home_goals"]].set_axis(
                    ["game_id", "club_id", "gf"], axis=1
                ),
                g[["game_id", "away_club_id", "away_goals"]].set_axis(
                    ["game_id", "club_id", "gf"], axis=1
                ),
            ]
        )
        shots = lg.player_match.groupby(["game_id", "club_id"], as_index=False)["shots"].sum()
        team = shots.merge(goals, on=["game_id", "club_id"])
        return float(np.corrcoef(team["shots"], team["gf"])[0, 1])

    # 720 team-matches: SE of a correlation ~0.04.
    assert abs(shots_goals_corr(0.0)) < 0.15
    assert shots_goals_corr(0.5) > 0.2


def test_one_league_default_draws_are_those_before_the_multi_league_extension() -> None:
    # Totals of the generator as it was before league_means existed, seed 7.
    lg = simulate_league(np.random.default_rng(7))
    assert set(lg.games["competition_id"]) == {"SYN"}
    assert len(lg.stints) == 1226
    assert lg.stints[["home_goals", "away_goals"]].sum().tolist() == [268, 234]
    assert lg.player_match[["shots", "goals"]].sum().tolist() == [5504, 643]
    assert lg.truth["offence"].sum() == pytest.approx(-1.4059210315288806, abs=1e-12)
    assert (lg.truth["share_SYN"] == 1).all() and (lg.truth["league_mean"] == 0).all()


def _multi(n_seasons: int = 4, **kw: Any) -> League:
    """Three leagues of 6 clubs x 16 players, planted means 0, -0.1, -0.2."""
    return simulate_league(
        np.random.default_rng(21),
        n_teams=6,
        n_seasons=n_seasons,
        transfer_rate=0.0,
        league_means=(0.0, -0.1, -0.2),
        **kw,
    )


def test_leagues_get_their_planted_means_their_own_games_and_a_cup() -> None:
    lg = _multi(cross_transfer_rate=0.1)
    games, truth = lg.games, lg.truth
    league_of = (games[["home_club_id", "away_club_id"]] - 1) // 6
    cup = games["competition_id"] == CUP
    # League games stay inside a league, cup games never do.
    assert (league_of.loc[~cup, "home_club_id"] == league_of.loc[~cup, "away_club_id"]).all()
    assert (league_of.loc[cup, "home_club_id"] != league_of.loc[cup, "away_club_id"]).all()
    assert games.loc[~cup].groupby("competition_id").size().to_dict() == {
        f"SYN{k}": 4 * 30 for k in (1, 2, 3)
    }
    # Twelve cup clubs, each meets the eight of the other two leagues once: 48 games.
    assert games[cup].groupby("season").size().tolist() == [48] * 4
    # From the second season on the entrants are last season's top four on points.
    home, away = games["home_goals"], games["away_goals"]
    points = pd.concat(
        [
            pd.DataFrame(
                {"season": games["season"], "club": games[f"{side}_club_id"], "points": pts}
            )[~cup]
            for side, pts in (
                ("home", 3 * (home > away) + (home == away)),
                ("away", 3 * (away > home) + (home == away)),
            )
        ]
    )
    table = points.groupby(["season", "club"])["points"].sum()
    seasons = sorted(games["season"].unique())
    for last, now in zip(seasons[:-1], seasons[1:], strict=True):
        entrants = set(games.loc[cup & (games["season"] == now), "home_club_id"]) | set(
            games.loc[cup & (games["season"] == now), "away_club_id"]
        )
        for k in range(3):
            clubs = [c for c in entrants if (c - 1) // 6 == k]
            others = [c for c in range(6 * k + 1, 6 * k + 7) if c not in clubs]
            assert len(clubs) == CUP_TEAMS
            assert table[last][clubs].min() >= table[last][others].max()
    # Planted means: in both effects of the players who start in the league.
    assert truth["league_mean"].tolist() == list(np.repeat([0.0, -0.1, -0.2], 6 * 16))
    by_league = truth.groupby("league_mean")[["offence", "defence"]].mean()
    assert np.abs(by_league.to_numpy() - by_league.index.to_numpy()[:, None]).max() < 0.02
    # Shares: each player's league minutes split over the leagues he played in.
    shares = truth.set_index("player_id").filter(like="share_")
    assert list(shares.columns) == ["share_SYN1", "share_SYN2", "share_SYN3"]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    pm = lg.player_match.merge(games[["game_id", "competition_id"]], on="game_id")
    minutes = pm[pm["competition_id"] != CUP].groupby(["player_id", "competition_id"])["minutes"]
    expected = minutes.sum().unstack(fill_value=0)
    np.testing.assert_allclose(shares.to_numpy(), expected.div(expected.sum(axis=1), axis=0))
    # 14 swaps between two seasons (10% of 288 players, fewer if one is drawn twice);
    # squads stay at 16.
    club = lg.player_match.groupby(["season", "player_id"])["club_id"].first().unstack(0)
    assert (club.apply(pd.Series.value_counts) == 16).all().all()
    moved = ((club[seasons[1]] - 1) // 6 != (club[seasons[0]] - 1) // 6).sum()
    assert 22 <= moved <= 28


def test_weakly_linked_league_has_no_cup_game_and_its_swaps_only() -> None:
    lg = _multi(weak_link_swaps=1)
    cup = lg.games[lg.games["competition_id"] == CUP]
    assert (cup[["home_club_id", "away_club_id"]].to_numpy() <= 12).all()
    assert len(cup) == 4 * 16
    # One swap in each of the three breaks: three players leave and three arrive.
    shares = lg.truth.set_index("player_id").filter(like="share_")
    assert ((shares["share_SYN3"] > 0) & (shares["share_SYN3"] < 1)).sum() == 6
    assert ((shares["share_SYN1"] > 0) & (shares["share_SYN2"] > 0)).sum() == 0


def test_age_variant_sends_older_players_down_and_younger_players_up() -> None:
    lg = _multi(cross_transfer_rate=0.2, age_slope=0.01, n_seasons=2)
    truth = lg.truth.set_index("player_id")
    assert truth["age"].between(AGE_MIN, AGE_MAX).all()
    first = lg.player_match.groupby(["season", "player_id"])["club_id"].first().unstack(0)
    before, after = ((first[s] - 1) // 6 for s in sorted(first.columns))
    down, up = truth.loc[after > before, "age"], truth.loc[after < before, "age"]
    # 28 swaps, each between the older half of one league and the younger half of a
    # weaker one (ages 18 to 33, so the halves' means are about 8 years apart).
    assert min(len(down), len(up)) >= 22
    assert down.mean() > up.mean() + 5
    assert "age" not in _multi().truth.columns
