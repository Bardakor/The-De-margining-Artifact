import numpy as np
import pandas as pd

from footy.players.synthetic import STARTERS, simulate_league


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
