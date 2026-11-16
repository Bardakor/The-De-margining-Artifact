import json
from pathlib import Path

import pandas as pd
import pytest

from footy.players import panel, stints

FIXTURE = Path(__file__).parent / "fixtures" / "panel_sb_events.json"


def _match() -> panel.SbMatch:
    return panel.parse_sb_match(1, json.loads(FIXTURE.read_text()))


TEAMS = pd.DataFrame(
    {"sb_match_id": [1, 1], "team": ["Home FC", "Away FC"], "game_id": [500, 500],
     "club_id": [10, 20]}
)  # fmt: skip
GAMES = pd.DataFrame(
    {
        "game_id": [500],
        "competition_id": ["FIWC"],
        "competition_type": ["national_team_competition"],
        "season": ["2022/23"],
        "date": pd.to_datetime(["2022-12-01"]),
        "home_club_id": [10],
        "away_club_id": [20],
        "home_goals": [1],
        "away_goals": [2],
        "knockout": [True],
    }
)
# StatsBomb 23 has no Transfermarkt id: he keeps his place under the pseudo id -23.
MAPPING = pd.Series({11: 1011, 12: 1012, 13: 1013, 14: 1014, 21: 1021, 22: 1022})


def test_fold_minute() -> None:
    assert panel.fold_minute(0, 1) == 1
    assert panel.fold_minute(46, 1) == 45  # first-half stoppage
    assert panel.fold_minute(45, 2) == 46
    assert panel.fold_minute(93, 2) == 90
    assert panel.fold_minute(100, 3) == 101
    assert panel.fold_minute(125, 4) == 120


def test_parse_sb_match() -> None:
    m = _match()
    assert m.end_min == 120  # a period-3 event; the period-5 shoot-out kick is ignored
    assert len(m.shots) == 3
    kinds = m.timeline.sort_values("minute")[["minute", "kind"]].values.tolist()
    assert kinds == [[11, "goal"], [60, "sub"], [71, "goal"], [81, "red"], [90, "goal"]]
    counts = m.counts.set_index("sb_player_id")
    assert counts.loc[22, "yellow"] == 2 and counts.loc[22, "red"] == 0
    assert counts.loc[23, "penalty_goals"] == 1 and counts.loc[12, "assists"] == 1


def test_sb_tables_stints_and_appearances() -> None:
    sb_stints, sb_players, apps = panel.sb_tables([_match()], TEAMS, GAMES, MAPPING)
    assert sb_stints[["start_min", "end_min"]].values.tolist() == [[0, 60], [60, 81], [81, 120]]
    assert sb_stints[["home_goals", "away_goals"]].values.tolist() == [[1, 0], [0, 1], [0, 1]]
    assert sb_stints["away_n"].tolist() == [3, 3, 2]
    assert stints.reconcile(sb_stints, GAMES)["match"].all()
    a = apps.set_index("player_id")
    assert a["minutes"].to_dict() == {
        1011: 120,
        1012: 120,
        1013: 60,
        1014: 60,
        1021: 120,
        1022: 81,
        -23: 120,
    }
    assert a.loc[1013, "goals"] == 1 and a.loc[1012, "goals"] == 0  # own goal not counted
    assert a.loc[-23, "penalty_goals"] == 1
    assert a.loc[1022, "yellow"] == 2 and a.loc[1022, "red"] == 0
    assert not a.loc[1014, "started"] and a.loc[1014, "position"] == "Centre-Forward"
    assert a.loc[1012, "position"] == "Centre-Back"
    assert (apps["source"] == "statsbomb").all()
    assert set(sb_players["player_id"]) == set(a.index)


def test_sb_player_columns() -> None:
    m = _match()
    sb_stints, sb_players, _ = panel.sb_tables([m], TEAMS, GAMES, MAPPING)
    cols = panel.sb_player_columns([m], TEAMS, MAPPING, sb_stints, sb_players, GAMES)
    c = cols.set_index("player_id")
    assert c.loc[1013, "sb_xg"] == pytest.approx(0.3)
    assert c.loc[-23, "sb_np_xg"] == 0.0 and c.loc[-23, "sb_xg"] == pytest.approx(0.78)
    assert c.loc[1012, "sb_xa"] == pytest.approx(0.3)
    assert c.loc[1011, "on_pitch_sb_xg_for"] == pytest.approx(0.4)
    assert c.loc[1011, "on_pitch_sb_xg_against"] == pytest.approx(0.78)
    assert c.loc[1022, "on_pitch_sb_xg_for"] == 0.0  # the penalty came after his red card
    assert c.loc[1022, "on_pitch_sb_xg_against"] == pytest.approx(0.3)


def test_scope_and_tournament_seasons() -> None:
    assert panel.scope("CL", "international_cup") == "ucl"
    assert panel.scope("CLQ", "international_cup") == "other"
    assert panel.scope("EL", "other") == "other"
    assert panel.scope("SA1", "domestic_league") == "league"
    assert panel.scope("EURO", "national_team_competition") == "intl"
    games = pd.DataFrame(
        {
            "competition_id": ["EURO", "EURO", "GB1", "KLUB", "FIWC"],
            "competition_type": ["national_team_competition"] * 2
            + ["domestic_league", "international_cup", "national_team_competition"],
            "season": ["2021/22", "2021/22", "2019/20", "2025/26", "2022/23"],
            "date": ["2021-06-11", "2021-07-11", "2020-07-26", "2025-06-15", "2022-11-20"],
        }
    )
    assert panel.study_seasons(games).tolist() == [
        "2020/21", "2020/21", "2019/20", "2024/25", "2022/23"
    ]  # fmt: skip


def test_assign_to_stints_boundary_rule() -> None:
    st = pd.DataFrame(
        {"stint_id": [1, 2], "game_id": [9, 9], "start_min": [0, 60], "end_min": [60, 90]}
    )
    events = pd.DataFrame(
        {"game_id": 9, "minute": [0, 60, 61], "side": ["home", "away", "home"],
         "value": [1.0, 2.0, 4.0]}
    )  # fmt: skip
    out = panel.assign_to_stints(st, events).set_index("stint_id")
    assert out.loc[1, "home_value"] == 1.0 and out.loc[1, "away_value"] == 2.0
    assert out.loc[2, "home_value"] == 4.0 and out.loc[2, "away_value"] == 0.0


def _toy_panel() -> pd.DataFrame:
    """Two games of one club: a league game with stints and a cup game without."""
    games = pd.DataFrame(
        {
            "game_id": [1, 2],
            "competition_id": ["GB1", "FAC"],
            "competition_type": ["domestic_league", "domestic_cup"],
            "season": ["2021/22", "2021/22"],
            "date": pd.to_datetime(["2021-09-01", "2021-09-08"]),
            "home_club_id": [10, 30],
            "away_club_id": [20, 10],
            "knockout": [False, True],
        }
    )
    apps = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 2, 2],
            "player_id": [7, 8, 9, 7, 8],
            "club_id": [10, 10, 20, 10, 10],
            "date": pd.to_datetime(["2021-09-01"] * 3 + ["2021-09-08"] * 2),
            "competition_id": ["GB1"] * 3 + ["FAC"] * 2,
            "minutes": [90, 0, 90, 90, 45],
            "goals": [2, 0, 1, 1, 0],
            "assists": [0, 0, 0, 0, 1],
            "yellow": [0, 0, 1, 0, 0],
            "red": [0, 0, 0, 0, 0],
            "started": [True, False, True, True, False],
            "position": ["Centre-Forward", None, "Goalkeeper", "Left Winger", "Right-Back"],
            "captain": [True, False, False, True, False],
            "penalty_goals": [1, 0, 0, 0, 0],
            "source": "stints",
        }
    )
    st = pd.DataFrame(
        {"stint_id": [100, 101], "game_id": [1, 1], "start_min": [0, 50],
         "end_min": [50, 90], "minutes": [50, 40], "home_goals": [1, 1], "away_goals": [1, 0],
         "home_n": [1, 1], "away_n": [1, 1]}
    )  # fmt: skip
    sp = pd.DataFrame(
        {"stint_id": [100, 100, 101, 101], "player_id": [7, 9, 7, 9],
         "side": ["home", "away", "home", "away"]}
    )  # fmt: skip
    elo = pd.DataFrame(
        {"game_id": [1, 2], "home_elo_pre": [1600.0, 1400.0], "away_elo_pre": [1500.0, 1650.0]}
    )
    profiles = pd.DataFrame({"player_id": [7, 8, 9], "sub_position": ["Centre-Forward"] * 3})
    groups = panel.position_groups(apps, profiles)
    sb_cols = pd.DataFrame(columns=["game_id", "player_id", *panel.SB_COLUMNS])
    return panel.player_match(apps, games, st, sp, elo, groups, sb_cols)


def test_toy_panel_build() -> None:
    pm = _toy_panel()
    assert list(pm.columns) == panel.PANEL_COLUMNS
    assert len(pm) == 4  # the 0-minute row is dropped
    row = pm.set_index(["game_id", "player_id"])
    assert row.loc[(1, 7), "on_pitch_gf"] == 2 and row.loc[(1, 7), "on_pitch_ga"] == 1
    assert row.loc[(1, 9), "on_pitch_gf"] == 1 and row.loc[(1, 9), "on_pitch_ga"] == 2
    assert pd.isna(row.loc[(2, 7), "on_pitch_gf"])  # no stints for the cup game
    assert row.loc[(1, 7), "np_goals"] == 1
    assert row.loc[(1, 7), "team_elo"] == 1600.0 and row.loc[(1, 9), "team_elo"] == 1500.0
    assert row.loc[(2, 7), "team_elo"] == 1650.0 and row.loc[(2, 7), "opp_elo"] == 1400.0
    assert row.loc[(1, 7), "scope"] == "league" and row.loc[(2, 7), "scope"] == "other"
    assert bool(row.loc[(2, 7), "knockout"])
    # a 90-90 tie between CF and W goes to W, listed first in GROUP_ORDER:
    assert row.loc[(1, 7), "position_group"] == "W"
    assert row.loc[(1, 7), "match_position_group"] == "CF"
    assert row.loc[(2, 8), "position_group"] == "FB"
    assert not pm["sb_linked"].any()


def test_player_season_rates_use_their_own_minutes() -> None:
    ps = panel.player_season(_toy_panel())
    s = ps.set_index(["player_id", "season", "scope"])
    all7 = s.xs((7, "2021/22", "all"))
    assert all7["minutes"] == 180 and all7["games"] == 2 and all7["goals"] == 3
    assert all7["goals_p90"] == pytest.approx(1.5)
    assert all7["minutes_stints"] == 90  # only game 1 has stints
    assert all7["on_pitch_gf_p90"] == pytest.approx(2.0)
    assert s.loc[(7, "2021/22", "league"), "goals"] == 2
    assert pd.isna(all7["sb_xg"]) and pd.isna(all7["sb_xg_p90"])
    assert ps["sb_xg"].dtype == float


def test_position_groups_profile_fallback() -> None:
    apps = pd.DataFrame(
        {"player_id": [1, 1, 1], "position": ["Centre-Back", "Right-Back", "Right-Back"],
         "minutes": [90, 30, 30]}
    )  # fmt: skip
    profiles = pd.DataFrame({"player_id": [1, 2], "sub_position": ["Centre-Back", "Goalkeeper"]})
    out = panel.position_groups(apps, profiles).set_index("player_id")
    assert out.loc[1, "position_group"] == "CB" and out.loc[1, "group_source"] == "lineups"
    assert out.loc[2, "position_group"] == "GK" and out.loc[2, "group_source"] == "profile"


def test_pool_names_and_matching() -> None:
    v1 = pd.DataFrame({"player": ["Vitinha", "Vinícius Júnior", "Nobody"]})
    bdo = pd.DataFrame(
        {
            "edition": [2019, 2024, 2025, 2025, 2022],
            "player": ["Old Star", "Vitinha", "Vitinha", "Vinicius Junior", "Rodri"],
            "club": ["X", "Paris Saint-Germain", "Paris Saint-Germain", "Real Madrid", "City"],
        }
    )
    pool = panel.pool_names(v1, bdo)
    assert "Old Star" not in set(pool["player"])  # 2019 only
    row = pool.set_index("player")
    assert row.loc["Vitinha", "bdo_editions"] == "2024;2025" and row.loc["Vitinha", "in_v1"]
    assert row.loc["Vinícius Júnior", "bdo_editions"] == "2025"
    players = pd.DataFrame(
        {"player_id": [1, 2, 3, 4, 5],
         "name": ["Vitinha", "Vitinha", "Vinicius Junior", "Rodri", "Rodri"]}
    )  # fmt: skip
    club_minutes = pd.DataFrame(
        {"player_id": [1, 2, 3, 4, 5], "minutes": [9000, 100, 9000, 50, 60],
         "club_name": ["Olympique Marseille", "Paris Saint-Germain", "Real Madrid", "A", "B"]}
    )  # fmt: skip
    groups = pd.DataFrame({"player_id": [1, 2, 3, 4], "position_group": ["CF", "DM", "W", "DM"]})
    out, cands = panel.match_pool(
        pool, players, club_minutes, groups, {"Rodri": (4, "checked by hand")}
    )
    got = out.set_index("player")
    assert got.loc["Vitinha", "tm_player_id"] == 2  # the PSG one, not the striker
    assert got.loc["Vitinha", "match_method"] == "name+club"
    assert got.loc["Vitinha", "position_group"] == "DM"
    assert got.loc["Vinícius Júnior", "tm_player_id"] == 3
    assert got.loc["Rodri", "match_method"] == "manual" and got.loc["Rodri", "tm_player_id"] == 4
    assert got.loc["Nobody", "match_method"] == "unmapped"
    assert not out["tm_player_id"].dropna().duplicated().any()
    assert len(cands[cands["player"] == "Vitinha"]) == 2
