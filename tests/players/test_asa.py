import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from footy.players import asa


def test_study_season_is_the_panel_label_of_an_mls_year() -> None:
    # Transfermarkt files MLS 2025 under its season 2024, which the panel labels 2024/25.
    assert asa.study_season(2025) == "2024/25"
    assert asa.study_season(2023) == "2022/23"
    assert asa.study_season(2000) == "1999/00"


def test_match_needs_the_name_and_the_birth_date() -> None:
    pool = pd.DataFrame(
        {
            "tm_player_id": [1, 2, 3],
            "player": ["Ana Pérez", "Kim Min-su", "Zed"],
            "tm_name": ["Ana Perez", "Min-su Kim", "Zed"],
            "date_of_birth": ["1990-01-01", "1991-02-02", "1992-03-03"],
        }
    )
    players = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d", "e"],
            "player_name": ["Ana Pérez", "Kim Min-Su", "Ana Perez", "Someone Else", "Nobody"],
            "birth_date": ["1990-01-01", "1991-02-02", "1985-05-05", "1992-03-03", None],
        }
    )
    pairs = asa.match_players(pool, players).set_index(["tm_player_id", "asa_player_id"])
    # a namesake born another day and a stranger born the same day are listed, not matched
    assert pairs[["same_name", "same_birth_date"]].to_dict("index") == {
        (1, "a"): {"same_name": True, "same_birth_date": True},
        (1, "c"): {"same_name": True, "same_birth_date": False},
        (2, "b"): {"same_name": True, "same_birth_date": True},
        (3, "d"): {"same_name": False, "same_birth_date": True},
    }


def _row(player: str, team: str, season: str, minutes: int, goals: int, assists: int) -> asa.Json:
    return {
        "player_id": player,
        "team_id": team,
        "season_name": season,
        "minutes_played": minutes,
        "goals": goals,
        "primary_assists": assists,
    }


MATCHED = pd.DataFrame({"tm_player_id": [1], "player": ["Ana Pérez"], "asa_player_id": ["a"]})
TEAMS = [{"team_id": "t1", "team_name": "First FC"}, {"team_id": "t2", "team_name": "Second SC"}]
ROWS = [
    _row("a", "t1", "2023", 900, 5, 2),
    _row("a", "t2", "2023", 450, 3, 1),
    _row("a", "t2", "2024", 2000, 10, 4),
]


def test_player_seasons_adds_the_clubs_of_a_season_and_splits_penalties() -> None:
    penalties = [
        _row("a", "t1", "2023", 900, 1, 0),
        _row("a", "t2", "2023", 450, 0, 0),
        _row("a", "t2", "2024", 2000, 3, 0),
    ]
    got = asa.player_seasons(MATCHED, ROWS, penalties, TEAMS)
    assert got.to_dict("records") == [
        {
            "tm_player_id": 1,
            "player": "Ana Pérez",
            "asa_player_id": "a",
            "season": "2022/23",
            "asa_season_name": 2023,
            "club": "First FC;Second SC",
            "minutes": 1350,
            "goals": 8,
            "np_goals": 7,
            "penalty_goals": 1,
            "assists": 3,
        },
        {
            "tm_player_id": 1,
            "player": "Ana Pérez",
            "asa_player_id": "a",
            "season": "2023/24",
            "asa_season_name": 2024,
            "club": "Second SC",
            "minutes": 2000,
            "goals": 10,
            "np_goals": 7,
            "penalty_goals": 3,
            "assists": 4,
        },
    ]


def test_player_seasons_refuses_a_row_without_its_penalty_count() -> None:
    with pytest.raises(ValueError, match="no penalty row"):
        asa.player_seasons(MATCHED, ROWS, [_row("a", "t1", "2023", 900, 1, 0)], TEAMS)


def test_player_games_puts_minutes_on_the_scale_of_a_90_minute_match() -> None:
    def game_row(game: str, minutes: int, goals: int, assists: int) -> asa.Json:
        row = _row("a", "t1", "", minutes, goals, assists) | {"game_id": game}
        del row["season_name"]  # rows split by game carry no season
        return row

    games = [
        {"game_id": "g1", "date_time_utc": "2024-03-02 23:30:00 UTC", "season_name": "2024",
         "expanded_minutes": 100},
        {"game_id": "g2", "date_time_utc": "2024-03-10 00:30:00 UTC", "season_name": "2024",
         "expanded_minutes": 96},
    ]  # fmt: skip
    rows = [game_row("g1", 100, 2, 1), game_row("g2", 32, 0, 0)]
    penalties = [game_row("g1", 100, 1, 0), game_row("g2", 32, 0, 0)]
    got = asa.player_games(MATCHED, rows, penalties, games)
    assert got["tm_player_id"].tolist() == [1, 1]
    assert got["asa_season_name"].tolist() == [2024, 2024]
    assert got["kickoff"].dt.strftime("%Y-%m-%d %H:%M %Z").tolist() == [
        "2024-03-02 23:30 UTC",
        "2024-03-10 00:30 UTC",
    ]
    assert got["minutes"].tolist() == [100, 32]
    assert got["minutes_scaled"].tolist() == [90.0, 30.0]
    assert got[["goals", "penalty_goals", "assists"]].to_numpy().tolist() == [[2, 1, 1], [0, 0, 0]]
    with pytest.raises(ValueError, match="no penalty row"):
        asa.player_games(MATCHED, rows, penalties[:1], games)


def test_ensure_reads_the_cache_and_refuses_another_query(tmp_path: Path) -> None:
    url = f"{asa.BASE_URL}/mls/teams?team_id=t1,t2"
    (tmp_path / "teams.json.gz").write_bytes(gzip.compress(json.dumps(TEAMS).encode()))
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"teams": {"url": url}}}))
    assert asa.ensure(tmp_path, "teams", "mls/teams", {"team_id": "t1,t2"}) == TEAMS
    with pytest.raises(ValueError, match="cached for another query"):
        asa.ensure(tmp_path, "teams", "mls/teams", {"team_id": "t1"})
