"""Transfermarkt normalisers and the end-to-end build on raw-format fixtures (no network)."""

import gzip
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from footy.players import transfermarkt as tm

FIXTURES = Path(__file__).parent / "fixtures"
RAW_NAMES = ("games", "game_events", "game_lineups", "appearances", "players", "player_valuations")


def test_position_groups() -> None:
    expected = {
        "Goalkeeper": "GK",
        "Centre-Back": "CB",
        "Sweeper": "CB",
        "Left-Back": "FB",
        "Right-Back": "FB",
        "Defensive Midfield": "DM",
        "Central Midfield": "CM",
        "Attacking Midfield": "AM",
        "Left Winger": "W",
        "Right Winger": "W",
        "Left Midfield": "W",
        "Right Midfield": "W",
        "Centre-Forward": "CF",
        "Second Striker": "CF",
        "midfield": "CM",
        "Attack": "CF",
        "Defender": "CB",
    }
    assert {p: tm.position_group(p) for p in expected} == expected
    with pytest.raises(KeyError):
        tm.position_group("Libero")


@pytest.mark.parametrize(
    ("round_name", "knockout"),
    [
        ("1. Matchday", False),
        ("Group A", False),
        ("group I", False),
        ("Group Stage", False),
        ("Final", True),
        ("Round of 16", True),
        ("Quarter-Finals 1st leg", True),
        ("intermediate stage", True),
        ("First Round Replay", True),
        ("Third Place Play-Off", True),
        ("Qualifying Round 2nd leg", True),
        ("|", False),
        ("", False),
    ],
)
def test_knockout_from_round(round_name: str, knockout: bool) -> None:
    assert tm.is_knockout(round_name) is knockout


def test_stint_events_encodings() -> None:
    events = pd.read_csv(FIXTURES / "tm_game_events.csv")
    norm = tm.stint_events(events[events["game_id"] == 1])
    assert norm["kind"].tolist() == ["goal", "sub", "goal", "red", "sub", "goal"]
    assert norm.loc[norm["kind"] == "red", "player_id"].tolist() == [205]  # second yellow
    shootout = tm.stint_events(events[events["game_id"] == 4])
    assert set(shootout["kind"]) == {"goal"} and len(shootout) == 2


def test_shootout_kicks_removed_unless_score_never_included_them() -> None:
    games = pd.read_csv(FIXTURES / "tm_games.csv").head(4)
    awarded_game = games.head(1).assign(
        game_id=99, home_club_id=10, away_club_id=30, home_club_goals=0, away_club_goals=3
    )
    games = pd.concat([games, awarded_game], ignore_index=True)
    events = pd.read_csv(FIXTURES / "tm_game_events.csv")
    awarded = pd.DataFrame(
        {
            "game_id": 99,
            "minute": -1,
            "type": "Shootout",
            "club_id": [10, 10, 30],
            "player_id": [1, 2, 3],
            "description": ", Scored",
            "player_in_id": None,
        }
    )
    out = tm.build_games(games, pd.concat([events, awarded])).set_index("game_id")
    score = out.loc[[4, 99, 1], ["home_goals", "away_goals"]].to_numpy().tolist()
    assert score == [[1, 1], [0, 3], [2, 1]]  # 1-1 after 4-3 kicks; awarded 0-3 kept; no kicks


def test_club_season_is_the_competitions_own() -> None:
    """The COVID-extended 2019/20 ran into July 2020: the date rule would say 2020/21."""
    games = pd.read_csv(FIXTURES / "tm_games.csv").head(1).assign(date="2020-07-26")
    events = pd.read_csv(FIXTURES / "tm_game_events.csv").head(0)
    assert tm.build_games(games, events)["season"].tolist() == ["2019/20"]


def _raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    for name in RAW_NAMES:
        with (
            (FIXTURES / f"tm_{name}.csv").open("rb") as src,
            gzip.open(root / f"{name}.csv.gz", "wb") as dst,
        ):
            shutil.copyfileobj(src, dst)
    return root


def test_build_on_fixtures(tmp_path: Path) -> None:
    tables = tm.build(_raw_root(tmp_path), tmp_path / "out")
    for name in (
        "tm_games",
        "tm_appearances",
        "tm_players",
        "tm_valuations",
        "tm_stints",
        "tm_stint_players",
        "team_elo",
    ):
        assert (tmp_path / "out" / f"{name}.csv.gz").exists()

    games = tables["tm_games"].set_index("game_id")
    assert sorted(games.index) == [1, 4, 6, 7, 8, 9]  # 2018-19 games feed Elo only
    assert games.loc[4, "knockout"] and not games.loc[1, "knockout"]
    assert games.loc[[6, 7], "season"].tolist() == ["2024/25", "2024/25"]  # final in July
    assert games.loc[[8, 9], "season"].tolist() == ["2025/26", "2025/26"]
    assert games.loc[8, "competition_type"] == "international_cup"

    stints = tables["tm_stints"]
    assert stints["game_id"].unique().tolist() == [1]  # the only game with full lineups
    assert stints["minutes"].tolist() == [60, 15, 5, 10]
    assert stints["home_goals"].tolist() == [1, 0, 0, 1]
    assert stints["away_goals"].tolist() == [1, 0, 0, 0]
    assert stints["away_n"].tolist() == [11, 11, 10, 10]

    apps = tables["tm_appearances"].set_index("player_id")
    assert apps.loc[209, "penalty_goals"] == 1 and apps.loc[203, "penalty_goals"] == 0
    assert bool(apps.loc[101, "captain"]) and bool(apps.loc[101, "started"])
    assert not bool(apps.loc[112, "started"]) and apps.loc[112, "position"] == "Second Striker"

    assert set(tables["tm_players"]["player_id"]) == {109, 209}
    assert tables["tm_valuations"]["date"].dt.year.tolist() == [2021]

    elo = tables["team_elo"].set_index("game_id")
    assert sorted(elo.index) == sorted(games.index)
    report = tables["elo_report"].set_index("pool")
    assert report.loc["club", "n_calibration"] == 2 and report.loc["national", "n_calibration"] == 1


def test_download_uses_cache_and_detects_drift(tmp_path: Path) -> None:
    for name in tm.TABLES:
        (tmp_path / f"{name}.csv.gz").write_bytes(name.encode())
    assert set(tm.download(tmp_path).values()) == {"cached"}
    manifest = json.loads((tmp_path / "manifest.json").read_text())["files"]
    assert [e["name"] for e in manifest] == list(tm.TABLES)
    assert all(len(e["sha256"]) == 64 for e in manifest)
    (tmp_path / "games.csv.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest"):
        tm.download(tmp_path)
