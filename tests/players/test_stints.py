"""Stints on a toy match with a planted truth (design §4.1)."""

import pandas as pd
import pytest

from footy.players.stints import build_stints, player_minutes, reconcile

HOME, AWAY = 10, 20
GAMES = pd.DataFrame(
    {"game_id": [1], "home_club_id": [HOME], "away_club_id": [AWAY], "end_min": [90]}
)
LINEUPS = pd.DataFrame(
    {
        "game_id": 1,
        "player_id": [*range(101, 112), *range(201, 212)],
        "club_id": [HOME] * 11 + [AWAY] * 11,
        "started": True,
    }
)


def toy_events() -> pd.DataFrame:
    """Two subs, a red card, a goal at the minute of a change, an own goal. Final 2-1."""
    rows = [
        (30, "goal", HOME, 109, None),
        (60, "sub", HOME, 110, 112),
        (60, "goal", AWAY, 209, None),  # same minute as the sub: belongs to the stint ending at 60
        (75, "red", AWAY, 205, None),
        (80, "sub", AWAY, 211, 212),
        (85, "goal", HOME, 203, None),  # own goal by away player 203, credited to home
    ]
    frame = pd.DataFrame(rows, columns=["minute", "kind", "club_id", "player_id", "player_in_id"])
    return frame.assign(game_id=1)


def test_toy_match_stints_match_planted_truth() -> None:
    stints, players = build_stints(GAMES, LINEUPS, toy_events())
    expected = pd.DataFrame(
        {
            "start_min": [0, 60, 75, 80],
            "end_min": [60, 75, 80, 90],
            "minutes": [60, 15, 5, 10],
            "home_goals": [1, 0, 0, 1],
            "away_goals": [1, 0, 0, 0],
            "home_n": [11, 11, 11, 11],
            "away_n": [11, 11, 10, 10],
        }
    )
    pd.testing.assert_frame_equal(stints[expected.columns], expected)
    assert stints["stint_id"].tolist() == [100, 101, 102, 103]
    on = players.groupby("stint_id")["player_id"].apply(set)
    assert 110 in on[100] and 110 not in on[101] and 112 in on[101]
    assert 205 in on[101] and 205 not in on[102]
    assert 211 in on[102] and 212 in on[103] and 211 not in on[103]
    assert set(players.loc[players["player_id"] == 112, "side"]) == {"home"}


def test_toy_match_minutes_and_score_reconcile() -> None:
    stints, players = build_stints(GAMES, LINEUPS, toy_events())
    minutes = player_minutes(stints, players).set_index("player_id")["minutes"]
    assert minutes[110] == 60 and minutes[112] == 30
    assert minutes[205] == 75 and minutes[211] == 80 and minutes[212] == 10
    assert minutes[101] == 90 and minutes[201] == 90
    assert len(minutes) == 24
    score = pd.DataFrame({"game_id": [1], "home_goals": [2], "away_goals": [1]})
    assert reconcile(stints, score)["match"].tolist() == [True]


def test_goal_at_minute_zero_and_unknown_minute() -> None:
    events = pd.DataFrame(
        {
            "game_id": 1,
            "minute": [0, -1, 50],
            "kind": ["goal", "goal", "sub"],
            "club_id": [HOME, AWAY, HOME],
            "player_id": [101, 201, 102],
            "player_in_id": [None, None, 112],
        }
    )
    stints, _ = build_stints(GAMES, LINEUPS, events)
    assert stints["home_goals"].tolist() == [1, 0]
    assert stints["away_goals"].sum() == 0  # unknown-minute goal is not assigned


def test_sub_without_replacement_reduces_side() -> None:
    events = pd.DataFrame(
        {
            "game_id": 1,
            "minute": [70],
            "kind": ["sub"],
            "club_id": [AWAY],
            "player_id": [204],
            "player_in_id": [float("nan")],
        }
    )
    stints, _ = build_stints(GAMES, LINEUPS, events)
    assert stints["away_n"].tolist() == [11, 10]


def test_change_at_unknown_minute_raises() -> None:
    events = pd.DataFrame(
        {
            "game_id": 1,
            "minute": [-1],
            "kind": ["red"],
            "club_id": [AWAY],
            "player_id": [204],
            "player_in_id": [None],
        }
    )
    with pytest.raises(ValueError, match="unknown minute"):
        build_stints(GAMES, LINEUPS, events)
