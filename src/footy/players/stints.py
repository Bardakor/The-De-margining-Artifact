"""Stints: maximal intervals of a game with no substitution or red card (design §4.1). Pure.

Inputs are already normalised by ``footy.players.transfermarkt``:

- games: game_id, home_club_id, away_club_id, end_min (90, or 120 after extra time)
- lineups: game_id, player_id, club_id, started (starters only are used)
- events: game_id, minute, kind in {goal, sub, red}, club_id, player_id, player_in_id
  For a goal, club_id is the club credited with it (own goals included); for a sub,
  player_id goes off and player_in_id (may be missing) comes on for club_id.

Rules (stated assumptions, each tested):

- Minutes are integers with stoppage time folded into 45/90 (Transfermarkt's own
  convention), so stint boundaries are the distinct change minutes plus 0 and end_min.
- A change at minute m takes effect at the start of the stint beginning at m.
- A goal at minute m belongs to the stint (s, t] with s < m <= t, i.e. a goal at the
  minute of a change goes to the stint that ends there; a goal at minute 0 goes to the
  first stint. Goals with an unknown minute (< 0) are not assigned to any stint.
- A sub whose outgoing player is not on the pitch still brings the incoming player on,
  so a data error shows up as a side count outside 7..11 instead of being hidden.
- A change with an unknown minute (< 0) cannot be placed: the caller excludes the game.
- stint_id = game_id * 100 + stint index (a game has far fewer than 100 stints).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

STINT_COLUMNS = [
    "stint_id",
    "game_id",
    "start_min",
    "end_min",
    "minutes",
    "home_goals",
    "away_goals",
    "home_n",
    "away_n",
]


def _by_game(frame: pd.DataFrame, columns: list[str]) -> dict[int, list[tuple[Any, ...]]]:
    grouped: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    for row in frame[["game_id", *columns]].itertuples(index=False, name=None):
        grouped[row[0]].append(row[1:])
    return grouped


def build_stints(
    games: pd.DataFrame, lineups: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tm_stints, tm_stint_players) for every game in ``games``."""
    starters = _by_game(lineups[lineups["started"]], ["player_id", "club_id"])
    game_events = _by_game(events, ["minute", "kind", "club_id", "player_id", "player_in_id"])
    stints: list[tuple[int, ...]] = []
    players: list[tuple[int, int, str]] = []

    for game_id, home, away, end in games[
        ["game_id", "home_club_id", "away_club_id", "end_min"]
    ].itertuples(index=False, name=None):
        side_of = {home: "home", away: "away"}
        on: dict[str, set[int]] = {"home": set(), "away": set()}
        for player_id, club_id in starters.get(game_id, []):
            on[side_of[club_id]].add(player_id)

        evs = game_events.get(game_id, [])
        changes = sorted((e for e in evs if e[1] != "goal"), key=lambda e: e[0])
        if any(e[0] < 0 for e in changes):
            raise ValueError(f"game {game_id}: change with unknown minute")
        goals = [(min(e[0], end), side_of[e[2]]) for e in evs if e[1] == "goal" and e[0] >= 0]
        bounds = sorted({0, end, *(min(e[0], end) for e in changes)})

        for index, (start, stop) in enumerate(zip(bounds, bounds[1:], strict=False)):
            for minute, kind, club_id, player_id, player_in_id in changes:
                if min(minute, end) != start:
                    continue
                on["home"].discard(player_id)
                on["away"].discard(player_id)
                if kind == "sub" and not pd.isna(player_in_id):
                    on[side_of[club_id]].add(int(player_in_id))
            scored = [s for m, s in goals if start < m <= stop or (index == 0 and m == 0)]
            stint_id = game_id * 100 + index
            stints.append(
                (
                    stint_id,
                    game_id,
                    start,
                    stop,
                    stop - start,
                    scored.count("home"),
                    scored.count("away"),
                    len(on["home"]),
                    len(on["away"]),
                )
            )
            players.extend((stint_id, p, side) for side in ("home", "away") for p in on[side])

    return (
        pd.DataFrame(stints, columns=STINT_COLUMNS),
        pd.DataFrame(players, columns=["stint_id", "player_id", "side"]),
    )


def player_minutes(stints: pd.DataFrame, stint_players: pd.DataFrame) -> pd.DataFrame:
    """Minutes on the pitch per (game_id, player_id) implied by the stints."""
    merged = stint_players.merge(stints[["stint_id", "game_id", "minutes"]], on="stint_id")
    return merged.groupby(["game_id", "player_id"])["minutes"].sum().reset_index()


def reconcile(stints: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Per game: stint goal totals next to the final score (home_goals, away_goals)."""
    totals = stints.groupby("game_id", as_index=False)[["home_goals", "away_goals"]].sum()
    out = games[["game_id", "home_goals", "away_goals"]].merge(
        totals, on="game_id", suffixes=("", "_stints")
    )
    out["match"] = (out["home_goals"] == out["home_goals_stints"]) & (
        out["away_goals"] == out["away_goals_stints"]
    )
    return out
