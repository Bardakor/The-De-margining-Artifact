"""Panel tables of paper 2: StatsBomb-derived stints, player-match, player-season, position
groups and the player pool (design §4.4, §6). Pure.

Assumptions (each stated once here):

- Scope (Claude's call): ``league`` = competition_type ``domestic_league``, which in this
  Transfermarkt snapshot is first-tier leagues only (GB1 ES1 L1 IT1 FR1 SA1 MLS1 and A1
  ARG1 AUS1 BE1 BRA1 C1 COL1 DK1 GR1 JAP1 KR1 MEX1 NL1 NO1 PL1 PO1 RO1 RSK1 RU1 SC1 SE1
  SER1 TR1 TS1 UKR1, all ``first_tier`` in competitions.csv except COL1, absent there,
  Colombia's Primera A); ``ucl`` = competition_id CL (its qualifying rounds, CLQ, are
  ``other``); ``intl`` = competition_type ``national_team_competition``; ``other`` =
  everything else (domestic cups and super cups, EL and its qualifying, Conference
  League, Club World Cup, UEFA Super Cup).
- Season (orchestrator decision, overriding design §3): club competitions keep
  Transfermarkt's season; a national-team tournament or the Club World Cup (KLUB) takes
  ``season_label`` of its edition's first match, an edition being one (competition_id,
  Transfermarkt season).
- StatsBomb minutes in Transfermarkt's convention (``fold_minute``): the minute being
  played (StatsBomb's elapsed minute + 1), stoppage time folded into 45/90/105/120.
  Extra time is kept: a game with a period-3 event ends at 120. Penalty shoot-outs
  (period 5) are ignored.
- StatsBomb stints use ``footy.players.stints.build_stints`` unchanged: starters from the
  Starting XI events, substitutions, sendings-off (Red Card or Second Yellow, from Foul
  Committed or Bad Behaviour), goals (Shot outcome Goal for the shooter's team, Own Goal
  For for the team credited). A StatsBomb player without a Transfermarkt id keeps his
  place on the pitch under the pseudo id ``-sb_player_id``, so side counts stay right.
- StatsBomb appearances in the tm_appearances layout and Transfermarkt's conventions:
  goals exclude own goals and include penalties; assists are passes flagged
  goal_assist; a second yellow is yellow 2, red 0 (straight Red Card only is red);
  position is the Starting XI slot or, for a substitute, the position of the player he
  replaced, mapped to Transfermarkt's lineup labels (``SB_TO_TM_POSITION``); captain is
  unknown (missing).
- Panel rows have minutes >= 1. position_group is the player's group over the window:
  minutes-weighted modal group of his lineup positions (ties: the group listed first in
  ``GROUP_ORDER``); a player with no known lineup position takes the group of
  Transfermarkt's profile sub_position.
- on_pitch_* sum, over the stints a player was on the pitch, the value for his side and
  for the other side; missing for games without stints. Per-90 rates in the season
  panel divide by the minutes of the rows where that column is not missing.
- StatsBomb columns carry an ``sb_`` prefix and are missing where no StatsBomb match
  is linked. There are no ESPN columns: that source was withdrawn (A15).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from footy.players import stints
from footy.players.linking import name_similarity, normalise_name, team_similarity
from footy.players.transfermarkt import position_group
from footy.players.window import season_label

END_OF_PERIOD = {1: 45, 2: 90, 3: 105, 4: 120}
SENDING_OFF = ("Red Card", "Second Yellow")
YELLOWS = ("Yellow Card", "Second Yellow")
ON_TARGET = ("Goal", "Saved", "Saved to Post")
SB_TO_TM_POSITION = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Centre-Back", "Left Center Back": "Centre-Back",
    "Right Center Back": "Centre-Back",
    "Left Back": "Left-Back", "Left Wing Back": "Left-Back",
    "Right Back": "Right-Back", "Right Wing Back": "Right-Back",
    "Center Defensive Midfield": "Defensive Midfield",
    "Left Defensive Midfield": "Defensive Midfield",
    "Right Defensive Midfield": "Defensive Midfield",
    "Center Midfield": "Central Midfield", "Left Center Midfield": "Central Midfield",
    "Right Center Midfield": "Central Midfield",
    "Center Attacking Midfield": "Attacking Midfield",
    "Left Attacking Midfield": "Attacking Midfield",
    "Right Attacking Midfield": "Attacking Midfield",
    "Left Wing": "Left Winger", "Right Wing": "Right Winger",
    "Left Midfield": "Left Midfield", "Right Midfield": "Right Midfield",
    "Center Forward": "Centre-Forward", "Left Center Forward": "Centre-Forward",
    "Right Center Forward": "Centre-Forward", "Secondary Striker": "Second Striker",
}  # fmt: skip
GROUP_ORDER = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "CF")
SB_COLUMNS = [
    "sb_shots", "sb_shots_on_target", "sb_xg", "sb_np_xg", "sb_xa",
    "on_pitch_sb_xg_for", "on_pitch_sb_xg_against",
]  # fmt: skip
SEASON_COUNTS = [
    "goals", "np_goals", "assists", "penalty_goals", "yellow", "red",
    "on_pitch_gf", "on_pitch_ga", *SB_COLUMNS,
]  # fmt: skip
PANEL_COLUMNS = [
    "game_id", "player_id", "date", "season", "competition_id", "competition_type", "scope",
    "club_id", "position_group", "match_position_group", "started", "minutes", "goals",
    "np_goals", "assists", "penalty_goals", "yellow", "red", "on_pitch_gf", "on_pitch_ga",
    "team_elo", "opp_elo", "knockout", "sb_linked", *SB_COLUMNS,
]  # fmt: skip
APPEARANCE_COLUMNS = [
    "game_id", "player_id", "club_id", "date", "competition_id", "minutes", "goals",
    "assists", "yellow", "red", "started", "position", "captain", "penalty_goals", "source",
]  # fmt: skip

Json = Any


# --- scope and season -------------------------------------------------------------------


def scope(competition_id: str, competition_type: str) -> str:
    if competition_id == "CL":
        return "ucl"
    if competition_type == "national_team_competition":
        return "intl"
    if competition_type == "domestic_league":
        return "league"
    return "other"


def study_seasons(games: pd.DataFrame) -> pd.Series:
    """Season per game: Transfermarkt's for club competitions, first-match label for
    national-team tournaments and the Club World Cup."""
    tournament = (games["competition_type"] == "national_team_competition") | (
        games["competition_id"] == "KLUB"
    )
    first = (
        pd.to_datetime(games["date"])
        .groupby([games["competition_id"], games["season"]])
        .transform("min")
    )
    return games["season"].where(~tournament, first.dt.date.map(season_label))


# --- StatsBomb matches -------------------------------------------------------------------


def fold_minute(minute: int, period: int) -> int:
    return min(minute + 1, END_OF_PERIOD[period])


@dataclass(frozen=True)
class SbMatch:
    """One StatsBomb match, in StatsBomb ids. starters: team, sb_player_id, position.
    timeline: minute, kind (goal/sub/red), team, sb_player_id, sb_player_in_id, position,
    own_goal. shots: minute, team, sb_player_id, assister, xg, penalty, on_target.
    counts: sb_player_id, goals, penalty_goals, assists, yellow, red."""

    sb_match_id: int
    end_min: int
    starters: pd.DataFrame
    timeline: pd.DataFrame
    shots: pd.DataFrame
    counts: pd.DataFrame


def _card(event: Json) -> str | None:
    for key in ("foul_committed", "bad_behaviour"):
        card = (event.get(key) or {}).get("card")
        if card is not None:
            return str(card["name"])
    return None


def parse_sb_match(match_id: int, events: Json) -> SbMatch:
    play = [e for e in events if e["period"] <= 4]
    passer = {e["id"]: e["player"]["id"] for e in play if e["type"]["name"] == "Pass"}
    starters, timeline, shots, counts = [], [], [], []
    for e in play:
        etype, team = e["type"]["name"], e["team"]["name"]
        minute = fold_minute(e["minute"], e["period"])
        player = (e.get("player") or {}).get("id")
        if etype == "Starting XI":
            starters += [
                (team, s["player"]["id"], s["position"]["name"]) for s in e["tactics"]["lineup"]
            ]
        elif etype == "Substitution":
            replacement = e["substitution"]["replacement"]["id"]
            position = (e.get("position") or {}).get("name")
            timeline.append((minute, "sub", team, player, replacement, position, False))
        elif etype == "Own Goal For":
            timeline.append((minute, "goal", team, None, None, None, True))
        elif etype == "Shot":
            shot = e["shot"]
            goal = shot["outcome"]["name"] == "Goal"
            penalty = shot["type"]["name"] == "Penalty"
            assister = passer.get(shot.get("key_pass_id"))
            shots.append(
                (minute, team, player, assister, shot.get("statsbomb_xg", 0.0), penalty,
                 shot["outcome"]["name"] in ON_TARGET)
            )  # fmt: skip
            if goal:
                timeline.append((minute, "goal", team, player, None, None, False))
                counts += [(player, "goals")] + ([(player, "penalty_goals")] if penalty else [])
        elif etype == "Pass" and e["pass"].get("goal_assist"):
            counts.append((player, "assists"))
        card = _card(e)
        if card is not None and player is not None:
            if card in YELLOWS:
                counts.append((player, "yellow"))
            if card == "Red Card":
                counts.append((player, "red"))
            if card in SENDING_OFF:
                timeline.append((minute, "red", team, player, None, None, False))
    line = pd.DataFrame(
        timeline,
        columns=[
            "minute", "kind", "team", "sb_player_id", "sb_player_in_id", "position", "own_goal",
        ],
    )  # fmt: skip
    line[["sb_player_id", "sb_player_in_id"]] = line[["sb_player_id", "sb_player_in_id"]].astype(
        "Int64"
    )
    shot_frame = pd.DataFrame(
        shots,
        columns=[
            "minute", "team", "sb_player_id", "assister", "xg", "penalty", "on_target",
        ],
    )  # fmt: skip
    shot_frame[["sb_player_id", "assister"]] = shot_frame[["sb_player_id", "assister"]].astype(
        "Int64"
    )
    tally = pd.DataFrame(counts, columns=["sb_player_id", "stat"])
    count_cols = ["goals", "penalty_goals", "assists", "yellow", "red"]
    per_player = (
        tally.groupby(["sb_player_id", "stat"]).size().unstack(fill_value=0)
        .reindex(columns=count_cols, fill_value=0).reset_index()
    )  # fmt: skip
    return SbMatch(
        sb_match_id=match_id,
        end_min=120 if any(e["period"] >= 3 for e in play) else 90,
        starters=pd.DataFrame(starters, columns=["team", "sb_player_id", "position"]),
        timeline=line,
        shots=shot_frame,
        counts=per_player,
    )


def _concat(matches: Sequence[SbMatch], field: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = [getattr(m, field) for m in matches]
    tagged = [f.assign(sb_match_id=m.sb_match_id) for f, m in zip(frames, matches, strict=True)]
    return pd.concat(tagged, ignore_index=True)


def _tm_id(sb_ids: pd.Series, mapping: pd.Series) -> pd.Series:
    """Transfermarkt id per StatsBomb id; -sb_player_id where unlinked; missing stays."""
    mapped = sb_ids.map(mapping)
    return mapped.fillna(-sb_ids).astype("Int64")


def sb_tables(
    matches: Sequence[SbMatch], teams: pd.DataFrame, games: pd.DataFrame, mapping: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(sb_stints, sb_stint_players, sb_appearances) in the Transfermarkt layouts.

    teams: sb_match_id, team, game_id, club_id. games: tm_games rows of those games.
    mapping: StatsBomb id -> Transfermarkt id (linked players only).
    """
    club = teams.set_index(["sb_match_id", "team"])

    def located(frame: pd.DataFrame) -> pd.DataFrame:
        keys = pd.MultiIndex.from_frame(frame[["sb_match_id", "team"]])
        return frame.assign(
            game_id=club["game_id"].reindex(keys).to_numpy(),
            club_id=club["club_id"].reindex(keys).to_numpy(),
        )

    starters = located(_concat(matches, "starters"))
    timeline = located(_concat(matches, "timeline"))
    starters["player_id"] = _tm_id(starters["sb_player_id"], mapping)
    timeline["player_id"] = _tm_id(timeline["sb_player_id"], mapping)
    timeline["player_in_id"] = _tm_id(timeline["sb_player_in_id"], mapping)
    ends = pd.DataFrame(
        {"sb_match_id": [m.sb_match_id for m in matches], "end_min": [m.end_min for m in matches]}
    )
    game_rows = (
        teams.drop_duplicates("sb_match_id")[["sb_match_id", "game_id"]]
        .merge(ends, on="sb_match_id")
        .merge(games[["game_id", "home_club_id", "away_club_id"]], on="game_id")
    )
    sb_stints, sb_stint_players = stints.build_stints(
        game_rows,
        starters.assign(started=True),
        timeline[["game_id", "minute", "kind", "club_id", "player_id", "player_in_id"]],
    )
    subs = timeline[timeline["kind"] == "sub"]
    rows = pd.concat(
        [
            starters[
                ["sb_match_id", "game_id", "club_id", "sb_player_id", "player_id", "position"]
            ].assign(started=True),
            pd.DataFrame(
                {
                    "sb_match_id": subs["sb_match_id"],
                    "game_id": subs["game_id"],
                    "club_id": subs["club_id"],
                    "sb_player_id": subs["sb_player_in_id"],
                    "player_id": subs["player_in_id"],
                    "position": subs["position"],
                    "started": False,
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["game_id", "player_id"])
    counts = _concat(matches, "counts")
    rows = (
        rows.merge(counts, on=["sb_match_id", "sb_player_id"], how="left")
        .merge(stints.player_minutes(sb_stints, sb_stint_players), on=["game_id", "player_id"],
               how="left")
        .merge(games[["game_id", "date", "competition_id"]], on="game_id")
    )  # fmt: skip
    appearances = rows.assign(
        minutes=rows["minutes"].fillna(0).astype(int),
        position=rows["position"].map(SB_TO_TM_POSITION),
        captain=pd.Series(pd.NA, index=rows.index, dtype="boolean"),
        started=rows["started"].astype("boolean"),
        source="statsbomb",
    )
    stat_cols = ["goals", "assists", "yellow", "red", "penalty_goals"]
    appearances[stat_cols] = appearances[stat_cols].fillna(0).astype(int)
    return sb_stints, sb_stint_players, appearances[APPEARANCE_COLUMNS]


def sb_player_columns(
    matches: Sequence[SbMatch],
    teams: pd.DataFrame,
    mapping: pd.Series,
    all_stints: pd.DataFrame,
    all_stint_players: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """SB_COLUMNS per (game_id, player_id) for every player on the pitch in a linked
    StatsBomb match (zeros for players without a shot)."""
    club = teams.set_index(["sb_match_id", "team"])
    shots = _concat(matches, "shots")
    keys = pd.MultiIndex.from_frame(shots[["sb_match_id", "team"]])
    shots["game_id"] = club["game_id"].reindex(keys).to_numpy()
    shots["club_id"] = club["club_id"].reindex(keys).to_numpy()
    shots["player_id"] = _tm_id(shots["sb_player_id"], mapping)
    shots["assister_id"] = _tm_id(shots["assister"], mapping)
    shots["np_xg"] = shots["xg"].where(~shots["penalty"].astype(bool), 0.0)
    shooter = shots.groupby(["game_id", "player_id"]).agg(
        sb_shots=("xg", "size"),
        sb_shots_on_target=("on_target", "sum"),
        sb_xg=("xg", "sum"),
        sb_np_xg=("np_xg", "sum"),
    )
    assist = shots.dropna(subset=["assister_id"]).groupby(["game_id", "assister_id"])["xg"].sum()
    assist.index = assist.index.set_names(["game_id", "player_id"])
    side = games.set_index("game_id")
    shots["side"] = np.where(
        shots["club_id"].to_numpy() == side["home_club_id"].reindex(shots["game_id"]).to_numpy(),
        "home",
        "away",
    )
    per_stint = assign_to_stints(all_stints, shots.rename(columns={"xg": "value"}))
    on = on_pitch_totals(all_stints.merge(per_stint, on="stint_id"), all_stint_players)
    on = on.rename(columns={"for": "on_pitch_sb_xg_for", "against": "on_pitch_sb_xg_against"})
    on = on[on["game_id"].isin(shots["game_id"])].set_index(["game_id", "player_id"])
    out = on.join(shooter, how="left").join(assist.rename("sb_xa"), how="left")
    out[SB_COLUMNS[:5]] = out[SB_COLUMNS[:5]].fillna(0)  # players on the pitch without a shot
    return out.reset_index()[["game_id", "player_id", *SB_COLUMNS]]


# --- on-pitch sums ---------------------------------------------------------------------


def assign_to_stints(all_stints: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Sum event values per stint and side with the stint rule of ``stints``: an event at
    minute m belongs to the stint (s, t] with s < m <= t (minute 0 to the first stint).

    events: game_id, minute, side (home/away), value. Returns stint_id, home_value,
    away_value for every stint of the games given (0 where nothing happened).
    """
    st = all_stints[["stint_id", "game_id", "start_min", "end_min"]]
    merged = events[["game_id", "minute", "side", "value"]].merge(st, on="game_id")
    inside = (merged["start_min"] < merged["minute"]) & (merged["minute"] <= merged["end_min"])
    hit = merged[inside | ((merged["start_min"] == 0) & (merged["minute"] == 0))]
    sums = hit.pivot_table(index="stint_id", columns="side", values="value", aggfunc="sum")
    sums = sums.reindex(columns=["home", "away"]).fillna(0.0)
    sums.columns = ["home_value", "away_value"]
    played = st[st["game_id"].isin(events["game_id"])][["stint_id"]]
    return played.merge(sums.reset_index(), on="stint_id", how="left").fillna(0.0)


def on_pitch_totals(
    stint_values: pd.DataFrame, stint_players: pd.DataFrame, home: str = "home_value",
    away: str = "away_value",
) -> pd.DataFrame:  # fmt: skip
    """Per (game_id, player_id): sum of his side's value (``for``) and the other side's
    (``against``) over the stints he was on the pitch."""
    merged = stint_players.merge(stint_values[["stint_id", "game_id", home, away]], on="stint_id")
    is_home = merged["side"] == "home"
    merged["for"] = merged[home].where(is_home, merged[away])
    merged["against"] = merged[away].where(is_home, merged[home])
    return merged.groupby(["game_id", "player_id"], as_index=False)[["for", "against"]].sum()


# --- panels ------------------------------------------------------------------------------


def position_groups(appearances: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """player_id, position_group, group_source (lineups or profile).

    appearances: player_id, position (lineup label), minutes. profiles: player_id,
    sub_position (Transfermarkt profile).
    """
    known = appearances.dropna(subset=["position"])
    known = known.assign(group=known["position"].map(position_group))
    weight = known.groupby(["player_id", "group"])["minutes"].sum().reset_index()
    weight["order"] = weight["group"].map(GROUP_ORDER.index)
    weight = weight.sort_values(["player_id", "minutes", "order"], ascending=[True, False, True])
    modal = weight.drop_duplicates("player_id")[["player_id", "group"]]
    modal = modal.assign(group_source="lineups")
    rest = profiles[~profiles["player_id"].isin(modal["player_id"])].dropna(subset=["sub_position"])
    fallback = pd.DataFrame(
        {
            "player_id": rest["player_id"],
            "group": rest["sub_position"].map(position_group),
            "group_source": "profile",
        }
    )
    out = pd.concat([modal, fallback], ignore_index=True)
    return out.rename(columns={"group": "position_group"})


def player_match(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    all_stints: pd.DataFrame,
    all_stint_players: pd.DataFrame,
    elo: pd.DataFrame,
    groups: pd.DataFrame,
    sb_columns: pd.DataFrame,
) -> pd.DataFrame:
    """panel_player_match (``PANEL_COLUMNS``): one row per (game_id, player_id) with
    minutes >= 1."""
    played = appearances[appearances["minutes"] >= 1]
    g = games.assign(season=study_seasons(games))[
        ["game_id", "season", "competition_type", "knockout", "home_club_id", "away_club_id"]
    ]
    out = played.merge(g, on="game_id").merge(elo, on="game_id", how="left")
    home = out["club_id"] == out["home_club_id"]
    out["team_elo"] = out["home_elo_pre"].where(home, out["away_elo_pre"])
    out["opp_elo"] = out["away_elo_pre"].where(home, out["home_elo_pre"])
    out["scope"] = [
        scope(c, t) for c, t in zip(out["competition_id"], out["competition_type"], strict=True)
    ]
    out["np_goals"] = out["goals"] - out["penalty_goals"]
    out["match_position_group"] = out["position"].map(
        lambda p: position_group(p) if isinstance(p, str) else None
    )
    on = on_pitch_totals(all_stints, all_stint_players, "home_goals", "away_goals")
    on = on.rename(columns={"for": "on_pitch_gf", "against": "on_pitch_ga"})
    out = out.merge(on, on=["game_id", "player_id"], how="left")
    out = out.merge(groups[["player_id", "position_group"]], on="player_id", how="left")
    out = out.merge(sb_columns, on=["game_id", "player_id"], how="left")
    out[SB_COLUMNS] = out[SB_COLUMNS].astype(float)
    out["sb_linked"] = out["sb_xg"].notna()
    out["started"] = out["started"].astype("boolean")
    return out[PANEL_COLUMNS].sort_values(["date", "game_id", "player_id"]).reset_index(drop=True)


def player_season(panel: pd.DataFrame) -> pd.DataFrame:
    """panel_player_season: per (player_id, season, scope), plus scope ``all``: games,
    starts, minutes, the minutes behind each column family (stints, sb), sums of
    ``SEASON_COUNTS`` (missing when no row has the column) and their per-90 rates."""
    long = pd.concat([panel, panel.assign(scope="all")], ignore_index=True)
    keys = ["player_id", "season", "scope"]
    grouped = long.groupby(keys)
    out = grouped.agg(
        games=("game_id", "size"), starts=("started", "sum"), minutes=("minutes", "sum")
    )
    for column in SEASON_COUNTS:
        total = grouped[column].sum(min_count=1)
        behind = (
            long["minutes"]
            .where(long[column].notna())
            .groupby([long[k] for k in keys])
            .sum(min_count=1)
        )
        out[column] = total
        out[f"{column}_p90"] = 90.0 * total / behind.where(behind > 0)
        if column in ("on_pitch_gf", "sb_xg"):
            family = {"on_pitch_gf": "stints", "sb_xg": "sb"}[column]
            out[f"minutes_{family}"] = behind.fillna(0).astype(int)
    return out.reset_index()


# --- pool ---------------------------------------------------------------------------------

POOL_COLUMNS = [
    "player", "in_v1", "bdo_editions", "tm_player_id", "tm_name", "position_group",
    "is_goalkeeper", "minutes_in_window", "match_method", "confidence",
]  # fmt: skip
POOL_NAME_THRESHOLD = 0.85
CLUB_THRESHOLD = 1.0


def pool_names(v1: pd.DataFrame, ballon_dor: pd.DataFrame) -> pd.DataFrame:
    """player, in_v1, bdo_editions, clubs: V1's names and every player ranked in a
    2021-2026 edition, one row per normalised name (V1's spelling kept when both)."""
    bdo = ballon_dor[ballon_dor["edition"].between(2021, 2026)]
    rows: dict[str, dict[str, Any]] = {}
    for name in v1["player"]:
        rows[normalise_name(name)] = {"player": name, "in_v1": True, "editions": [], "clubs": []}
    for edition, name, club in bdo[["edition", "player", "club"]].itertuples(index=False):
        row = rows.setdefault(
            normalise_name(name), {"player": name, "in_v1": False, "editions": [], "clubs": []}
        )
        row["editions"].append(int(edition))
        row["clubs"].append(club)
    return pd.DataFrame(
        [
            {
                "player": r["player"],
                "in_v1": r["in_v1"],
                "bdo_editions": ";".join(str(e) for e in sorted(r["editions"])),
                "clubs": tuple(dict.fromkeys(r["clubs"])),
            }
            for r in rows.values()
        ]
    )


def match_pool(
    pool: pd.DataFrame,
    players: pd.DataFrame,
    club_minutes: pd.DataFrame,
    groups: pd.DataFrame,
    manual: dict[str, tuple[int, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map pool names to Transfermarkt ids; return (pool table, every candidate looked at).

    players: player_id, name. club_minutes: player_id, club_name, minutes (window).
    groups: player_id, position_group (``position_groups``).
    Candidates: names scoring >= ``POOL_NAME_THRESHOLD``. Rule, in order: ``manual`` (an
    explicit decision, listed); a single candidate who played for a club the Ballon d'Or
    lists for him (``name+club``); several such: the one with most window minutes
    (``name+club+minutes``, medium); no club evidence possible (V1-only names) and a
    single candidate (``name_unique``, medium); otherwise unmapped with the reason.
    confidence: high when the name scores >= 0.95 and the club confirms, else medium.
    """
    minutes = club_minutes.groupby("player_id")["minutes"].sum()
    prefix: dict[str, set[int]] = {}
    for pid, name in players[["player_id", "name"]].itertuples(index=False):
        for token in normalise_name(name).split():
            prefix.setdefault(token[:3], set()).add(int(pid))
    names = players.set_index("player_id")["name"]
    rows: list[tuple[Any, ...]] = []
    looked: list[tuple[Any, ...]] = []
    for player, clubs in pool[["player", "clubs"]].itertuples(index=False):
        ids = set().union(*(prefix.get(t[:3], set()) for t in normalise_name(player).split()))
        cands = [(pid, name_similarity(player, names[pid])) for pid in ids]
        cands = [(pid, s) for pid, s in cands if s >= POOL_NAME_THRESHOLD]
        played = club_minutes[club_minutes["player_id"].isin([pid for pid, _ in cands])]
        confirmed = {
            int(pid)
            for pid, club in played[["player_id", "club_name"]].itertuples(index=False)
            if any(team_similarity(club, listed) >= CLUB_THRESHOLD for listed in clubs)
        }
        for pid, s in cands:
            looked.append((player, pid, names[pid], s, pid in confirmed, minutes.get(pid, 0)))
        score = dict(cands)
        if player in manual:
            pid, reason = manual[player]
            rows.append((player, pid, "manual", "high", reason))
        elif len(confirmed) == 1:
            pid = next(iter(confirmed))
            rows.append((player, pid, "name+club", "high" if score[pid] >= 0.95 else "medium", ""))
        elif confirmed:
            pid = max(confirmed, key=lambda p: minutes.get(p, 0))
            rows.append((player, pid, "name+club+minutes", "medium", f"{len(confirmed)} confirmed"))
        elif not clubs and len(cands) == 1:
            rows.append((player, cands[0][0], "name_unique", "medium", ""))
        else:
            reason = f"{len(cands)} name candidates, {len(confirmed)} club-confirmed"
            rows.append((player, None, "unmapped", "", reason))
    matched = pd.DataFrame(
        rows, columns=["player", "tm_player_id", "match_method", "confidence", "note"]
    )
    matched["tm_player_id"] = matched["tm_player_id"].astype("Int64")
    out = pool.drop(columns="clubs").merge(matched, on="player")
    out["tm_name"] = out["tm_player_id"].map(names)
    out["minutes_in_window"] = out["tm_player_id"].map(minutes).fillna(0).astype(int)
    group = groups.set_index("player_id")["position_group"]
    out["position_group"] = out["tm_player_id"].map(group)
    out["is_goalkeeper"] = out["position_group"] == "GK"
    candidates = pd.DataFrame(
        looked,
        columns=["player", "tm_player_id", "tm_name", "name_score", "club_confirmed", "minutes"],
    )
    return out[[*POOL_COLUMNS, "note"]], candidates
