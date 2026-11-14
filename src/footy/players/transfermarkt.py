"""Transfermarkt tables: download, load, and the processed study tables (design §3, §4.1).

Source: the dcaribou Transfermarkt datasets, R2 mirror, snapshot Last-Modified
2026-09-05. Files are cached under ``data/players/raw/transfermarkt/`` and never
re-fetched; ``manifest.json`` records URL, retrieval time, Last-Modified and SHA-256
per file (paper 1's convention, ``footy/study/acquire.py``).

Encodings verified on the snapshot's 2020-01-01..2026-08-31 games:

- Substitution events: ``player_id`` goes off, ``player_in_id`` comes on (missing for
  "Substitution without replacement"); ``club_id`` is the side making the change.
  Checked against lineups: ``player_id`` is a starter in 96.5% of 382,950 subs,
  ``player_in_id`` a listed substitute in 97.0% and a starter in 0.002%.
- Sendings-off: Cards events whose description contains "Red card" or "Second yellow";
  a second yellow is never also logged as a "Red card".
- Goal events: ``club_id`` is the club credited with the goal, own goals included
  (``player_id`` is then the defender who put it into his own net: his lineup club
  differs from ``club_id`` in all 3,476 own goals whose scorer is in a lineup, and
  reading ``club_id`` as the credited side reconciles 98.3% of own-goal games against
  1.9% for the opposite reading); an own goal's description contains "Own-goal"; a
  penalty goal's first descriptor is "Penalty" ("Penalty rebound" is not one).
- Minutes are integers with stoppage time folded into 45/90; -1 means unknown.
- ``home_club_goals``/``away_club_goals`` include penalty-shootout kicks scored
  (93.7% of shootout games since 2015 reconcile only that way).

Processing assumptions:

- tm_games ``home_goals``/``away_goals`` exclude shootout kicks: Shootout events
  described "Scored" are subtracted per club, unless that would make a side negative,
  which marks a raw score that never included them (awarded or already-net results).
- Season: a club competition keeps Transfermarkt's own season (start year, labelled
  e.g. 2019/20), not the 1 July rule applied to the match date, because the date rule
  mislabels 1,597 window club games: the COVID-extended 2019/20 (Premier League, LaLiga,
  Serie A to July-August 2020, the Champions League final eight in August 2020) and
  calendar-year leagues split in two (MLS, Brasileirao, Allsvenskan...). Calendar-year
  leagues therefore carry Transfermarkt's season id: MLS 2025 is "2024/25". A
  national-team tournament takes the 1 July label of the last game of its edition in
  the data, i.e. of its final (design §3); the 2026 World Cup is in the data only to
  2026-06-28 (group stage), so it is labelled 2025/26, not 2026/27.
- Competition types missing from the raw games (codes absent from competitions.csv)
  are filled from ``UNLISTED_COMPETITION_TYPES``.
- knockout: the round names neither a league matchday nor a group (the 2024+ UEFA
  league phase is labelled "Group Stage" and is therefore not knockout).
- A game is extra time (end_min 120) when any event minute or any appearance exceeds 90.
- Stints are built only for window games with 11 distinct starters on each side and
  no substitution or sending-off at an unknown minute.
- tm_appearances (``source`` column): for every game with stints the rows are derived
  (``stint_appearances``: minutes from stints, counts from game_events), because the raw
  appearances table has no rows at all for Copa America 2024, Saudi Pro League and MLS
  games and misses 31% of players on the pitch in stint games. Games without stints
  keep Transfermarkt's own rows, with started/position/captain from the lineups
  (missing where the lineup is).
- tm_players keeps players with an appearance or a lineup entry in the window.
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from footy.data.football_data import sha256_file
from footy.players import elo, stints
from footy.players.paths import PROCESSED, RAW
from footy.players.window import WINDOW_END, WINDOW_START, season_label

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = (
    "games",
    "appearances",
    "game_events",
    "game_lineups",
    "players",
    "player_valuations",
    "club_games",
    "competitions",
    "clubs",
)
RAW_DIR = RAW / "transfermarkt"
# Generic client name: no personal or project identifiers go to third parties.
USER_AGENT = "footy-research/0.1"
TIMEOUT_SECONDS = 120
ELO_START = pd.Timestamp("2015-07-01")

UNLISTED_COMPETITION_TYPES = {
    "KLUB": "international_cup",  # FIFA Club World Cup
    "CGB": "domestic_cup",  # EFL Cup
    "POCP": "domestic_cup",  # Taça da Liga
    "UKRS": "other",  # Ukrainian Super Cup
    "COL1": "domestic_league",  # Colombian Primera A
}

POSITION_GROUPS = {
    "goalkeeper": "GK",
    "centre-back": "CB",
    "sweeper": "CB",
    "defender": "CB",
    "left-back": "FB",
    "right-back": "FB",
    "defensive midfield": "DM",
    "central midfield": "CM",
    "midfield": "CM",
    "attacking midfield": "AM",
    "left winger": "W",
    "right winger": "W",
    "left midfield": "W",
    "right midfield": "W",
    "centre-forward": "CF",
    "second striker": "CF",
    "attack": "CF",
}
_NOT_KNOCKOUT = re.compile(r"matchday|group", re.IGNORECASE)
_SENDING_OFF = re.compile(r"Red card|Second yellow")
_PENALTY_GOAL = re.compile(r"^\s*,\s*Penalty\s*(?:,|$)")
_YELLOW = re.compile(r"Yellow card|Second yellow")
_OWN_GOAL = re.compile(r"Own-goal")


# --- pure ---------------------------------------------------------------------------


def position_group(lineup_position: str) -> str:
    """Design §4.4 group of a lineup position string.

    The rare generic labels ("Defender", "Midfield", "Attack") map to the most common
    group of their line (CB, CM, CF). An unknown label raises, so new labels surface.
    """
    return POSITION_GROUPS[lineup_position.strip().lower()]


def is_knockout(round_name: str) -> bool:
    name = round_name.strip()
    return bool(name) and name != "|" and not _NOT_KNOCKOUT.search(name)


def stint_events(events: pd.DataFrame) -> pd.DataFrame:
    """Goals, substitutions and sendings-off in the form ``stints.build_stints`` takes."""
    kind = pd.Series(None, index=events.index, dtype="object")
    kind[events["type"] == "Goals"] = "goal"
    kind[events["type"] == "Substitutions"] = "sub"
    sent_off = events["description"].fillna("").str.contains(_SENDING_OFF)
    kind[(events["type"] == "Cards") & sent_off] = "red"
    out = events.assign(kind=kind)[kind.notna()]
    return out[["game_id", "minute", "kind", "club_id", "player_id", "player_in_id"]]


def _per_side(games: pd.DataFrame, counts: pd.Series, side: str) -> pd.Series:
    """Look up a (game_id, club_id)-indexed count for each game's home or away club."""
    key = pd.MultiIndex.from_arrays([games["game_id"], games[f"{side}_club_id"]])
    return pd.Series(counts.reindex(key).fillna(0).to_numpy(int), index=games.index)


def build_games(games: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """tm_games columns (plus knockout) for every game given, shootout kicks removed."""
    scored = events[
        (events["type"] == "Shootout") & events["description"].fillna("").str.contains("Scored")
    ]
    kicks = scored.groupby(["game_id", "club_id"]).size()
    home_kicks, away_kicks = _per_side(games, kicks, "home"), _per_side(games, kicks, "away")
    net = (games["home_club_goals"] >= home_kicks) & (games["away_club_goals"] >= away_kicks)
    day = pd.to_datetime(games["date"])
    comp_type = games["competition_type"].fillna(
        games["competition_id"].map(UNLISTED_COMPETITION_TYPES)
    )
    final = day.groupby([games["competition_id"], games["season"]]).transform("max")
    own = games["season"].map(lambda start: season_label(date(int(start), 7, 1)))
    national = comp_type == "national_team_competition"
    season = own.where(~national, final.dt.date.map(season_label))
    return pd.DataFrame(
        {
            "game_id": games["game_id"],
            "competition_id": games["competition_id"],
            "competition_type": comp_type,
            "season": season,
            "date": day,
            "round": games["round"],
            "home_club_id": games["home_club_id"],
            "away_club_id": games["away_club_id"],
            "home_goals": games["home_club_goals"] - home_kicks.where(net, 0),
            "away_goals": games["away_club_goals"] - away_kicks.where(net, 0),
            "home_club_name": games["home_club_name"],
            "away_club_name": games["away_club_name"],
            "knockout": games["round"].fillna("").map(is_knockout),
        }
    )


STATS = ["goals", "assists", "yellow", "red", "penalty_goals"]
KEY = ["game_id", "player_id"]


def event_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, player_id) counts from game_events, in Transfermarkt's own conventions.

    Checked against the appearances table on stint games: goals exclude own goals
    (99.999% agreement); assists include assists to own goals (99.68%, vs 99.56%
    without); a second yellow counts as a yellow, not a red, so a player sent off for two
    yellows has yellow 2, red 0 (yellow 99.955%, red 99.937%).
    """
    description = events["description"].fillna("")
    goal = events["type"] == "Goals"
    own = description.str.contains(_OWN_GOAL)
    card = events["type"] == "Cards"
    assists = events.loc[goal, ["game_id", "player_assist_id"]]
    long = pd.concat(
        [
            events.loc[goal & ~own, KEY].assign(stat="goals"),
            assists.rename(columns={"player_assist_id": "player_id"}).assign(stat="assists"),
            events.loc[card & description.str.contains(_YELLOW), KEY].assign(stat="yellow"),
            events.loc[card & description.str.contains("Red card"), KEY].assign(stat="red"),
            events.loc[goal & ~own & description.str.contains(_PENALTY_GOAL), KEY].assign(
                stat="penalty_goals"
            ),
        ]
    ).dropna()
    long["player_id"] = long["player_id"].astype(int)
    return (
        long.groupby([*KEY, "stat"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATS, fill_value=0)
    )


def _stats_for(rows: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(rows[KEY])
    return stats.reindex(index, fill_value=0).set_index(rows.index)


def build_appearances(
    appearances: pd.DataFrame, lineups: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Transfermarkt's own appearance rows in the tm_appearances layout."""
    lineup = lineups.drop_duplicates(KEY)
    merged = appearances.merge(
        lineup[[*KEY, "type", "position", "team_captain"]], on=KEY, how="left"
    )
    listed = merged["type"].notna()
    return pd.DataFrame(
        {
            "game_id": merged["game_id"],
            "player_id": merged["player_id"],
            "club_id": merged["player_club_id"],
            "date": pd.to_datetime(merged["date"]),
            "competition_id": merged["competition_id"],
            "minutes": merged["minutes_played"],
            "goals": merged["goals"],
            "assists": merged["assists"],
            "yellow": merged["yellow_cards"],
            "red": merged["red_cards"],
            "started": (merged["type"] == "starting_lineup").astype("boolean").where(listed),
            "position": merged["position"],
            "captain": (merged["team_captain"] == 1).astype("boolean").where(listed),
            "penalty_goals": _stats_for(merged, event_stats(events))["penalty_goals"],
            "source": "tm_appearances",
        }
    )


def stint_appearances(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    events: pd.DataFrame,
    tm_stints: pd.DataFrame,
    tm_stint_players: pd.DataFrame,
) -> pd.DataFrame:
    """Appearance rows derived for every game with stints.

    One row per starter and per player brought on (a player brought on at the folded
    minute 90 has 0 minutes). Minutes come from the stints; started, position and
    captain from the lineups (a player brought on who is missing from the lineup has
    started False and no position or captain); counts from ``event_stats``.
    """
    ids = tm_stints["game_id"].unique()
    lineup = lineups[lineups["game_id"].isin(ids)].drop_duplicates(KEY)
    subs = events[(events["type"] == "Substitutions") & events["game_id"].isin(ids)]
    subs = subs.dropna(subset=["player_in_id"])
    rows = (
        pd.concat(
            [
                lineup.loc[lineup["type"] == "starting_lineup", [*KEY, "club_id"]],
                pd.DataFrame(
                    {
                        "game_id": subs["game_id"],
                        "player_id": subs["player_in_id"].astype(int),
                        "club_id": subs["club_id"],
                    }
                ),
            ]
        )
        .drop_duplicates(KEY)
        .merge(stints.player_minutes(tm_stints, tm_stint_players), on=KEY, how="left")
        .merge(lineup[[*KEY, "type", "position", "team_captain"]], on=KEY, how="left")
        .merge(games[["game_id", "date", "competition_id"]], on="game_id")
    )
    stats = _stats_for(rows, event_stats(events))
    listed = rows["type"].notna()
    return pd.DataFrame(
        {
            "game_id": rows["game_id"],
            "player_id": rows["player_id"],
            "club_id": rows["club_id"],
            "date": rows["date"],
            "competition_id": rows["competition_id"],
            "minutes": rows["minutes"].fillna(0).astype(int),
            **{stat: stats[stat] for stat in STATS[:4]},
            "started": (rows["type"] == "starting_lineup").astype("boolean"),
            "position": rows["position"],
            "captain": (rows["team_captain"] == 1).astype("boolean").where(listed),
            "penalty_goals": stats["penalty_goals"],
            "source": "stints",
        }
    )


def coverage_table(
    games: pd.DataFrame,
    appearances: pd.DataFrame,
    tm_rows: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """Per competition x season: games, games with stints, player-games (minutes >= 1),
    those player-games present only through stints, and the date range."""
    played = appearances[appearances["minutes"] >= 1]
    in_tm = pd.MultiIndex.from_frame(played[KEY]).isin(pd.MultiIndex.from_frame(tm_rows[KEY]))
    per_game = (
        played.assign(filled=(played["source"] == "stints") & ~in_tm)
        .groupby("game_id")
        .agg(player_games=("player_id", "size"), filled=("filled", "sum"))
    )
    frame = (
        games[["game_id", "competition_id", "season", "date"]]
        .merge(selection[["game_id", "has_stints"]], on="game_id")
        .join(per_game, on="game_id")
        .fillna({"player_games": 0, "filled": 0})
    )
    return (
        frame.groupby(["competition_id", "season"])
        .agg(
            games=("game_id", "size"),
            games_with_stints=("has_stints", "sum"),
            player_games=("player_games", "sum"),
            player_games_only_from_stints=("filled", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .astype({"player_games": int, "player_games_only_from_stints": int})
        .reset_index()
    )


def stint_game_selection(
    games: pd.DataFrame, lineups: pd.DataFrame, events: pd.DataFrame, appearances: pd.DataFrame
) -> pd.DataFrame:
    """Per game: starters per side, unknown-minute counts, end_min, and whether stints are built."""
    starters = lineups[lineups["type"] == "starting_lineup"].drop_duplicates(
        ["game_id", "player_id"]
    )
    counts = starters.groupby(["game_id", "club_id"]).size()
    norm = stint_events(events)
    unknown = norm[norm["minute"] < 0]
    unknown_changes = unknown[unknown["kind"] != "goal"].groupby("game_id").size()
    unknown_goals = unknown[unknown["kind"] == "goal"].groupby("game_id").size()
    timed = events[events["type"].isin(["Goals", "Substitutions", "Cards"])]
    last_event = games["game_id"].map(timed.groupby("game_id")["minute"].max())
    longest = games["game_id"].map(appearances.groupby("game_id")["minutes_played"].max())
    extra_time = (last_event.fillna(0) > 90) | (longest.fillna(0) > 90)
    out = games[["game_id", "home_club_id", "away_club_id"]].copy()
    out["home_starters"] = _per_side(games, counts, "home")
    out["away_starters"] = _per_side(games, counts, "away")
    out["complete"] = (out["home_starters"] == 11) & (out["away_starters"] == 11)
    out["unknown_minute_changes"] = out["game_id"].map(unknown_changes).fillna(0).astype(int)
    out["unknown_minute_goals"] = out["game_id"].map(unknown_goals).fillna(0).astype(int)
    out["end_min"] = np.where(extra_time, 120, 90)
    out["has_stints"] = out["complete"] & (out["unknown_minute_changes"] == 0)
    return out


def build_players(players: pd.DataFrame, keep: pd.Series) -> pd.DataFrame:
    kept = players[players["player_id"].isin(keep)]
    return pd.DataFrame(
        {
            "player_id": kept["player_id"],
            "name": kept["name"],
            "date_of_birth": pd.to_datetime(kept["date_of_birth"]).dt.date,
            "position": kept["position"],
            "sub_position": kept["sub_position"],
            "citizenship": kept["country_of_citizenship"],
        }
    )


def _window_mask(dates: pd.Series) -> pd.Series:
    return (dates >= pd.Timestamp(WINDOW_START)) & (dates <= pd.Timestamp(WINDOW_END))


# --- I/O ----------------------------------------------------------------------------


def download(root: Path = RAW_DIR) -> dict[str, str]:
    """Fetch every table not already on disk; write the manifest. Returns name -> status.

    A file whose bytes no longer match the SHA-256 already in the manifest raises:
    the cache is the provenance, so silent drift is an error.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    previous: dict[str, dict[str, object]] = {}
    if manifest_path.exists():
        previous = {e["name"]: e for e in json.loads(manifest_path.read_text())["files"]}

    status: dict[str, str] = {}
    for name in TABLES:
        url = f"{BASE_URL}/{name}.csv.gz"
        dest = root / f"{name}.csv.gz"
        entry = dict(previous.get(name, {"name": name, "url": url}))
        if dest.exists() and dest.stat().st_size > 0:
            status[name] = "cached"
        else:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            part = dest.with_suffix(".part")
            with (
                urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response,
                part.open("wb") as fh,
            ):
                shutil.copyfileobj(response, fh, 1 << 20)
                entry["last_modified"] = response.headers.get("Last-Modified")
            part.replace(dest)
            entry["retrieved_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            status[name] = "downloaded"
        digest = sha256_file(dest)
        if "sha256" in entry and entry["sha256"] != digest:
            raise ValueError(f"{dest} no longer matches its manifest SHA-256")
        entry.update(path=str(dest.relative_to(root)), sha256=digest, bytes=dest.stat().st_size)
        previous[name] = entry
        manifest_path.write_text(json.dumps({"files": list(previous.values())}, indent=2) + "\n")
    return status


def load(name: str, usecols: list[str], root: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(root / f"{name}.csv.gz", usecols=usecols, low_memory=False)


def build(root: Path = RAW_DIR, out: Path = PROCESSED) -> dict[str, pd.DataFrame]:
    """Write every processed table; return them plus the Elo report, the stint selection,
    Transfermarkt's own appearance rows (``tm_rows``, for validation) and the coverage."""
    raw_games = load(
        "games",
        [
            "game_id",
            "competition_id",
            "season",
            "round",
            "date",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
            "home_club_name",
            "away_club_name",
            "competition_type",
        ],
        root,
    )
    events = load(
        "game_events",
        [
            "game_id",
            "minute",
            "type",
            "club_id",
            "player_id",
            "description",
            "player_in_id",
            "player_assist_id",
        ],
        root,
    )
    lineups = load(
        "game_lineups",
        ["game_id", "player_id", "club_id", "type", "position", "team_captain"],
        root,
    )
    raw_apps = load(
        "appearances",
        [
            "game_id",
            "player_id",
            "player_club_id",
            "date",
            "competition_id",
            "yellow_cards",
            "red_cards",
            "goals",
            "assists",
            "minutes_played",
        ],
        root,
    )

    all_games = build_games(raw_games, events)
    games = all_games[_window_mask(all_games["date"])].reset_index(drop=True)
    window_ids = set(games["game_id"])
    events = events[events["game_id"].isin(window_ids)]
    lineups = lineups[lineups["game_id"].isin(window_ids)]
    raw_apps = raw_apps[raw_apps["game_id"].isin(window_ids)]

    selection = stint_game_selection(games, lineups, events, raw_apps)
    starters = lineups[lineups["type"] == "starting_lineup"].drop_duplicates(
        ["game_id", "player_id"]
    )
    tm_stints, tm_stint_players = stints.build_stints(
        selection[selection["has_stints"]],
        starters.assign(started=True),
        stint_events(events),
    )

    elo_games = all_games[all_games["date"] >= ELO_START].assign(
        pool=lambda g: (g["competition_type"] == "national_team_competition").map(
            {True: "national", False: "club"}
        )
    )
    ratings, elo_report = elo.team_elo(elo_games)

    tm_rows = build_appearances(raw_apps, lineups, events)
    derived = stint_appearances(games, lineups, events, tm_stints, tm_stint_players)
    appearances = pd.concat(
        [derived, tm_rows[~tm_rows["game_id"].isin(tm_stints["game_id"])]], ignore_index=True
    )

    valuations = load("player_valuations", ["player_id", "date", "market_value_in_eur"], root)
    valuations["date"] = pd.to_datetime(valuations["date"])
    tables = {
        "tm_games": games,
        "tm_appearances": appearances,
        "tm_players": build_players(
            load(
                "players",
                [
                    "player_id",
                    "name",
                    "date_of_birth",
                    "position",
                    "sub_position",
                    "country_of_citizenship",
                ],
                root,
            ),
            pd.concat([raw_apps["player_id"], lineups["player_id"], derived["player_id"]]),
        ),
        "tm_valuations": valuations[_window_mask(valuations["date"])].rename(
            columns={"market_value_in_eur": "market_value_eur"}
        ),
        "tm_stints": tm_stints,
        "tm_stint_players": tm_stint_players,
        "team_elo": ratings[ratings["game_id"].isin(window_ids)].reset_index(drop=True),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv.gz", index=False, date_format="%Y-%m-%d")
    return {
        **tables,
        "elo_report": elo_report,
        "stint_selection": selection,
        "tm_rows": tm_rows,
        "coverage": coverage_table(games, appearances, tm_rows, selection),
    }
