"""American Soccer Analysis (ASA) API: MLS seasons of pool players (pre-registration A20 vi).

Source. https://app.americansocceranalysis.com/api/v1, public and without
authentication; its documentation (``/api/v1/__docs__/``, OpenAPI at
``/api/v1/openapi.json``) states "Data returned by this API is licensed under CC BY-SA
4.0" (``docs/unmeasured/data-licences.md`` §5). The documentation lists the query
parameters but not the response fields; the fields named below were read from the
responses.

Download. One request per call of ``ensure``, gzipped on disk under
``data/players/raw/asa/`` and never re-fetched once cached. ``manifest.json`` records,
per file, the URL with its query, the UTC retrieval time and the SHA-256 of the bytes as
served (before gzip). A cached file is refused when the query asked for is not the one
it was fetched with. Responses are parsed with ``json`` only.

Fields and assumptions:

- ``mls/players``: ``player_id``, ``player_name``, ``birth_date``. A pool player is an
  ASA player only when the birth date is the same day and the name has the same words
  (``linking.normalise_name``, any order: "Son Heung-min" = "Heung-Min Son"), under the
  pool's name or Transfermarkt's. Pairs that share only one of the two are returned for
  review and never used.
- ``mls/players/xgoals`` with ``split_by_seasons`` and ``split_by_teams``: per player,
  team and ``season_name`` (the MLS calendar year), ``minutes_played``, ``goals`` and
  ``primary_assists`` (ASA glossary: "Primary assists"; the league's official assists
  add a secondary one, Transfermarkt's do not).
- Penalty goals: ``goals`` of the same query with ``shot_pattern=Penalty``, which keeps
  every row and counts only penalty shots. np_goals = goals - penalty_goals.
- Stage: ``stage_name=Regular Season`` only. The study's league scope is Transfermarkt's
  MLS1, whose games are the regular season's matchdays; ASA files the playoffs under
  another stage.
- Season label: the panel's (``study_season``).
- Minutes. ``minutes_played`` includes stoppage time (ASA glossary: "Minutes played
  (including stoppage time)"): about 100 for a full match, where Transfermarkt and the
  panel give 90. ``minutes`` keeps the value as served. ``minutes_scaled`` puts it on
  the panel's scale: per game, from the same endpoint with ``split_by_games``,
  ``minutes_played`` x 90 / ``expanded_minutes`` (the length of the game with stoppage
  time, from ``mls/games``). It is 90 for a full match and an estimate for a partial one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from footy.players.linking import normalise_name
from footy.players.paths import RAW
from footy.players.statsbomb import read_json
from footy.players.window import season_label

BASE_URL = "https://app.americansocceranalysis.com/api/v1"
LICENCE = "CC BY-SA 4.0"
USER_AGENT = "footy-research/0.1 (academic study)"
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 60
RAW_DIR = RAW / "asa"

Json = Any


def ensure(root: Path, name: str, endpoint: str, params: dict[str, str]) -> Json:
    """Rows of one API call, cached as ``<name>.json.gz`` and requested only if absent."""
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote, safe=",")
    url = f"{BASE_URL}/{endpoint}" + (f"?{query}" if query else "")
    dest, manifest_path = root / f"{name}.json.gz", root / "manifest.json"
    manifest: dict[str, Json] = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"source": BASE_URL, "licence": LICENCE, "files": {}}
    )
    if dest.exists():
        cached_url = manifest["files"][name]["url"]
        if cached_url != url:
            raise ValueError(f"{dest} is cached for another query: {cached_url}")
        return read_json(dest)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        body: bytes = response.read()
    rows = json.loads(body)
    if not isinstance(rows, list):
        raise ValueError(f"{url}: expected a JSON list, got {body[:200]!r}")
    root.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    tmp.replace(dest)
    manifest["files"][name] = {
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    time.sleep(REQUEST_DELAY_SECONDS)
    return rows


def study_season(year: int) -> str:
    """The panel's label of the MLS season played in ``year``.

    Transfermarkt files a calendar-year league under the season that starts the year
    before, and the panel keeps that (``transfermarkt.py``): MLS 2025 is "2024/25".
    """
    return season_label(date(year - 1, 7, 1))


def _name_key(name: str) -> str:
    return " ".join(sorted(normalise_name(name).split()))


def match_players(pool: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Pool x ASA player pairs that share a name or a birth date, one flag for each.

    ``pool``: tm_player_id, player, tm_name, date_of_birth. ``players``: the rows of
    ``mls/players``. Only a pair with both flags is a match.
    """
    left = pool[["tm_player_id", "player", "date_of_birth"]].assign(
        key_pool=pool["player"].map(_name_key), key_tm=pool["tm_name"].map(_name_key)
    )
    right = pd.DataFrame(
        {
            "asa_player_id": players["player_id"],
            "asa_player_name": players["player_name"],
            "asa_birth_date": players["birth_date"],
            "key": players["player_name"].map(_name_key),
        }
    )
    pairs = left.merge(right, how="cross")
    pairs["same_name"] = (pairs["key"] == pairs["key_pool"]) | (pairs["key"] == pairs["key_tm"])
    pairs["same_birth_date"] = pairs["asa_birth_date"] == pairs["date_of_birth"]
    keep = pairs["same_name"] | pairs["same_birth_date"]
    return pairs[keep].drop(columns=["key", "key_pool", "key_tm"]).reset_index(drop=True)


def player_seasons(
    matched: pd.DataFrame, rows: Json, penalty_rows: Json, teams: Json
) -> pd.DataFrame:
    """One row per matched player x MLS season, from ``mls/players/xgoals`` split by
    season and team.

    ``matched``: tm_player_id, player, asa_player_id. ``penalty_rows`` is the same query
    with ``shot_pattern=Penalty``. The clubs of one season are added up and their names
    joined with ";".
    """
    keys = ["player_id", "team_id", "season_name"]
    penalties = pd.DataFrame(penalty_rows)[[*keys, "goals"]]
    frame = (
        pd.DataFrame(rows)
        .merge(penalties.rename(columns={"goals": "penalty_goals"}), on=keys, validate="1:1")
        .merge(pd.DataFrame(teams)[["team_id", "team_name"]], on="team_id", validate="m:1")
        .merge(matched, left_on="player_id", right_on="asa_player_id", validate="m:1")
    )
    if len(frame) != len(rows):
        raise ValueError("a player-team-season has no penalty row, no team or no pool player")
    out = (
        frame.assign(asa_season_name=frame["season_name"].astype(int))
        .groupby(["tm_player_id", "player", "asa_player_id", "asa_season_name"], as_index=False)
        .agg(
            club=("team_name", lambda names: ";".join(sorted(names))),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            penalty_goals=("penalty_goals", "sum"),
            assists=("primary_assists", "sum"),
        )
    )
    out["season"] = out["asa_season_name"].map(study_season)
    out["np_goals"] = out["goals"] - out["penalty_goals"]
    return out[
        [
            "tm_player_id", "player", "asa_player_id", "season", "asa_season_name", "club",
            "minutes", "goals", "np_goals", "penalty_goals", "assists",
        ]
    ]  # fmt: skip


def player_games(
    matched: pd.DataFrame, game_rows: Json, game_penalty_rows: Json, games: Json
) -> pd.DataFrame:
    """One row per matched player x game, from ``mls/players/xgoals`` split by game and
    the rows of ``mls/games`` for those games.

    ``minutes`` is ``minutes_played`` (stoppage time included); ``minutes_scaled`` is
    ``minutes_played`` x 90 / ``expanded_minutes``. ``kickoff`` is UTC.
    """
    keys = ["player_id", "game_id"]
    penalties = pd.DataFrame(game_penalty_rows)[[*keys, "goals"]]
    info = pd.DataFrame(games)[["game_id", "date_time_utc", "season_name", "expanded_minutes"]]
    frame = (
        pd.DataFrame(game_rows)
        .merge(penalties.rename(columns={"goals": "penalty_goals"}), on=keys, validate="1:1")
        .merge(info, on="game_id", validate="m:1")
        .merge(matched, left_on="player_id", right_on="asa_player_id", validate="m:1")
    )
    if len(frame) != len(game_rows):
        raise ValueError("a player-game has no penalty row, no game or no pool player")
    return pd.DataFrame(
        {
            "tm_player_id": frame["tm_player_id"],
            "player": frame["player"],
            "asa_player_id": frame["asa_player_id"],
            "asa_game_id": frame["game_id"],
            "kickoff": pd.to_datetime(frame["date_time_utc"], utc=True),
            "asa_season_name": frame["season_name"].astype(int),
            "minutes": frame["minutes_played"],
            "minutes_scaled": frame["minutes_played"] * 90 / frame["expanded_minutes"],
            "goals": frame["goals"],
            "penalty_goals": frame["penalty_goals"],
            "assists": frame["primary_assists"],
        }
    )
