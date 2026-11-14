"""StatsBomb open data: download, cache and parse into processed tables (design §3, §4.3).

Download. Files come from the public open-data repository, one request at a time
with a short delay, gzipped on disk under ``data/players/raw/statsbomb/`` and never
re-fetched once cached. ``manifest.json`` records, per file, the URL, the UTC
retrieval time and the SHA-256 of the bytes as served (before gzip). The
competitions list is cached too, so the season ids are frozen at first download.
Downloaded files are parsed with ``json`` only; nothing in them is executed.

Parsing rules (assumptions, each stated once here):

- Ids. ``player_id`` is StatsBomb's player id and ``sb_match_id`` its match id;
  linking to Transfermarkt ids is done elsewhere (design §4.4). ``season`` is
  StatsBomb's season name verbatim (``2019/2020``, ``2022``).
- Penalty shoot-outs (period 5) are dropped from every table: they are not play and
  the final score excludes them. Periods 1-4 are kept.
- Coordinates are StatsBomb's 120 x 80 pitch, the acting team attacking towards
  x = 120; the opponent goal centre is (120, 40) and the opponent penalty area is
  x >= 102, 18 <= y <= 62.
- Minutes come from events, not from the lineups' ``positions`` (those are malformed in
  some matches: overlapping or inverted from/to, e.g. Euro 2020 Italy-Austria). A
  player is on from 0 if listed in a Starting XI event, or from the time of the
  Substitution event that brings him on; he is off at the first Substitution event
  that takes him off or the first Red Card / Second Yellow (Foul Committed or Bad
  Behaviour), else at the end of period 4. Time is elapsed playing time: lengths of
  earlier periods plus the event's period-relative timestamp; a period's length is
  its last event's timestamp, so stoppage time counts. Temporary absences (Player
  Off / Player On, e.g. treatment) are not subtracted. Players never on the pitch
  get no row. Lineup files are cached for linking but not used here.
- Open play: a pass whose StatsBomb pass type is not Corner, Free Kick, Goal Kick,
  Kick Off or Throw-in. A pass is completed when it has no outcome. Carries have
  no outcome in StatsBomb and all count as successful.
- Progressive: a completed open-play pass or any carry that starts at x >= 48 (outside
  the actor's own 40% of the pitch) and ends at most 0.75 times as far from the
  opponent goal centre as it started (at least 25% closer). No minimum length.
- passes_into_box: completed open-play passes ending in the opponent penalty area
  from outside it. key_passes: passes (set pieces included) flagged shot_assist or
  goal_assist. sb_xa: StatsBomb xG of the shots those passes assisted, via the
  shot's key_pass_id.
- tackles: Duel events of duel type Tackle, any outcome. interceptions, blocks,
  clearances, pressures: event counts, any outcome. dribbles_completed: Dribble
  events with outcome Complete.
- touches_in_box: a touch is a spell of consecutive on-ball events (completed Ball
  Receipt*, Carry, Pass, Shot, Dribble, Miscontrol, Ball Recovery) by one player in
  one possession, so receive-carry-shoot is one touch; it is in the box when any of
  its events starts in the opponent penalty area.
- xT (``footy.players.xt``) is fitted on open-play shots (shot type Open Play; goal =
  outcome Goal) and open-play passes and carries of every downloaded match.
  xt_added credits a player with xT(end) - xT(start) of each completed open-play
  pass and each carry; failed passes and everything else get 0 (Singh 2019).
- shots and sb_xg in ``sb_player_match`` include penalties; ``sb_shots.shot_type``
  lets the caller drop them. Goals = Shot outcome Goal plus "Own Goal For" events.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from footy.players import xt
from footy.players.paths import RAW

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
USER_AGENT = "footy-research/0.1 (academic study; statsbomb open data)"
REQUEST_DELAY_SECONDS = 0.2
TIMEOUT_SECONDS = 60
RAW_DIR = RAW / "statsbomb"

# Men's competition-seasons in the study (design §3), as named in competitions.json.
SEASONS: tuple[tuple[str, str], ...] = (
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2020"),
    ("UEFA Euro", "2024"),
    ("Copa America", "2024"),
    ("African Cup of Nations", "2023"),
    ("La Liga", "2019/2020"),
    ("La Liga", "2020/2021"),
    ("Ligue 1", "2021/2022"),
    ("Ligue 1", "2022/2023"),
    ("1. Bundesliga", "2023/2024"),
    ("Major League Soccer", "2023"),
)

Json = Any


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                body: bytes = response.read()
                return body
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise AssertionError("unreachable")


def ensure(root: Path, rel: str, manifest: dict[str, Json], delay: float) -> Path:
    """Path of the cached gzip of ``rel``; downloads it first if it is not on disk."""
    dest = root / f"{rel}.gz"
    if dest.exists():
        return dest
    url = f"{BASE_URL}/{rel}"
    body = _download(url)
    json.loads(body)  # refuse to cache anything that is not JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(gzip.compress(body, mtime=0))
    tmp.replace(dest)
    manifest[rel] = {
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    time.sleep(delay)
    return dest


def read_json(path: Path) -> Json:
    return json.loads(gzip.decompress(path.read_bytes()))


def _load_manifest(root: Path) -> dict[str, Json]:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    files: dict[str, Json] = json.loads(path.read_text())["files"]
    return files


def _save_manifest(root: Path, manifest: dict[str, Json]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    body = {"source": BASE_URL, "files": dict(sorted(manifest.items()))}
    (root / "manifest.json").write_text(json.dumps(body, indent=2) + "\n")


def resolve_seasons(competitions: Json) -> list[tuple[str, str, int, int]]:
    """(competition, season, competition_id, season_id) for every study season.

    Raises if a season is missing or ambiguous, so coverage never shrinks silently.
    """
    out = []
    for name, season in SEASONS:
        hits = [
            c
            for c in competitions
            if c["competition_name"] == name
            and c["season_name"] == season
            and c["competition_gender"] == "male"
        ]
        if len(hits) != 1:
            raise ValueError(f"{name} {season}: {len(hits)} entries in competitions.json")
        out.append((name, season, int(hits[0]["competition_id"]), int(hits[0]["season_id"])))
    return out


def acquire(root: Path = RAW_DIR, *, delay: float = REQUEST_DELAY_SECONDS) -> int:
    """Download every file the study needs (sequentially); return the number of matches."""
    manifest = _load_manifest(root)
    n_matches = 0
    try:
        competitions = read_json(ensure(root, "competitions.json", manifest, delay))
        for _, _, cid, sid in resolve_seasons(competitions):
            matches = read_json(ensure(root, f"matches/{cid}/{sid}.json", manifest, delay))
            for match in matches:
                for kind in ("events", "lineups"):
                    ensure(root, f"{kind}/{match['match_id']}.json", manifest, delay)
                n_matches += 1
                if n_matches % 25 == 0:
                    _save_manifest(root, manifest)
    finally:
        _save_manifest(root, manifest)
    return n_matches


# --- parsing -------------------------------------------------------------------------

SET_PIECES = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in")
ON_BALL = ("Ball Receipt*", "Carry", "Pass", "Shot", "Dribble", "Miscontrol", "Ball Recovery")
GROUP_STAGES = ("Regular Season", "Group Stage")
SENDING_OFF = ("Red Card", "Second Yellow")
GOAL = (120.0, 40.0)
_DETAIL = {
    "Pass": "pass",
    "Carry": "carry",
    "Shot": "shot",
    "Dribble": "dribble",
    "Duel": "duel",
    "Interception": "interception",
    "Ball Receipt*": "ball_receipt",
}
EVENT_COLUMNS = [
    "sb_match_id", "id", "index", "period", "minute", "second", "type", "team",
    "player_id", "player_name", "possession", "x", "y", "end_x", "end_y",
    "outcome", "kind", "body_part", "technique", "key_pass", "key_pass_id", "xg",
]  # fmt: skip
SHOT_COLUMNS = [
    "sb_match_id", "minute", "second", "player_id", "player_name", "team", "x", "y",
    "body_part", "shot_type", "technique", "outcome", "sb_xg",
]  # fmt: skip
COUNT_COLUMNS = [
    "passes", "passes_completed", "progressive_passes", "progressive_carries",
    "passes_into_box", "key_passes", "sb_xa", "pressures", "tackles", "interceptions",
    "blocks", "clearances", "dribbles_completed", "touches_in_box", "xt_added", "shots",
    "sb_xg",
]  # fmt: skip
PLAYER_COLUMNS = ["sb_match_id", "player_id", "player_name", "team", "minutes", *COUNT_COLUMNS]
MATCH_COLUMNS = [
    "sb_match_id", "competition", "season", "date", "home_team", "away_team",
    "home_score", "away_score", "stage", "knockout",
]  # fmt: skip

BoolArray = npt.NDArray[np.bool_]


def _name(detail: Json, key: str) -> str | None:
    value = detail.get(key)
    return None if value is None else str(value["name"])


def _timestamp_seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split(":")
    return 3600.0 * int(hours) + 60.0 * int(minutes) + float(seconds)


def matches_table(competition: str, season: str, matches: Json) -> pd.DataFrame:
    rows = [
        (
            m["match_id"],
            competition,
            season,
            m["match_date"],
            m["home_team"]["home_team_name"],
            m["away_team"]["away_team_name"],
            m["home_score"],
            m["away_score"],
            m["competition_stage"]["name"],
            m["competition_stage"]["name"] not in GROUP_STAGES,
        )
        for m in matches
    ]
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def flatten_events(match_id: int, events: Json) -> pd.DataFrame:
    """One row per event of periods 1-4, with the fields the tables need."""
    rows = []
    for e in events:
        if e["period"] > 4:
            continue
        detail = e.get(_DETAIL.get(e["type"]["name"], ""), {})
        loc = e.get("location") or (np.nan, np.nan)
        end = detail.get("end_location") or (np.nan, np.nan)
        player = e.get("player") or {}
        rows.append(
            (
                match_id,
                e["id"],
                e["index"],
                e["period"],
                e["minute"],
                e["second"],
                e["type"]["name"],
                e["team"]["name"],
                player.get("id"),
                player.get("name"),
                e["possession"],
                loc[0],
                loc[1],
                end[0],
                end[1],
                _name(detail, "outcome"),
                _name(detail, "type"),
                _name(detail, "body_part"),
                _name(detail, "technique"),
                bool(detail.get("shot_assist") or detail.get("goal_assist")),
                detail.get("key_pass_id"),
                detail.get("statsbomb_xg"),
            )
        )
    flat = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    flat["player_id"] = flat["player_id"].astype("Int64")
    flat[["x", "y", "end_x", "end_y", "xg"]] = flat[["x", "y", "end_x", "end_y", "xg"]].astype(
        float
    )
    return flat


def period_lengths(events: Json) -> dict[int, float]:
    """Seconds played in each of periods 1-4: the period's last event timestamp."""
    out: dict[int, float] = {}
    for e in events:
        if e["period"] <= 4:
            out[e["period"]] = max(out.get(e["period"], 0.0), _timestamp_seconds(e["timestamp"]))
    return out


def event_minutes(match_id: int, events: Json) -> pd.DataFrame:
    """Minutes on the pitch per player, from Starting XI, Substitution and card events."""
    lengths = period_lengths(events)
    total = sum(lengths.values())

    def elapsed(event: Json) -> float:
        period = event["period"]
        if period > 4:
            return total
        before = sum(v for p, v in lengths.items() if p < period)
        return before + min(_timestamp_seconds(event["timestamp"]), lengths[period])

    on: dict[int, float] = {}
    off: dict[int, float] = {}
    who: dict[int, tuple[str, str]] = {}
    for e in events:
        etype, team = e["type"]["name"], e["team"]["name"]
        if etype == "Starting XI":
            for slot in e["tactics"]["lineup"]:
                on[slot["player"]["id"]] = 0.0
                who[slot["player"]["id"]] = (slot["player"]["name"], team)
        elif etype == "Substitution":
            off.setdefault(e["player"]["id"], elapsed(e))
            replacement = e["substitution"]["replacement"]
            on.setdefault(replacement["id"], elapsed(e))
            who.setdefault(replacement["id"], (replacement["name"], team))
        elif "player" in e:
            card = (e.get("foul_committed") or e.get("bad_behaviour") or {}).get("card")
            if card is not None and card["name"] in SENDING_OFF:
                off.setdefault(e["player"]["id"], elapsed(e))
    rows = [
        (match_id, pid, *who[pid], max(off.get(pid, total) - start, 0.0) / 60.0)
        for pid, start in on.items()
    ]
    out = pd.DataFrame(rows, columns=["sb_match_id", "player_id", "player_name", "team", "minutes"])
    out["player_id"] = out["player_id"].astype("Int64")
    return out


def in_box(x: npt.ArrayLike, y: npt.ArrayLike) -> BoolArray:
    """Inside the opponent penalty area (x >= 102, 18 <= y <= 62)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out: BoolArray = (xa >= 102.0) & (ya >= 18.0) & (ya <= 62.0)
    return out


def is_progressive(
    x: npt.ArrayLike, y: npt.ArrayLike, end_x: npt.ArrayLike, end_y: npt.ArrayLike
) -> BoolArray:
    """Starts at x >= 48 and ends at least 25% closer to the opponent goal centre."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    before = np.hypot(GOAL[0] - xa, GOAL[1] - ya)
    after = np.hypot(
        GOAL[0] - np.asarray(end_x, dtype=float), GOAL[1] - np.asarray(end_y, dtype=float)
    )
    out: BoolArray = (xa >= 48.0) & (after <= 0.75 * before)
    return out


def shots_table(flat: pd.DataFrame) -> pd.DataFrame:
    shots = flat[flat["type"] == "Shot"].rename(columns={"kind": "shot_type", "xg": "sb_xg"})
    return shots[SHOT_COLUMNS].reset_index(drop=True)


def own_goals(flat: pd.DataFrame) -> pd.DataFrame:
    """Own goals per (match, benefiting team), from "Own Goal For" events."""
    og = flat[flat["type"] == "Own Goal For"]
    return og.groupby(["sb_match_id", "team"]).size().rename("own_goals").reset_index()


def moves(flat: pd.DataFrame) -> pd.DataFrame:
    """Open-play passes (with completion) and carries: the xT move attempts."""
    is_pass = (flat["type"] == "Pass") & ~flat["kind"].isin(SET_PIECES)
    is_carry = flat["type"] == "Carry"
    out = flat.loc[is_pass | is_carry, ["sb_match_id", "player_id", "x", "y", "end_x", "end_y"]]
    out["success"] = (is_carry | flat["outcome"].isna())[is_pass | is_carry]
    return out.reset_index(drop=True)


def _touches_in_box(flat: pd.DataFrame) -> pd.Series:
    receipt_failed = (flat["type"] == "Ball Receipt*") & flat["outcome"].notna()
    ball = flat[flat["type"].isin(ON_BALL) & ~receipt_failed].sort_values(["sb_match_id", "index"])
    new = (
        (ball["player_id"] != ball["player_id"].shift())
        | (ball["possession"] != ball["possession"].shift())
        | (ball["sb_match_id"] != ball["sb_match_id"].shift())
    )
    spells = pd.DataFrame(
        {
            "sb_match_id": ball["sb_match_id"],
            "player_id": ball["player_id"],
            "spell": new.cumsum(),
            "box": in_box(ball["x"], ball["y"]),
        }
    )
    per_spell = spells.groupby(["sb_match_id", "player_id", "spell"])["box"].any()
    return per_spell.groupby(level=["sb_match_id", "player_id"]).sum().rename("touches_in_box")


def player_counts(flat: pd.DataFrame) -> pd.DataFrame:
    """Per (match, player) event counts; xt_added is left at 0 for ``add_xt``."""
    kind, outcome, etype = flat["kind"], flat["outcome"], flat["type"]
    is_pass = etype == "Pass"
    completed_open = is_pass & outcome.isna() & ~kind.isin(SET_PIECES)
    progressive = is_progressive(flat["x"], flat["y"], flat["end_x"], flat["end_y"])
    shot = etype == "Shot"
    xg_by_pass = flat[shot].groupby("key_pass_id")["xg"].sum()
    entering = in_box(flat["end_x"], flat["end_y"]) & ~in_box(flat["x"], flat["y"])
    counts = pd.DataFrame(
        {
            "sb_match_id": flat["sb_match_id"],
            "player_id": flat["player_id"],
            "passes": is_pass,
            "passes_completed": is_pass & outcome.isna(),
            "progressive_passes": completed_open & progressive,
            "progressive_carries": (etype == "Carry") & progressive,
            "passes_into_box": completed_open & entering,
            "key_passes": is_pass & flat["key_pass"],
            "sb_xa": flat["id"].map(xg_by_pass).where(is_pass, 0.0).fillna(0.0),
            "pressures": etype == "Pressure",
            "tackles": (etype == "Duel") & (kind == "Tackle"),
            "interceptions": etype == "Interception",
            "blocks": etype == "Block",
            "clearances": etype == "Clearance",
            "dribbles_completed": (etype == "Dribble") & (outcome == "Complete"),
            "xt_added": 0.0,
            "shots": shot,
            "sb_xg": flat["xg"].where(shot, 0.0),
        }
    )
    counts = counts[counts["player_id"].notna()]
    out = counts.groupby(["sb_match_id", "player_id"]).sum()
    return out.join(_touches_in_box(flat), how="left").reset_index()


def player_match_table(minutes: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Minutes joined to event counts; a player with no counted event gets zeros."""
    out = minutes.merge(counts, on=["sb_match_id", "player_id"], how="left")
    out[COUNT_COLUMNS] = out[COUNT_COLUMNS].fillna(0)
    int_columns = [c for c in COUNT_COLUMNS if c not in ("sb_xa", "xt_added", "sb_xg")]
    out[int_columns] = out[int_columns].astype(int)
    return out[PLAYER_COLUMNS]


def fit_xt(shots: pd.DataFrame, attempts: pd.DataFrame) -> xt.XTGrid:
    open_play = shots[shots["shot_type"] == "Open Play"]
    return xt.fit(
        open_play["x"],
        open_play["y"],
        open_play["outcome"] == "Goal",
        attempts["x"],
        attempts["y"],
        attempts["end_x"],
        attempts["end_y"],
        attempts["success"],
    )


def add_xt(player_match: pd.DataFrame, attempts: pd.DataFrame, grid: xt.XTGrid) -> pd.DataFrame:
    """Fill xt_added: sum of xT(end) - xT(start) over each player's successful moves."""
    ok = attempts[attempts["success"]]
    gained = ok.assign(xt=xt.xt_added(grid, ok["x"], ok["y"], ok["end_x"], ok["end_y"]))
    per_player = gained.groupby(["sb_match_id", "player_id"])["xt"].sum()
    keys = pd.MultiIndex.from_frame(player_match[["sb_match_id", "player_id"]])
    out = player_match.copy()
    out["xt_added"] = per_player.reindex(keys).fillna(0.0).to_numpy()
    return out


@dataclass(frozen=True)
class Tables:
    matches: pd.DataFrame
    shots: pd.DataFrame
    player_match: pd.DataFrame
    own_goals: pd.DataFrame
    match_minutes: pd.Series  # match length in minutes (periods 1-4), by sb_match_id
    on_pitch_without_events: int
    event_players_without_minutes: int
    grid: xt.XTGrid
    attempts: pd.DataFrame


def build(root: Path = RAW_DIR) -> Tables:
    """Parse every cached file into the processed tables and fit xT on all of them."""
    competitions = read_json(root / "competitions.json.gz")
    match_frames, shot_frames, og_frames, player_frames, move_frames = [], [], [], [], []
    lengths: dict[int, float] = {}
    no_events = no_minutes = 0
    for competition, season, cid, sid in resolve_seasons(competitions):
        matches = read_json(root / f"matches/{cid}/{sid}.json.gz")
        match_frames.append(matches_table(competition, season, matches))
        for match in matches:
            mid = int(match["match_id"])
            events = read_json(root / f"events/{mid}.json.gz")
            periods = period_lengths(events)
            lengths[mid] = sum(periods.values()) / 60.0
            flat = flatten_events(mid, events)
            minutes = event_minutes(mid, events)
            counts = player_counts(flat)
            shot_frames.append(shots_table(flat))
            og_frames.append(own_goals(flat))
            move_frames.append(moves(flat))
            player_frames.append(player_match_table(minutes, counts))
            no_events += len(set(minutes["player_id"]) - set(counts["player_id"]))
            no_minutes += len(set(counts["player_id"]) - set(minutes["player_id"]))
    shots = pd.concat(shot_frames, ignore_index=True)
    attempts = pd.concat(move_frames, ignore_index=True)
    grid = fit_xt(shots, attempts)
    player_match = add_xt(pd.concat(player_frames, ignore_index=True), attempts, grid)
    return Tables(
        matches=pd.concat(match_frames, ignore_index=True),
        shots=shots,
        player_match=player_match,
        own_goals=pd.concat(og_frames, ignore_index=True),
        match_minutes=pd.Series(lengths, name="match_minutes").rename_axis("sb_match_id"),
        on_pitch_without_events=no_events,
        event_players_without_minutes=no_minutes,
        grid=grid,
        attempts=attempts,
    )


def grid_frame(grid: xt.XTGrid) -> pd.DataFrame:
    """Long form of the grid: one row per cell with its pitch bounds."""
    i, j = np.meshgrid(np.arange(xt.N_X), np.arange(xt.N_Y), indexing="ij")
    dx, dy = xt.LENGTH / xt.N_X, xt.WIDTH / xt.N_Y
    return pd.DataFrame(
        {
            "cell_x": i.ravel(),
            "cell_y": j.ravel(),
            "x_lo": i.ravel() * dx,
            "x_hi": (i.ravel() + 1) * dx,
            "y_lo": j.ravel() * dy,
            "y_hi": (j.ravel() + 1) * dy,
            "xt": grid.values.ravel(),
        }
    )
