"""Link StatsBomb records to Transfermarkt ids (design §4.4). Pure.

Names. ``normalise_name`` folds accents (NFKD) and the letters NFKD leaves alone (ß ø æ
ł đ ı ð þ œ), lowercases, deletes apostrophes and turns every other non-alphanumeric
character into a space, so "N'Golo Kanté" -> "ngolo kante" and "Heung-Min Son" ->
"heung min son". ``name_similarity`` ignores word order and lets either name be the
longer one, because StatsBomb gives full legal names ("Rodrigo Hernández Cascante") and
Transfermarkt short ones ("Rodri"):

- tokens are the words minus the particles in ``PARTICLES`` (al, de, van, ...), unless
  nothing else is left;
- 1.0 when the two token multisets are equal;
- otherwise the larger of (a) 0.95 x the mean, over the tokens of the shorter name, of
  each token's best match in the longer one: 1.0 equal, 0.9 when one is a prefix of the
  other and both have at least 4 letters ("rodri"/"rodrigo"), else the difflib ratio
  when it is at least 0.8 ("mohamed"/"mohammed"), else 0; and (b) 0.9 x the difflib
  ratio of the two whole normalised names (transliterations such as "Hassan Al Heidos"
  / "Hasan Al-Haydos" that no single token match catches).
A StatsBomb nickname ("Rodri", "Vinícius Júnior"), when present, is scored too and the
higher score is kept.

Teams. ``normalise_team`` applies ``normalise_name``, drops digits and generic club words
(``GENERIC_TEAM_WORDS``) and maps the remaining spelling to ``TEAM_ALIASES`` (the
StatsBomb/Transfermarkt pairs that share no word, e.g. "Côte d'Ivoire"/"Ivory Coast").
``team_similarity`` is the share of the shorter name's words found in the longer one
("Lyon" vs "Olympique Lyon" = 1).

Games (``link_sb_games``). Candidates are Transfermarkt games dated within one day of the
StatsBomb date. A candidate's score is the lower of its two team similarities, in the
better of the two orientations (neutral-venue tournaments may list home and away the
other way round; ``swapped`` records it); ties break on the mean of the two. The best
candidate is linked when its score is at least ``GAME_THRESHOLD`` and no other candidate
reaches the same (score, tie-break); otherwise the match is ``ambiguous`` or
``no_candidate``.

Players (``link_sb_players``), per StatsBomb player id, in tiers:

1. ``game``: inside one linked game and one side, StatsBomb players against the
   Transfermarkt players attached to that side (lineups, or events where there are no
   lineups) by a one-to-one assignment maximising total similarity
   (``scipy.optimize.linear_sum_assignment``); pairs under ``PLAYER_THRESHOLD`` are
   dropped. A StatsBomb id takes the Transfermarkt id it was assigned in most games
   (ties: higher mean score); every disagreement is reported.
2. ``team_pool``: still unlinked ids against every Transfermarkt player attached to the
   same Transfermarkt club or national team anywhere in the data given;
3. ``country``: national-team players still unlinked against every Transfermarkt player
   whose citizenship is that country.
Tiers 2 and 3 need a score of at least ``POOL_THRESHOLD`` and a lead of at least
``AMBIGUITY_MARGIN`` over the runner-up, and never reuse a Transfermarkt id already
linked to another StatsBomb id. An id linked by tier 1 whose runner-up was within
``AMBIGUITY_MARGIN`` in some game is listed as ambiguous (kept, but reported).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

GAME_THRESHOLD = 0.75
PLAYER_THRESHOLD = 0.6
POOL_THRESHOLD = 0.85
AMBIGUITY_MARGIN = 0.05

_TRANSLITERATE = str.maketrans(
    {
        "ß": "ss",
        "ø": "o",
        "Ø": "O",
        "æ": "ae",
        "Æ": "AE",
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "Đ": "D",
        "ı": "i",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "œ": "oe",
        "Œ": "OE",
    }
)
PARTICLES = frozenset(
    {"al", "el", "de", "da", "do", "dos", "das", "di", "del", "della", "van", "von", "der",
     "den", "ter", "la", "le", "bin", "ben"}
)  # fmt: skip
GENERIC_TEAM_WORDS = frozenset(
    {"fc", "cf", "sc", "ac", "as", "afc", "cd", "ud", "sd", "rc", "rcd", "club", "de", "del",
     "of", "the", "football", "fussballclub"}
)  # fmt: skip
TEAM_ALIASES = {
    "cote divoire": "ivory coast",
    "congo dr": "democratic republic congo",
    "turkey": "turkiye",
    "korea south": "south korea",
    "rennes": "stade rennais",
    "usa": "united states",
}


def normalise_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.translate(_TRANSLITERATE))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"['’`´]", "", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _token_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio if ratio >= 0.8 else 0.0


def _tokens(normalised: str) -> list[str]:
    words = normalised.split()
    return [w for w in words if w not in PARTICLES] or words


def name_similarity(a: str, b: str) -> float:
    na, nb = normalise_name(a), normalise_name(b)
    if not na or not nb:
        return 0.0
    ta, tb = _tokens(na), _tokens(nb)
    if sorted(ta) == sorted(tb):
        return 1.0
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    best = [max(_token_similarity(t, u) for u in long) for t in short]
    return max(0.95 * sum(best) / len(short), 0.9 * SequenceMatcher(None, na, nb).ratio())


def best_name_similarity(names: Iterable[str | None], other: str) -> float:
    """Highest similarity of ``other`` to any of ``names`` (missing names skipped)."""
    return max((name_similarity(n, other) for n in names if isinstance(n, str)), default=0.0)


def normalise_team(name: str) -> str:
    words = [w for w in normalise_name(name).split() if w not in GENERIC_TEAM_WORDS]
    key = " ".join(w for w in words if not w.isdigit())
    return TEAM_ALIASES.get(key, key)


def team_similarity(a: str, b: str) -> float:
    wa, wb = set(normalise_team(a).split()), set(normalise_team(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


# --- games ----------------------------------------------------------------------------

GAME_LINK_COLUMNS = ["sb_match_id", "game_id", "swapped", "score", "n_candidates", "status"]


def link_sb_games(sb_matches: pd.DataFrame, tm_games: pd.DataFrame) -> pd.DataFrame:
    """One row per StatsBomb match: the Transfermarkt game it is linked to, if any.

    sb_matches: sb_match_id, date, home_team, away_team.
    tm_games: game_id, date, home_club_name, away_club_name.
    """
    tm = tm_games.assign(day=pd.to_datetime(tm_games["date"]).dt.normalize())
    tm = tm.fillna({"home_club_name": "", "away_club_name": ""})
    by_day = {day: group for day, group in tm.groupby("day")}
    rows: list[tuple[Any, ...]] = []
    for mid, day, home, away in sb_matches[
        ["sb_match_id", "date", "home_team", "away_team"]
    ].itertuples(index=False, name=None):
        stamp = pd.Timestamp(day).normalize()
        near = [
            by_day[d] for d in (stamp + pd.Timedelta(days=k) for k in (-1, 0, 1)) if d in by_day
        ]
        cands = pd.concat(near) if near else tm.iloc[:0]
        scored = []
        for gid, th, ta in cands[["game_id", "home_club_name", "away_club_name"]].itertuples(
            index=False, name=None
        ):
            straight = (team_similarity(home, th), team_similarity(away, ta))
            crossed = (team_similarity(home, ta), team_similarity(away, th))
            for swapped, pair in ((False, straight), (True, crossed)):
                scored.append((min(pair), sum(pair) / 2, gid, swapped))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < GAME_THRESHOLD:
            rows.append(
                (mid, None, None, scored[0][0] if scored else None, len(cands), "no_candidate")
            )
            continue
        top = scored[0]
        rivals = [s for s in scored[1:] if s[2] != top[2] and s[:2] == top[:2]]
        status = "ambiguous" if rivals else "linked"
        rows.append((mid, top[2] if not rivals else None, top[3], top[0], len(cands), status))
    out = pd.DataFrame(rows, columns=GAME_LINK_COLUMNS)
    out["game_id"] = out["game_id"].astype("Int64")
    return out


def team_clubs(
    sb_matches: pd.DataFrame, game_links: pd.DataFrame, tm_games: pd.DataFrame
) -> pd.DataFrame:
    """Per linked StatsBomb match and team: the Transfermarkt game and club id."""
    linked = game_links[game_links["status"] == "linked"]
    frame = sb_matches[["sb_match_id", "home_team", "away_team"]].merge(
        linked[["sb_match_id", "game_id", "swapped"]], on="sb_match_id"
    )
    frame = frame.merge(tm_games[["game_id", "home_club_id", "away_club_id"]], on="game_id")
    swapped = frame["swapped"].astype(bool)
    home_club = frame["home_club_id"].where(~swapped, frame["away_club_id"])
    away_club = frame["away_club_id"].where(~swapped, frame["home_club_id"])
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "sb_match_id": frame["sb_match_id"],
                    "team": frame[f"{side}_team"],
                    "game_id": frame["game_id"].astype(int),
                    "club_id": club.astype(int),
                }
            )
            for side, club in (("home", home_club), ("away", away_club))
        ],
        ignore_index=True,
    )


# --- players --------------------------------------------------------------------------

MAPPING_COLUMNS = ["sb_player_id", "tm_player_id", "method", "score", "runner_up", "games"]


def _scores(sb: pd.DataFrame, names: list[str]) -> np.ndarray:
    return np.array(
        [
            [best_name_similarity((full, nick), name) for name in names]
            for full, nick in sb[["player_name", "player_nickname"]].itertuples(
                index=False, name=None
            )
        ]
    ).reshape(len(sb), len(names))


def _in_game_votes(rows: pd.DataFrame, tm_game_players: pd.DataFrame) -> pd.DataFrame:
    """Tier 1: one-to-one assignment inside each (game, club)."""
    pools = {key: group for key, group in tm_game_players.groupby(["game_id", "club_id"])}
    votes = []
    for (game_id, club_id), sb in rows.groupby(["game_id", "club_id"]):
        tm = pools.get((game_id, club_id))
        if tm is None:
            continue
        tm = tm.drop_duplicates("player_id")
        scores = _scores(sb, tm["name"].tolist())
        for i, j in zip(*linear_sum_assignment(-scores), strict=True):
            if scores[i, j] < PLAYER_THRESHOLD:
                continue
            runner = np.delete(scores[i], j).max(initial=0.0)
            votes.append(
                (sb["sb_player_id"].iloc[i], tm["player_id"].iloc[j], scores[i, j], runner)
            )
    return pd.DataFrame(votes, columns=["sb_player_id", "tm_player_id", "score", "runner_up"])


def _pool_match(
    sb: pd.Series, pool: pd.DataFrame, used: set[int]
) -> tuple[int | None, float, float]:
    """Best pool player for one StatsBomb player: (id or None, score, runner-up)."""
    pool = pool[~pool["player_id"].isin(used)].drop_duplicates("player_id")
    if pool.empty:
        return None, 0.0, 0.0
    scores = _scores(sb.to_frame().T, pool["name"].tolist())[0]
    order = np.argsort(-scores)
    best = float(scores[order[0]])
    runner = float(scores[order[1]]) if len(order) > 1 else 0.0
    if best < POOL_THRESHOLD or best - runner < AMBIGUITY_MARGIN:
        return None, best, runner
    return int(pool["player_id"].iloc[order[0]]), best, runner


def link_sb_players(
    sb_players: pd.DataFrame,
    sb_teams: pd.DataFrame,
    tm_game_players: pd.DataFrame,
    tm_team_players: pd.DataFrame,
    tm_country_players: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map StatsBomb player ids to Transfermarkt ids; return (mapping, issues).

    sb_players: sb_match_id, sb_player_id, player_name, player_nickname, team.
    sb_teams: sb_match_id, team, game_id, club_id (``team_clubs``).
    tm_game_players: game_id, club_id, player_id, name (who Transfermarkt puts in that game).
    tm_team_players: club_id, player_id, name.  tm_country_players: country, player_id,
    name, with ``country`` already passed through ``normalise_team``.

    mapping: one row per StatsBomb id seen in a linked match (``MAPPING_COLUMNS``;
    tm_player_id missing when unlinked). issues: kind in {conflict, ambiguous, unlinked},
    sb_player_id, detail.
    """
    rows = sb_players.merge(sb_teams, on=["sb_match_id", "team"])
    votes = _in_game_votes(rows, tm_game_players)
    issues: list[tuple[str, Any, str]] = []

    tally = (
        votes.groupby(["sb_player_id", "tm_player_id"])
        .agg(games=("score", "size"), score=("score", "mean"), runner_up=("runner_up", "max"))
        .reset_index()
        .sort_values(["games", "score"], ascending=False)
    )
    for sb_id, group in tally.groupby("sb_player_id"):
        if len(group) > 1:
            issues.append(("conflict", sb_id, f"tm ids {group['tm_player_id'].tolist()}"))
    best = tally.drop_duplicates("sb_player_id")
    taken = best.drop_duplicates("tm_player_id")
    for tm_id, group in best.groupby("tm_player_id"):
        if len(group) > 1:
            issues.append(("conflict", group["sb_player_id"].iloc[1], f"tm id {tm_id} taken"))
    mapping = taken.assign(method="game")
    for sb_id in mapping.loc[
        mapping["score"] - mapping["runner_up"] < AMBIGUITY_MARGIN, "sb_player_id"
    ]:
        issues.append(("ambiguous", sb_id, "runner-up within margin in a game"))

    used = set(mapping["tm_player_id"].astype(int))
    people = rows.drop_duplicates("sb_player_id").set_index("sb_player_id")
    clubs = rows.groupby("sb_player_id")["club_id"].agg(set)
    countries = (
        rows.assign(country=rows["team"].map(normalise_team))
        .groupby("sb_player_id")["country"]
        .agg(set)
    )
    extra = []
    for sb_id in people.index.difference(pd.Index(mapping["sb_player_id"])):
        sb = people.loc[sb_id, ["player_name", "player_nickname"]]
        tiers = (
            ("team_pool", tm_team_players[tm_team_players["club_id"].isin(clubs[sb_id])]),
            ("country", tm_country_players[tm_country_players["country"].isin(countries[sb_id])]),
        )
        for method, pool in tiers:
            found, score, runner = _pool_match(sb, pool, used)
            if found is not None:
                extra.append((sb_id, found, method, score, runner, 0))
                used.add(found)
                break
        else:
            issues.append(
                ("unlinked", sb_id, f"best pool score {score:.2f}, runner-up {runner:.2f}")
            )
    games = rows.groupby("sb_player_id")["game_id"].nunique()
    out = pd.concat(
        [mapping[MAPPING_COLUMNS], pd.DataFrame(extra, columns=MAPPING_COLUMNS)], ignore_index=True
    )
    out = pd.DataFrame({"sb_player_id": people.index}).merge(out, on="sb_player_id", how="left")
    out["games"] = out["sb_player_id"].map(games)
    out["tm_player_id"] = out["tm_player_id"].astype("Int64")
    return out, pd.DataFrame(issues, columns=["kind", "sb_player_id", "detail"])
