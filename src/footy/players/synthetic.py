"""Synthetic leagues with a planted truth (design §5, M9).

Every estimator in paper 2 is tested on a league whose answer is known, and the
recovery study runs the whole pipeline on many such leagues. The tables come
out in the same schemas as the real processed tables (design §4.1, §4.4), so
estimators run on them unchanged.

Model, stated so tests can rely on it:

- Each player has an offensive effect ``offence`` and a defensive effect
  ``defence``, in goals per 90 minutes. A team's scoring rate in a stint is
  ``base_rate ± home_advantage / 2 + sum(offence of its players on the pitch)
  - sum(defence of the opponents on the pitch)``, floored at ``RATE_FLOOR``;
  goals are Poisson with that rate times minutes / 90.
- ``offence = offence_sd * (sqrt(m) * z_seen + sqrt(1 - m) * z_unseen)`` with
  ``m = measurable_share``. Only ``z_seen`` drives the player's own shot rate,
  so in the population exactly a share ``m`` of the variance of ``offence`` is
  visible in his box score. Defence is never visible in the box score.
- Box score and team goals are independent given the true effects: a player's
  goals come from his own shots, not from the stint goals. The exception is
  ``match_shock_sd > 0``: each team-match then draws one shock
  ``s ~ Normal(0, match_shock_sd)``, added to that team's scoring rate in every
  stint of the match and multiplying its players' shot rates in that match by
  ``exp(s)``, so the same matches move both sides.
- Every match: each side starts 11 players drawn uniformly from its squad and
  makes ``subs_per_side`` substitutions at distinct minutes in 46..89. No red
  cards, no stoppage time.
- Between seasons a share ``transfer_rate`` of players moves to a random other
  club; this is what makes teammates separable for plus-minus models.

Several leagues (pre-registration A20 vii), one per entry of ``league_means``;
with the default single entry nothing below applies and the draws are those of
the one-league generator:

- Each league has ``n_teams`` clubs and its own double round robin, competition
  ``SYN1``, ``SYN2``, ... (``SYN`` when there is one league). A player's
  ``offence`` and ``defence`` each include the planted mean of the league he
  starts in, and he keeps them when he moves. Transfers at ``transfer_rate``
  stay inside a league.
- Cup (competition ``CUP``): each season the ``CUP_TEAMS`` clubs of every
  league with the most points in its previous season (first season: drawn at
  random) play every cup club of the other leagues once, home side by coin flip.
- Between seasons a share ``cross_transfer_rate`` of players swaps clubs with a
  player of another league (a player drawn for two swaps counts once).
- ``age_slope > 0``: a player is ``AGE_MIN``..``AGE_MAX`` years old in the first
  season and loses ``age_slope`` of offence and of defence per year above
  ``AGE_PEAK``. In a swap the player who leaves the league with the higher mean
  is then drawn from its older half and the other from the younger half of his
  league. The truth table holds the effects before any decline and ``age``, the
  age in the first season.
- ``weak_link_swaps``: when set, the last league plays no cup and swaps exactly
  that many players per season with the other leagues.
- The truth table has ``league_mean``, the planted mean of the player's first
  league, and one ``share_<competition>`` per league: the share of his league
  minutes (no cup) played there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

RATE_FLOOR = 0.02
SHOT_RATE_FLOOR = 0.05
STARTERS = 11
CUP = "CUP"
CUP_TEAMS = 4
AGE_MIN, AGE_PEAK, AGE_MAX = 18, 27, 33


@dataclass(frozen=True)
class League:
    games: pd.DataFrame
    stints: pd.DataFrame
    stint_players: pd.DataFrame
    player_match: pd.DataFrame
    truth: pd.DataFrame


def simulate_league(
    rng: np.random.Generator,
    *,
    n_teams: int = 10,
    squad_size: int = 16,
    n_seasons: int = 2,
    offence_sd: float = 0.05,
    defence_sd: float = 0.05,
    measurable_share: float = 0.5,
    base_rate: float = 1.35,
    home_advantage: float = 0.3,
    shot_rate_mean: float = 1.5,
    shot_rate_sd: float = 0.8,
    conversion: float = 0.11,
    subs_per_side: int = 3,
    transfer_rate: float = 0.2,
    match_shock_sd: float = 0.0,
    league_means: Sequence[float] = (0.0,),
    cross_transfer_rate: float = 0.0,
    age_slope: float = 0.0,
    weak_link_swaps: int | None = None,
) -> League:
    n_leagues = len(league_means)
    n_clubs = n_leagues * n_teams
    n_players = n_clubs * squad_size
    player_ids = np.arange(1, n_players + 1)
    z_seen = rng.standard_normal(n_players)
    z_unseen = rng.standard_normal(n_players)
    club = np.repeat(np.arange(1, n_clubs + 1), squad_size)
    league_of = (np.arange(n_clubs + 1) - 1) // n_teams  # by club id
    means = np.asarray(league_means, dtype=np.float64)
    league_mean = means[league_of[club]]
    offence = league_mean + offence_sd * (
        np.sqrt(measurable_share) * z_seen + np.sqrt(1 - measurable_share) * z_unseen
    )
    defence = league_mean + defence_sd * rng.standard_normal(n_players)
    shot_rate = np.maximum(SHOT_RATE_FLOOR, shot_rate_mean + shot_rate_sd * z_seen)
    age = rng.integers(AGE_MIN, AGE_MAX + 1, size=n_players) if age_slope > 0 else None
    comps = ["SYN"] if n_leagues == 1 else [f"SYN{k + 1}" for k in range(n_leagues)]
    n_linked = n_leagues - (weak_link_swaps is not None)  # leagues in the cup

    games: list[tuple[object, ...]] = []
    stints: list[tuple[object, ...]] = []
    stint_players: list[tuple[object, ...]] = []
    player_match: list[tuple[object, ...]] = []
    game_id = stint_id = 0
    points = np.zeros(n_clubs + 1, dtype=np.int64)  # by club id, league games of one season
    league_minutes = np.zeros((n_players, n_leagues))
    for season_index in range(n_seasons):
        if season_index > 0:
            for k in range(n_leagues):
                members = np.flatnonzero(league_of[club] == k)
                club[members] = _transfer(rng, club[members], transfer_rate)
            swaps = [
                rng.choice(n_linked, size=2, replace=False)
                for _ in range(int(cross_transfer_rate * n_linked * n_teams * squad_size) // 2)
            ] + [(n_linked, rng.integers(n_linked)) for _ in range(weak_link_swaps or 0)]
            for a, b in swaps:
                if means[b] > means[a]:
                    a, b = b, a
                down = _pick(rng, np.flatnonzero(league_of[club] == a), age, oldest=True)
                up = _pick(rng, np.flatnonzero(league_of[club] == b), age, oldest=False)
                club[down], club[up] = club[up], club[down]
        season = f"{2020 + season_index}/{(21 + season_index) % 100:02d}"
        squads = {c: player_ids[club == c] for c in range(1, n_clubs + 1)}
        fade = 0.0 if age is None else age_slope * np.maximum(0, age + season_index - AGE_PEAK)
        off_now, def_now = offence - fade, defence - fade
        fixtures = [
            (comps[k], k * n_teams + h, k * n_teams + a)
            for k in range(n_leagues)
            for h, a in _double_round_robin(n_teams)
        ]
        if n_linked > 1:
            fixtures += _cup(rng, points, n_linked, n_teams, first=season_index == 0)
        points = np.zeros(n_clubs + 1, dtype=np.int64)
        for comp, home, away in fixtures:
            game_id += 1
            on = {"home": _starters(rng, squads[home]), "away": _starters(rng, squads[away])}
            subs = {
                side: _subs(rng, squads[t], on[side], subs_per_side)
                for side, t in (("home", home), ("away", away))
            }
            shock = {"home": 0.0, "away": 0.0}
            if match_shock_sd > 0:  # no draw at 0, so default leagues are unchanged
                shock = {side: float(rng.normal(0.0, match_shock_sd)) for side in shock}
            cuts = sorted({0, 90} | {m for s in subs.values() for m, _, _ in s})
            minutes_on: dict[int, int] = {}
            gf = {p: 0 for p in np.concatenate([squads[home], squads[away]]).tolist()}
            ga = dict(gf)
            totals = {"home": 0, "away": 0}
            for start, end in zip(cuts[:-1], cuts[1:], strict=True):
                for side in ("home", "away"):
                    for minute, out, inn in subs[side]:
                        if minute == start:
                            on[side] = [inn if p == out else p for p in on[side]]
                length = end - start
                goals: dict[str, int] = {}
                for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
                    rate = (
                        base_rate
                        + sign * home_advantage / 2
                        + off_now[np.array(on[side]) - 1].sum()
                        - def_now[np.array(on[other]) - 1].sum()
                        + shock[side]
                    )
                    goals[side] = int(rng.poisson(max(RATE_FLOOR, rate) * length / 90))
                stint_id += 1
                stints.append(
                    (
                        stint_id,
                        game_id,
                        start,
                        end,
                        length,
                        goals["home"],
                        goals["away"],
                        STARTERS,
                        STARTERS,
                    )
                )
                for side, other in (("home", "away"), ("away", "home")):
                    totals[side] += goals[side]
                    for p in on[side]:
                        stint_players.append((stint_id, p, side))
                        minutes_on[p] = minutes_on.get(p, 0) + length
                        gf[p] += goals[side]
                        ga[p] += goals[other]
            games.append((game_id, comp, season, home, away, totals["home"], totals["away"]))
            if comp != CUP:
                margin = totals["home"] - totals["away"]
                points[home] += 3 * (margin > 0) + (margin == 0)
                points[away] += 3 * (margin < 0) + (margin == 0)
            starters_now = {
                side: set(_starting_ids(on[side], subs[side])) for side in ("home", "away")
            }
            for side, team in (("home", home), ("away", away)):
                for p, mins in minutes_on.items():
                    if club[p - 1] != team:
                        continue
                    if comp != CUP:
                        league_minutes[p - 1, league_of[team]] += mins
                    shots = int(rng.poisson(shot_rate[p - 1] * np.exp(shock[side]) * mins / 90))
                    scored = int(rng.binomial(shots, conversion))
                    player_match.append(
                        (
                            game_id,
                            p,
                            team,
                            season,
                            mins,
                            p in starters_now[side],
                            shots,
                            scored,
                            gf[p],
                            ga[p],
                        )
                    )

    shares = league_minutes / np.maximum(league_minutes.sum(axis=1, keepdims=True), 1.0)
    truth = pd.DataFrame(
        {
            "player_id": player_ids,
            "offence": offence,
            "defence": defence,
            "seen": offence_sd * np.sqrt(measurable_share) * z_seen,
            "shot_rate": shot_rate,
            "league_mean": league_mean,
            **{f"share_{c}": shares[:, k] for k, c in enumerate(comps)},
            **({} if age is None else {"age": age}),
        }
    )
    return League(
        games=pd.DataFrame(
            games,
            columns=[
                "game_id",
                "competition_id",
                "season",
                "home_club_id",
                "away_club_id",
                "home_goals",
                "away_goals",
            ],
        ),
        stints=pd.DataFrame(
            stints,
            columns=[
                "stint_id",
                "game_id",
                "start_min",
                "end_min",
                "minutes",
                "home_goals",
                "away_goals",
                "home_n",
                "away_n",
            ],
        ),
        stint_players=pd.DataFrame(stint_players, columns=["stint_id", "player_id", "side"]),
        player_match=pd.DataFrame(
            player_match,
            columns=[
                "game_id",
                "player_id",
                "club_id",
                "season",
                "minutes",
                "started",
                "shots",
                "goals",
                "on_pitch_gf",
                "on_pitch_ga",
            ],
        ),
        truth=truth,
    )


def _double_round_robin(n_teams: int) -> list[tuple[int, int]]:
    return [(h, a) for h in range(1, n_teams + 1) for a in range(1, n_teams + 1) if h != a]


def _starters(rng: np.random.Generator, squad: NDArray[np.int64]) -> list[int]:
    return [int(p) for p in rng.choice(squad, size=STARTERS, replace=False)]


def _subs(
    rng: np.random.Generator, squad: NDArray[np.int64], starters: list[int], n: int
) -> list[tuple[int, int, int]]:
    """(minute, player off, player on); a player who comes off does not return."""
    bench = [int(p) for p in squad if int(p) not in starters]
    minutes = sorted(int(m) for m in rng.choice(np.arange(46, 90), size=n, replace=False))
    current, out = list(starters), []
    for minute in minutes:
        off = int(rng.choice(current))
        inn = int(bench.pop(int(rng.integers(len(bench)))))
        current = [inn if p == off else p for p in current]
        out.append((minute, off, inn))
    return out


def _starting_ids(final_on: list[int], subs: list[tuple[int, int, int]]) -> list[int]:
    """Undo the substitutions to recover who started."""
    current = list(final_on)
    for _, off, inn in reversed(subs):
        current = [off if p == inn else p for p in current]
    return current


def _cup(
    rng: np.random.Generator,
    points: NDArray[np.int64],
    n_leagues: int,
    n_teams: int,
    *,
    first: bool,
) -> list[tuple[str, int, int]]:
    """Cup fixtures among the first ``n_leagues`` leagues; ``points`` (by club id)
    are last season's."""
    entrants = []
    for k in range(n_leagues):
        clubs = k * n_teams + 1 + np.arange(n_teams)
        best = (
            rng.choice(clubs, size=CUP_TEAMS, replace=False)
            if first
            else clubs[np.argsort(-points[clubs], kind="stable")[:CUP_TEAMS]]
        )
        entrants += [(k, int(c)) for c in best]
    fixtures = []
    for i, (league_a, a) in enumerate(entrants):
        for league_b, b in entrants[i + 1 :]:
            if league_a != league_b:
                fixtures.append((CUP, a, b) if rng.random() < 0.5 else (CUP, b, a))
    return fixtures


def _pick(
    rng: np.random.Generator,
    members: NDArray[np.int64],
    age: NDArray[np.int64] | None,
    *,
    oldest: bool,
) -> int:
    """One of ``members`` at random; with ages, from their older or younger half."""
    if age is not None:
        by_age = members[np.argsort(age[members], kind="stable")]
        members = by_age[len(by_age) // 2 :] if oldest else by_age[: len(by_age) // 2]
    return int(rng.choice(members))


def _transfer(rng: np.random.Generator, club: NDArray[np.int64], rate: float) -> NDArray[np.int64]:
    """Move a share of players to other clubs, swapping pairs so squad sizes hold."""
    club = club.copy()
    movers = rng.permutation(len(club))[: int(rate * len(club)) // 2 * 2]
    for a, b in zip(movers[0::2], movers[1::2], strict=True):
        if club[a] != club[b]:
            club[a], club[b] = club[b], club[a]
    return club
