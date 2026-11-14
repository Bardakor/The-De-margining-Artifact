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
  goals come from his own shots, not from the stint goals.
- Every match: each side starts 11 players drawn uniformly from its squad and
  makes ``subs_per_side`` substitutions at distinct minutes in 46..89. No red
  cards, no stoppage time.
- Between seasons a share ``transfer_rate`` of players moves to a random other
  club; this is what makes teammates separable for plus-minus models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

RATE_FLOOR = 0.02
SHOT_RATE_FLOOR = 0.05
STARTERS = 11


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
) -> League:
    n_players = n_teams * squad_size
    player_ids = np.arange(1, n_players + 1)
    z_seen = rng.standard_normal(n_players)
    z_unseen = rng.standard_normal(n_players)
    offence = offence_sd * (
        np.sqrt(measurable_share) * z_seen + np.sqrt(1 - measurable_share) * z_unseen
    )
    defence = defence_sd * rng.standard_normal(n_players)
    shot_rate = np.maximum(SHOT_RATE_FLOOR, shot_rate_mean + shot_rate_sd * z_seen)
    club = np.repeat(np.arange(1, n_teams + 1), squad_size)

    games: list[tuple[object, ...]] = []
    stints: list[tuple[object, ...]] = []
    stint_players: list[tuple[object, ...]] = []
    player_match: list[tuple[object, ...]] = []
    game_id = stint_id = 0
    for season_index in range(n_seasons):
        if season_index > 0:
            club = _transfer(rng, club, transfer_rate)
        season = f"{2020 + season_index}/{(21 + season_index) % 100:02d}"
        squads = {c: player_ids[club == c] for c in range(1, n_teams + 1)}
        for home, away in _double_round_robin(n_teams):
            game_id += 1
            on = {"home": _starters(rng, squads[home]), "away": _starters(rng, squads[away])}
            subs = {
                side: _subs(rng, squads[t], on[side], subs_per_side)
                for side, t in (("home", home), ("away", away))
            }
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
                        + offence[np.array(on[side]) - 1].sum()
                        - defence[np.array(on[other]) - 1].sum()
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
            games.append((game_id, "SYN", season, home, away, totals["home"], totals["away"]))
            starters_now = {
                side: set(_starting_ids(on[side], subs[side])) for side in ("home", "away")
            }
            for side, team in (("home", home), ("away", away)):
                for p, mins in minutes_on.items():
                    if club[p - 1] != team:
                        continue
                    shots = int(rng.poisson(shot_rate[p - 1] * mins / 90))
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

    truth = pd.DataFrame(
        {
            "player_id": player_ids,
            "offence": offence,
            "defence": defence,
            "seen": offence_sd * np.sqrt(measurable_share) * z_seen,
            "shot_rate": shot_rate,
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


def _transfer(rng: np.random.Generator, club: NDArray[np.int64], rate: float) -> NDArray[np.int64]:
    """Move a share of players to other clubs, swapping pairs so squad sizes hold."""
    club = club.copy()
    movers = rng.permutation(len(club))[: int(rate * len(club)) // 2 * 2]
    for a, b in zip(movers[0::2], movers[1::2], strict=True):
        if club[a] != club[b]:
            club[a], club[b] = club[b], club[a]
    return club
