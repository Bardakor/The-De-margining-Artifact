"""Team Elo with a goal-difference multiplier and home advantage (design §4.3). Pure.

Model (World Football Elo form):

    E_home = 1 / (1 + 10 ** (-(R_home + H - R_away) / 400))
    S_home = 1 / 0.5 / 0 for a home win / draw / loss
    G      = 1 if |gd| <= 1, 1.5 if |gd| == 2, (11 + |gd|) / 8 otherwise
    dR     = K * G * (S_home - E_home);  R_home += dR;  R_away -= dR

so every game's two rating changes sum to zero. Assumptions (stated, not hidden):

- Every team enters its pool at 1500 the first time it appears.
- Home advantage H applies to every game, neutral-venue tournament games included:
  the source has no neutral-venue flag.
- Games are processed in the order given; the caller sorts by (date, game_id).
- Clubs and national teams are separate pools, each with its own K and H, chosen by
  grid search minimising the mean squared error of E_home against S_home over the
  pool's calibration games only, then frozen for every later game. The objective is
  squared error, as design §4.3 states.
- The no-skill baseline predicts the calibration period's mean home score for every game.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

INITIAL_RATING = 1500.0
K_GRID = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0)
HOME_GRID = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0)
CALIBRATION_END = date(2019, 12, 31)

FloatArray = NDArray[np.float64]


def goal_multiplier(goal_difference: int) -> float:
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def expected_home(home_elo: FloatArray, away_elo: FloatArray, home_advantage: float) -> FloatArray:
    return 1.0 / (1.0 + 10.0 ** (-(home_elo + home_advantage - away_elo) / 400.0))


def home_score(home_goals: FloatArray, away_goals: FloatArray) -> FloatArray:
    return np.where(home_goals > away_goals, 1.0, np.where(home_goals == away_goals, 0.5, 0.0))


def update(
    home_elo: float,
    away_elo: float,
    home_goals: int,
    away_goals: int,
    k: float,
    home_advantage: float,
) -> tuple[float, float]:
    """Post-match ratings of one game."""
    expected = 1.0 / (1.0 + 10.0 ** (-(home_elo + home_advantage - away_elo) / 400.0))
    actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    delta = k * goal_multiplier(home_goals - away_goals) * (actual - expected)
    return home_elo + delta, away_elo - delta


def run(
    home_ids: Sequence[int],
    away_ids: Sequence[int],
    home_goals: Sequence[int],
    away_goals: Sequence[int],
    k: float,
    home_advantage: float,
) -> tuple[FloatArray, FloatArray]:
    """Pre-match ratings of both sides for each game, in the order given."""
    ratings: dict[int, float] = {}
    pre_home: list[float] = []
    pre_away: list[float] = []
    for home, away, hg, ag in zip(home_ids, away_ids, home_goals, away_goals, strict=True):
        r_home = ratings.get(home, INITIAL_RATING)
        r_away = ratings.get(away, INITIAL_RATING)
        pre_home.append(r_home)
        pre_away.append(r_away)
        ratings[home], ratings[away] = update(r_home, r_away, hg, ag, k, home_advantage)
    return np.asarray(pre_home), np.asarray(pre_away)


def squared_error(
    pre_home: FloatArray,
    pre_away: FloatArray,
    home_goals: FloatArray,
    away_goals: FloatArray,
    home_advantage: float,
) -> float:
    """Mean squared error of the one-step-ahead expected home score."""
    expected = expected_home(pre_home, pre_away, home_advantage)
    return float(np.mean((home_score(home_goals, away_goals) - expected) ** 2))


def grid_search(
    games: pd.DataFrame, ks: Sequence[float], home_advantages: Sequence[float]
) -> tuple[float, float, dict[tuple[float, float], float]]:
    """(K, H) minimising one-step-ahead squared error over ``games`` (calibration only)."""
    home_ids, away_ids = games["home_club_id"].tolist(), games["away_club_id"].tolist()
    home_goals, away_goals = games["home_goals"].tolist(), games["away_goals"].tolist()
    hg = games["home_goals"].to_numpy(np.float64)
    ag = games["away_goals"].to_numpy(np.float64)
    errors: dict[tuple[float, float], float] = {}
    for k in ks:
        for h in home_advantages:
            pre_h, pre_a = run(home_ids, away_ids, home_goals, away_goals, k, h)
            errors[(k, h)] = squared_error(pre_h, pre_a, hg, ag, h)
    best_k, best_h = min(errors, key=errors.__getitem__)
    return best_k, best_h, errors


def team_elo(
    games: pd.DataFrame,
    calibration_end: date = CALIBRATION_END,
    ks: Sequence[float] = K_GRID,
    home_advantages: Sequence[float] = HOME_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-match Elo of both sides for every game, and a per-pool fit report.

    ``games``: game_id, date (datetime64), pool, home_club_id, away_club_id, home_goals,
    away_goals. Returns (game_id, home_elo_pre, away_elo_pre) and one report row per pool.
    """
    ordered = games.sort_values(["date", "game_id"], kind="stable")
    ratings: list[pd.DataFrame] = []
    report: list[dict[str, object]] = []
    for pool, pool_games in ordered.groupby("pool", sort=True):
        calibration = pool_games["date"].dt.date <= calibration_end
        k, h, _ = grid_search(pool_games[calibration], ks, home_advantages)
        pre_h, pre_a = run(
            pool_games["home_club_id"].tolist(),
            pool_games["away_club_id"].tolist(),
            pool_games["home_goals"].tolist(),
            pool_games["away_goals"].tolist(),
            k,
            h,
        )
        ratings.append(
            pd.DataFrame(
                {
                    "game_id": pool_games["game_id"].to_numpy(),
                    "home_elo_pre": pre_h,
                    "away_elo_pre": pre_a,
                }
            )
        )
        hg = pool_games["home_goals"].to_numpy(np.float64)
        ag = pool_games["away_goals"].to_numpy(np.float64)
        actual = home_score(hg, ag)
        expected = expected_home(pre_h, pre_a, h)
        base_rate = float(actual[calibration].mean())
        row: dict[str, object] = {"pool": pool, "k": k, "home_advantage": h}
        for label, mask in (("calibration", calibration.to_numpy()), ("after", ~calibration)):
            mask = np.asarray(mask)
            row[f"n_{label}"] = int(mask.sum())
            empty = not mask.any()
            row[f"mse_{label}"] = (
                np.nan if empty else float(np.mean((actual[mask] - expected[mask]) ** 2))
            )
            row[f"mse_baseline_{label}"] = (
                np.nan if empty else float(np.mean((actual[mask] - base_rate) ** 2))
            )
        row["k_on_grid_edge"] = k in (min(ks), max(ks))
        row["h_on_grid_edge"] = h in (min(home_advantages), max(home_advantages))
        report.append(row)
    return pd.concat(ratings, ignore_index=True), pd.DataFrame(report)
