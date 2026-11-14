"""Elo: hand-computed example, conservation, calibration-only fit, separate pools."""

import numpy as np
import pandas as pd
import pytest

from footy.players.elo import goal_multiplier, grid_search, run, team_elo, update


def test_goal_multiplier() -> None:
    assert [goal_multiplier(d) for d in (0, 1, -1, 2, 3, -4)] == [1.0, 1.0, 1.0, 1.5, 1.75, 1.875]


def test_hand_computed_two_games() -> None:
    """K = 20, H = 100, both teams start at 1500.

    Game 1, A hosts B, 3-0: E = 1 / (1 + 10^(-100/400)) = 0.6400650, G = (11+3)/8 = 1.75,
    dR = 20 * 1.75 * (1 - 0.6400650) = 12.597725 -> A 1512.597725, B 1487.402275.
    Game 2, B hosts A, 1-1: E = 1 / (1 + 10^(-(1487.402275 + 100 - 1512.597725)/400))
    = 0.6060192, dR = 20 * (0.5 - 0.6060192) = -2.120384 -> B 1485.281890, A 1514.718110.
    """
    a, b = update(1500.0, 1500.0, 3, 0, k=20.0, home_advantage=100.0)
    assert a == pytest.approx(1512.597725, abs=1e-6)
    assert b == pytest.approx(1487.402275, abs=1e-6)
    b2, a2 = update(b, a, 1, 1, k=20.0, home_advantage=100.0)
    assert b2 == pytest.approx(1485.281890, abs=1e-6)
    assert a2 == pytest.approx(1514.718110, abs=1e-6)
    pre_home, pre_away = run([1, 2], [2, 1], [3, 1], [0, 1], k=20.0, home_advantage=100.0)
    assert pre_home.tolist() == pytest.approx([1500.0, 1487.402275])
    assert pre_away.tolist() == pytest.approx([1500.0, 1512.597725])


def test_rating_changes_sum_to_zero() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        home, away = rng.uniform(1200, 1800, size=2)
        hg, ag = (int(x) for x in rng.integers(0, 6, size=2))
        new_home, new_away = update(
            home,
            away,
            hg,
            ag,
            k=float(rng.uniform(5, 60)),
            home_advantage=float(rng.uniform(0, 150)),
        )
        assert (new_home - home) + (new_away - away) == pytest.approx(0.0, abs=1e-9)


def _games(rows: list[tuple[str, str, int, int, int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["date", "pool", "home_club_id", "away_club_id", "home_goals", "away_goals"]
    )
    return frame.assign(game_id=range(1, len(rows) + 1), date=pd.to_datetime(frame["date"]))


def _league(
    rng: np.random.Generator, year: int, n: int, pool: str, teams: int
) -> list[tuple[str, str, int, int, int, int]]:
    """One random game a day from 1 January of ``year``."""
    start = pd.Timestamp(f"{year}-01-01")
    rows = []
    for day in range(n):
        home, away = (int(x) for x in rng.choice(teams, size=2, replace=False))
        date = (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
        rows.append((date, pool, home, away, int(rng.poisson(1.6)), int(rng.poisson(1.1))))
    return rows


def test_parameters_are_chosen_on_calibration_games_only() -> None:
    rng = np.random.default_rng(3)
    calibration = _league(rng, 2016, 300, "club", 8)
    later = [(d, p, h, a, 9, 0) for d, p, h, a, _, _ in _league(rng, 2021, 200, "club", 8)]
    _, report_short = team_elo(_games(calibration))
    _, report_long = team_elo(_games(calibration + later))
    chosen = ["k", "home_advantage"]
    assert report_short[chosen].to_numpy().tolist() == report_long[chosen].to_numpy().tolist()
    assert report_long.loc[0, "n_calibration"] == 300 and report_long.loc[0, "n_after"] == 200


def test_grid_search_returns_the_minimum() -> None:
    rng = np.random.default_rng(5)
    games = _games(_league(rng, 2016, 200, "club", 6))
    k, h, errors = grid_search(games, [10.0, 30.0], [0.0, 80.0])
    assert errors[(k, h)] == min(errors.values()) and len(errors) == 4


def test_pools_do_not_share_ratings() -> None:
    """The same id in two pools is two teams: national games never move club ratings."""
    rows = [
        ("2016-01-01", "club", 1, 2, 1, 0),
        ("2016-01-02", "national", 1, 2, 5, 0),
        ("2016-01-03", "club", 1, 2, 0, 0),
    ]
    alone = _games([rows[0], rows[2]])
    ratings_mixed, _ = team_elo(_games(rows))
    ratings_alone, _ = team_elo(alone)
    club_mixed = ratings_mixed.set_index("game_id").loc[[1, 3]].to_numpy()
    assert np.allclose(club_mixed, ratings_alone.set_index("game_id").loc[[1, 2]].to_numpy())
