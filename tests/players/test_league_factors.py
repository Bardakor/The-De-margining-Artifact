"""A20 (vii), A21: the league-factor estimator on simulated cells where the truth is known.

``simulate`` draws player x season cells directly: a player effect, a league
factor ``GAMMA`` on the goal rate, an age curve ``B1 * a + B2 * a**2`` around 26,
Poisson goals on uniform minutes. GB1, A and B exchange players; FEW is closed
apart from ``few`` players who leave it once. Variants:

- ``random``: 8% of the players of the open leagues change league each season.
- ``form``: a player of A or B moves to GB1 with probability 0.25 after a season
  above his own expected count and 0.02 otherwise; GB1 players leave at random.
- ``ageing``: nobody leaves B and players of GB1 or A join it only from the age
  of 30 (30% per season); younger ones move between GB1 and A.
- ``context`` (A21): moves as in ``random``, and the goal rate also carries
  ``TEAM`` = 0.15 per unit of ``elo``, the player's team strength centred within
  league-season (minutes-weighted). A player who arrives in GB1 joins a team one
  unit weaker than his own level, one who leaves GB1 a team one unit stronger;
  on top, every cell has noise of SD 0.5.

What 100 simulated datasets per variant gave (1,200 players, 6 seasons; error of
gamma for A, B and FEW; the standard error of a mean is near 0.003, 0.003, 0.012):

- ``random``: bias -0.001, -0.003, -0.010; SD 0.031, 0.028, 0.120; largest error
  0.08, 0.06, 0.36. With every planted gamma at zero the mean estimates were
  -0.003, -0.002, -0.008. Over 200 datasets with B = 200 the mean bootstrap SE
  was 0.032, 0.031, 0.119 (SD of the estimates 0.032, 0.032, 0.105) and the 90%
  interval covered 0.880, 0.855 and 0.900: slightly under nominal.
- ``form``: gamma of the leagues the movers leave is biased up, +0.083 (A) and
  +0.076 (B): they look easier than they are and their gap to GB1 is
  exaggerated. With the before-a-move dummy +0.044 and +0.036 remain; without
  the cells before a move the sign flips, -0.027 and -0.029 (the seasons that
  did not trigger a move sit below the player's mean). Neither check removes
  the bias; here the two bracket the truth.
- ``ageing``: -0.003 for B with the age terms, -0.190 without them: the league
  joined late then looks harder than it is and its gap is understated.
- ``context``: the players seen in GB1 and in A or B have ``elo`` 0.74 higher
  outside GB1. Without the control gamma is biased up, +0.112, +0.118, +0.148
  (GB1 looks harder than it is); with ``elo`` as a covariate the bias is +0.001,
  +0.006, +0.022 and the coefficient is recovered (bias -0.001, SD 0.010).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.optimize import approx_fprime
from scipy.special import logsumexp

from footy.players.league_factors import (
    POOLED,
    Cells,
    _score,
    _score_jacobian,
    bootstrap,
    fit,
    league_parameters,
    pre_move,
    prepare_cells,
    summarise,
)

LEAGUES = np.array(["GB1", "A", "B", "FEW"])
GAMMA = (0.0, 0.10, 0.25, 0.25)
B_LEAGUE, FEW = 2, 3
B1, B2 = -0.02, -0.004
AGE_REF = 26.0
MOVE, FORM_HI, FORM_LO, LATE = 0.08, 0.25, 0.02, 0.3
TEAM, TEAM_SHIFT, TEAM_NOISE = 0.15, 1.0, 0.5
OWN = {str(x): str(x) for x in LEAGUES}
REPEATS = 40


def simulate(
    rng: np.random.Generator,
    *,
    gamma: Sequence[float] = GAMMA,
    variant: str = "random",
    n_players: int = 1200,
    n_seasons: int = 6,
    few: int = 12,
) -> pd.DataFrame:
    alpha = rng.normal(np.log(0.25), 0.6, n_players)
    age = rng.uniform(18.0, 31.0, n_players) - AGE_REF
    league = rng.integers(0, len(LEAGUES), n_players)
    leaves = np.full(n_players, n_seasons)
    leaves[rng.choice(np.flatnonzero(league == FEW), few, replace=False)] = rng.integers(
        1, n_seasons, few
    )
    home = league.copy()
    strength = rng.normal(0.0, 1.0, n_players) if variant == "context" else np.zeros(n_players)
    seasons = []
    for t in range(n_seasons):
        league = np.where(leaves == t, rng.integers(0, FEW, n_players), league)
        a = age + t
        minutes = rng.uniform(300.0, 3000.0, n_players)
        elo = strength
        if variant == "context":
            away = np.where(league == 0, -1.0 * (home != 0), 1.0 * (home == 0))
            raw = strength + TEAM_SHIFT * away + rng.normal(0.0, TEAM_NOISE, n_players)
            elo = raw - (np.bincount(league, minutes * raw) / np.bincount(league, minutes))[league]
        mean = (
            minutes
            / 90
            * np.exp(alpha + np.asarray(gamma)[league] + B1 * a + B2 * a**2 + TEAM * elo)
        )
        goals = rng.poisson(mean)
        seasons.append(
            pd.DataFrame(
                {
                    "player_id": np.arange(n_players),
                    "season": t,
                    "league": LEAGUES[league],
                    "minutes": minutes,
                    "goals": goals,
                    "age": a,
                    "age2": a**2,
                    "elo": elo,
                }
            )
        )
        other = rng.integers(1, FEW, n_players)
        if variant == "form":
            rate = np.where(league == 0, MOVE, np.where(goals > mean, FORM_HI, FORM_LO))
            dest = np.where(league == 0, other, 0)
        elif variant == "ageing":
            old = a + AGE_REF >= 30
            rate = np.where(league == B_LEAGUE, 0.0, np.where(old, LATE, MOVE))
            dest = np.where(old, B_LEAGUE, 1 - league)
        else:
            rate = np.full(n_players, MOVE)
            dest = (league + other) % FEW
        league = np.where((league != FEW) & (rng.random(n_players) < rate), dest, league)
    return pd.concat(seasons, ignore_index=True)


def fit_as(cells: pd.DataFrame, how: str) -> pd.Series:
    if how == "dummy":
        return fit(prepare_cells(cells, "goals", OWN, covariates=("age", "age2", "pre_move")))
    if how == "dropped":
        return fit(prepare_cells(cells[cells["pre_move"] == 0], "goals", OWN))
    if how == "no_age":
        return fit(prepare_cells(cells, "goals", OWN, covariates=()))
    if how == "context":
        return fit(prepare_cells(cells, "goals", OWN, covariates=("age", "age2", "elo")))
    return fit(prepare_cells(cells, "goals", OWN))


def gamma_errors(test: int, variant: str, fits: Sequence[str]) -> pd.DataFrame:
    """Mean error of gamma over ``REPEATS`` datasets: one row per league, one column per fit."""
    truth = pd.Series(GAMMA[1:], index=LEAGUES[1:])
    rows = []
    for r in range(REPEATS):
        cells = simulate(np.random.default_rng([test, r]), variant=variant)
        cells["pre_move"] = pre_move(cells, "season").astype(np.float64)
        rows.append(pd.DataFrame({how: fit_as(cells, how)[truth.index] - truth for how in fits}))
    return pd.concat(rows).groupby(level=0).mean()


def log_likelihood(theta: NDArray[np.float64], c: Cells, weight: NDArray[np.float64]) -> float:
    """The weighted conditional log-likelihood, player by player."""
    eta = c.offset + c.design @ theta
    value = 0.0
    for p in range(len(c.start)):
        cell = c.player == p
        value += weight[p] * c.count[cell] @ (eta[cell] - logsumexp(eta[cell]))
    return float(value)


def test_score_and_jacobian_match_finite_differences() -> None:
    rng = np.random.default_rng(0)
    c = prepare_cells(simulate(rng, n_players=80, few=3), "goals", OWN)
    weight = rng.integers(0, 4, len(c.start)).astype(np.float64)
    theta = rng.normal(0.0, 0.1, len(c.names))
    scale = weight @ c.total
    step = 1e-5 * np.eye(len(theta))
    numeric = [
        (log_likelihood(theta - h, c, weight) - log_likelihood(theta + h, c, weight)) / 2e-5
        for h in step
    ]
    np.testing.assert_allclose(_score(theta, c, weight), np.array(numeric) / scale, atol=1e-7)
    jacobian = approx_fprime(theta, _score, 1e-7, c, weight)
    np.testing.assert_allclose(_score_jacobian(theta, c, weight), jacobian, atol=1e-4)


def test_planted_gamma_is_recovered_and_intervals_cover() -> None:
    truth = pd.Series([*GAMMA[1:], B1, B2], index=["A", "B", "FEW", "age", "age2"])
    errors, covered, ses, used = [], [], [], []
    for r in range(REPEATS):
        c = prepare_cells(simulate(np.random.default_rng([1, r])), "goals", OWN)
        estimate = fit(c)
        table = summarise(
            estimate, bootstrap(c, estimate, np.random.default_rng([2, r]), n_boot=100)
        )
        used.append(table["n_boot"].min())
        errors.append(estimate - truth)
        covered.append((table["lo"] <= truth) & (truth <= table["hi"]))
        ses.append(table["se"])
    error, cover, se = pd.DataFrame(errors), pd.DataFrame(covered).mean(), pd.DataFrame(ses).mean()
    assert error[["A", "B"]].abs().to_numpy().max() < 0.12
    assert error["FEW"].abs().max() < 0.45
    assert (error.mean().abs() < [0.015, 0.015, 0.06, 0.0015, 0.0002]).all()
    assert ((se / error.std()).between(0.75, 1.35)).all()
    assert cover[["A", "B", "age", "age2"]].mean() > 0.8
    assert cover["FEW"] > 0.75
    assert sum(used) == 100 * REPEATS - 1  # one replicate drew none of FEW's 12 movers


def test_zero_planted_gamma_gives_estimates_within_noise() -> None:
    zero = (0.0, 0.0, 0.0, 0.0)
    means = pd.DataFrame(
        [
            fit(prepare_cells(simulate(np.random.default_rng([3, r]), gamma=zero), "goals", OWN))
            for r in range(REPEATS)
        ]
    ).mean()
    assert (means[["A", "B", "FEW"]].abs() < [0.015, 0.015, 0.06]).all()
    c = prepare_cells(simulate(np.random.default_rng(4), gamma=zero), "goals", OWN)
    estimate = fit(c)
    table = summarise(estimate, bootstrap(c, estimate, np.random.default_rng(5), n_boot=100))
    assert (table["estimate"].abs() < 3 * table["se"])[["A", "B", "FEW"]].all()


def test_league_with_few_movers_is_pooled_by_the_rule_and_has_a_wide_interval() -> None:
    cells = simulate(np.random.default_rng(6))
    rule = league_parameters(cells).set_index("league")
    assert rule.loc["FEW", "n_linking"] == 12
    assert rule["parameter"].to_dict() == {"A": "A", "B": "B", "FEW": POOLED, "GB1": "GB1"}
    assert (rule.loc[["A", "B", "GB1"], "n_linking"] > 150).all()
    c = prepare_cells(cells, "goals", OWN)
    estimate = fit(c)
    table = summarise(estimate, bootstrap(c, estimate, np.random.default_rng(7), n_boot=200))
    width = table["hi"] - table["lo"]
    assert width["FEW"] > 2.5 * width[["A", "B"]].max()
    assert width["FEW"] > 0.25


def test_moves_selected_on_form_bias_gamma_up_and_the_checks_bracket_the_truth() -> None:
    error = gamma_errors(8, "form", ("plain", "dummy", "dropped")).loc[["A", "B"]].mean()
    assert 0.05 < error["plain"] < 0.10
    assert 0.015 < error["dummy"] < 0.06
    assert -0.065 < error["dropped"] < -0.01


def test_late_moves_to_a_weak_league_need_the_age_terms() -> None:
    error = gamma_errors(9, "ageing", ("plain", "no_age"))
    plain, no_age = error["plain"], error["no_age"]
    assert abs(plain["B"]) < 0.02
    assert -0.23 < no_age["B"] < -0.14
    assert abs(no_age["A"]) < 0.02


def test_team_context_control_recovers_gamma_when_movers_change_team_strength() -> None:
    error = gamma_errors(11, "context", ("plain", "context")).loc[["A", "B"]]
    assert error["plain"].between(0.08, 0.15).all()
    assert (error["context"].abs() < 0.02).all()
    cells = simulate(np.random.default_rng(12), variant="context")
    c = prepare_cells(cells, "goals", OWN, covariates=("age", "age2", "elo"))
    estimate = fit(c)
    table = summarise(estimate, bootstrap(c, estimate, np.random.default_rng(13), n_boot=100))
    assert abs(table["estimate"]["elo"] - TEAM) < 3 * table["se"]["elo"] < 0.06


def small_cells() -> pd.DataFrame:
    """40 two-season players in GB1, 40 in C, and player 0, who links them."""
    cells = pd.DataFrame(
        {
            "player_id": np.repeat(np.arange(81), 2),
            "season": np.tile([0, 1], 81),
            "league": ["GB1", "C"] + ["GB1"] * 80 + ["C"] * 80,
            "minutes": 900.0,
            "goals": 3,
        }
    )
    cells.loc[1, "minutes"] = 1800.0
    cells.loc[1, "goals"] = 9
    return cells


def test_single_cell_and_scoreless_players_are_dropped_and_counted() -> None:
    cells = small_cells().iloc[:-1]
    cells.loc[cells["player_id"].isin([5, 6, 7]), "goals"] = 0
    c = prepare_cells(cells, "goals", OWN | {"C": "C"}, covariates=())
    assert (c.n_single_cell, c.n_zero_count, len(c.start), len(c.count)) == (1, 3, 77, 154)
    assert c.names == ("C",)
    np.testing.assert_allclose(c.total, [12.0] + [6.0] * 76)


def test_gamma_of_a_league_with_one_link_and_its_missing_replicates() -> None:
    cells = small_cells()
    c = prepare_cells(cells, "goals", OWN | {"C": "C"}, covariates=())
    estimate = fit(c)
    assert estimate["C"] == pytest.approx(np.log((9 / 1800) / (3 / 900)))
    reps = bootstrap(c, estimate, np.random.default_rng(10), n_boot=100)
    missing = np.isnan(reps[:, 0])
    assert 0.2 < missing.mean() < 0.55  # the linking player is left out with probability 0.37
    np.testing.assert_allclose(reps[~missing, 0], estimate["C"])
    assert summarise(estimate, reps).loc["C", "n_boot"] == (~missing).sum()
    with pytest.raises(ValueError, match="C"):
        fit(prepare_cells(cells[cells["player_id"] > 0], "goals", OWN | {"C": "C"}, covariates=()))


def test_pre_move_flags_the_cell_before_a_league_change() -> None:
    cells = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 2, 2],
            "season": [3, 0, 1, 2, 0, 1],
            "league": ["A", "A", "B", "B", "A", "A"],
        }
    )
    assert pre_move(cells, "season").tolist() == [False, True, False, True, False, False]
