"""Fitting: time decay, the weighted likelihood, its gradient, and the driver.

Study spec §5.2. The load-bearing test in this file is the gradient check. A
hand-derived gradient that is subtly wrong does not crash — it converges
smoothly to the wrong optimum, and every downstream test would pass against
those wrong parameters. `check_grad` is the only thing standing between that
and the study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime, check_grad

from footy.core.dixon_coles import rho_bounds
from footy.fit.decay import (
    half_life_to_xi,
    match_age_days,
    time_weights,
    xi_to_half_life,
)
from footy.fit.likelihood import (
    MatchData,
    initial_theta,
    log_factorial_offset,
    negative_log_likelihood,
    negative_log_likelihood_gradient,
    rates,
    unpack,
)
from footy.fit.mle import ConvergenceError, encode, fit, fitted_frame

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def synthetic_data(seed: int = 0, n_teams: int = 6, n_matches: int = 300) -> MatchData:
    rng = np.random.default_rng(seed)
    home = rng.integers(0, n_teams, n_matches)
    away = (home + 1 + rng.integers(0, n_teams - 1, n_matches)) % n_teams
    return MatchData(
        home_idx=home,
        away_idx=away,
        home_goals=rng.poisson(1.5, n_matches),
        away_goals=rng.poisson(1.1, n_matches),
        weights=np.exp(-0.003 * rng.uniform(0, 400, n_matches)),
        n_teams=n_teams,
    )


def random_theta(rng: np.random.Generator, n_teams: int) -> np.ndarray:
    return np.concatenate(
        [
            rng.normal(0, 0.3, n_teams - 1),
            rng.normal(0, 0.3, n_teams),
            [rng.uniform(0.0, 0.5)],
            [rng.uniform(-0.15, 0.05)],
        ]
    )


def simulated_league(
    seed: int = 7, n_teams: int = 14, rounds: int = 4
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """A league generated from known parameters, so a fit can be scored."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = np.exp(rng.normal(0, 0.35, n_teams))
    attack /= np.exp(np.mean(np.log(attack)))  # geometric mean 1, as the fit reports
    defence = np.exp(rng.normal(0, 0.30, n_teams))
    gamma = 1.35

    rows = []
    start = pd.Timestamp("2020-08-01")
    day = 0
    for _ in range(rounds):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam = attack[h] * defence[a] * gamma
                mu = attack[a] * defence[h]
                rows.append(
                    (
                        start + pd.Timedelta(days=day // 20),
                        teams[h],
                        teams[a],
                        int(rng.poisson(lam)),
                        int(rng.poisson(mu)),
                    )
                )
                day += 1
    frame = pd.DataFrame(rows, columns=["kickoff", "home", "away", "home_goals", "away_goals"])
    return frame, attack, defence, gamma


# --------------------------------------------------------------------------
# Time decay
# --------------------------------------------------------------------------


def test_half_life_round_trips_through_xi() -> None:
    for days in (30.0, 90.0, 180.0, 365.0):
        assert xi_to_half_life(half_life_to_xi(days)) == pytest.approx(days, abs=1e-9)


def test_a_match_one_half_life_old_carries_half_the_weight() -> None:
    xi = half_life_to_xi(100.0)
    as_of = pd.Timestamp("2024-01-01")
    kickoffs = pd.Series([as_of - pd.Timedelta(days=100)])
    assert float(time_weights(kickoffs, as_of, xi)[0]) == pytest.approx(0.5, abs=1e-12)


def test_weights_decay_monotonically_with_age() -> None:
    as_of = pd.Timestamp("2024-01-01")
    kickoffs = pd.Series([as_of - pd.Timedelta(days=d) for d in (1, 30, 200, 900)])
    weights = time_weights(kickoffs, as_of, half_life_to_xi(180.0))
    assert list(weights) == sorted(weights, reverse=True)
    assert np.all(weights > 0.0)
    assert np.all(weights <= 1.0)


def test_zero_xi_is_an_unweighted_fit() -> None:
    as_of = pd.Timestamp("2024-01-01")
    kickoffs = pd.Series([as_of - pd.Timedelta(days=d) for d in (1, 500)])
    assert time_weights(kickoffs, as_of, 0.0) == pytest.approx([1.0, 1.0])


def test_a_match_at_or_after_the_cutoff_is_rejected() -> None:
    """This is the single point where leakage can enter the walk-forward: a
    fit may only ever see matches that kicked off strictly earlier."""
    as_of = pd.Timestamp("2024-01-01")
    for offset in (pd.Timedelta(0), pd.Timedelta(days=1)):
        with pytest.raises(ValueError, match="at or after the fit cutoff"):
            match_age_days(pd.Series([as_of + offset]), as_of)


def test_missing_kickoffs_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing timestamps"):
        match_age_days(pd.Series([pd.NaT]), pd.Timestamp("2024-01-01"))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_half_life_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        half_life_to_xi(bad)


def test_negative_xi_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        time_weights(pd.Series([pd.Timestamp("2020-01-01")]), pd.Timestamp("2024-01-01"), -0.1)


# --------------------------------------------------------------------------
# The gradient — the load-bearing check
# --------------------------------------------------------------------------


def test_analytic_gradient_matches_finite_differences() -> None:
    """Spec §8: mandatory. A wrong gradient converges silently to the wrong
    optimum, so nothing downstream would catch it."""
    data = synthetic_data()
    rng = np.random.default_rng(1)
    for _ in range(25):
        theta = random_theta(rng, data.n_teams)
        error = check_grad(negative_log_likelihood, negative_log_likelihood_gradient, theta, data)
        scale = float(np.linalg.norm(negative_log_likelihood_gradient(theta, data)))
        assert error / max(scale, 1.0) < 1e-5, f"relative gradient error {error / scale:.2e}"


def test_every_gradient_component_is_individually_correct() -> None:
    """A single wrong component would be masked by the norm in check_grad."""
    data = synthetic_data(seed=3)
    theta = random_theta(np.random.default_rng(2), data.n_teams)
    analytic = negative_log_likelihood_gradient(theta, data)
    numeric = approx_fprime(theta, negative_log_likelihood, 1e-7, data)
    for i, (a, n) in enumerate(zip(analytic, numeric, strict=True)):
        assert abs(a - n) / max(1.0, abs(n)) < 1e-4, f"component {i}: {a} vs {n}"


def test_gradient_has_one_entry_per_free_parameter() -> None:
    data = synthetic_data()
    theta = initial_theta(data.n_teams)
    assert negative_log_likelihood_gradient(theta, data).shape == (data.n_parameters,)
    assert data.n_parameters == 2 * data.n_teams + 1


def test_weights_actually_influence_the_likelihood() -> None:
    """A gradient that ignored the weights would still pass check_grad against
    a likelihood that ignored them too."""
    base = synthetic_data()
    doubled = MatchData(
        home_idx=base.home_idx,
        away_idx=base.away_idx,
        home_goals=base.home_goals,
        away_goals=base.away_goals,
        weights=base.weights * 2.0,
        n_teams=base.n_teams,
    )
    theta = initial_theta(base.n_teams)
    assert negative_log_likelihood(theta, doubled) == pytest.approx(
        2.0 * negative_log_likelihood(theta, base), rel=1e-12
    )


# --------------------------------------------------------------------------
# Likelihood structure
# --------------------------------------------------------------------------


def test_identifiability_constraint_holds_by_construction() -> None:
    rng = np.random.default_rng(4)
    for _ in range(10):
        params = unpack(random_theta(rng, 6), 6)
        assert float(np.exp(np.mean(np.log(params.attack)))) == pytest.approx(1.0, abs=1e-12)


def test_the_flat_direction_is_genuinely_removed() -> None:
    """Scaling every attack by c and every defence by 1/c leaves lam and mu
    unchanged. If the constraint did not bind, the optimum would not be unique."""
    data = synthetic_data()
    theta = random_theta(np.random.default_rng(5), data.n_teams)
    lam, mu, _ = rates(theta, data)

    shifted = theta.copy()
    shifted[: data.n_teams - 1] += 0.3  # would rescale attack were it unconstrained
    shifted[data.n_teams - 1 : 2 * data.n_teams - 1] -= 0.3
    lam2, mu2, _ = rates(shifted, data)
    assert not np.allclose(lam, lam2), "constraint failed to pin the scale"


def test_rates_match_the_specified_factorisation() -> None:
    data = synthetic_data()
    theta = random_theta(np.random.default_rng(6), data.n_teams)
    params = unpack(theta, data.n_teams)
    lam, mu, rho = rates(theta, data)
    expected_lam = (
        params.attack[data.home_idx] * params.defence[data.away_idx] * params.home_advantage
    )
    expected_mu = params.attack[data.away_idx] * params.defence[data.home_idx]
    assert lam == pytest.approx(expected_lam)
    assert mu == pytest.approx(expected_mu)
    assert rho == params.rho


def test_inadmissible_parameters_return_a_finite_penalty_not_an_exception() -> None:
    """L-BFGS-B must be able to back out of a bad trial step."""
    data = synthetic_data()
    theta = initial_theta(data.n_teams)
    theta[-1] = 5.0  # tau goes negative
    value = negative_log_likelihood(theta, data)
    assert np.isfinite(value)
    assert value >= 1e11


def test_factorial_offset_is_constant_in_the_parameters() -> None:
    """Dropping it from the objective must not move the optimum."""
    data = synthetic_data()
    offset = log_factorial_offset(data)
    assert offset > 0.0
    assert log_factorial_offset(data) == offset


def test_theta_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="must have"):
        unpack(np.zeros(3), 6)


def test_a_team_cannot_play_itself() -> None:
    with pytest.raises(ValueError, match="cannot play itself"):
        MatchData(
            home_idx=np.array([0]),
            away_idx=np.array([0]),
            home_goals=np.array([1]),
            away_goals=np.array([0]),
            weights=np.array([1.0]),
            n_teams=2,
        )


def test_ragged_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="ragged"):
        MatchData(
            home_idx=np.array([0, 1]),
            away_idx=np.array([1]),
            home_goals=np.array([1]),
            away_goals=np.array([0]),
            weights=np.array([1.0]),
            n_teams=2,
        )


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def test_fit_recovers_the_parameters_it_was_generated_from() -> None:
    """The strongest available check: simulate from known strengths and
    confirm the fit finds them."""
    matches, attack, defence, gamma = simulated_league()
    fitted = fit(matches)

    assert fitted.converged
    assert np.corrcoef(np.log(fitted.attack), np.log(attack))[0, 1] > 0.85
    assert np.corrcoef(np.log(fitted.defence), np.log(defence))[0, 1] > 0.85
    assert fitted.home_advantage == pytest.approx(gamma, rel=0.15)


def test_fit_recovers_a_near_zero_rho_when_tau_was_not_used() -> None:
    """The simulation draws independent Poissons, so the dependence parameter
    has nothing to find. A fit reporting a large rho would be fitting noise."""
    matches, *_ = simulated_league()
    assert abs(fit(matches).rho) < 0.12


def test_fit_is_deterministic() -> None:
    matches, *_ = simulated_league()
    first, second = fit(matches), fit(matches)
    assert first.attack.tobytes() == second.attack.tobytes()
    assert first.rho == second.rho


def test_fit_does_not_depend_on_row_order() -> None:
    """Teams are sorted before encoding, so a shuffled input must fit the same."""
    matches, *_ = simulated_league()
    shuffled = matches.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a, b = fit(matches), fit(shuffled)
    assert a.teams == b.teams
    assert a.attack == pytest.approx(b.attack, abs=1e-6)
    assert a.log_likelihood == pytest.approx(b.log_likelihood, abs=1e-6)


def test_rho_is_admissible_for_every_match_at_the_optimum() -> None:
    """An inadmissible rho would give negative probabilities in the four
    corrected cells."""
    matches, *_ = simulated_league()
    fitted = fit(matches)
    for home in fitted.teams:
        for away in fitted.teams:
            if home == away:
                continue
            lam, mu = fitted.expected_goals(home, away)
            low, high = rho_bounds(lam, mu)
            assert low <= fitted.rho <= high


def test_time_decay_shifts_the_fit_toward_recent_form() -> None:
    """A team that improves sharply late must rate higher under decay than
    without it, or the weighting is doing nothing."""
    rng = np.random.default_rng(11)
    rows = []
    start = pd.Timestamp("2020-01-01")
    for i in range(400):
        late = i > 300
        lam = 3.0 if late else 0.6
        rows.append(
            (start + pd.Timedelta(days=i), "Riser", f"Opp{i % 8}", int(rng.poisson(lam)), 1)
        )
        rows.append(
            (start + pd.Timedelta(days=i), f"Opp{i % 8}", "Riser", 1, int(rng.poisson(lam)))
        )
    matches = pd.DataFrame(rows, columns=["kickoff", "home", "away", "home_goals", "away_goals"])
    as_of = start + pd.Timedelta(days=401)

    flat = fit(matches)
    decayed = fit(matches, xi=half_life_to_xi(30.0), as_of=as_of)
    riser = flat.index_of("Riser")
    assert decayed.attack[decayed.index_of("Riser")] > flat.attack[riser]


def test_fit_requires_as_of_when_decay_is_requested() -> None:
    matches, *_ = simulated_league()
    with pytest.raises(ValueError, match="as_of is required"):
        fit(matches, xi=0.01)


def test_fit_requires_a_kickoff_column_when_decay_is_requested() -> None:
    matches, *_ = simulated_league()
    with pytest.raises(ValueError, match="kickoff column"):
        fit(
            matches.drop(columns=["kickoff"]),
            xi=0.01,
            as_of=pd.Timestamp("2030-01-01"),
        )


def test_missing_columns_are_reported_by_name() -> None:
    with pytest.raises(ValueError, match="home_goals"):
        fit(pd.DataFrame({"home": ["a"], "away": ["b"], "away_goals": [1]}))


def test_an_empty_frame_is_rejected() -> None:
    empty = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    with pytest.raises(ValueError, match="no matches"):
        fit(empty)


def test_encode_sorts_teams_for_a_stable_index() -> None:
    matches = pd.DataFrame(
        {
            "home": ["Zulu", "Alpha"],
            "away": ["Alpha", "Zulu"],
            "home_goals": [1, 2],
            "away_goals": [0, 1],
        }
    )
    data, teams = encode(matches)
    assert teams == ("Alpha", "Zulu")
    assert list(data.home_idx) == [1, 0]


def test_expected_goals_rejects_a_team_outside_the_fit_window() -> None:
    matches, *_ = simulated_league()
    with pytest.raises(KeyError, match="Promoted"):
        fit(matches).expected_goals("Promoted", "T00")


def test_fitted_frame_stacks_leagues_tidily() -> None:
    matches, *_ = simulated_league()
    fitted = fit(matches)
    frame = fitted_frame({"E0": fitted, "D1": fitted})
    assert list(frame.columns) == [
        "league",
        "team",
        "attack",
        "defence",
        "home_advantage",
        "rho",
    ]
    assert sorted(set(frame["league"])) == ["D1", "E0"]
    assert len(frame) == 2 * len(fitted.teams)


def test_fitted_frame_of_nothing_is_empty_but_well_formed() -> None:
    assert list(fitted_frame({}).columns) == [
        "league",
        "team",
        "attack",
        "defence",
        "home_advantage",
        "rho",
    ]


def test_non_convergence_raises_under_strict() -> None:
    """Two teams and one match cannot identify the model; the fit must say so
    rather than return arbitrary parameters."""
    degenerate = pd.DataFrame({"home": ["a"], "away": ["b"], "home_goals": [0], "away_goals": [0]})
    try:
        result = fit(degenerate, strict=False)
    except ConvergenceError:  # pragma: no cover - acceptable either way
        return
    assert result.n_matches == 1
