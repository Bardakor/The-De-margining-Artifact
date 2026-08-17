"""Weighted Dixon-Coles log-likelihood and its analytic gradient.

MODEL.md §4, study spec §5.2. The likelihood over matches m is

    l = sum_m w_m [ log tau(x,y,lam,mu,rho)
                    + x log lam - lam + y log mu - mu - log x! - log y! ]

with lam = alpha_home * beta_away * gamma and mu = alpha_away * beta_home.

Two decisions make this fast enough for a walk-forward with thousands of
refits, and both are why this module exists rather than a one-line call to a
generic optimiser:

**Log parameterisation.** We optimise over a = log(alpha), b = log(beta),
g = log(gamma). Positivity is then structural rather than a bound, and the
derivatives collapse: d(lam)/d(a_home) = lam.

**Analytic gradient.** Hand-derived rather than finite-differenced. Finite
differences cost one likelihood evaluation per parameter — around 80 for a
20-team league — so the analytic form is the difference between minutes and
hours over a full study. It is verified against ``scipy.optimize.check_grad``
in the test suite, because a subtly wrong gradient converges silently to the
wrong optimum and no downstream test would notice.

The factorial terms do not depend on any parameter, so they are dropped from
the objective. They shift the likelihood by a constant and change neither the
optimum nor the gradient; :func:`log_factorial_offset` recovers them where a
comparable absolute likelihood is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

UNREACHABLE = 1e12
"""Objective value returned for parameters outside the admissible region.

Returned rather than raised so L-BFGS-B can back out of a bad trial step
instead of aborting the whole fit.
"""


@dataclass(frozen=True)
class MatchData:
    """Matches encoded as integer team indices, ready for the likelihood."""

    home_idx: IntArray
    away_idx: IntArray
    home_goals: IntArray
    away_goals: IntArray
    weights: FloatArray
    n_teams: int

    def __post_init__(self) -> None:
        lengths = {
            len(self.home_idx),
            len(self.away_idx),
            len(self.home_goals),
            len(self.away_goals),
            len(self.weights),
        }
        if len(lengths) != 1:
            raise ValueError(f"ragged match arrays: lengths {sorted(lengths)}")
        if self.n_matches == 0:
            raise ValueError("no matches to fit")
        if self.n_teams < 2:
            raise ValueError(f"need at least two teams, got {self.n_teams}")
        for name, idx in (("home", self.home_idx), ("away", self.away_idx)):
            if idx.min() < 0 or idx.max() >= self.n_teams:
                raise ValueError(f"{name} index outside [0, {self.n_teams})")
        if np.any(self.home_idx == self.away_idx):
            raise ValueError("a team cannot play itself")
        if np.any(self.weights < 0.0):
            raise ValueError("weights must be non-negative")
        if self.home_goals.min() < 0 or self.away_goals.min() < 0:
            raise ValueError("goals must be non-negative")

    @property
    def n_matches(self) -> int:
        return int(len(self.home_idx))

    @property
    def n_parameters(self) -> int:
        """Free parameters: attack (one fewer, see below), defence, gamma, rho."""
        return (self.n_teams - 1) + self.n_teams + 2


@dataclass(frozen=True)
class Parameters:
    """Unpacked parameters on their natural scale."""

    attack: FloatArray
    defence: FloatArray
    home_advantage: float
    rho: float


def initial_theta(n_teams: int) -> FloatArray:
    """A neutral start: every team average, mild home advantage, DC's rho.

    Starting at the reference value rho = -0.10 rather than 0 matters: at
    rho = 0 the tau derivatives for the (0,1) and (1,0) cells vanish, so the
    optimiser gets no first-order signal about rho from those cells.
    """
    if n_teams < 2:
        raise ValueError(f"need at least two teams, got {n_teams}")
    theta = np.zeros(2 * n_teams + 1, dtype=np.float64)
    theta[-2] = np.log(1.35)  # home advantage, roughly the league-average value
    theta[-1] = -0.10
    return theta


def unpack(theta: FloatArray, n_teams: int) -> Parameters:
    """Split the free vector into parameters on their natural scale.

    The likelihood has exactly one flat direction: scaling every alpha by c
    and every beta by 1/c leaves lam and mu unchanged. One constraint removes
    it. We drop the last attack parameter and set it to minus the sum of the
    rest, so sum(log alpha) = 0 — the geometric mean of alpha is 1.

    MODEL.md §2 states the constraint as an arithmetic mean of 1. Both fix the
    same flat direction and identify the same model; they differ only in where
    along it the solution is reported. Documented rather than silently adopted.
    """
    expected = 2 * n_teams + 1
    if theta.shape != (expected,):
        raise ValueError(f"theta must have {expected} entries, got {theta.shape}")
    free_attack = theta[: n_teams - 1]
    log_attack = np.concatenate([free_attack, [-free_attack.sum()]])
    log_defence = theta[n_teams - 1 : 2 * n_teams - 1]
    return Parameters(
        attack=np.exp(log_attack),
        defence=np.exp(log_defence),
        home_advantage=float(np.exp(theta[-2])),
        rho=float(theta[-1]),
    )


def rates(theta: FloatArray, data: MatchData) -> tuple[FloatArray, FloatArray, float]:
    """Expected goals for each match, plus rho."""
    params = unpack(theta, data.n_teams)
    lam = params.attack[data.home_idx] * params.defence[data.away_idx] * params.home_advantage
    mu = params.attack[data.away_idx] * params.defence[data.home_idx]
    return lam, mu, params.rho


def _tau_terms(
    lam: FloatArray, mu: FloatArray, rho: float, x: IntArray, y: IntArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """tau and its partial derivatives, non-zero in four cells only.

    MODEL.md §3:
        (0,0)  tau = 1 - lam*mu*rho   d/dlam = -mu*rho   d/dmu = -lam*rho   d/drho = -lam*mu
        (0,1)  tau = 1 + lam*rho      d/dlam = rho                          d/drho = lam
        (1,0)  tau = 1 + mu*rho                          d/dmu = rho        d/drho = mu
        (1,1)  tau = 1 - rho                                                d/drho = -1
    """
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    at_00 = (x == 0) & (y == 0)
    at_01 = (x == 0) & (y == 1)
    at_10 = (x == 1) & (y == 0)
    at_11 = (x == 1) & (y == 1)

    tau[at_00] = 1.0 - lam[at_00] * mu[at_00] * rho
    d_lam[at_00] = -mu[at_00] * rho
    d_mu[at_00] = -lam[at_00] * rho
    d_rho[at_00] = -lam[at_00] * mu[at_00]

    tau[at_01] = 1.0 + lam[at_01] * rho
    d_lam[at_01] = rho
    d_rho[at_01] = lam[at_01]

    tau[at_10] = 1.0 + mu[at_10] * rho
    d_mu[at_10] = rho
    d_rho[at_10] = mu[at_10]

    tau[at_11] = 1.0 - rho
    d_rho[at_11] = -1.0

    return tau, d_lam, d_mu, d_rho


def log_factorial_offset(data: MatchData) -> float:
    """The constant the objective drops: sum_m w_m [log x! + log y!]."""
    terms = gammaln(data.home_goals + 1.0) + gammaln(data.away_goals + 1.0)
    return float(np.sum(data.weights * terms))


def negative_log_likelihood(theta: FloatArray, data: MatchData) -> float:
    """Objective for the minimiser. Constant factorial terms are omitted."""
    lam, mu, rho = rates(theta, data)
    tau, *_ = _tau_terms(lam, mu, rho, data.home_goals, data.away_goals)
    if np.any(tau <= 0.0) or not np.all(np.isfinite(lam)) or not np.all(np.isfinite(mu)):
        return UNREACHABLE
    terms = np.log(tau) + data.home_goals * np.log(lam) - lam + data.away_goals * np.log(mu) - mu
    value = -float(np.sum(data.weights * terms))
    return value if np.isfinite(value) else UNREACHABLE


def negative_log_likelihood_gradient(theta: FloatArray, data: MatchData) -> FloatArray:
    """Analytic gradient of :func:`negative_log_likelihood`.

    Per-team sums are scatter-adds over the match arrays (``np.bincount``), so
    the whole gradient is computed without a Python-level loop over matches.

    The last attack parameter is not free — it is minus the sum of the others —
    so its gradient is folded into the free entries by the chain rule:
    ``d/d a_i = g_i - g_last``.
    """
    n = data.n_teams
    lam, mu, rho = rates(theta, data)
    tau, d_lam, d_mu, d_rho = _tau_terms(lam, mu, rho, data.home_goals, data.away_goals)
    if np.any(tau <= 0.0):
        return np.zeros(data.n_parameters, dtype=np.float64)

    # d(loglik)/d(log lam_m) and d(loglik)/d(log mu_m)
    s_lam = data.weights * ((d_lam / tau) * lam + data.home_goals - lam)
    s_mu = data.weights * ((d_mu / tau) * mu + data.away_goals - mu)

    grad_attack = np.bincount(data.home_idx, s_lam, minlength=n) + np.bincount(
        data.away_idx, s_mu, minlength=n
    )
    grad_defence = np.bincount(data.away_idx, s_lam, minlength=n) + np.bincount(
        data.home_idx, s_mu, minlength=n
    )
    grad_attack_free = grad_attack[:-1] - grad_attack[-1]
    grad_gamma = float(np.sum(s_lam))
    grad_rho = float(np.sum(data.weights * d_rho / tau))

    gradient = np.concatenate([grad_attack_free, grad_defence, [grad_gamma], [grad_rho]])
    return -gradient
