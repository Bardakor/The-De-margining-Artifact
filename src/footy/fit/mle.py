"""Maximum-likelihood fit of the Dixon-Coles parameters.

Study spec §5.2. L-BFGS-B over the log-parameterisation with the analytic
gradient supplied. Only rho carries a box; the attack, defence and home
advantage parameters are unbounded because the log scale already enforces
positivity.

The admissibility bound on rho,

    max(-1/lam, -1/mu) <= rho <= min(1/(lam*mu), 1)

depends on the parameters and so cannot be imposed as a constant box. It is
checked at the optimum instead, and a violation triggers a refit inside a
tightened box rather than returning a fit that produces negative cell
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from footy.fit.decay import time_weights
from footy.fit.likelihood import (
    MatchData,
    initial_theta,
    negative_log_likelihood,
    negative_log_likelihood_gradient,
    rates,
    unpack,
)

FloatArray = npt.NDArray[np.float64]

DEFAULT_RHO_BOX = (-0.20, 0.20)
MAX_REFITS = 4
REQUIRED_COLUMNS = ("home", "away", "home_goals", "away_goals")


class ConvergenceError(RuntimeError):
    """The optimiser did not converge, and no fallback was requested."""


@dataclass(frozen=True)
class FittedParameters:
    """Fitted strengths, indexed by team name."""

    teams: tuple[str, ...]
    attack: FloatArray
    defence: FloatArray
    home_advantage: float
    rho: float
    log_likelihood: float
    n_matches: int
    n_iterations: int
    converged: bool

    def index_of(self, team: str) -> int:
        try:
            return self.teams.index(team)
        except ValueError as exc:
            raise KeyError(f"team {team!r} was not in the fit window") from exc

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """(lam, mu) for a fixture between two teams present in the fit."""
        h, a = self.index_of(home), self.index_of(away)
        lam = float(self.attack[h] * self.defence[a] * self.home_advantage)
        mu = float(self.attack[a] * self.defence[h])
        return lam, mu

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})


def encode(
    matches: pd.DataFrame, weights: FloatArray | None = None
) -> tuple[MatchData, tuple[str, ...]]:
    """Encode a match frame into integer indices, returning the team ordering.

    Teams are sorted so the encoding — and therefore the fit — does not depend
    on the order matches happened to arrive in.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing columns: {missing}")
    if matches.empty:
        raise ValueError("no matches to fit")

    teams = tuple(sorted(set(matches["home"]) | set(matches["away"])))
    lookup = {team: i for i, team in enumerate(teams)}
    if weights is None:
        weights = np.ones(len(matches), dtype=np.float64)

    data = MatchData(
        home_idx=matches["home"].map(lookup).to_numpy(dtype=np.int_),
        away_idx=matches["away"].map(lookup).to_numpy(dtype=np.int_),
        home_goals=matches["home_goals"].to_numpy(dtype=np.int_),
        away_goals=matches["away_goals"].to_numpy(dtype=np.int_),
        weights=np.asarray(weights, dtype=np.float64),
        n_teams=len(teams),
    )
    return data, teams


def fit_encoded(
    data: MatchData,
    teams: tuple[str, ...],
    *,
    rho_box: tuple[float, float] = DEFAULT_RHO_BOX,
    strict: bool = True,
) -> FittedParameters:
    """Run L-BFGS-B, then enforce rho admissibility at the optimum."""
    box: list[tuple[float | None, float | None]] = [(None, None)] * (data.n_parameters - 1)
    lo, hi = rho_box

    for _attempt in range(MAX_REFITS):
        result = minimize(
            negative_log_likelihood,
            initial_theta(data.n_teams),
            args=(data,),
            jac=negative_log_likelihood_gradient,
            method="L-BFGS-B",
            bounds=[*box, (lo, hi)],
            options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 500},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        violation = _worst_rho_violation(theta, data)
        if violation <= 0.0:
            break
        # Shrink the box toward zero and refit; rho = 0 is always admissible.
        lo, hi = lo * 0.5, hi * 0.5
    else:
        if strict:
            raise ConvergenceError(
                f"rho remained inadmissible after {MAX_REFITS} refits "
                f"(worst violation {violation:.3e})"
            )

    if strict and not result.success:
        raise ConvergenceError(f"L-BFGS-B did not converge: {result.message}")

    params = unpack(theta, data.n_teams)
    return FittedParameters(
        teams=teams,
        attack=params.attack,
        defence=params.defence,
        home_advantage=params.home_advantage,
        rho=params.rho,
        log_likelihood=-float(result.fun),
        n_matches=data.n_matches,
        n_iterations=int(result.nit),
        converged=bool(result.success),
    )


def _worst_rho_violation(theta: FloatArray, data: MatchData) -> float:
    """How far outside the admissible region rho sits, over all matches.

    Positive means inadmissible. Zero or below means every match's four
    corrected cells stay non-negative.
    """
    lam, mu, rho = rates(theta, data)
    # Vectorised form of core.dixon_coles.rho_bounds. This runs on every fit,
    # and a Python loop over matches here would dominate the fit itself.
    low = np.maximum(-1.0 / lam, -1.0 / mu)
    high = np.minimum(1.0 / (lam * mu), 1.0)
    return float(max(0.0, np.max(low - rho), np.max(rho - high)))


def fit(
    matches: pd.DataFrame,
    *,
    xi: float = 0.0,
    as_of: pd.Timestamp | None = None,
    rho_box: tuple[float, float] = DEFAULT_RHO_BOX,
    strict: bool = True,
) -> FittedParameters:
    """Fit Dixon-Coles to a match frame.

    Args:
        matches: Columns ``home``, ``away``, ``home_goals``, ``away_goals``,
            and ``kickoff`` when ``xi > 0``.
        xi: Exponential decay rate per day. Zero gives an unweighted fit.
        as_of: Fit cutoff. Every match must have kicked off strictly before it.
            Required when ``xi > 0``.
        rho_box: Initial box for rho, shrunk on an admissibility violation.
        strict: Raise on non-convergence rather than returning the best effort.
    """
    if xi > 0.0:
        if as_of is None:
            raise ValueError("as_of is required when xi > 0")
        if "kickoff" not in matches.columns:
            raise ValueError("matches must carry a kickoff column when xi > 0")
        weights = time_weights(matches["kickoff"], as_of, xi)
    else:
        weights = None

    data, teams = encode(matches, weights)
    return fit_encoded(data, teams, rho_box=rho_box, strict=strict)


def fitted_frame(fits: dict[str, FittedParameters]) -> pd.DataFrame:
    """Stack several league fits into one tidy parameter table."""
    frames: list[pd.DataFrame] = []
    for league, params in sorted(fits.items()):
        frame = params.as_frame()
        frame.insert(0, "league", league)
        frame["home_advantage"] = params.home_advantage
        frame["rho"] = params.rho
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["league", "team", "attack", "defence", "home_advantage", "rho"]
        )
    return pd.concat(frames, ignore_index=True)
