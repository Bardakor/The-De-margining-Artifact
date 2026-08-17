"""The walk-forward protocol.

Study spec §5.3 and §5.4. Each league's matches are partitioned in time into
three disjoint periods, fixed before any fitting:

    burn-in       first N seasons    fitting only, never forecast or scored
    calibration   next M seasons     forecast and scored solely to select xi
    evaluation    the remainder      the study; every reported number

Within calibration and evaluation the protocol is: before each matchday, refit
on every match that kicked off strictly earlier, weighted by exp(-xi * age);
predict that matchday; roll forward.

The strict inequality is the single point where leakage could enter, so it is
enforced in one place — :func:`footy.fit.decay.match_age_days` raises on any
match at or after the cutoff — and asserted directly by the test suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from footy.core.dixon_coles import rho_bounds
from footy.core.markets import match_odds, totals
from footy.core.matrix import scoreline_matrix
from footy.data.football_data import season_start_year
from footy.fit.mle import ConvergenceError, FittedParameters, fit

REQUIRED_COLUMNS = ("league", "season", "kickoff", "home", "away", "home_goals", "away_goals")

FORECAST_COLUMNS = (
    "league",
    "season",
    "kickoff",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "outcome",
    "p_home",
    "p_draw",
    "p_away",
    "p_over25",
    "p_under25",
    "lambda_home",
    "lambda_away",
    "rho",
    "n_fit_matches",
)


@dataclass(frozen=True)
class Periods:
    """Season codes assigned to each period, in chronological order."""

    burn_in: tuple[str, ...]
    calibration: tuple[str, ...]
    evaluation: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = (
            set(self.burn_in) & set(self.calibration)
            | set(self.burn_in) & set(self.evaluation)
            | set(self.calibration) & set(self.evaluation)
        )
        if overlap:
            raise ValueError(f"periods must be disjoint; {sorted(overlap)} appear twice")

    @property
    def scored(self) -> tuple[str, ...]:
        return self.calibration + self.evaluation


def split_periods(
    seasons: list[str] | tuple[str, ...],
    *,
    burn_in: int = 3,
    calibration: int = 2,
) -> Periods:
    """Partition an ordered season list into the three periods.

    Raises:
        ValueError: If there are not enough seasons to leave an evaluation
            period. Silently returning an empty evaluation would produce a
            study with no results and no error.
    """
    # Sort CHRONOLOGICALLY, not lexicographically. Season codes wrap the
    # century — "9394" sorts after "0001" as a string — so plain sorted()
    # scrambles the partition: it once assigned burn-in to 2000-2002,
    # calibration to 2023-2024 and evaluation to 1993-1999, which both leaked
    # the newest seasons into parameter selection and left the study measuring
    # years that predate closing odds entirely.
    ordered = tuple(sorted(seasons, key=season_start_year))
    if burn_in < 0 or calibration < 0:
        raise ValueError("period lengths must be non-negative")
    if len(ordered) <= burn_in + calibration:
        raise ValueError(
            f"{len(ordered)} seasons cannot fill burn-in ({burn_in}) plus "
            f"calibration ({calibration}) and still leave an evaluation period"
        )
    return Periods(
        burn_in=ordered[:burn_in],
        calibration=ordered[burn_in : burn_in + calibration],
        evaluation=ordered[burn_in + calibration :],
    )


def outcome_index(home_goals: int, away_goals: int) -> int:
    """0 = home win, 1 = draw, 2 = away win, matching the scoring rules."""
    if home_goals > away_goals:
        return 0
    return 1 if home_goals == away_goals else 2


def matchdays(matches: pd.DataFrame) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """Group by calendar day of kickoff, in chronological order.

    A matchday is the unit of refitting: every fixture on a day is predicted
    from one fit, using only matches that kicked off before that day began.
    """
    days = matches["kickoff"].dt.normalize()
    for day, block in matches.groupby(days, sort=True):
        yield pd.Timestamp(day), block


def walk_forward(
    matches: pd.DataFrame,
    *,
    xi: float,
    scored_seasons: tuple[str, ...],
    min_fit_matches: int = 100,
) -> pd.DataFrame:
    """Run the protocol over one league, returning one row per forecast.

    Args:
        matches: One league's matches, columns per ``REQUIRED_COLUMNS``.
        xi: Frozen decay rate. See spec §5.3 — never tuned on evaluation data.
        scored_seasons: Seasons to forecast. Earlier seasons are fit-only.
        min_fit_matches: Skip a matchday whose fit window is smaller than this.
            Recorded as skipped rather than silently forecast from noise.

    Returns:
        A frame with :data:`FORECAST_COLUMNS`. Matchdays that were skipped, or
        whose fit failed to converge, are absent — the caller compares against
        the input to see what was dropped.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing columns: {missing}")
    if matches.empty:
        return pd.DataFrame(columns=list(FORECAST_COLUMNS))

    frame = matches.sort_values("kickoff", kind="stable").reset_index(drop=True)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"])
    scored = set(scored_seasons)

    rows: list[dict[str, object]] = []
    for day, block in matchdays(frame):
        if not set(block["season"]) & scored:
            continue
        history = frame[frame["kickoff"] < day]
        if len(history) < min_fit_matches:
            continue

        try:
            params = fit(history, xi=xi, as_of=day, strict=True)
        except (ConvergenceError, ValueError):
            continue

        for match in block.itertuples(index=False):
            if match.season not in scored:
                continue
            row = _forecast_row(params, match, len(history))
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows, columns=list(FORECAST_COLUMNS))


MAX_EXPECTED_GOALS = 6.0
"""Above this a fitted rate is degenerate, not a strong team.

The highest per-side expected goals in senior league football sits near 4. A
fit reporting 8.8 has produced an unidentified parameter for a side with too
little effective weight in the decayed window to determine one, which happens
at aggressive half-lives where a team has only two or three matches carrying
any weight at all.
"""

MIN_EXPECTED_GOALS = 0.05


def _forecast_row(params: FittedParameters, match: Any, n_fit: int) -> dict[str, object] | None:
    """One forecast, or None when this fixture cannot be priced from this fit.

    Three ways that happens, all recorded as a dropped fixture rather than a
    fabricated price:

    A team absent from the fit window has no fitted strength — a promoted side.
    Predicting it from a league prior is a modelling decision this study does
    not need, since the transform comparison holds the model fixed.

    A degenerate rate (see :data:`MAX_EXPECTED_GOALS`) means the fit failed to
    identify that team even though it converged.

    An inadmissible rho for THIS fixture. Admissibility depends on the
    fixture's own (lam, mu), so a rho valid for every match in the fit window
    can still be invalid for a pairing that was not in it — the fit-time check
    is structurally unable to catch that, and this is where it surfaces.
    """
    home = str(match.home)
    away = str(match.away)
    try:
        lam, mu = params.expected_goals(home, away)
    except KeyError:
        return None

    if not (MIN_EXPECTED_GOALS < lam < MAX_EXPECTED_GOALS):
        return None
    if not (MIN_EXPECTED_GOALS < mu < MAX_EXPECTED_GOALS):
        return None
    low, high = rho_bounds(lam, mu)
    if not low <= params.rho <= high:
        return None

    matrix = scoreline_matrix(lam, mu, params.rho)
    p_home, p_draw, p_away = match_odds(matrix)
    p_over, p_under = totals(matrix, 2.5)
    home_goals = int(match.home_goals)
    away_goals = int(match.away_goals)

    return {
        "league": match.league,
        "season": match.season,
        "kickoff": match.kickoff,
        "home": home,
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "outcome": outcome_index(home_goals, away_goals),
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over,
        "p_under25": p_under,
        "lambda_home": lam,
        "lambda_away": mu,
        "rho": params.rho,
        "n_fit_matches": n_fit,
    }


def select_xi(
    matches: pd.DataFrame,
    *,
    calibration_seasons: tuple[str, ...],
    candidates: tuple[float, ...],
    min_fit_matches: int = 100,
) -> tuple[float, pd.DataFrame]:
    """Choose xi by minimising mean RPS over the calibration period only.

    Spec §5.3. The returned frame records every candidate's score so the
    choice is auditable, and the winner is then frozen into the
    pre-registration before the evaluation period is ever touched.
    """
    from footy.eval.scoring import ranked_probability_score

    if not candidates:
        raise ValueError("need at least one candidate xi")

    records: list[dict[str, object]] = []
    for xi in candidates:
        forecasts = walk_forward(
            matches,
            xi=xi,
            scored_seasons=calibration_seasons,
            min_fit_matches=min_fit_matches,
        )
        if forecasts.empty:
            records.append({"xi": xi, "mean_rps": np.nan, "n": 0})
            continue
        probabilities = forecasts[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
        outcomes = forecasts["outcome"].to_numpy(dtype=int)
        scores = [
            ranked_probability_score(row, int(outcome))
            for row, outcome in zip(probabilities, outcomes, strict=True)
        ]
        records.append({"xi": xi, "mean_rps": float(np.mean(scores)), "n": len(scores)})

    table = pd.DataFrame(records)
    usable = table.dropna(subset=["mean_rps"])
    if usable.empty:
        raise ValueError("no candidate xi produced any calibration forecasts")
    best = float(
        usable["xi"].to_numpy(dtype=float)[int(np.argmin(usable["mean_rps"].to_numpy(dtype=float)))]
    )
    return best, table
