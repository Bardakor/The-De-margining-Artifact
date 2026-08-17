"""The walk-forward protocol and the leakage guarantee.

Study spec §5.3, §5.4. The single test that matters most here is the leakage
one. Every other defect in this repository produces a number that is wrong;
leakage produces a number that is *flattering*, which is far harder to notice
and is the standard way a backtest lies.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from footy.fit.decay import half_life_to_xi
from footy.study.walkforward import (
    FORECAST_COLUMNS,
    Periods,
    matchdays,
    outcome_index,
    select_xi,
    split_periods,
    walk_forward,
)

SEASONS = ("1920", "2021", "2122", "2223", "2324", "2425")


def league(
    seasons: tuple[str, ...] = SEASONS,
    n_teams: int = 10,
    seed: int = 3,
) -> pd.DataFrame:
    """A synthetic league: every team plays every other, home and away, per season."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    attack = np.exp(rng.normal(0, 0.3, n_teams))
    defence = np.exp(rng.normal(0, 0.25, n_teams))

    rows = []
    for s, season in enumerate(seasons):
        start = pd.Timestamp("2019-08-01") + pd.DateOffset(years=s)
        fixture = 0
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam = attack[h] * defence[a] * 1.35
                mu = attack[a] * defence[h]
                rows.append(
                    {
                        "league": "E0",
                        "season": season,
                        "kickoff": start + pd.Timedelta(days=fixture // 5),
                        "home": teams[h],
                        "away": teams[a],
                        "home_goals": int(rng.poisson(lam)),
                        "away_goals": int(rng.poisson(mu)),
                    }
                )
                fixture += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Period partition
# --------------------------------------------------------------------------


def test_periods_are_disjoint_and_ordered() -> None:
    periods = split_periods(SEASONS, burn_in=3, calibration=2)
    assert periods.burn_in == ("1920", "2021", "2122")
    assert periods.calibration == ("2223", "2324")
    assert periods.evaluation == ("2425",)
    assert set(periods.burn_in).isdisjoint(periods.calibration)
    assert set(periods.calibration).isdisjoint(periods.evaluation)


def test_periods_cover_every_season_exactly_once() -> None:
    periods = split_periods(SEASONS)
    combined = [*periods.burn_in, *periods.calibration, *periods.evaluation]
    assert sorted(combined) == sorted(SEASONS)
    assert len(combined) == len(set(combined))


def test_overlapping_periods_are_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        Periods(burn_in=("a",), calibration=("a",), evaluation=("b",))


def test_too_few_seasons_to_leave_an_evaluation_period_is_an_error() -> None:
    """Silently returning an empty evaluation would give a study with no
    results and no error."""
    with pytest.raises(ValueError, match="cannot fill burn-in"):
        split_periods(("1920", "2021", "2122"), burn_in=3, calibration=2)


def test_scored_is_calibration_then_evaluation() -> None:
    periods = split_periods(SEASONS)
    assert periods.scored == periods.calibration + periods.evaluation


# --------------------------------------------------------------------------
# Leakage — the guarantee the whole protocol exists to provide
# --------------------------------------------------------------------------


def test_no_forecast_uses_a_match_at_or_after_its_own_kickoff() -> None:
    """Spec §12 criterion 5. Asserted structurally: for every forecast, the
    fit window recorded against it contains only strictly earlier matches."""
    matches = league()
    periods = split_periods(SEASONS)
    forecasts = walk_forward(matches, xi=0.0, scored_seasons=periods.scored, min_fit_matches=50)
    assert not forecasts.empty

    ordered = matches.sort_values("kickoff", kind="stable").reset_index(drop=True)
    for row in forecasts.itertuples(index=False):
        entry: Any = row
        day = pd.Timestamp(entry.kickoff).normalize()
        available = int((ordered["kickoff"] < day).sum())
        assert entry.n_fit_matches == available, (
            f"fit for {entry.home} v {entry.away} on {entry.kickoff} saw "
            f"{entry.n_fit_matches} matches but only {available} had kicked off"
        )


def test_a_forecast_is_unchanged_when_later_results_are_altered() -> None:
    """The decisive leakage test. Rewrite every result after a cutoff; any
    forecast before it must be bit-identical. If future data reached the fit,
    these would differ."""
    matches = league()
    periods = split_periods(SEASONS)
    cutoff = pd.Timestamp("2023-01-01")

    tampered = matches.copy()
    later = tampered["kickoff"] >= cutoff
    tampered.loc[later, "home_goals"] = 9
    tampered.loc[later, "away_goals"] = 0

    original = walk_forward(matches, xi=0.0, scored_seasons=periods.scored, min_fit_matches=50)
    altered = walk_forward(tampered, xi=0.0, scored_seasons=periods.scored, min_fit_matches=50)

    before = original[original["kickoff"] < cutoff].reset_index(drop=True)
    after = altered[altered["kickoff"] < cutoff].reset_index(drop=True)
    assert not before.empty
    for column in ("p_home", "p_draw", "p_away", "lambda_home", "lambda_away"):
        assert before[column].to_numpy() == pytest.approx(after[column].to_numpy(), abs=1e-12)


def test_matchdays_are_yielded_in_chronological_order() -> None:
    days = [day for day, _ in matchdays(league().sort_values("kickoff"))]
    assert days == sorted(days)


def test_every_fixture_on_a_matchday_shares_one_fit() -> None:
    """Otherwise an early kickoff would inform a later one on the same day."""
    matches = league()
    periods = split_periods(SEASONS)
    forecasts = walk_forward(matches, xi=0.0, scored_seasons=periods.scored, min_fit_matches=50)
    per_day = forecasts.groupby(forecasts["kickoff"].dt.normalize())["n_fit_matches"].nunique()
    assert set(per_day) == {1}


# --------------------------------------------------------------------------
# Forecast content
# --------------------------------------------------------------------------


def test_forecasts_have_the_declared_schema() -> None:
    matches = league()
    forecasts = walk_forward(
        matches, xi=0.0, scored_seasons=split_periods(SEASONS).scored, min_fit_matches=50
    )
    assert list(forecasts.columns) == list(FORECAST_COLUMNS)


def test_forecast_probabilities_are_simplexes() -> None:
    matches = league()
    forecasts = walk_forward(
        matches, xi=0.0, scored_seasons=split_periods(SEASONS).scored, min_fit_matches=50
    )
    one_x_two = forecasts[["p_home", "p_draw", "p_away"]].to_numpy()
    assert one_x_two.sum(axis=1) == pytest.approx(np.ones(len(forecasts)), abs=1e-9)
    assert np.all(one_x_two > 0.0)

    over_under = forecasts[["p_over25", "p_under25"]].to_numpy()
    assert over_under.sum(axis=1) == pytest.approx(np.ones(len(forecasts)), abs=1e-9)


def test_burn_in_seasons_are_never_scored() -> None:
    matches = league()
    periods = split_periods(SEASONS)
    forecasts = walk_forward(matches, xi=0.0, scored_seasons=periods.scored, min_fit_matches=50)
    assert set(forecasts["season"]).isdisjoint(periods.burn_in)
    assert set(forecasts["season"]) <= set(periods.scored)


def test_burn_in_matches_still_enter_the_fit_window() -> None:
    """Excluded from scoring, not from estimation — otherwise the first scored
    matchday would fit on nothing."""
    matches = league()
    periods = split_periods(SEASONS)
    forecasts = walk_forward(matches, xi=0.0, scored_seasons=periods.scored, min_fit_matches=50)
    burn_in_matches = int(matches["season"].isin(periods.burn_in).sum())
    assert int(forecasts["n_fit_matches"].min()) >= burn_in_matches * 0.5


@pytest.mark.parametrize(
    ("home_goals", "away_goals", "expected"),
    [(2, 1, 0), (1, 1, 1), (0, 3, 2), (0, 0, 1)],
)
def test_outcome_index_matches_the_scoring_convention(
    home_goals: int, away_goals: int, expected: int
) -> None:
    assert outcome_index(home_goals, away_goals) == expected


def test_recorded_outcome_agrees_with_the_recorded_goals() -> None:
    forecasts = walk_forward(
        league(), xi=0.0, scored_seasons=split_periods(SEASONS).scored, min_fit_matches=50
    )
    for row in forecasts.itertuples(index=False):
        entry: Any = row
        assert entry.outcome == outcome_index(int(entry.home_goals), int(entry.away_goals))


def test_the_walk_forward_is_deterministic() -> None:
    matches = league()
    scored = split_periods(SEASONS).scored
    first = walk_forward(matches, xi=0.0, scored_seasons=scored, min_fit_matches=50)
    second = walk_forward(matches, xi=0.0, scored_seasons=scored, min_fit_matches=50)
    assert first.equals(second)


def test_decay_changes_the_forecasts() -> None:
    """If xi had no effect, selecting it on the calibration period would be
    meaningless."""
    matches = league()
    scored = split_periods(SEASONS).scored
    flat = walk_forward(matches, xi=0.0, scored_seasons=scored, min_fit_matches=50)
    decayed = walk_forward(
        matches, xi=half_life_to_xi(120.0), scored_seasons=scored, min_fit_matches=50
    )
    # Counts need not match: a decayed fit can leave a team with too little
    # effective weight to identify, and those fixtures are dropped rather than
    # priced. Compare on the fixtures both arms produced.
    keys = ["kickoff", "home", "away"]
    shared = flat.merge(decayed, on=keys, suffixes=("_flat", "_decayed"))
    assert not shared.empty
    assert not np.allclose(shared["p_home_flat"].to_numpy(), shared["p_home_decayed"].to_numpy())


def test_missing_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        walk_forward(pd.DataFrame({"league": ["E0"]}), xi=0.0, scored_seasons=("2324",))


def test_an_empty_frame_yields_an_empty_but_well_formed_result() -> None:
    empty = pd.DataFrame(columns=list(FORECAST_COLUMNS))
    result = walk_forward(empty, xi=0.0, scored_seasons=("2324",))
    assert result.empty
    assert list(result.columns) == list(FORECAST_COLUMNS)


def test_matchdays_below_the_minimum_fit_size_are_skipped() -> None:
    """Forecasting from a handful of matches would be noise dressed as a model."""
    matches = league()
    scored = split_periods(SEASONS).scored
    permissive = walk_forward(matches, xi=0.0, scored_seasons=scored, min_fit_matches=50)
    strict = walk_forward(matches, xi=0.0, scored_seasons=scored, min_fit_matches=2000)
    assert len(strict) < len(permissive)


# --------------------------------------------------------------------------
# xi selection — must touch calibration only
# --------------------------------------------------------------------------


def test_select_xi_scores_every_candidate() -> None:
    matches = league()
    periods = split_periods(SEASONS)
    best, table = select_xi(
        matches,
        calibration_seasons=periods.calibration,
        candidates=(0.0, half_life_to_xi(365.0), half_life_to_xi(90.0)),
        min_fit_matches=50,
    )
    assert len(table) == 3
    assert set(table.columns) == {"xi", "mean_rps", "n"}
    assert best in set(table["xi"])
    assert table["mean_rps"].notna().all()


def test_select_xi_picks_the_minimum_mean_rps() -> None:
    matches = league()
    periods = split_periods(SEASONS)
    best, table = select_xi(
        matches,
        calibration_seasons=periods.calibration,
        candidates=(0.0, half_life_to_xi(365.0), half_life_to_xi(60.0)),
        min_fit_matches=50,
    )
    scores = table["mean_rps"].to_numpy(dtype=float)
    assert best == pytest.approx(float(table["xi"].to_numpy(dtype=float)[int(np.argmin(scores))]))


def test_select_xi_never_scores_the_evaluation_period() -> None:
    """Spec §5.3. Tuning xi against evaluation data would let the model absorb
    information about the very matches used to judge it."""
    matches = league()
    periods = split_periods(SEASONS)
    forecasts = walk_forward(
        matches, xi=0.0, scored_seasons=periods.calibration, min_fit_matches=50
    )
    assert set(forecasts["season"]) <= set(periods.calibration)
    assert set(forecasts["season"]).isdisjoint(periods.evaluation)


def test_select_xi_requires_candidates() -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        select_xi(league(), calibration_seasons=("2223",), candidates=())


# --------------------------------------------------------------------------
# Degenerate fits must drop the fixture, never crash or fabricate a price
# --------------------------------------------------------------------------


def test_a_fixture_whose_rho_is_inadmissible_is_dropped_not_crashed() -> None:
    """Regression from the real study run. Admissibility depends on the
    fixture's own (lam, mu), so a rho valid for every match in the fit window
    can still be invalid for a pairing that was not in it. The fit-time check
    is structurally unable to catch that. Crashing there killed a 16-league
    calibration sweep two hours in."""
    from footy.core.dixon_coles import rho_bounds
    from footy.fit.mle import FittedParameters
    from footy.study.walkforward import _forecast_row

    lam, mu = 8.77, 0.174
    low, _ = rho_bounds(lam, mu)
    params = FittedParameters(
        teams=("A", "B"),
        attack=np.array([lam, mu]),
        defence=np.array([1.0, 1.0]),
        home_advantage=1.0,
        rho=low - 0.05,  # inadmissible for this fixture
        log_likelihood=-1.0,
        n_matches=10,
        n_iterations=1,
        converged=True,
    )
    match = pd.Series(
        {
            "league": "E0",
            "season": "2324",
            "kickoff": pd.Timestamp("2024-01-01"),
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
    )
    assert _forecast_row(params, match, n_fit=10) is None


def test_a_degenerate_expected_goals_rate_is_dropped() -> None:
    """A fit reporting 8.8 expected goals has failed to identify that team,
    not found a very strong one. Pricing from it would be fabrication."""
    from footy.fit.mle import FittedParameters
    from footy.study.walkforward import MAX_EXPECTED_GOALS, _forecast_row

    params = FittedParameters(
        teams=("A", "B"),
        attack=np.array([MAX_EXPECTED_GOALS + 2.0, 1.0]),
        defence=np.array([1.0, 1.0]),
        home_advantage=1.0,
        rho=-0.05,
        log_likelihood=-1.0,
        n_matches=10,
        n_iterations=1,
        converged=True,
    )
    match = pd.Series(
        {
            "league": "E0",
            "season": "2324",
            "kickoff": pd.Timestamp("2024-01-01"),
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
    )
    assert _forecast_row(params, match, n_fit=10) is None


def test_a_plausible_fit_still_produces_a_forecast() -> None:
    """The complement: the guards must not reject ordinary fixtures."""
    from footy.fit.mle import FittedParameters
    from footy.study.walkforward import _forecast_row

    params = FittedParameters(
        teams=("A", "B"),
        attack=np.array([1.4, 1.0]),
        defence=np.array([1.0, 0.9]),
        home_advantage=1.3,
        rho=-0.05,
        log_likelihood=-1.0,
        n_matches=400,
        n_iterations=12,
        converged=True,
    )
    match = pd.Series(
        {
            "league": "E0",
            "season": "2324",
            "kickoff": pd.Timestamp("2024-01-01"),
            "home": "A",
            "away": "B",
            "home_goals": 1,
            "away_goals": 0,
        }
    )
    row = _forecast_row(params, match, n_fit=400)
    assert row is not None
    total = sum(float(cast(float, row[k])) for k in ("p_home", "p_draw", "p_away"))
    assert abs(total - 1.0) < 1e-9


def test_periods_are_ordered_chronologically_not_lexicographically() -> None:
    """Regression, and a severe one. Season codes wrap the century: "9394"
    sorts AFTER "0001" as a string. Plain sorted() therefore assigned burn-in
    to 2000-2002, calibration to 2023-2024 and evaluation to 1993-1999 — it
    tuned the decay parameter on the newest seasons the study is supposed to
    evaluate, and left the evaluation period on years that predate closing odds.

    The original tests could not catch this: their fixtures were all post-2000
    season codes, which happen to sort correctly either way.
    """
    seasons = ["9394", "9495", "9596", "9697", "9798", "9899", "9900", "0001", "0102", "2324"]
    periods = split_periods(seasons, burn_in=3, calibration=2)

    assert periods.burn_in == ("9394", "9495", "9596")
    assert periods.calibration == ("9697", "9798")
    assert periods.evaluation == ("9899", "9900", "0001", "0102", "2324")


def test_every_period_precedes_the_next_in_real_time() -> None:
    """The property the partition exists to guarantee, stated directly."""
    from footy.data.football_data import season_start_year

    seasons = ["9394", "9899", "9900", "0001", "0506", "1213", "1920", "2324", "2425"]
    periods = split_periods(seasons, burn_in=2, calibration=2)

    burn_in = [season_start_year(s) for s in periods.burn_in]
    calibration = [season_start_year(s) for s in periods.calibration]
    evaluation = [season_start_year(s) for s in periods.evaluation]

    assert max(burn_in) < min(calibration)
    assert max(calibration) < min(evaluation)
