"""M1 on a synthetic league where the truth is known.

Shots per 90 are Poisson around a planted per-player rate, so at ``M`` minutes
the per-90 rate has noise variance ``90 * mean_rate / M`` and the reliability is
``var_true / (var_true + 90 * mean_rate / M)``, i.e. ``M / (M + k)`` with
``k = 90 * mean_rate / var_true``. ``shot_rate_sd=0.3`` puts the season-level
reliability near 0.6, where a missing step-up or a wrong weighting shows.
Tolerances were set from 20-40 seeds: split-half minus theory had sd 0.012,
k / k_theory had sd 0.054.
"""

from functools import cache

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from footy.players.reliability import bootstrap_rates, split_half, stabilisation
from footy.players.synthetic import League, simulate_league


@cache
def _league() -> League:
    return simulate_league(
        np.random.default_rng(11), n_teams=30, squad_size=24, n_seasons=2, shot_rate_sd=0.3
    )


def _true_rates(units: pd.DataFrame) -> NDArray[np.float64]:
    rates = _league().truth.set_index("player_id")["shot_rate"]
    return rates.loc[units["player_id"]].to_numpy(dtype=np.float64)


def test_split_half_recovers_planted_reliability() -> None:
    pm = _league().player_match
    seasons = pm.groupby(["player_id", "season"])["minutes"].sum().reset_index()
    lam = _true_rates(seasons)
    harmonic = 1 / (1 / seasons["minutes"]).mean()
    theory = lam.var() / (lam.var() + 90 * lam.mean() / harmonic)
    est = split_half(pm, "shots", np.random.default_rng(0), n_splits=50)
    assert est.n_units > 0.95 * len(seasons)
    assert abs(est.estimate - theory) < 0.05
    assert 0 < est.sd < 0.05


def test_reliability_rises_with_minutes_and_k_matches_theory() -> None:
    pm = _league().player_match
    lam = _true_rates(pm.groupby(["player_id", "season"]).size().reset_index())
    k_theory = 90 * lam.mean() / lam.var()
    st = stabilisation(pm, "shots", np.random.default_rng(1), n_splits=20)
    assert (np.diff(st.curve["reliability"]) > 0).all()
    assert abs(st.k / k_theory - 1) < 0.2
    assert st.minutes_05 == st.k
    assert abs(st.minutes_07 / (st.minutes_07 + st.k) - 0.7) < 1e-9


def test_pure_noise_metric_has_reliability_near_zero() -> None:
    pm = _league().player_match.copy()
    rng = np.random.default_rng(2)
    for season in pm["season"].unique():
        rows = pm["season"] == season
        pm.loc[rows, "player_id"] = rng.permutation(pm.loc[rows, "player_id"].to_numpy())
    est = split_half(pm, "shots", np.random.default_rng(3), n_splits=50)
    assert abs(est.estimate) < 0.1


def test_bootstrap_intervals_cover_true_rates() -> None:
    pm = _league().player_match
    out = bootstrap_rates(pm, "shots", np.random.default_rng(4), n_boot=200, level=0.9)
    assert len(out) == pm.groupby(["player_id", "season"]).ngroups
    lam = _true_rates(out)
    covered = ((out["lo"] <= lam) & (lam <= out["hi"])).mean()
    # Percentile intervals on ~25 matches run slightly under nominal (0.86-0.88 seen).
    assert 0.83 < covered < 0.95
    poisson_se = np.sqrt(90 * lam / out["minutes"])
    assert abs((out["se"] / poisson_se).median() - 1) < 0.1


def test_bootstrap_by_player_pools_seasons() -> None:
    pm = _league().player_match
    out = bootstrap_rates(pm, "shots", np.random.default_rng(5), keys=("player_id",), n_boot=20)
    assert len(out) == pm["player_id"].nunique()
    assert out["minutes"].sum() == pm["minutes"].sum()
    assert (out["lo"] <= out["rate"]).all()
    assert (out["rate"] <= out["hi"]).all()
