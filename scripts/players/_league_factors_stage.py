"""Stage ``factors`` of run_league.py: box-score league factors from movers (A20 ii).

Cells are player x season x league sums over the panel's league matches
(scope == "league", minutes > 0). One conditional fixed-effects Poisson fit per
metric (``footy.players.league_factors``), B = 200 player bootstrap seeded by
task ``factors/<metric>``, 90% percentile intervals.

Choices A20 does not fix (post hoc, Claude's calls):

- Linking players: at least 450 league minutes, summed over the window, in the
  league and in another one, the count of docs/unmeasured/league-strength.md §2.2
  that §3.2 applies the threshold of 100 to. Counted on every cell, before the
  birth-date filter. The number of leagues that pass with any minutes is recorded.
- Age: at the midpoint between the first and the last match of the league-season
  in the panel (a calendar-year league's season is half a year later than its
  label), in years of 365.25 days, centred at 26. A player without a birth date
  is dropped; counted.
- Selection check (§3.2): a cell is "before a move" when the player's next cell,
  ordered by the date of the cell's last match, is in another league. Two point
  fits per metric: with that dummy, and without those cells.
- Leagues holding pool player-seasons: a pool player with at least 900 league
  minutes in the league in one season.

Outputs in results/players/league/: factors.csv (one row per metric and league),
age_curve.csv (rate relative to age 26), selection_check.csv (one row per metric
and league parameter), and ``factors_*`` entries in summary.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from run_impact import record, rng, to_csv

from footy.players.league_factors import (
    POOLED,
    REFERENCE,
    bootstrap,
    fit,
    league_parameters,
    pre_move,
    prepare_cells,
    summarise,
)
from footy.players.paths import PAPER, PROCESSED, RESULTS

LEAGUE_DIR = RESULTS / "league"
FACTOR_METRICS = ("goals", "np_goals", "assists")
FACTOR_BOOT = 200
FACTOR_LEVEL = 0.9
AGE_REF = 26.0
AGES = np.arange(17.0, 41.0)
POOL_SEASON_MINUTES = 900.0


def league_cells() -> pd.DataFrame:
    """Player x season x league cells: minutes, counts, last match date, centred age."""
    pm = pd.read_csv(
        PROCESSED / "panel_player_match.csv.gz",
        usecols=["player_id", "season", "competition_id", "scope", "minutes", "date"]
        + list(FACTOR_METRICS),
        parse_dates=["date"],
    )
    pm = pm[pm["scope"] == "league"].rename(columns={"competition_id": "league"})
    sums = {m: (m, "sum") for m in ("minutes", *FACTOR_METRICS)}
    cells = pm.groupby(["player_id", "season", "league"], as_index=False).agg(
        **sums, last_date=("date", "max")
    )
    cells = cells[cells["minutes"] > 0]
    span = pm.groupby(["season", "league"])["date"].agg(["min", "max"])
    midpoint = (span["min"] + (span["max"] - span["min"]) / 2).rename("midpoint")
    born = pd.read_csv(
        PROCESSED / "tm_players.csv.gz",
        usecols=["player_id", "date_of_birth"],
        parse_dates=["date_of_birth"],
    ).set_index("player_id")["date_of_birth"]
    cells = cells.join(midpoint, on=["season", "league"])
    age = (cells["midpoint"] - cells["player_id"].map(born)).dt.days / 365.25 - AGE_REF
    return cells.assign(age=age, age2=age**2).drop(columns="midpoint")


def stage_factors() -> None:
    cells = league_cells()
    rule = league_parameters(cells)
    rule["status"] = np.select(
        [rule["league"] == REFERENCE, rule["parameter"] == POOLED], ["reference", "pooled"], "own"
    )
    parameter_of = dict(zip(rule["league"], rule["parameter"], strict=True))
    n_any_minutes = int((league_parameters(cells, min_minutes=0.0)["parameter"] != POOLED).sum())
    no_age = cells["age"].isna()
    entries: dict[str, tuple[Any, str]] = {
        "factors_cells": (len(cells), "player x season x league cells with league minutes > 0"),
        "factors_players": (cells["player_id"].nunique(), "players with league minutes"),
        "factors_players_no_birth_date": (
            cells.loc[no_age, "player_id"].nunique(),
            "players dropped: no date of birth",
        ),
        "factors_cells_no_birth_date": (int(no_age.sum()), "their cells"),
        "factors_leagues_own": (
            int((rule["status"] == "own").sum()),
            "leagues with their own gamma, the reference not counted",
        ),
        "factors_leagues_pooled": (
            int((rule["status"] == "pooled").sum()),
            "leagues sharing one gamma",
        ),
        "factors_leagues_any_minutes": (
            n_any_minutes,
            "leagues, the reference included, with >= 100 linking players if any minutes count",
        ),
    }
    cells = cells[~no_age].reset_index(drop=True)
    cells["pre_move"] = pre_move(cells, "last_date").astype(np.float64)
    pool = pd.read_csv(PAPER / "data" / "pool.csv")["tm_player_id"]
    held = cells["player_id"].isin(pool) & (cells["minutes"] >= POOL_SEASON_MINUTES)
    pool_leagues = sorted(cells.loc[held, "league"].unique())
    pool_parameters = sorted({parameter_of[x] for x in pool_leagues} - {REFERENCE})
    entries["factors_cells_pre_move"] = (int(cells["pre_move"].sum()), "cells before a move")
    entries["factors_pool_leagues"] = (pool_leagues, "leagues holding a pool player-season")

    factors, curves, checks = [], [], []
    years = AGES - AGE_REF
    quantiles = [(1 - FACTOR_LEVEL) / 2, (1 + FACTOR_LEVEL) / 2]
    for metric in FACTOR_METRICS:
        c = prepare_cells(cells, metric, parameter_of)
        estimate = fit(c)
        reps = bootstrap(c, estimate, rng(f"factors/{metric}"), n_boot=FACTOR_BOOT)
        table = summarise(estimate, reps, level=FACTOR_LEVEL).rename(columns={"estimate": "gamma"})

        f = rule.join(table[["gamma", "se", "lo", "hi"]], on="parameter")
        f.loc[f["league"] == REFERENCE, "gamma"] = 0.0
        f = f.assign(metric=metric, multiplier=np.exp(-f["gamma"]))
        factors.append(f)

        b1, b2 = estimate["age"], estimate["age2"]
        lo, hi = np.nanquantile(
            np.exp(np.outer(reps[:, -2], years) + np.outer(reps[:, -1], years**2)),
            quantiles,
            axis=0,
        )
        curve = {"age": AGES, "rate": np.exp(b1 * years + b2 * years**2), "lo": lo, "hi": hi}
        curves.append(pd.DataFrame({"metric": metric} | curve))

        with_dummy = ("age", "age2", "pre_move")
        dummy = fit(prepare_cells(cells, metric, parameter_of, covariates=with_dummy))
        dropped = fit(prepare_cells(cells[cells["pre_move"] == 0], metric, parameter_of))
        check = table[["gamma", "se"]].drop(index=["age", "age2"])
        check = check.assign(metric=metric, gamma_dummy=dummy, gamma_dropped=dropped)
        checks.append(check.rename_axis("parameter").reset_index())
        shift = check.loc[pool_parameters].eval("(gamma_dummy - gamma) / se").abs().max()
        shift_dropped = check.loc[pool_parameters].eval("(gamma_dropped - gamma) / se").abs().max()

        entries |= {
            f"factors_{metric}_cells_used": (len(c.count), f"{metric}: cells in the likelihood"),
            f"factors_{metric}_players_used": (len(c.start), f"{metric}: players in it"),
            f"factors_{metric}_players_single_cell": (
                c.n_single_cell,
                f"{metric}: players dropped, a single cell",
            ),
            f"factors_{metric}_players_zero_count": (
                c.n_zero_count,
                f"{metric}: players dropped, several cells and a total of zero",
            ),
            f"factors_{metric}_boot_min": (
                int(table["n_boot"].min()),
                f"{metric}: bootstrap replicates used, of {FACTOR_BOOT}",
            ),
            f"factors_{metric}_age_b1": (b1, f"{metric}: coefficient of age - {AGE_REF:g}"),
            f"factors_{metric}_age_b2": (b2, f"{metric}: coefficient of (age - {AGE_REF:g})^2"),
            f"factors_{metric}_age_peak": (AGE_REF - b1 / (2 * b2), f"{metric}: age of the peak"),
            f"factors_{metric}_pre_move": (
                dummy["pre_move"],
                f"{metric}: coefficient of the before-a-move dummy",
            ),
            f"factors_{metric}_shift_dummy_max_se": (
                shift,
                f"{metric}: largest |gamma with the dummy - gamma| over the pool's leagues, in SE",
            ),
            f"factors_{metric}_shift_dropped_max_se": (
                shift_dropped,
                f"{metric}: the same without the cells before a move",
            ),
        }
        shown = f[f["league"].isin(pool_leagues)].sort_values("gamma")
        for row in shown.itertuples():
            entries[f"factors_{metric}_gamma_{row.league}"] = (
                row.gamma,
                f"{metric}: gamma, {row.league} ({row.status}, {row.n_linking} linking players)",
            )
            entries[f"factors_{metric}_multiplier_{row.league}"] = (
                row.multiplier,
                f"{metric}: exp(-gamma), applied to counts in {row.league} league matches",
            )
        print(
            f"\n{metric}: {len(c.count)} cells, {len(c.start)} players; dropped "
            f"{c.n_single_cell} single-cell and {c.n_zero_count} zero-count players"
        )
        print(shown.drop(columns=["metric", "parameter"]).round(3).to_string(index=False))
        print(
            f"age peak {AGE_REF - b1 / (2 * b2):.1f}; dummy {dummy['pre_move']:+.3f}; largest "
            f"shift {shift:.2f} SE with the dummy, {shift_dropped:.2f} SE without the cells"
        )

    columns = ["metric", "league", "gamma", "se", "lo", "hi", "multiplier", "n_linking", "status"]
    to_csv(pd.concat(factors)[columns], LEAGUE_DIR / "factors.csv")
    to_csv(pd.concat(curves), LEAGUE_DIR / "age_curve.csv")
    columns = ["metric", "parameter", "gamma", "se", "gamma_dummy", "gamma_dropped"]
    to_csv(pd.concat(checks)[columns], LEAGUE_DIR / "selection_check.csv")
    record(LEAGUE_DIR, entries)
