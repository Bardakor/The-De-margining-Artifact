"""Stage ``factors`` of run_league.py: box-score league factors from movers (A20 ii, A21).

Cells are player x season x league sums over the panel's league matches
(scope == "league", minutes > 0). Per metric, two conditional fixed-effects
Poisson fits (``footy.players.league_factors``):

- ``context`` (A21, the one the league-adjusted ranking uses): age, age^2 and the
  team's Elo. It enters the estimation of gamma only; counts are divided by
  exp(gamma) alone.
- ``plain`` (A20 ii, sensitivity): age and age^2.

B = 200 player bootstrap per fit, 90% percentile intervals. Both models of a
metric draw from task ``factors/<metric>``: as long as no cell lacks team Elo
the same players inform both, so they see the same 200 resamples.

Choices the amendments do not fix (post hoc, Claude's calls):

- Linking players: at least 450 league minutes, summed over the window, in the
  league and in another one, the count of docs/unmeasured/league-strength.md §2.2
  that §3.2 applies the threshold of 100 to. Counted on every cell, before the
  birth-date filter. The number of leagues that pass with any minutes is recorded.
- Age: at the midpoint between the first and the last match of the league-season
  in the panel (a calendar-year league's season is half a year later than its
  label), in years of 365.25 days, centred at 26. A player without a birth date
  is dropped; counted.
- Team Elo: minutes-weighted mean of ``team_elo`` over the cell's matches, minus
  the same mean over every player-row of the league-season, in units of 100
  points. A cell with a match without ``team_elo`` is left out of ``context``;
  counted.
- Selection check (§3.2): a cell is "before a move" when the player's next cell,
  ordered by the date of the cell's last match, is in another league. Two point
  fits per model and metric: with that dummy, and without those cells.
- Leagues holding pool player-seasons: a pool player with at least 900 league
  minutes in the league in one season.

Outputs in results/players/league/, each with a ``model`` column: factors.csv
(one row per metric and league; ``team_elo`` is the coefficient per 100 points),
age_curve.csv (rate relative to age 26), selection_check.csv (one row per metric
and league parameter); ``factors_*`` entries in summary.json.
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
FACTOR_MODELS = {"context": ("age", "age2", "elo"), "plain": ("age", "age2")}
FACTOR_BOOT = 200
FACTOR_LEVEL = 0.9
AGE_REF = 26.0
AGES = np.arange(17.0, 41.0)
ELO_UNIT = 100.0
POOL_SEASON_MINUTES = 900.0


def league_cells() -> pd.DataFrame:
    """Player x season x league cells: minutes, counts, last match date, age, team Elo."""
    pm = pd.read_csv(
        PROCESSED / "panel_player_match.csv.gz",
        usecols=["player_id", "season", "competition_id", "scope", "minutes", "date", "team_elo"]
        + list(FACTOR_METRICS),
        parse_dates=["date"],
    )
    pm = pm[pm["scope"] == "league"].rename(columns={"competition_id": "league"})
    pm["elo_weight"] = pm["minutes"].where(pm["team_elo"].notna())
    pm["elo_minutes"] = pm["elo_weight"] * pm["team_elo"]
    sums = {m: (m, "sum") for m in ("minutes", "elo_weight", "elo_minutes", *FACTOR_METRICS)}
    cells = pm.groupby(["player_id", "season", "league"], as_index=False).agg(
        **sums, last_date=("date", "max")
    )
    cells = cells[cells["minutes"] > 0]
    season = pm.groupby(["season", "league"]).agg(
        first=("date", "min"),
        last=("date", "max"),
        elo_weight=("elo_weight", "sum"),
        elo_minutes=("elo_minutes", "sum"),
    )
    season["midpoint"] = season["first"] + (season["last"] - season["first"]) / 2
    season["level"] = season["elo_minutes"] / season["elo_weight"]
    born = pd.read_csv(
        PROCESSED / "tm_players.csv.gz",
        usecols=["player_id", "date_of_birth"],
        parse_dates=["date_of_birth"],
    ).set_index("player_id")["date_of_birth"]
    cells = cells.join(season[["midpoint", "level"]], on=["season", "league"])
    age = (cells["midpoint"] - cells["player_id"].map(born)).dt.days / 365.25 - AGE_REF
    elo = (cells["elo_minutes"] / cells["elo_weight"] - cells["level"]) / ELO_UNIT
    cells = cells.assign(
        age=age, age2=age**2, elo=elo.where(cells["elo_weight"] == cells["minutes"])
    )
    return cells.drop(columns=["midpoint", "level", "elo_minutes", "elo_weight"])


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
        "factors_cells_no_team_elo": (
            int(cells["elo"].isna().sum()),
            "cells left out of the context model: a match without team Elo",
        ),
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
    fits = [(model, metric) for model in FACTOR_MODELS for metric in FACTOR_METRICS]
    for model, metric in fits:
        covariates = FACTOR_MODELS[model]
        data = cells.dropna(subset=list(covariates))
        c = prepare_cells(data, metric, parameter_of, covariates=covariates)
        estimate = fit(c)
        reps = bootstrap(c, estimate, rng(f"factors/{metric}"), n_boot=FACTOR_BOOT)
        table = summarise(estimate, reps, level=FACTOR_LEVEL)
        gammas = table.rename(columns={"estimate": "gamma"})[["gamma", "se", "lo", "hi"]]
        team = table.reindex(["elo"]).iloc[0]  # NaN in the plain model

        f = rule.join(gammas, on="parameter")
        f.loc[f["league"] == REFERENCE, "gamma"] = 0.0
        f = f.assign(
            model=model,
            metric=metric,
            multiplier=np.exp(-f["gamma"]),
            team_elo=team["estimate"],
            team_elo_se=team["se"],
        )
        factors.append(f)

        b1, b2 = estimate["age"], estimate["age2"]
        age, age2 = c.names.index("age"), c.names.index("age2")
        lo, hi = np.nanquantile(
            np.exp(np.outer(reps[:, age], years) + np.outer(reps[:, age2], years**2)),
            quantiles,
            axis=0,
        )
        curve = {"age": AGES, "rate": np.exp(b1 * years + b2 * years**2), "lo": lo, "hi": hi}
        curves.append(pd.DataFrame({"model": model, "metric": metric} | curve))

        with_dummy = (*covariates, "pre_move")
        dummy = fit(prepare_cells(data, metric, parameter_of, covariates=with_dummy))
        stays = data[data["pre_move"] == 0]
        dropped = fit(prepare_cells(stays, metric, parameter_of, covariates=covariates))
        check = gammas[["gamma", "se"]].drop(index=list(covariates))
        check = check.assign(model=model, metric=metric, gamma_dummy=dummy, gamma_dropped=dropped)
        checks.append(check.rename_axis("parameter").reset_index())
        shift = check.loc[pool_parameters].eval("(gamma_dummy - gamma) / se").abs().max()
        shift_dropped = check.loc[pool_parameters].eval("(gamma_dropped - gamma) / se").abs().max()

        key, label = f"factors_{model}_{metric}", f"{model}, {metric}"
        entries |= {
            f"{key}_cells_used": (len(c.count), f"{label}: cells in the likelihood"),
            f"{key}_players_used": (len(c.start), f"{label}: players in it"),
            f"{key}_players_single_cell": (
                c.n_single_cell,
                f"{label}: players dropped, a single cell",
            ),
            f"{key}_players_zero_count": (
                c.n_zero_count,
                f"{label}: players dropped, several cells and a total of zero",
            ),
            f"{key}_boot_min": (
                int(table["n_boot"].min()),
                f"{label}: bootstrap replicates used, of {FACTOR_BOOT}",
            ),
            f"{key}_age_b1": (b1, f"{label}: coefficient of age - {AGE_REF:g}"),
            f"{key}_age_b2": (b2, f"{label}: coefficient of (age - {AGE_REF:g})^2"),
            f"{key}_age_peak": (AGE_REF - b1 / (2 * b2), f"{label}: age of the peak"),
            f"{key}_pre_move": (dummy["pre_move"], f"{label}: before-a-move dummy coefficient"),
            f"{key}_shift_dummy_max_se": (
                shift,
                f"{label}: largest |gamma with the dummy - gamma| over the pool's leagues, in SE",
            ),
            f"{key}_shift_dropped_max_se": (
                shift_dropped,
                f"{label}: the same without the cells before a move",
            ),
        }
        if "elo" in covariates:
            entries[f"{key}_team_elo"] = (
                team["estimate"],
                f"{label}: coefficient of the team's centred Elo, per {ELO_UNIT:g} points",
            )
            entries[f"{key}_team_elo_se"] = (team["se"], f"{label}: its bootstrap SE")
        shown = f[f["league"].isin(pool_leagues)].sort_values("gamma")
        for row in shown.itertuples():
            entries[f"{key}_gamma_{row.league}"] = (
                row.gamma,
                f"{label}: gamma, {row.league} ({row.status}, {row.n_linking} linking players)",
            )
            entries[f"{key}_multiplier_{row.league}"] = (
                row.multiplier,
                f"{label}: exp(-gamma), applied to counts in {row.league} league matches",
            )
        print(
            f"\n{label}: {len(c.count)} cells, {len(c.start)} players; dropped "
            f"{c.n_single_cell} single-cell and {c.n_zero_count} zero-count players"
        )
        columns = ["league", "n_linking", "status", "gamma", "se", "lo", "hi", "multiplier"]
        print(shown[columns].round(3).to_string(index=False))
        if "elo" in covariates:
            print(f"team Elo {team['estimate']:+.3f} per {ELO_UNIT:g} points (SE {team['se']:.3f})")
        print(
            f"age peak {AGE_REF - b1 / (2 * b2):.1f}; dummy {dummy['pre_move']:+.3f}; largest "
            f"shift {shift:.2f} SE with the dummy, {shift_dropped:.2f} SE without the cells"
        )

    columns = ["model", "metric", "league", "gamma", "se", "lo", "hi", "multiplier"]
    columns += ["n_linking", "status", "team_elo", "team_elo_se"]
    to_csv(pd.concat(factors)[columns], LEAGUE_DIR / "factors.csv")
    to_csv(pd.concat(curves), LEAGUE_DIR / "age_curve.csv")
    columns = ["model", "metric", "parameter", "gamma", "se", "gamma_dummy", "gamma_dropped"]
    to_csv(pd.concat(checks)[columns], LEAGUE_DIR / "selection_check.csv")
    record(LEAGUE_DIR, entries)
