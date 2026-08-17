"""The (league × season × book × market) coverage matrix.

Spec §3.1. Coverage is a result, not a preliminary: it is published as a
table in the paper. The study is restricted to cells with complete closing
odds. Opening-only cells are recorded as excluded.

Everything here is expressed as group-bys over the whole odds table rather
than a filter per cell. The archive carries millions of odds rows across tens
of thousands of cells, and a per-cell filter is quadratic in that product —
it made computing the coverage table slower than the study it gates.
"""

from __future__ import annotations

import pandas as pd

from footy.data.football_data import SeasonTables

REQUIRED_OUTCOMES = {"1X2": ("H", "D", "A"), "OU25": ("O", "U")}

COVERAGE_COLUMNS = [
    "league",
    "season",
    "book",
    "market",
    "opening",
    "closing",
    "n_matches",
    "n_with_closing",
]

CELL_KEYS = ["league", "season", "book", "market"]
MATCH_KEYS = ["kickoff", "home", "away"]


def _required_counts(markets: pd.Series[str]) -> pd.Series[int]:
    """How many distinct outcomes each row's market needs to be complete."""
    sizes = {market: len(outcomes) for market, outcomes in REQUIRED_OUTCOMES.items()}
    missing = set(markets.unique()) - set(sizes)
    if missing:
        raise KeyError(f"no required-outcome definition for market(s) {sorted(missing)}")
    return markets.map(sizes)


def coverage_matrix(tables: SeasonTables) -> pd.DataFrame:
    """One row per (league, season, book, market).

    ``closing`` is true when the file carries the closing columns.
    ``n_with_closing`` counts matches whose closing book is complete — every
    required outcome present at a payable price. Present is not the same as
    usable, and only the latter may enter the study.
    """
    matches, odds = tables.matches, tables.odds
    if odds.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)

    # Raises on an unknown market before any counting, so an unrecognised
    # market cannot silently score zero coverage.
    _required_counts(odds["market"].drop_duplicates())

    periods = (
        odds.assign(is_open=odds["period"].eq("open"), is_close=odds["period"].eq("close"))
        .groupby(CELL_KEYS, sort=False)[["is_open", "is_close"]]
        .any()
        .rename(columns={"is_open": "opening", "is_close": "closing"})
        .reset_index()
    )

    closing = odds[odds["period"] == "close"]
    if closing.empty:
        complete = pd.DataFrame(columns=[*CELL_KEYS, "n_with_closing"])
    else:
        per_match = (
            closing.groupby([*CELL_KEYS, *MATCH_KEYS], sort=False)["outcome"]
            .nunique()
            .rename("n_outcomes")
            .reset_index()
        )
        per_match["needed"] = _required_counts(per_match["market"])
        complete = (
            per_match[per_match["n_outcomes"] >= per_match["needed"]]
            .groupby(CELL_KEYS, sort=False)
            .size()
            .rename("n_with_closing")
            .reset_index()
        )

    league_season = (
        matches.groupby(["league", "season"], sort=False).size().rename("n_matches").reset_index()
    )

    coverage = (
        periods.merge(complete, on=CELL_KEYS, how="left")
        .merge(league_season, on=["league", "season"], how="left")
        .fillna({"n_with_closing": 0, "n_matches": 0})
    )
    coverage["n_with_closing"] = coverage["n_with_closing"].astype(int)
    coverage["n_matches"] = coverage["n_matches"].astype(int)
    coverage["opening"] = coverage["opening"].astype(bool)
    coverage["closing"] = coverage["closing"].astype(bool)
    return coverage[COVERAGE_COLUMNS].sort_values(CELL_KEYS).reset_index(drop=True)


def eligible_cells(coverage: pd.DataFrame) -> pd.DataFrame:
    """Cells with closing columns present and at least one complete book."""
    if coverage.empty:
        return coverage
    mask = coverage["closing"].astype(bool) & (coverage["n_with_closing"] > 0)
    return coverage.loc[mask].reset_index(drop=True)
