"""The (league × season × book × market) coverage matrix.

Spec §3.1. Coverage is a result, not a preliminary: it is published as a
table in the paper. The study is restricted to cells with complete closing
odds. Opening-only cells are recorded as excluded.
"""

from __future__ import annotations

import pandas as pd

from footy.data.football_data import SeasonTables

REQUIRED_OUTCOMES = {"1X2": ("H", "D", "A"), "OU25": ("O", "U")}


def coverage_matrix(tables: SeasonTables) -> pd.DataFrame:
    """One row per (league, season, book, market).

    ``closing`` is true when the file carries the closing columns.
    ``n_with_closing`` counts matches whose closing book is complete (every
    required outcome present and a payable price).
    """
    matches = tables.matches
    odds = tables.odds
    n_matches = (
        matches.groupby(["league", "season"], sort=False).size().rename("n_matches").reset_index()
    )

    empty = pd.DataFrame(
        columns=[
            "league",
            "season",
            "book",
            "market",
            "opening",
            "closing",
            "n_matches",
            "n_with_closing",
        ]
    )
    if odds.empty:
        return empty

    rows: list[dict[str, object]] = []
    keys = odds[["league", "season", "book", "market"]].drop_duplicates()
    for rec in keys.itertuples(index=False):
        league, season, book, market = rec.league, rec.season, rec.book, rec.market
        block = odds[
            (odds["league"] == league)
            & (odds["season"] == season)
            & (odds["book"] == book)
            & (odds["market"] == market)
        ]
        outcomes = REQUIRED_OUTCOMES[str(market)]
        opening = bool((block["period"] == "open").any())
        closing_block = block[block["period"] == "close"]
        closing = not closing_block.empty
        n_complete = _complete_books(closing_block, outcomes) if closing else 0
        n_league_season = int(
            n_matches.loc[
                (n_matches["league"] == league) & (n_matches["season"] == season),
                "n_matches",
            ].sum()
        )
        rows.append(
            {
                "league": league,
                "season": season,
                "book": book,
                "market": market,
                "opening": opening,
                "closing": closing,
                "n_matches": n_league_season,
                "n_with_closing": n_complete,
            }
        )
    coverage = pd.DataFrame(rows)
    return coverage.sort_values(["league", "season", "book", "market"]).reset_index(drop=True)


def _complete_books(closing: pd.DataFrame, outcomes: tuple[str, ...]) -> int:
    """Count matches that have every required closing outcome."""
    needed = set(outcomes)
    complete = 0
    grouped = closing.groupby(["kickoff", "home", "away"], sort=False)
    for _, group in grouped:
        present = {str(v) for v in group["outcome"].tolist()}
        if needed <= present:
            complete += 1
    return complete


def eligible_cells(coverage: pd.DataFrame) -> pd.DataFrame:
    """Cells with closing columns present and at least one complete book."""
    if coverage.empty:
        return coverage
    mask = coverage["closing"].astype(bool) & (coverage["n_with_closing"] > 0)
    return coverage.loc[mask].reset_index(drop=True)
