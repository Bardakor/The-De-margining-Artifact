"""The (league × season × book × market) coverage matrix.

Spec §3.1. Coverage is a result, not a preliminary — it is published as a table
in the paper and it decides which cells the study runs on. Over-reporting
coverage would admit cells whose closing book is incomplete, and a de-margining
transform applied to an incomplete book returns a benchmark that is simply
wrong rather than merely noisy.
"""

from __future__ import annotations

import io

import pytest

from footy.data.coverage import coverage_matrix, eligible_cells
from footy.data.football_data import SeasonTables, read_season_csv

BASE = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG"
CLOSING_1X2 = ",B365CH,B365CD,B365CA"


def read(header: str, *rows: str, season: str = "2324") -> SeasonTables:
    csv = io.StringIO("\n".join((header, *rows)) + "\n")
    return read_season_csv(csv, season=season, league="E0")


def only(tables: SeasonTables, book: str, market: str) -> dict[str, object]:
    frame = coverage_matrix(tables)
    row = frame[(frame["book"] == book) & (frame["market"] == market)]
    assert len(row) == 1, f"expected exactly one {book}/{market} row, got {len(row)}"
    return {str(k): v for k, v in row.iloc[0].to_dict().items()}


def test_one_row_per_league_season_book_market() -> None:
    tables = read(
        BASE + CLOSING_1X2 + ",PSCH,PSCD,PSCA,B365C>2.5,B365C<2.5",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50,1.82,3.55,4.40,1.90,2.00",
    )
    frame = coverage_matrix(tables)
    assert set(frame.columns) == {
        "league",
        "season",
        "book",
        "market",
        "opening",
        "closing",
        "n_matches",
        "n_with_closing",
    }
    keys = list(zip(frame["book"], frame["market"], strict=True))
    assert sorted(keys) == [("Bet365", "1X2"), ("Bet365", "OU25"), ("Pinnacle", "1X2")]
    assert len(keys) == len(set(keys))


def test_opening_and_closing_are_reported_separately() -> None:
    """Benchmarking against opening odds would be a methodological error, so
    the two must never be collapsed into one availability flag."""
    opening_only = read(
        BASE + ",B365H,B365D,B365A",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50",
    )
    row = only(opening_only, "Bet365", "1X2")
    assert row["opening"] is True
    assert row["closing"] is False
    assert row["n_with_closing"] == 0


def test_a_complete_closing_book_is_counted() -> None:
    tables = read(
        BASE + CLOSING_1X2,
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50",
        "E0,13/08/2023,Spurs,Everton,1,1,2.10,3.40,3.60",
    )
    row = only(tables, "Bet365", "1X2")
    assert row["closing"] is True
    assert row["n_matches"] == 2
    assert row["n_with_closing"] == 2


def test_a_partial_closing_book_is_not_counted() -> None:
    """H and A without D is not a simplex. Any transform applied to it would
    normalise over two outcomes and silently misprice the third."""
    tables = read(
        BASE + ",B365CH,B365CD,B365CA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,,4.50",
        "E0,13/08/2023,Spurs,Everton,1,1,2.10,3.40,3.60",
    )
    row = only(tables, "Bet365", "1X2")
    assert row["n_matches"] == 2
    assert row["n_with_closing"] == 1


def test_over_under_requires_both_sides() -> None:
    tables = read(
        BASE + ",B365C>2.5,B365C<2.5",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.90,2.00",
        "E0,13/08/2023,Spurs,Everton,1,1,1.85,",
    )
    row = only(tables, "Bet365", "OU25")
    assert row["n_with_closing"] == 1


def test_unpayable_prices_do_not_complete_a_book() -> None:
    """A price at or below 1.0 is discarded at ingest, so the book it belonged
    to must not be reported as complete."""
    tables = read(
        BASE + ",B365CH,B365CD,B365CA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,1.00,4.50",
    )
    row = only(tables, "Bet365", "1X2")
    assert row["n_with_closing"] == 0


def test_n_matches_counts_the_league_season_not_the_book() -> None:
    """The denominator is fixture count, so a book present for only some
    matches shows as low coverage rather than as full coverage of itself."""
    tables = read(
        BASE + ",B365CH,B365CD,B365CA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50",
        "E0,13/08/2023,Spurs,Everton,1,1,,,",
        "E0,14/08/2023,Leeds,Hull,0,0,,,",
    )
    row = only(tables, "Bet365", "1X2")
    assert row["n_matches"] == 3
    assert row["n_with_closing"] == 1


def test_books_are_counted_independently_of_each_other() -> None:
    tables = read(
        BASE + ",B365CH,B365CD,B365CA,PSCH,PSCD,PSCA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50,1.82,3.55,4.40",
        "E0,13/08/2023,Spurs,Everton,1,1,2.10,3.40,3.60,,,",
    )
    assert only(tables, "Bet365", "1X2")["n_with_closing"] == 2
    assert only(tables, "Pinnacle", "1X2")["n_with_closing"] == 1


def test_no_odds_yields_an_empty_but_well_formed_matrix() -> None:
    frame = coverage_matrix(read(BASE, "E0,12/08/2023,Arsenal,Chelsea,2,1"))
    assert frame.empty
    assert "n_with_closing" in frame.columns


def test_rows_are_deterministically_ordered() -> None:
    """The matrix is published as a table; row order must not depend on the
    order columns happened to appear in the source file."""
    tables = read(
        BASE + ",PSCH,PSCD,PSCA,B365CH,B365CD,B365CA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.82,3.55,4.40,1.80,3.60,4.50",
    )
    frame = coverage_matrix(tables)
    keys = list(zip(frame["league"], frame["season"], frame["book"], frame["market"], strict=True))
    assert keys == sorted(keys)


def test_eligible_cells_keeps_only_complete_closing_coverage() -> None:
    tables = read(
        BASE + ",B365CH,B365CD,B365CA,PSH,PSD,PSA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50,1.82,3.55,4.40",
    )
    coverage = coverage_matrix(tables)
    eligible = eligible_cells(coverage)
    assert list(eligible["book"]) == ["Bet365"]
    assert bool(eligible["closing"].all())
    assert int(eligible["n_with_closing"].min()) > 0


def test_eligible_cells_drops_closing_columns_with_no_complete_book() -> None:
    """The columns exist but every row is unusable — present is not the same
    as usable, and only the latter may enter the study."""
    tables = read(
        BASE + ",B365CH,B365CD,B365CA",
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,,4.50",
    )
    coverage = coverage_matrix(tables)
    assert bool(coverage["closing"].iloc[0]) is True
    assert int(coverage["n_with_closing"].iloc[0]) == 0
    assert eligible_cells(coverage).empty


def test_eligible_cells_on_an_empty_matrix_is_empty() -> None:
    empty = coverage_matrix(read(BASE, "E0,12/08/2023,Arsenal,Chelsea,2,1"))
    assert eligible_cells(empty).empty


def test_eligible_cells_does_not_mutate_its_input() -> None:
    tables = read(
        BASE + CLOSING_1X2,
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50",
    )
    coverage = coverage_matrix(tables)
    before = coverage.copy(deep=True)
    eligible_cells(coverage)
    assert coverage.equals(before)


def test_an_unknown_market_would_be_a_loud_failure() -> None:
    """REQUIRED_OUTCOMES is keyed by market; a market reaching the matrix
    without an entry must raise rather than silently score zero coverage."""
    tables = read(
        BASE + CLOSING_1X2,
        "E0,12/08/2023,Arsenal,Chelsea,2,1,1.80,3.60,4.50",
    )
    tampered = tables.odds.copy()
    tampered.loc[:, "market"] = "BTTS"
    with pytest.raises(KeyError):
        coverage_matrix(SeasonTables(matches=tables.matches, odds=tampered, source="x"))
