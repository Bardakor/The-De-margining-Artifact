"""Ingest of football-data.co.uk season CSVs.

Spec §3.1. This layer decides which (league, season, book, market) cells enter
the study, so its failure modes matter more than its happy path. The rule the
spec is emphatic about is that an unrecognised header must fail loudly rather
than be silently mis-mapped.

No test here touches the network. `fetch_season` accepts an opener, and the
stub below is what every download test uses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from footy.data.football_data import (
    ARCHIVE_ROOT,
    LEAGUES,
    SeasonTables,
    UnrecognisedHeaderError,
    concat_seasons,
    discover_header,
    fetch_season,
    read_season_csv,
    season_code,
    season_codes,
    season_start_year,
    season_url,
    sha256_file,
    write_manifest,
)

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG"
ROW_A = "E0,12/08/2023,Arsenal,Chelsea,2,1"
ROW_B = "E0,13/08/2023,Spurs,Everton,1,1"


def make_csv(header: str, *rows: str) -> io.StringIO:
    return io.StringIO("\n".join((header, *rows)) + "\n")


def read(header: str, *rows: str, league: str | None = "E0") -> SeasonTables:
    return read_season_csv(make_csv(header, *rows), season="2324", league=league)


def text_at(frame: pd.DataFrame, row: int, column: str) -> str:
    """Typed cell accessors. pandas returns a wide union that mypy --strict
    will not narrow, so the tests go through these rather than sprinkling
    casts across every assertion."""
    return str(frame.iloc[row][column])


def int_at(frame: pd.DataFrame, row: int, column: str) -> int:
    return int(cast(Any, frame.iloc[row][column]))


def stamp_at(frame: pd.DataFrame, row: int, column: str) -> pd.Timestamp:
    return cast(pd.Timestamp, frame.iloc[row][column])


# --------------------------------------------------------------------------
# Season codes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "code"),
    [(1993, "9394"), (1999, "9900"), (2000, "0001"), (2019, "1920"), (2025, "2526")],
)
def test_season_code_matches_the_archive_convention(year: int, code: str) -> None:
    assert season_code(year) == code


def test_season_code_round_trips_across_the_archive_range() -> None:
    for year in range(1993, 2027):
        assert season_start_year(season_code(year)) == year


def test_season_start_year_boundary_is_documented_and_pinned() -> None:
    """Codes 90-99 are 1990s, 00-89 are 2000s. The scheme is inherently
    ambiguous and cannot survive 2090; pinned so the limit is deliberate."""
    assert season_start_year("9394") == 1993
    assert season_start_year("8990") == 2089
    assert season_start_year(season_code(2090)) == 1990  # the documented break


@pytest.mark.parametrize("bad", ["", "93", "93945", "abcd", "20xx"])
def test_season_start_year_rejects_malformed_codes(bad: str) -> None:
    with pytest.raises(ValueError, match="four digits"):
        season_start_year(bad)


def test_season_codes_is_inclusive_and_ordered() -> None:
    codes = season_codes(2020, 2022)
    assert codes == ["2021", "2122", "2223"]


def test_season_codes_rejects_a_reversed_range() -> None:
    with pytest.raises(ValueError, match="precedes"):
        season_codes(2020, 2019)


def test_season_url_is_built_from_the_archive_root() -> None:
    assert season_url("2324", "E0") == f"{ARCHIVE_ROOT}/2324/E0.csv"


def test_leagues_are_unique() -> None:
    assert len(set(LEAGUES)) == len(LEAGUES)


# --------------------------------------------------------------------------
# Header discovery — the loud-failure requirement
# --------------------------------------------------------------------------


def test_unrecognised_header_raises_rather_than_being_skipped() -> None:
    """Spec §3.1: never silently mis-map a column."""
    with pytest.raises(UnrecognisedHeaderError) as excinfo:
        discover_header(["Div", "Date", "MysteryColumn"], source="probe.csv")
    assert excinfo.value.columns == ["MysteryColumn"]
    assert excinfo.value.source == "probe.csv"
    assert "probe.csv" in str(excinfo.value)


def test_every_unrecognised_column_is_reported_not_just_the_first() -> None:
    with pytest.raises(UnrecognisedHeaderError) as excinfo:
        discover_header(["Div", "Nonsense1", "Nonsense2"])
    assert excinfo.value.columns == ["Nonsense1", "Nonsense2"]


def test_blank_and_unnamed_columns_are_ignored() -> None:
    """Trailing commas in the archive produce these; they are not a schema
    change and must not trip the loud failure."""
    assert discover_header(["Div", "Date", "", "  ", "Unnamed: 7", "Unnamed: 12"]) == {}


def test_result_columns_are_recognised_but_not_priced() -> None:
    assert discover_header(["Div", "Date", "HomeTeam", "FTHG", "Referee", "HS"]) == {}


def test_asian_handicap_columns_are_tolerated_but_not_ingested() -> None:
    """They carry a line rather than a simplex, so they are out of scope for
    this study — but they must not be mistaken for an unknown schema."""
    assert discover_header(["Div", "B365AHH", "B365AHA", "AHh", "BbAHh"]) == {}


def test_odds_columns_are_returned_with_their_specs() -> None:
    found = discover_header(["Div", "B365H", "B365CH", "PSC>2.5"])
    assert set(found) == {"B365H", "B365CH", "PSC>2.5"}
    assert found["B365H"].period == "open"
    assert found["B365CH"].period == "close"
    assert found["PSC>2.5"].market == "OU25"


def test_headers_are_matched_after_stripping_whitespace() -> None:
    assert set(discover_header([" B365H ", "  Div"])) == {"B365H"}


# --------------------------------------------------------------------------
# Pinnacle aliasing — regression for a duplicate-price defect
# --------------------------------------------------------------------------


def test_pinnacle_1x2_alias_is_dropped_when_canonical_is_present() -> None:
    found = discover_header(["PSH", "PSD", "PSA", "PH", "PD", "PA"])
    assert set(found) == {"PSH", "PSD", "PSA"}


def test_pinnacle_over_under_alias_is_dropped_when_canonical_is_present() -> None:
    """Regression. Only the 1X2 triple was de-aliased, so `P>2.5` survived
    beside `PS>2.5` and put the same Pinnacle price into the odds table twice
    under one (book, market, outcome, period) key — double-counting it in any
    book sum, which is the quantity this study measures."""
    found = discover_header(["PS>2.5", "PS<2.5", "P>2.5", "P<2.5"])
    assert set(found) == {"PS>2.5", "PS<2.5"}

    closing = discover_header(["PSC>2.5", "PSC<2.5", "PC>2.5", "PC<2.5"])
    assert set(closing) == {"PSC>2.5", "PSC<2.5"}


def test_pinnacle_alias_is_kept_when_the_canonical_is_absent() -> None:
    """Early archive files carry only the alias; dropping it unconditionally
    would discard Pinnacle entirely for those seasons."""
    assert set(discover_header(["PH", "PD", "PA"])) == {"PH", "PD", "PA"}
    assert set(discover_header(["P>2.5", "P<2.5"])) == {"P>2.5", "P<2.5"}


def test_partial_canonical_does_not_drop_the_alias() -> None:
    """A file with PSH but no PSD/PSA has an incomplete canonical triple, so
    the alias is still the only complete source."""
    found = discover_header(["PSH", "PH", "PD", "PA"])
    assert {"PH", "PD", "PA"} <= set(found)


def test_ingested_odds_contain_no_duplicate_outcome_rows() -> None:
    """End-to-end guard on the same defect: one price per
    (match, book, market, outcome, period)."""
    header = HEADER + ",PSH,PSD,PSA,PH,PD,PA,PS>2.5,PS<2.5,P>2.5,P<2.5"
    row = ROW_A + ",1.80,3.60,4.50,1.81,3.55,4.45,1.90,2.00,1.91,1.99"
    tables = read(header, row)
    keys = ["home", "away", "book", "market", "outcome", "period"]
    assert not tables.odds.duplicated(subset=keys).any()
    assert len(tables.odds) == 5


# --------------------------------------------------------------------------
# read_season_csv
# --------------------------------------------------------------------------


def test_parses_matches_into_the_tidy_schema() -> None:
    tables = read(HEADER, ROW_A, ROW_B)
    assert list(tables.matches.columns) == [
        "league",
        "season",
        "kickoff",
        "home",
        "away",
        "home_goals",
        "away_goals",
    ]
    assert len(tables.matches) == 2
    assert text_at(tables.matches, 0, "home") == "Arsenal"
    assert int_at(tables.matches, 0, "home_goals") == 2
    assert text_at(tables.matches, 0, "league") == "E0"
    assert text_at(tables.matches, 0, "season") == "2324"


def test_dates_are_parsed_day_first() -> None:
    """12/08/2023 is 12 August, not 8 December. Month-first parsing would
    reorder the walk-forward protocol and leak future matches into fits."""
    tables = read(HEADER, ROW_A)
    kickoff = stamp_at(tables.matches, 0, "kickoff")
    assert kickoff.day == 12
    assert kickoff.month == 8


def test_time_column_is_folded_into_the_kickoff() -> None:
    tables = read(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG",
        "E0,12/08/2023,15:00,Arsenal,Chelsea,2,1",
    )
    kickoff = stamp_at(tables.matches, 0, "kickoff")
    assert (kickoff.hour, kickoff.minute) == (15, 0)


def test_missing_time_values_do_not_destroy_the_date() -> None:
    tables = read(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG",
        "E0,12/08/2023,,Arsenal,Chelsea,2,1",
    )
    assert stamp_at(tables.matches, 0, "kickoff").day == 12


def test_rows_with_and_without_a_time_both_survive() -> None:
    """Regression, and the realistic archive shape: football-data added Time
    part-way through, so files mix populated and blank values.

    `astype(str)` preserves NA under pandas 3 instead of rendering "nan", so
    the blank-time branch was never taken and the date was concatenated with
    NA — yielding NaT, which the validity filter then dropped. The effect was
    silent, selective loss of exactly those matches with no kickoff time,
    while their neighbours survived.
    """
    tables = read(
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG",
        "E0,12/08/2023,15:00,Arsenal,Chelsea,2,1",
        "E0,13/08/2023,,Spurs,Everton,1,1",
        "E0,14/08/2023,17:30,Leeds,Hull,0,0",
    )
    assert len(tables.matches) == 3
    assert list(tables.matches["home"]) == ["Arsenal", "Spurs", "Leeds"]
    assert [k.hour for k in tables.matches["kickoff"]] == [15, 0, 17]


def test_whitespace_only_team_names_are_dropped() -> None:
    tables = read(HEADER, ROW_A, "E0,14/08/2023,   ,Wolves,1,0")
    assert list(tables.matches["home"]) == ["Arsenal"]


def test_hg_and_ag_are_accepted_as_goal_column_aliases() -> None:
    """Some divisions publish HG/AG rather than FTHG/FTAG."""
    tables = read("Div,Date,HomeTeam,AwayTeam,HG,AG", "E0,12/08/2023,Arsenal,Chelsea,3,0")
    assert int_at(tables.matches, 0, "home_goals") == 3


def test_rows_without_a_usable_result_are_dropped() -> None:
    """Postponed and future fixtures appear with blank scores."""
    tables = read(
        HEADER,
        ROW_A,
        "E0,13/08/2023,Spurs,Everton,,",
        "E0,not-a-date,Leeds,Hull,1,0",
        "E0,14/08/2023,,Wolves,1,0",
    )
    assert len(tables.matches) == 1
    assert text_at(tables.matches, 0, "home") == "Arsenal"


def test_goals_are_integers_not_floats() -> None:
    """Downstream code indexes the scoreline matrix with these."""
    tables = read(HEADER, ROW_A)
    assert tables.matches["home_goals"].dtype.kind == "i"
    assert tables.matches["away_goals"].dtype.kind == "i"


def test_league_is_inferred_from_div_when_not_supplied() -> None:
    tables = read(HEADER, ROW_A, ROW_B, league=None)
    assert text_at(tables.matches, 0, "league") == "E0"


def test_div_disagreeing_with_the_declared_league_is_an_error() -> None:
    """A mis-filed row would attribute matches to the wrong competition, and
    home advantage is fit per league."""
    with pytest.raises(ValueError, match="Div does not match"):
        read(HEADER, "D1,12/08/2023,Bayern,Mainz,2,1", league="E0")


def test_multiple_divs_without_a_declared_league_is_an_error() -> None:
    with pytest.raises(ValueError, match="expected one Div"):
        read(HEADER, ROW_A, "D1,13/08/2023,Bayern,Mainz,2,1", league=None)


def test_league_is_required_when_div_is_absent() -> None:
    with pytest.raises(ValueError, match="league is required"):
        read_season_csv(
            make_csv("Date,HomeTeam,AwayTeam,FTHG,FTAG", "12/08/2023,Arsenal,Chelsea,2,1"),
            season="2324",
            league=None,
        )


def test_missing_team_columns_are_an_error() -> None:
    with pytest.raises(ValueError, match="HomeTeam and AwayTeam"):
        read_season_csv(
            make_csv("Div,Date,FTHG,FTAG", "E0,12/08/2023,2,1"), season="2324", league="E0"
        )


def test_missing_goal_columns_are_an_error() -> None:
    with pytest.raises(ValueError, match="missing goal column"):
        read(
            "Div,Date,HomeTeam,AwayTeam,FTAG",
            "E0,12/08/2023,Arsenal,Chelsea,1",
        )


def test_unpayable_prices_are_discarded() -> None:
    """A decimal price at or below 1.0 cannot be paid and is archive noise;
    admitting it would produce a negative-probability book."""
    tables = read(HEADER + ",B365H,B365D,B365A", ROW_A + ",1.00,0.00,4.50")
    assert set(tables.odds["outcome"]) == {"A"}


def test_odds_table_is_tidy_and_carries_the_match_key() -> None:
    tables = read(HEADER + ",B365H,B365CH", ROW_A + ",1.80,1.75")
    assert list(tables.odds.columns) == [
        "league",
        "season",
        "kickoff",
        "home",
        "away",
        "book",
        "market",
        "period",
        "outcome",
        "decimal",
    ]
    periods = dict(zip(tables.odds["period"], tables.odds["decimal"], strict=True))
    assert periods == {"open": 1.80, "close": 1.75}


def test_odds_rows_align_with_the_surviving_matches() -> None:
    """Rows are dropped from matches before odds are emitted; a misalignment
    would attach prices to the wrong fixture."""
    tables = read(
        HEADER + ",B365H",
        "E0,12/08/2023,Arsenal,Chelsea,,1.80",  # dropped: no scores
        ROW_B + ",2.50",
    )
    assert len(tables.matches) == 1
    assert list(tables.odds["home"]) == ["Spurs"]
    assert list(tables.odds["decimal"]) == [2.50]


def test_a_file_with_no_odds_yields_an_empty_but_well_formed_table() -> None:
    tables = read(HEADER, ROW_A)
    assert tables.odds.empty
    assert "decimal" in tables.odds.columns


def test_unrecognised_column_in_a_real_file_names_the_source() -> None:
    with pytest.raises(UnrecognisedHeaderError, match="E0/2324"):
        read(HEADER + ",TotallyNewColumn", ROW_A + ",1.5")


# --------------------------------------------------------------------------
# concat_seasons
# --------------------------------------------------------------------------


def test_concat_stacks_matches_and_odds() -> None:
    first = read(HEADER + ",B365H", ROW_A + ",1.80")
    second = read(HEADER + ",B365H", ROW_B + ",2.50")
    combined = concat_seasons([first, second])
    assert len(combined.matches) == 2
    assert len(combined.odds) == 2
    assert combined.source == "concat"


def test_concat_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="no season tables"):
        concat_seasons([])


def test_season_tables_is_immutable() -> None:
    tables = read(HEADER, ROW_A)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tables.source = "tampered"  # type: ignore[misc]


# --------------------------------------------------------------------------
# Provenance: checksums and the manifest
# --------------------------------------------------------------------------


def test_sha256_matches_the_reference_digest(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == expected


def test_sha256_reads_files_larger_than_one_chunk(tmp_path: Path) -> None:
    path = tmp_path / "big.csv"
    payload = b"x" * (1 << 17)
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_manifest_records_a_digest_per_file(tmp_path: Path) -> None:
    """Spec §9: every result carries the SHA-256 of each input CSV."""
    first, second = tmp_path / "b.csv", tmp_path / "a.csv"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    dest = tmp_path / "out" / "manifest.json"

    write_manifest([first, second], dest)

    payload = json.loads(dest.read_text())
    assert [Path(r["path"]).name for r in payload["files"]] == ["a.csv", "b.csv"]
    assert all(len(r["sha256"]) == 64 for r in payload["files"])


# --------------------------------------------------------------------------
# fetch_season — network is stubbed, never called
# --------------------------------------------------------------------------


class StubOpener:
    """Stands in for urllib's OpenerDirector. Records what it was asked for."""

    def __init__(self, body: bytes = b"", error: urllib.error.HTTPError | None = None) -> None:
        self.body = body
        self.error = error
        self.urls: list[str] = []

    def open(self, request: Any, *args: Any, **kwargs: Any) -> io.BytesIO:
        self.urls.append(request.full_url)
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.body)


def as_opener(stub: StubOpener) -> urllib.request.OpenerDirector:
    return cast(urllib.request.OpenerDirector, stub)


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://example.invalid", code, "boom", {}, None)  # type: ignore[arg-type]


def test_fetch_writes_the_body_and_requests_the_right_url(tmp_path: Path) -> None:
    stub = StubOpener(body=b"Div,Date\nE0,12/08/2023\n")
    dest = tmp_path / "nested" / "E0.csv"

    result = fetch_season("2324", "E0", dest, opener=as_opener(stub))

    assert result == dest
    assert dest.read_bytes() == stub.body
    assert stub.urls == [f"{ARCHIVE_ROOT}/2324/E0.csv"]


def test_absent_league_season_returns_none_rather_than_raising(tmp_path: Path) -> None:
    """Not every league has every season; a 404 is expected, not exceptional."""
    stub = StubOpener(error=http_error(404))
    dest = tmp_path / "E0.csv"

    assert fetch_season("9394", "G1", dest, opener=as_opener(stub)) is None
    assert not dest.exists()


@pytest.mark.parametrize("code", [403, 500, 503])
def test_other_http_errors_propagate(tmp_path: Path, code: int) -> None:
    """A server fault must not be mistaken for an absent season, which would
    silently shrink the study's coverage."""
    stub = StubOpener(error=http_error(code))
    with pytest.raises(urllib.error.HTTPError):
        fetch_season("2324", "E0", tmp_path / "E0.csv", opener=as_opener(stub))
