"""Ingest football-data.co.uk season CSVs with header discovery.

Spec §3.1. Coverage is not assumed uniform. Opening and closing odds are
different columns; closing odds arrived part-way through the archive. The
ingest layer therefore matches each header against the registry in
``columns.py`` and fails loudly on anything it does not recognise.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from footy.data.columns import (
    KNOWN_COLUMNS,
    ODDS_MAP,
    PINNACLE_ALIASES,
    RESULT_COLUMNS,
    OddsColumn,
)

LEAGUES: tuple[str, ...] = (
    "E0",
    "E1",
    "D1",
    "D2",
    "SP1",
    "SP2",
    "I1",
    "I2",
    "F1",
    "F2",
    "N1",
    "B1",
    "P1",
    "T1",
    "SC0",
    "G1",
)

ARCHIVE_ROOT = "https://www.football-data.co.uk/mmz4281"
USER_AGENT = "footy-research/0.1 (+https://github.com/Bardakor/The-De-margining-Artifact)"
_UNNAMED = re.compile(r"^Unnamed:\s*\d+$")


class UnrecognisedHeaderError(ValueError):
    """A CSV column is not in the football-data.co.uk registry."""

    def __init__(self, columns: list[str], source: str | None = None) -> None:
        where = f" in {source}" if source else ""
        listed = ", ".join(columns)
        super().__init__(f"unrecognised header{where}: {listed}")
        self.columns = columns
        self.source = source


@dataclass(frozen=True)
class SeasonTables:
    """One season file, split into results and a tidy odds table."""

    matches: pd.DataFrame
    odds: pd.DataFrame
    source: str


def season_code(start_year: int) -> str:
    """1993 → '9394', 1999 → '9900', 2000 → '0001'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_start_year(code: str) -> int:
    """Invert ``season_code``. Codes from 90–99 are 1990s; 00–89 are 2000s."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"season code must be four digits, got {code!r}")
    yy = int(code[:2])
    return 1900 + yy if yy >= 90 else 2000 + yy


def season_codes(start_year: int = 1993, end_year: int = 2026) -> list[str]:
    """Inclusive range of football-data season codes."""
    if end_year < start_year:
        raise ValueError(f"end_year {end_year} precedes start_year {start_year}")
    return [season_code(year) for year in range(start_year, end_year + 1)]


def season_url(season: str, league: str) -> str:
    return f"{ARCHIVE_ROOT}/{season}/{league}.csv"


def discover_header(headers: list[str], source: str | None = None) -> dict[str, OddsColumn]:
    """Return the odds columns present in ``headers``.

    Raises:
        UnrecognisedHeaderError: a non-empty header is not in the registry.
    """
    unknown: list[str] = []
    present: dict[str, OddsColumn] = {}
    names = [h.strip() for h in headers if h.strip() and not _UNNAMED.match(h.strip())]
    for name in names:
        if name in RESULT_COLUMNS:
            continue
        if name in ODDS_MAP:
            present[name] = ODDS_MAP[name]
            continue
        if name in KNOWN_COLUMNS:
            continue
        unknown.append(name)
    if unknown:
        raise UnrecognisedHeaderError(unknown, source)
    return _drop_pinnacle_aliases(present, names)


def _drop_pinnacle_aliases(
    present: dict[str, OddsColumn], names: list[str]
) -> dict[str, OddsColumn]:
    """Prefer Pinnacle's canonical PS* columns over its one-letter P* aliases.

    Applies to the 1X2 triple and to both over/under pairs. Dropping only the
    1X2 triple left `P>2.5` alongside `PS>2.5`, which put the same Pinnacle
    price into the odds table twice under one (book, market, outcome, period)
    key and would have double-counted it in any book sum.
    """
    name_set = set(names)
    drop: set[str] = set()
    for preferred, aliases in PINNACLE_ALIASES:
        if all(col in name_set for col in preferred):
            drop.update(aliases)
    if not drop:
        return present
    return {k: v for k, v in present.items() if k not in drop}


def _first_present(frame: pd.DataFrame, *candidates: str) -> str | None:
    """First of ``candidates`` present as a column. Greek files use HT/AT."""
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _goal_column(frame: pd.DataFrame, primary: str, alias: str) -> pd.Series[Any]:
    if primary in frame.columns:
        return pd.to_numeric(frame[primary], errors="coerce")
    if alias in frame.columns:
        return pd.to_numeric(frame[alias], errors="coerce")
    raise ValueError(f"missing goal column {primary} (or alias {alias})")


_MISSING_TOKENS = frozenset({"nan", "NaN", "NAN", "None", "NaT", "<NA>", "null", "NULL"})


def _text(series: pd.Series[Any]) -> pd.Series[Any]:
    """Coerce a column to stripped strings, with every missing form as "".

    Missing values must be normalised BEFORE `astype(str)`. Under pandas 3 that
    call preserves NA rather than rendering it as the literal "nan" earlier
    versions produced, so a post-hoc replace of "nan" silently does nothing and
    NA survives into comparisons — where `!= ""` is true and the value passes
    validation. Both known defects in this module came from that.
    """
    filled = series.where(series.notna(), "")
    text = filled.astype(str).str.strip()
    return text.where(~text.isin(_MISSING_TOKENS), "")


def _kickoff(frame: pd.DataFrame) -> pd.Series[Any]:
    """Kickoff timestamps, with the Time column folded in where present.

    Dates are parsed on their own and the time added as an offset, rather than
    concatenating the two into one string. Concatenation produced a column of
    mixed shapes whenever a file carried times for some fixtures and not others
    — "12/08/2023 15:00" beside "13/08/2023" — and pandas infers a single format
    from the first element, coercing every row that does not match to NaT. Those
    rows then failed the validity filter, so the archive's blank-time fixtures
    were silently and selectively dropped while their neighbours survived.
    """
    if "Date" not in frame.columns:
        raise ValueError("missing Date column")

    kickoff = pd.to_datetime(_text(frame["Date"]), dayfirst=True, format="mixed", errors="coerce")
    if "Time" not in frame.columns:
        return kickoff

    times = _text(frame["Time"])
    # to_timedelta needs HH:MM:SS; the archive publishes HH:MM.
    padded = times.where(times.eq("") | times.str.count(":").ge(2), times + ":00")
    offset = pd.to_timedelta(padded, errors="coerce").fillna(pd.Timedelta(0))
    return kickoff + offset


_PAD_PREFIX = "__pad_"


def _source_text(source: str | Path | IO[str]) -> str:
    """Read the whole file once, so the header can be inspected before parsing."""
    text = (
        str(source.read())
        if hasattr(source, "read")
        else Path(str(source)).read_text(encoding="utf-8", errors="replace")
    )
    # Some archive files are UTF-8 with a BOM, which otherwise attaches to the
    # first header name and turns "Div" into "\ufeffDiv".
    return text.lstrip("\ufeff")


def _read_padded_csv(source: str | Path | IO[str], label: str) -> pd.DataFrame:
    """Parse a season CSV, tolerating rows padded with trailing empty fields.

    Real archive files carry rows with more commas than the header — E0/0304
    mixes widths of 57, 62 and 72 against a 57-column header. Every extra field
    is empty, so this is comma padding, not a schema change, and pandas' default
    strictness rejects the whole file over it. Losing a league-season to
    formatting noise would silently shrink the study.

    Tolerating padding must not become tolerating unknown data, so any padded
    column that actually contains a value raises :class:`UnrecognisedHeaderError`
    exactly as an unrecognised header would.
    """
    text = _source_text(source)
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{label}: file is empty")

    raw_header = lines[0].split(",")
    widest = max((len(line.split(",")) for line in lines if line.strip()), default=len(raw_header))
    if widest <= len(raw_header):
        return pd.read_csv(io.StringIO(text), encoding_errors="replace")

    # Blank header fields are unnamed columns. Left as "" they collide with each
    # other and pandas rejects the file for duplicate names, which cost 24 real
    # league-seasons before this was handled.
    header = [
        name.strip() if name.strip() else f"{_PAD_PREFIX}blank{i}"
        for i, name in enumerate(raw_header)
    ]
    names = [*header, *(f"{_PAD_PREFIX}{i}" for i in range(widest - len(header)))]
    frame = pd.read_csv(io.StringIO(text), names=names, skiprows=1, encoding_errors="replace")

    padded = [c for c in frame.columns if str(c).startswith(_PAD_PREFIX)]
    populated = [c for c in padded if _text(frame[c]).ne("").any()]
    if populated:
        raise UnrecognisedHeaderError(
            [f"{len(header) + padded.index(c)} (unnamed, populated)" for c in populated],
            label,
        )
    return frame.drop(columns=padded)


def read_season_csv(
    source: str | Path | IO[str],
    *,
    season: str,
    league: str | None = None,
) -> SeasonTables:
    """Parse one football-data season file into matches and tidy odds."""
    label = str(source) if not hasattr(source, "read") else f"{league}/{season}"
    frame = _read_padded_csv(source, label)
    frame = frame.loc[
        :,
        [c for c in frame.columns if str(c).strip() != "" and not _UNNAMED.match(str(c).strip())],
    ]
    frame.columns = [str(c).strip() for c in frame.columns]
    odds_cols = discover_header(list(frame.columns), source=label)

    home_column = _first_present(frame, "HomeTeam", "HT")
    away_column = _first_present(frame, "AwayTeam", "AT")
    if home_column is None or away_column is None:
        raise ValueError(f"{label}: HomeTeam and AwayTeam are required")

    kickoff = _kickoff(frame)
    home_goals = _goal_column(frame, "FTHG", "HG")
    away_goals = _goal_column(frame, "FTAG", "AG")
    if "Div" in frame.columns:
        file_league = _text(frame["Div"])
        if league is None:
            unique = [v for v in file_league.unique() if v not in ("", "nan")]
            if len(unique) != 1:
                raise ValueError(f"{label}: expected one Div, got {unique}")
            league = unique[0]
        mismatch = file_league.notna() & file_league.ne("") & file_league.ne(league)
        if bool(mismatch.any()):
            raise ValueError(f"{label}: Div does not match league {league}")
    if league is None:
        raise ValueError(f"{label}: league is required when Div is absent")

    matches = pd.DataFrame(
        {
            "league": league,
            "season": season,
            "kickoff": kickoff,
            "home": _text(frame[home_column]),
            "away": _text(frame[away_column]),
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
    )
    valid = (
        matches["kickoff"].notna()
        & matches["home"].ne("")
        & matches["away"].ne("")
        & matches["home_goals"].notna()
        & matches["away_goals"].notna()
        & (matches["home_goals"] >= 0)
        & (matches["away_goals"] >= 0)
    )
    matches = matches.loc[valid].reset_index(drop=True)
    matches["home_goals"] = matches["home_goals"].astype(np.int64)
    matches["away_goals"] = matches["away_goals"].astype(np.int64)

    odds = _tidy_odds(frame.loc[valid].reset_index(drop=True), matches, odds_cols)
    return SeasonTables(matches=matches, odds=odds, source=label)


ODDS_COLUMNS: tuple[str, ...] = (
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
)


def _tidy_odds(
    usable: pd.DataFrame, matches: pd.DataFrame, odds_cols: dict[str, OddsColumn]
) -> pd.DataFrame:
    """Melt the wide odds columns into one tidy row per priced outcome.

    One block per odds column, concatenated once. The obvious loop — over
    columns, then over rows, taking ``matches.iloc[i]`` for each — costs a
    pandas row lookup per (column, row) pair. At roughly a hundred odds columns
    across five hundred season files that is tens of millions of lookups, and
    it made simply reading the archive take longer than fitting the model to it.
    """
    if not odds_cols:
        return pd.DataFrame(columns=list(ODDS_COLUMNS))

    key_columns = ["league", "season", "kickoff", "home", "away"]
    blocks: list[pd.DataFrame] = []
    for column, spec in odds_cols.items():
        prices = pd.to_numeric(usable[column], errors="coerce").to_numpy(dtype=np.float64)
        payable = np.isfinite(prices) & (prices > 1.0)
        if not payable.any():
            continue
        block = matches.loc[payable, key_columns].copy()
        block["book"] = spec.book
        block["market"] = spec.market
        block["period"] = spec.period
        block["outcome"] = spec.outcome
        block["decimal"] = prices[payable]
        blocks.append(block)

    if not blocks:
        return pd.DataFrame(columns=list(ODDS_COLUMNS))
    return pd.concat(blocks, ignore_index=True)[list(ODDS_COLUMNS)]


def concat_seasons(tables: list[SeasonTables]) -> SeasonTables:
    """Stack many season files."""
    if not tables:
        raise ValueError("no season tables to concatenate")
    matches = pd.concat([t.matches for t in tables], ignore_index=True)
    odds = pd.concat([t.odds for t in tables], ignore_index=True)
    return SeasonTables(matches=matches, odds=odds, source="concat")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_season(
    season: str,
    league: str,
    dest: Path,
    *,
    opener: urllib.request.OpenerDirector | None = None,
) -> Path | None:
    """Download one CSV. Returns None on HTTP 404 (league-season absent)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = season_url(season, league)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        if opener is None:
            with urllib.request.urlopen(request) as response:
                body = response.read()
        else:
            with opener.open(request) as response:
                body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    dest.write_bytes(body)
    return dest


def write_manifest(paths: list[Path], dest: Path) -> None:
    """SHA-256 of every downloaded CSV, for the results manifest."""
    records = [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"files": records}, indent=2) + "\n")
