"""Ingest football-data.co.uk season CSVs with header discovery.

Spec §3.1. Coverage is not assumed uniform. Opening and closing odds are
different columns; closing odds arrived part-way through the archive. The
ingest layer therefore matches each header against the registry in
``columns.py`` and fails loudly on anything it does not recognise.
"""

from __future__ import annotations

import hashlib
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
    ALIAS_PINNACLE_OPEN,
    KNOWN_COLUMNS,
    ODDS_MAP,
    PREFERRED_PINNACLE_OPEN,
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
USER_AGENT = "footy-research/0.1 (+https://github.com/Bardakor/betting-app-yami)"
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
    """Prefer PSH/PSD/PSA when both the canonical and PH/PD/PA aliases exist."""
    name_set = set(names)
    if all(col in name_set for col in PREFERRED_PINNACLE_OPEN):
        return {k: v for k, v in present.items() if k not in ALIAS_PINNACLE_OPEN}
    return present


def _goal_column(frame: pd.DataFrame, primary: str, alias: str) -> pd.Series[Any]:
    if primary in frame.columns:
        return pd.to_numeric(frame[primary], errors="coerce")
    if alias in frame.columns:
        return pd.to_numeric(frame[alias], errors="coerce")
    raise ValueError(f"missing goal column {primary} (or alias {alias})")


def _kickoff(frame: pd.DataFrame) -> pd.Series[Any]:
    if "Date" not in frame.columns:
        raise ValueError("missing Date column")
    dates = frame["Date"].astype(str).str.strip()
    if "Time" in frame.columns:
        times = frame["Time"].astype(str).str.strip().replace({"nan": "", "NaN": "", "None": ""})
        combined = dates.where(times.eq(""), dates + " " + times)
        kickoff = pd.to_datetime(combined, dayfirst=True, errors="coerce")
    else:
        kickoff = pd.to_datetime(dates, dayfirst=True, errors="coerce")
    return kickoff


def read_season_csv(
    source: str | Path | IO[str],
    *,
    season: str,
    league: str | None = None,
) -> SeasonTables:
    """Parse one football-data season file into matches and tidy odds."""
    label = str(source) if not hasattr(source, "read") else f"{league}/{season}"
    frame = pd.read_csv(source, encoding="utf-8", encoding_errors="replace")
    frame = frame.loc[
        :,
        [c for c in frame.columns if str(c).strip() != "" and not _UNNAMED.match(str(c).strip())],
    ]
    frame.columns = [str(c).strip() for c in frame.columns]
    odds_cols = discover_header(list(frame.columns), source=label)

    if "HomeTeam" not in frame.columns or "AwayTeam" not in frame.columns:
        raise ValueError(f"{label}: HomeTeam and AwayTeam are required")

    kickoff = _kickoff(frame)
    home_goals = _goal_column(frame, "FTHG", "HG")
    away_goals = _goal_column(frame, "FTAG", "AG")
    if "Div" in frame.columns:
        file_league = frame["Div"].astype(str).str.strip()
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
            "home": frame["HomeTeam"].astype(str).str.strip(),
            "away": frame["AwayTeam"].astype(str).str.strip(),
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

    odds_rows: list[dict[str, object]] = []
    usable = frame.loc[valid].reset_index(drop=True)
    for column, spec in odds_cols.items():
        values = pd.to_numeric(usable[column], errors="coerce")
        for i, price in enumerate(values):
            if not np.isfinite(price) or float(price) <= 1.0:
                continue
            row = matches.iloc[i]
            odds_rows.append(
                {
                    "league": row["league"],
                    "season": row["season"],
                    "kickoff": row["kickoff"],
                    "home": row["home"],
                    "away": row["away"],
                    "book": spec.book,
                    "market": spec.market,
                    "period": spec.period,
                    "outcome": spec.outcome,
                    "decimal": float(price),
                }
            )
    odds = pd.DataFrame(
        odds_rows,
        columns=[
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
        ],
    )
    return SeasonTables(matches=matches, odds=odds, source=label)


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
