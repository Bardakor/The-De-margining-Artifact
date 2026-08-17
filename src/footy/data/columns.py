"""Registry of football-data.co.uk column names.

Ingest discovers headers against this registry rather than assuming a fixed
schema. An unrecognised name is a hard error: silently mis-mapping an odds
column would corrupt the coverage matrix and every result that hangs on it.

Closing 1X2 columns insert C before the outcome (B365H → B365CH), as in
notes.txt. Over/under inserts C before the inequality (B365>2.5 → B365C>2.5).
Pinnacle's opening aliases PH/PD/PA and the P>2.5 over/under stem are
registered explicitly so they cannot be confused with a one-letter bookmaker.
"""

from dataclasses import dataclass
from typing import Literal

Market = Literal["1X2", "OU25"]
Period = Literal["open", "close"]

BOOKMAKERS: dict[str, str] = {
    "1XB": "1XBet",
    "B365": "Bet365",
    "BF": "Betfair",
    "BFD": "Betfred",
    "BMGM": "BetMGM",
    "BV": "BetVictor",
    "BS": "Blue Square",
    "BW": "Bet&Win",
    "CL": "Coral",
    "GB": "Gamebookers",
    "IW": "Interwetten",
    "LB": "Ladbrokes",
    "PP": "Paddy Power",
    "PS": "Pinnacle",
    "SK": "Skybet",
    "SO": "Sporting Odds",
    "SB": "Sportingbet",
    "SJ": "Stan James",
    "SY": "Stanleybet",
    "VC": "VC Bet",
    "WH": "William Hill",
    "BFE": "Betfair Exchange",
    "Max": "Market maximum",
    "Avg": "Market average",
    "BbMx": "Betbrain maximum",
    "BbAv": "Betbrain average",
}

RESULT_COLUMNS = frozenset(
    {
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "HG",
        "FTAG",
        "AG",
        "FTR",
        "Res",
        "HTHG",
        "HTAG",
        "HTR",
        "Attendance",
        "Referee",
        "HS",
        "AS",
        "HST",
        "AST",
        "HHW",
        "AHW",
        "HC",
        "AC",
        "HF",
        "AF",
        "HFKC",
        "AFKC",
        "HO",
        "AO",
        "HY",
        "AY",
        "HR",
        "AR",
        "HBP",
        "ABP",
    }
)

COUNT_COLUMNS = frozenset({"Bb1X2", "BbOU", "BbAH"})

_ASIAN_FIXED = frozenset(
    {
        "BbAHh",
        "AHh",
        "AHCh",
        "BbMxAHH",
        "BbAvAHH",
        "BbMxAHA",
        "BbAvAHA",
        "GBAHH",
        "GBAHA",
        "GBAH",
        "LBAHH",
        "LBAHA",
        "LBAH",
        "B365AHH",
        "B365AHA",
        "B365AH",
        "B365CAHH",
        "B365CAHA",
        "PAHH",
        "PAHA",
        "PCAHH",
        "PCAHA",
        "MaxAHH",
        "MaxAHA",
        "AvgAHH",
        "AvgAHA",
        "MaxCAHH",
        "MaxCAHA",
        "AvgCAHH",
        "AvgCAHA",
        "BbMxCAHH",
        "BbAvCAHH",
        "BbMxCAHA",
        "BbAvCAHA",
    }
)


@dataclass(frozen=True)
class OddsColumn:
    book: str
    market: Market
    outcome: str
    period: Period


def _asian_for_stems() -> frozenset[str]:
    names: set[str] = set()
    for stem in BOOKMAKERS:
        for side in ("AHH", "AHA", "AH"):
            names.add(f"{stem}{side}")
            names.add(f"{stem}C{side}")
    return frozenset(names)


def _odds_map() -> dict[str, OddsColumn]:
    mapping: dict[str, OddsColumn] = {}
    for stem, book in BOOKMAKERS.items():
        for outcome in ("H", "D", "A"):
            mapping[f"{stem}{outcome}"] = OddsColumn(book, "1X2", outcome, "open")
            mapping[f"{stem}C{outcome}"] = OddsColumn(book, "1X2", outcome, "close")
        for flag, outcome in ((">2.5", "O"), ("<2.5", "U")):
            mapping[f"{stem}{flag}"] = OddsColumn(book, "OU25", outcome, "open")
            mapping[f"{stem}C{flag}"] = OddsColumn(book, "OU25", outcome, "close")
    aliases: dict[str, OddsColumn] = {
        "PH": OddsColumn("Pinnacle", "1X2", "H", "open"),
        "PD": OddsColumn("Pinnacle", "1X2", "D", "open"),
        "PA": OddsColumn("Pinnacle", "1X2", "A", "open"),
        "P>2.5": OddsColumn("Pinnacle", "OU25", "O", "open"),
        "P<2.5": OddsColumn("Pinnacle", "OU25", "U", "open"),
        "PC>2.5": OddsColumn("Pinnacle", "OU25", "O", "close"),
        "PC<2.5": OddsColumn("Pinnacle", "OU25", "U", "close"),
    }
    mapping.update(aliases)
    return mapping


ASIAN_COLUMNS: frozenset[str] = _ASIAN_FIXED | _asian_for_stems()
ODDS_MAP: dict[str, OddsColumn] = _odds_map()
KNOWN_COLUMNS: frozenset[str] = RESULT_COLUMNS | COUNT_COLUMNS | ASIAN_COLUMNS | frozenset(ODDS_MAP)

PREFERRED_PINNACLE_OPEN = ("PSH", "PSD", "PSA")
ALIAS_PINNACLE_OPEN = ("PH", "PD", "PA")

# Pinnacle is published under a canonical stem (PS) and a one-letter alias (P).
# Where a file carries both, the alias is dropped so the same price does not
# enter the odds table twice under one (book, market, outcome, period) key — a
# duplicate would silently corrupt any book sum built from it, which is the
# quantity this whole study measures.
#
# Each entry is (preferred, alias): the alias is dropped only when EVERY
# preferred column is present, so a file carrying the alias alone keeps it.
PINNACLE_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (PREFERRED_PINNACLE_OPEN, ALIAS_PINNACLE_OPEN),
    (("PS>2.5", "PS<2.5"), ("P>2.5", "P<2.5")),
    (("PSC>2.5", "PSC<2.5"), ("PC>2.5", "PC<2.5")),
)
