"""Assemble the archive, run the walk-forward, and compare the four transforms.

Study spec §4, §5.4 and §6. This module is the study itself: everything else
in the package is apparatus it calls.

The comparison is deliberately narrow. One model, fit once per matchday, is
held constant across all four de-margining transforms. The only thing that
varies between the arms is how the offered odds are converted into a benchmark
probability, so any difference in measured edge is attributable to that choice
and nothing else.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from footy.data.coverage import coverage_matrix, eligible_cells
from footy.data.football_data import (
    SeasonTables,
    UnrecognisedHeaderError,
    concat_seasons,
    read_season_csv,
)
from footy.eval.economic import closing_line_value, settle
from footy.eval.inference import diebold_mariano, stationary_bootstrap
from footy.eval.scoring import ranked_probability_score
from footy.market.demargin import METHODS, demargin

MARKET_OUTCOMES: dict[str, tuple[str, ...]] = {"1X2": ("H", "D", "A"), "OU25": ("O", "U")}
MODEL_COLUMNS: dict[str, tuple[str, ...]] = {
    "1X2": ("p_home", "p_draw", "p_away"),
    "OU25": ("p_over25", "p_under25"),
}


@dataclass(frozen=True)
class Archive:
    """Everything parsed from disk, plus what could not be parsed."""

    tables: SeasonTables
    files_read: int
    unreadable: tuple[tuple[str, str], ...]

    @property
    def matches(self) -> pd.DataFrame:
        return self.tables.matches

    @property
    def odds(self) -> pd.DataFrame:
        return self.tables.odds


def load_archive(root: Path, *, leagues: Iterable[str] | None = None) -> Archive:
    """Parse every cached CSV under ``root``.

    A file that cannot be parsed is recorded rather than skipped silently: an
    unreadable league-season is missing data, and missing data that nobody
    counted is how a coverage table becomes a lie.
    """
    wanted = set(leagues) if leagues is not None else None
    tables: list[SeasonTables] = []
    unreadable: list[tuple[str, str]] = []

    for path in sorted(root.glob("*/*.csv")):
        season, league = path.parent.name, path.stem
        if wanted is not None and league not in wanted:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tables.append(read_season_csv(path, season=season, league=league))
        except (UnrecognisedHeaderError, ValueError) as exc:
            unreadable.append((f"{league}/{season}", type(exc).__name__))

    if not tables:
        raise ValueError(f"no readable season files under {root}")
    return Archive(concat_seasons(tables), len(tables), tuple(unreadable))


def coverage(archive: Archive) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The full coverage matrix and the subset eligible for the study."""
    full = coverage_matrix(archive.tables)
    return full, eligible_cells(full)


def closing_books(odds: pd.DataFrame, book: str, market: str) -> pd.DataFrame:
    """Pivot closing odds into one row per match with a column per outcome.

    Only matches whose closing book is complete survive: a transform applied to
    a partial book normalises over the wrong support and returns a benchmark
    that is wrong rather than merely noisy.
    """
    outcomes = MARKET_OUTCOMES[market]
    block = odds[(odds["book"] == book) & (odds["market"] == market) & (odds["period"] == "close")]
    if block.empty:
        return pd.DataFrame()

    wide = block.pivot_table(
        index=["league", "season", "kickoff", "home", "away"],
        columns="outcome",
        values="decimal",
        aggfunc="first",
    ).reset_index()
    missing = [o for o in outcomes if o not in wide.columns]
    if missing:
        return pd.DataFrame()
    return wide.dropna(subset=list(outcomes)).reset_index(drop=True)


@dataclass(frozen=True)
class CellResult:
    """One (league, book, market) cell, evaluated under all four transforms."""

    league: str
    book: str
    market: str
    n_matches: int
    model_rps: float
    per_method: pd.DataFrame

    @property
    def roi_spread(self) -> float:
        values = self.per_method["roi"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.max() - finite.min()) if finite.size else float("nan")

    @property
    def roi_signs_disagree(self) -> bool:
        values = self.per_method["roi"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        return bool(finite.size and (finite.max() > 0.0) and (finite.min() < 0.0))


def evaluate_cell(
    forecasts: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    league: str,
    book: str,
    market: str,
    seed: int = 20260817,
) -> CellResult | None:
    """Score one cell under every transform. None when nothing lines up."""
    wide = closing_books(odds, book, market)
    if wide.empty:
        return None

    keys = ["league", "season", "kickoff", "home", "away"]
    merged = forecasts.merge(wide, on=keys, how="inner", suffixes=("", "_odds"))
    if merged.empty:
        return None

    outcomes = MARKET_OUTCOMES[market]
    model = merged[list(MODEL_COLUMNS[market])].to_numpy(dtype=float)
    offered = merged[list(outcomes)].to_numpy(dtype=float)
    if market == "1X2":
        realised = merged["outcome"].to_numpy(dtype=int)
    else:
        total = merged["home_goals"].to_numpy(int) + merged["away_goals"].to_numpy(int)
        realised = (total <= 2).astype(int)  # 0 = over, 1 = under

    implied = 1.0 / offered
    model_rps = float(
        np.mean([ranked_probability_score(m, int(o)) for m, o in zip(model, realised, strict=True)])
    )

    won = np.zeros_like(offered, dtype=bool)
    won[np.arange(len(realised)), realised] = True

    records: list[dict[str, object]] = []
    for method in METHODS:
        benchmark = np.vstack([demargin(row, method).probabilities for row in implied])
        bench_rps = np.array(
            [ranked_probability_score(b, int(o)) for b, o in zip(benchmark, realised, strict=True)]
        )
        per_match_model = np.array(
            [ranked_probability_score(m, int(o)) for m, o in zip(model, realised, strict=True)]
        )
        test = diebold_mariano(per_match_model, bench_rps)
        result = settle(
            model.reshape(-1), benchmark.reshape(-1), offered.reshape(-1), won.reshape(-1)
        )
        interval = stationary_bootstrap(
            result.profit[result.selected] if result.n_bets > 1 else np.zeros(2),
            rng=np.random.default_rng(seed),
            n_resamples=2000,
            block_length=20.0,
        )
        records.append(
            {
                "method": method,
                "bench_rps": float(bench_rps.mean()),
                "rps_diff": test.mean_difference,
                "dm_stat": test.statistic,
                "dm_p": test.p_value,
                "n_bets": result.n_bets,
                "roi": result.roi,
                "roi_low": interval.low,
                "roi_high": interval.high,
            }
        )

    return CellResult(
        league=league,
        book=book,
        market=market,
        n_matches=len(merged),
        model_rps=model_rps,
        per_method=pd.DataFrame(records),
    )


def clv_for_cell(taken: np.ndarray, closing: np.ndarray, selected: np.ndarray) -> float:
    """Mean closing-line value over the backed selections."""
    values = closing_line_value(taken, closing, selected=selected)
    backed = values[np.asarray(selected, dtype=bool)]
    return float(backed.mean()) if backed.size else float("nan")
