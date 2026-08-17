"""Run the de-margining study end to end.

Stages, in the order the spec requires:

    coverage     parse the archive, publish what is actually available
    calibrate    select xi on the calibration period ONLY
    evaluate     run the evaluation period against the frozen xi

`calibrate` writes the chosen xi to results/calibration.json. The
pre-registration must then be committed with that value BEFORE `evaluate` is
run — the git timestamp is what makes the prediction a prediction. `evaluate`
refuses to start unless paper/preregistration.md exists and names the same xi.

Usage:
    python scripts/study.py coverage
    python scripts/study.py calibrate
    python scripts/study.py evaluate
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from footy.data.football_data import LEAGUES
from footy.fit.decay import half_life_to_xi, xi_to_half_life
from footy.study.run import evaluate_cell, load_archive
from footy.study.walkforward import split_periods, walk_forward

RAW = Path("data/raw")
RESULTS = Path("results")
PREREG = Path("paper/preregistration.md")

BURN_IN_SEASONS = 3
CALIBRATION_SEASONS = 2
MIN_FIT_MATCHES = 300
HALF_LIFE_CANDIDATES = (30.0, 60.0, 90.0, 150.0, 250.0, 400.0, 700.0)


def _archive() -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = load_archive(RAW)
    print(f"parsed {archive.files_read} files, {len(archive.matches):,} matches")
    if archive.unreadable:
        print(f"  unreadable: {len(archive.unreadable)} -> {archive.unreadable[:5]}")
    return archive.matches, archive.odds


def _eligible_leagues(matches: pd.DataFrame) -> list[str]:
    counts = matches.groupby("league")["season"].nunique()
    return sorted(
        str(league)
        for league, n in counts.items()
        if n > BURN_IN_SEASONS + CALIBRATION_SEASONS and league in LEAGUES
    )


def cmd_coverage() -> None:
    from footy.data.coverage import coverage_matrix, eligible_cells
    from footy.data.football_data import SeasonTables

    matches, odds = _archive()
    RESULTS.mkdir(parents=True, exist_ok=True)
    full = coverage_matrix(SeasonTables(matches=matches, odds=odds, source="archive"))
    eligible = eligible_cells(full)
    full.to_csv(RESULTS / "coverage.csv", index=False)
    eligible.to_csv(RESULTS / "coverage_eligible.csv", index=False)

    print(f"\ncoverage rows: {len(full):,}   eligible: {len(eligible):,}")
    if not eligible.empty:
        by_book = (
            eligible.groupby("book")["n_with_closing"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
        )
        print("\ntop books by complete closing matches:")
        print(by_book.head(12).to_string())
        print(f"\nleagues with usable history: {_eligible_leagues(matches)}")


def cmd_calibrate() -> None:
    matches, _ = _archive()
    RESULTS.mkdir(parents=True, exist_ok=True)
    leagues = _eligible_leagues(matches)
    print(f"calibrating on {len(leagues)} leagues: {leagues}")

    records: list[dict[str, object]] = []
    for half_life in HALF_LIFE_CANDIDATES:
        xi = half_life_to_xi(half_life)
        scores: list[float] = []
        t0 = time.time()
        for league in leagues:
            block = matches[matches["league"] == league]
            seasons = sorted(set(block["season"]))
            if len(seasons) <= BURN_IN_SEASONS + CALIBRATION_SEASONS:
                continue
            periods = split_periods(
                seasons, burn_in=BURN_IN_SEASONS, calibration=CALIBRATION_SEASONS
            )
            forecasts = walk_forward(
                block,
                xi=xi,
                scored_seasons=periods.calibration,
                min_fit_matches=MIN_FIT_MATCHES,
            )
            if forecasts.empty:
                continue
            from footy.eval.scoring import ranked_probability_score

            probs = forecasts[["p_home", "p_draw", "p_away"]].to_numpy(float)
            outs = forecasts["outcome"].to_numpy(int)
            scores.extend(
                ranked_probability_score(p, int(o)) for p, o in zip(probs, outs, strict=True)
            )
        mean_rps = float(np.mean(scores)) if scores else float("nan")
        records.append(
            {"half_life_days": half_life, "xi": xi, "mean_rps": mean_rps, "n": len(scores)}
        )
        print(
            f"  half-life {half_life:>6.0f}d  xi={xi:.6f}  "
            f"mean RPS={mean_rps:.6f}  n={len(scores):,}  ({time.time() - t0:.0f}s)"
        )

    table = pd.DataFrame(records)
    table.to_csv(RESULTS / "calibration.csv", index=False)
    usable = table.dropna(subset=["mean_rps"])
    if usable.empty:
        raise SystemExit("no candidate produced calibration forecasts")

    best_row = usable.iloc[int(np.argmin(usable["mean_rps"].to_numpy(float)))]
    payload = {
        "xi": float(best_row["xi"]),
        "half_life_days": float(best_row["half_life_days"]),
        "mean_rps": float(best_row["mean_rps"]),
        "n_forecasts": int(best_row["n"]),
        "candidates": records,
        "burn_in_seasons": BURN_IN_SEASONS,
        "calibration_seasons": CALIBRATION_SEASONS,
        "min_fit_matches": MIN_FIT_MATCHES,
        "leagues": leagues,
    }
    (RESULTS / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"\nSELECTED xi = {payload['xi']:.6f} "
        f"(half-life {payload['half_life_days']:.0f} days), mean RPS {payload['mean_rps']:.6f}"
    )
    print("\nNEXT: commit paper/preregistration.md naming this xi, THEN run evaluate.")


def _frozen_xi() -> float:
    """Read xi from the committed pre-registration, not from the results file.

    The pre-registration is the artefact whose git timestamp makes the
    predictions predictions. Reading xi from anywhere else would let the
    evaluation run against a value nobody committed to in advance.
    """
    if not PREREG.exists():
        raise SystemExit(
            f"{PREREG} does not exist. The pre-registration must be committed, naming the "
            "frozen xi, BEFORE the evaluation period is touched."
        )
    text = PREREG.read_text()
    match = re.search(r"xi\s*=\s*([0-9.]+)", text)
    if not match:
        raise SystemExit(f"{PREREG} does not name a frozen xi")
    xi = float(match.group(1))

    recorded = json.loads((RESULTS / "calibration.json").read_text())
    if abs(xi - float(recorded["xi"])) > 1e-9:
        raise SystemExit(
            f"pre-registered xi {xi} does not match calibration {recorded['xi']}; "
            "the evaluation must run against the value that was committed"
        )
    return xi


def cmd_evaluate() -> None:
    xi = _frozen_xi()
    print(f"frozen xi = {xi:.6f}  (half-life {xi_to_half_life(xi):.0f} days)")

    matches, odds = _archive()
    RESULTS.mkdir(parents=True, exist_ok=True)
    leagues = _eligible_leagues(matches)

    all_forecasts: list[pd.DataFrame] = []
    for league in leagues:
        block = matches[matches["league"] == league]
        seasons = sorted(set(block["season"]))
        periods = split_periods(seasons, burn_in=BURN_IN_SEASONS, calibration=CALIBRATION_SEASONS)
        t0 = time.time()
        forecasts = walk_forward(
            block, xi=xi, scored_seasons=periods.evaluation, min_fit_matches=MIN_FIT_MATCHES
        )
        print(f"  {league}: {len(forecasts):,} forecasts ({time.time() - t0:.0f}s)")
        if not forecasts.empty:
            all_forecasts.append(forecasts)

    if not all_forecasts:
        raise SystemExit("no evaluation forecasts produced")
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    forecasts.to_parquet(RESULTS / "forecasts.parquet")
    print(f"\ntotal evaluation forecasts: {len(forecasts):,}")

    books = sorted(set(odds["book"]))
    rows: list[dict[str, object]] = []
    for league in leagues:
        for book in books:
            for market in ("1X2", "OU25"):
                cell = evaluate_cell(
                    forecasts[forecasts["league"] == league],
                    odds[odds["league"] == league],
                    league=league,
                    book=book,
                    market=market,
                )
                if cell is None or cell.n_matches < 200:
                    continue
                for record in cell.per_method.to_dict("records"):
                    rows.append(
                        {
                            "league": league,
                            "book": book,
                            "market": market,
                            "n_matches": cell.n_matches,
                            "model_rps": cell.model_rps,
                            **record,
                        }
                    )
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "cells.csv", index=False)
    print(f"evaluated cells: {results[['league', 'book', 'market']].drop_duplicates().shape[0]}")
    print(f"wrote {RESULTS / 'cells.csv'}")


COMMANDS = {"coverage": cmd_coverage, "calibrate": cmd_calibrate, "evaluate": cmd_evaluate}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(f"usage: python scripts/study.py {{{'|'.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()
