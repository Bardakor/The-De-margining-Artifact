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
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
# Extended past 700 days because the first sweep put its minimum on the
# boundary, which means the grid was drawn too narrow to contain the optimum.
# The last entry is effectively "no decay": a half-life far longer than the
# archive weights every match almost equally, and is the honest baseline for
# whether decay earns its place at all.
HALF_LIFE_CANDIDATES = (30.0, 60.0, 90.0, 150.0, 250.0, 400.0, 700.0, 1200.0, 2000.0, 40000.0)


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
    """Select xi by mean RPS on the fixtures EVERY candidate forecast.

    Candidates do not all price the same fixtures: a fit can fail to converge,
    or leave a team unidentified, and those fixtures are dropped. The first
    sweep varied from 7,804 to 11,152 forecasts across candidates, so its means
    were computed over different match sets — an arm that happens to drop
    harder fixtures scores better for a reason unrelated to forecast quality.

    Scoring on the intersection makes the comparison paired, which is the only
    way the ranking can be attributed to the decay rate itself.
    """
    from footy.eval.scoring import ranked_probability_score

    matches, _ = _archive()
    RESULTS.mkdir(parents=True, exist_ok=True)
    leagues = _eligible_leagues(matches)
    print(f"calibrating on {len(leagues)} leagues: {leagues}")

    per_candidate: dict[float, dict[tuple[object, ...], float]] = {}
    for half_life in HALF_LIFE_CANDIDATES:
        xi = half_life_to_xi(half_life)
        scored: dict[tuple[object, ...], float] = {}
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
                block, xi=xi, scored_seasons=periods.calibration, min_fit_matches=MIN_FIT_MATCHES
            )
            if forecasts.empty:
                continue
            probs = forecasts[["p_home", "p_draw", "p_away"]].to_numpy(float)
            outs = forecasts["outcome"].to_numpy(int)
            keys = list(
                zip(
                    forecasts["league"],
                    forecasts["kickoff"],
                    forecasts["home"],
                    forecasts["away"],
                    strict=True,
                )
            )
            for key, prob, out in zip(keys, probs, outs, strict=True):
                scored[key] = ranked_probability_score(prob, int(out))
        per_candidate[half_life] = scored
        print(f"  half-life {half_life:>7.0f}d  n={len(scored):,}  ({time.time() - t0:.0f}s)")

    common: set[tuple[object, ...]] = set.intersection(*(set(v) for v in per_candidate.values()))
    print(f"\ncommon fixtures across all {len(per_candidate)} candidates: {len(common):,}")
    if not common:
        raise SystemExit("no fixture was forecast by every candidate")

    records: list[dict[str, object]] = []
    for half_life, scored in per_candidate.items():
        paired = float(np.mean([scored[k] for k in common]))
        records.append(
            {
                "half_life_days": half_life,
                "xi": half_life_to_xi(half_life),
                "mean_rps_paired": paired,
                "mean_rps_all": float(np.mean(list(scored.values()))),
                "n_all": len(scored),
                "n_paired": len(common),
            }
        )

    table = pd.DataFrame(records).sort_values("mean_rps_paired").reset_index(drop=True)
    table.to_csv(RESULTS / "calibration.csv", index=False)
    print("\npaired comparison (all candidates scored on the same fixtures):")
    print(table.to_string(index=False))

    best = table.iloc[0]
    at_boundary = best["half_life_days"] in (
        min(HALF_LIFE_CANDIDATES),
        max(HALF_LIFE_CANDIDATES),
    )
    spread = float(table["mean_rps_paired"].max() - table["mean_rps_paired"].min())

    payload = {
        "xi": float(best["xi"]),
        "half_life_days": float(best["half_life_days"]),
        "mean_rps_paired": float(best["mean_rps_paired"]),
        "n_paired": int(best["n_paired"]),
        "selected_at_grid_boundary": bool(at_boundary),
        "paired_rps_spread": spread,
        "candidates": records,
        "burn_in_seasons": BURN_IN_SEASONS,
        "calibration_seasons": CALIBRATION_SEASONS,
        "min_fit_matches": MIN_FIT_MATCHES,
        "leagues": leagues,
    }
    (RESULTS / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"\nSELECTED xi = {payload['xi']:.6f} "
        f"(half-life {payload['half_life_days']:.0f} days), "
        f"paired mean RPS {payload['mean_rps_paired']:.6f} on {len(common):,} fixtures"
    )
    if at_boundary:
        print("  WARNING: the minimum sits on the grid boundary; the grid is too narrow.")
    print(f"  RPS spread across the whole grid: {spread:.6f}")
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


def _league_evaluation(job: tuple[str, pd.DataFrame, float]) -> tuple[str, pd.DataFrame]:
    """One league's evaluation forecasts. Module-level so it can be pickled.

    Leagues are independent and every fit is deterministic, so running these in
    parallel produces bit-identical output to running them in sequence. This
    changes no part of the registered protocol — only how many cores it uses.
    """
    league, block, xi = job
    seasons = sorted(set(block["season"]))
    periods = split_periods(seasons, burn_in=BURN_IN_SEASONS, calibration=CALIBRATION_SEASONS)
    forecasts = walk_forward(
        block, xi=xi, scored_seasons=periods.evaluation, min_fit_matches=MIN_FIT_MATCHES
    )
    return league, forecasts


def cmd_evaluate() -> None:
    xi = _frozen_xi()
    print(f"frozen xi = {xi:.6f}  (half-life {xi_to_half_life(xi):.0f} days)")

    matches, odds = _archive()
    RESULTS.mkdir(parents=True, exist_ok=True)
    leagues = _eligible_leagues(matches)

    # Per-league forecasts are written as each league finishes, so a failure
    # anywhere downstream costs no recomputation. The first run of this stage
    # completed all sixteen leagues and then died on the final write, losing
    # half an hour of fits that were already correct.
    forecast_dir = RESULTS / "forecasts"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    pending = [lg for lg in leagues if not (forecast_dir / f"{lg}.csv").exists()]
    done = [lg for lg in leagues if lg not in pending]
    if done:
        print(f"reusing cached forecasts for {len(done)} leagues: {done}")

    jobs = [(league, matches[matches["league"] == league].copy(), xi) for league in pending]
    workers = max(1, min(max(len(jobs), 1), (os.cpu_count() or 2) - 2))
    print(f"running {len(jobs)} leagues across {workers} workers")

    t0 = time.time()
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for league, block in pool.map(_league_evaluation, jobs):
                block.to_csv(forecast_dir / f"{league}.csv", index=False)
                print(f"  {league}: {len(block):,} forecasts  ({time.time() - t0:.0f}s elapsed)")

    loaded = [
        pd.read_csv(forecast_dir / f"{lg}.csv", parse_dates=["kickoff"], dtype={"season": str})
        for lg in leagues
        if (forecast_dir / f"{lg}.csv").exists()
    ]
    loaded = [f for f in loaded if not f.empty]
    if not loaded:
        raise SystemExit("no evaluation forecasts produced")
    forecasts = pd.concat(loaded, ignore_index=True)
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
