"""M5 (multiverse), M8 (synthesis), M6 (recognition) and M9 (recovery) on the real data.

Design §5 (M5, M6, M8, M9); pre-registration A1, A11-A14, A16 (i). Stages, in order:

- inputs: M5's player_season (scope rows league / ucl / intl) and impact_season on the
  original data, checked against panel_player_season; written for inspection.
- multiverse: M5 on the original data and on the B = 200 match-bootstrap replicates
  (A13 iii) through ``synthesis.multiverse_runs``; point ranks, weights, fallbacks, rank
  probabilities, variance indices, specification curve, distinct rankings; replicate
  scores as one float32 array.
- synthesis: M8 with sigma = M4's clipped cross-half total share per group (A13 ii),
  omega from M5's weights, U = 4; verdicts H1, H2, H4, H5; Kendall's W (A13 v); the
  exploratory comparison with V1.
- synthesis_bias (post hoc, robustness): the same with the bias-corrected shares of
  results/players/construct_gap/assortative/summary.json, if that file exists.
- recognition: M6 on A14's universe and features; H6 by the OLS shares (A1).
- recovery: M9's rank recovery on the real schedule (exploratory).

Outputs: results/players/{multiverse,synthesis,recognition,recovery}/, each with a
summary.json of {"value", "description"} entries (macro names of
paper-unmeasured/macros-defaults.tex where a quantity maps) and timings.json.

Seeds: master 20261114 (A12); each task draws from
SeedSequence(20261114, spawn_key=(crc32(task name),)), as run_impact.py.

Choices the pre-registration does not fix (post hoc, Claude's calls):

- M5 rows: Transfermarkt ids only (StatsBomb pseudo ids dropped) with a known position
  group, as M1. team_elo of a scope row = minutes-weighted mean pre-match Elo of the
  player's side in that scope's matches (national-team Elo for intl rows; the module
  then averages it across scopes). on_pitch_gd of a row without stint minutes = the
  minutes-weighted on-pitch GD per 90 of his position group in that season (all scope
  rows with stint minutes pooled), recomputed in every replicate.
- impact_season: a player-season without a per-season RAPM row takes the pooled
  estimate if the player has his own RAPM columns (the per-season prior, no data that
  season), else the replacement column's estimate for that season; rapm_total_pooled of
  a replacement player is the replacement column's pooled total. Counted.
- Stabilisation (A16 i): per (metric, group) from M1's ``by_position_group``. M1 has no
  goalkeeper k for goals, non-penalty goals and assists: k = 1e6 minutes (full
  shrinkage toward the goalkeeper mean).
- Replicate b: row weight = counts[b] of its game; panel games outside the RAPM design
  keep weight 1 (counted). Weighted minutes enter the minutes thresholds.
- M8's mean rank (V1 comparison) = 1 + sum_j P(j above i) from the pairwise matrix
  (ties count half). V1's order is its rank column.
- M6: the individual-statistics block is read as goals and assists (season totals),
  minutes, and non-penalty goals per 90 (the alternative reading, per-90 goals, assists
  and non-penalty goals plus minutes, is fitted as robustness). Team success, a player
  counts for a club in a competition-season if he played at least one minute for it
  there. League title: first in the computed table of the competition-season
  (Transfermarkt's season); none is awarded for the unfinished 2026 season of
  calendar-year leagues (2025/26 of a competition whose 2024/25 lies in one calendar
  year). UCL and international finals: tm_games' "Final" round; a drawn final is
  decided by the shoot-out in the StatsBomb events where the match is in StatsBomb.
  Every national-team competition with a final in tm_games counts as a senior
  tournament. A missing final gives no indicator (counted, reported). Market value:
  candidate-editions without a valuation on or before 30 June are dropped and counted.
- M9: pool = outfield pool players with at least 900 stint minutes (others dropped and
  counted); positions as M3 (unknown group for the few stint players without one);
  simulations split over worker processes with spawned seeds; gap bins recomputed on
  the pooled simulations as simulation.rank_recovery does.

Usage:
    uv run python scripts/players/run_ranking.py                # every stage
    uv run python scripts/players/run_ranking.py --stage recognition
    uv run python scripts/players/run_ranking.py --from synthesis
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import math
import shutil
import time
import zlib
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from footy.players.multiverse import (
    BOX,
    DIMENSIONS,
    SCOPES,
    STAT_METRICS,
    Multiverse,
    distinct_rankings,
    rank_probabilities,
    spec_grid,
    specification_curve,
    variance_indices,
)
from footy.players.paths import PAPER, PROCESSED, RAW, RESULTS
from footy.players.perception import bootstrap_shares, fit_recognition
from footy.players.rapm import REPLACEMENT
from footy.players.simulation import SimParams, rank_recovery
from footy.players.synthesis import (
    DECIDED,
    multiverse_runs,
    stats_weight_share,
    synthesise,
    verdicts,
    view_agreement,
)

MASTER_SEED = 20261114
N_BOOT = 200
GK_BOX_K = 1e6  # no goalkeeper box-metric k in M1: full shrinkage
N_M5_WORKERS = 4
N_SIM_WORKERS = 3
N_RECOVERY_SIMS = 200
RECOVERY_BINS = 4
MIN_MINUTES = 900.0
LEVEL = 0.9
UNKNOWN = "unknown"
EDITION_SEASON = {e: f"{e - 1}/{e % 100:02d}" for e in range(2021, 2027)}
PREVIOUS_EDITION = {2021: 2019} | {e: e - 1 for e in range(2022, 2027)}  # A3
BLOCKS = {
    "stats": ["goals", "assists", "minutes", "np_goals_p90"],
    "impact": ["rapm_offence", "rapm_defence"],
    "team": ["league_title", "ucl_winner", "ucl_finalist", "intl_winner"],
    "reputation": ["log_prev_points", "log_market_value"],
}
BLOCKS_PER90 = BLOCKS | {"stats": ["goals_p90", "assists_p90", "np_goals_p90", "minutes"]}
SUM_COLUMNS = ("minutes", *BOX, "gd", "stint_minutes", "elo_minutes")
DIM_NAMES = {d: d for d in DIMENSIONS} | {"min_minutes": "minimum minutes"}
LEVEL_LABELS: dict[str, dict[Any, str]] = {
    "family": {"box": "box", "box_gd": "box+GD", "impact": "impact", "hybrid": "hybrid"},
    "weights": {"equal_z": "equal-z", "pc1": "first principal component",
                "rapm_ridge": "ridge on RAPM", "v1_like": "V1-like"},
    "position": {"none": "no position normalisation", "group_z": "within-group z",
                 "replacement": "replacement level"},
    "context": {"none": "no context adjustment", "elo_partialled": "Elo-partialled"},
    "aggregation": {"peak": "peak season", "best3": "best three seasons",
                    "mean": "window mean", "volume": "window volume",
                    "recency": "recency-weighted"},
    "competition": {"equal": "equal competition weights",
                    "ucl_intl_1.5": "UCL and internationals x1.5",
                    "league_only": "league only"},
    "min_minutes": {900: "900 minutes", 1800: "1,800 minutes"},
    "shrinkage": {"raw": "raw rates", "eb": "empirical Bayes"},
}  # fmt: skip

RAPM_DIR = RESULTS / "rapm"
GAP_DIR = RESULTS / "construct_gap"
REL_DIR = RESULTS / "reliability"
M5_DIR = RESULTS / "multiverse"
REP_DIR = M5_DIR / "replicates_parts"
M8_DIR = RESULTS / "synthesis"
M8_BIAS_DIR = M8_DIR / "bias_corrected"
M6_DIR = RESULTS / "recognition"
M9_DIR = RESULTS / "recovery"


# --- seeds and files ------------------------------------------------------------------


def seed(task: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(MASTER_SEED, spawn_key=(zlib.crc32(task.encode()),))


def rng(task: str) -> np.random.Generator:
    return np.random.default_rng(seed(task))


def plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text())) if path.exists() else {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def record(folder: Path, entries: dict[str, tuple[Any, str]]) -> None:
    """Merge headline numbers into ``folder/summary.json``."""
    path = folder / "summary.json"
    current = read_json(path)
    current.update({k: {"value": v, "description": d} for k, (v, d) in entries.items()})
    write_json(path, current)


def to_csv(df: pd.DataFrame, path: Path, index: bool = False, exact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, float_format=None if exact else "%.6g")


# --- inputs ---------------------------------------------------------------------------


@functools.cache
def pool() -> pd.DataFrame:
    p = pd.read_csv(PAPER / "data" / "pool.csv")
    p["player_id"] = p["tm_player_id"].astype(np.int64)
    return p


def pool_ids() -> np.ndarray:
    return pool()["player_id"].to_numpy(np.int64)


def names() -> pd.Series:
    return pool().set_index("player_id")["player"]


@dataclass(frozen=True)
class Panel:
    keys: pd.DataFrame  # player_id, season, scope, position_group: one row per unit
    unit: np.ndarray  # panel row -> unit
    game: np.ndarray  # panel row -> column of the bootstrap counts, -1 outside the design
    values: np.ndarray  # panel rows x SUM_COLUMNS
    counts: dict[str, int]


@functools.cache
def panel() -> Panel:
    cols = ["game_id", "player_id", "season", "scope", "position_group", "minutes", *BOX,
            "on_pitch_gf", "on_pitch_ga", "team_elo"]  # fmt: skip
    pm = pd.read_csv(PROCESSED / "panel_player_match.csv.gz", usecols=cols)
    pm = pm[pm["scope"].isin(SCOPES)]
    pseudo, no_group = pm["player_id"] < 0, pm["position_group"].isna()
    counts = {"rows_pseudo_id": int(pseudo.sum()), "rows_no_group": int((no_group & ~pseudo).sum())}
    pm = pm[~pseudo & ~no_group].reset_index(drop=True)
    unit, keys = pd.factorize(pd.MultiIndex.from_frame(pm[["player_id", "season", "scope"]]))
    frame = keys.to_frame(index=False, name=["player_id", "season", "scope"])
    frame["position_group"] = pm.groupby(unit)["position_group"].first().to_numpy()
    if (pm.groupby("player_id")["position_group"].nunique() > 1).any():
        raise ValueError("position_group varies within a player")
    game_ids = pd.Index(np.load(RAPM_DIR / "bootstrap_pooled.npz")["game_ids"])
    game = game_ids.get_indexer(pm["game_id"])
    counts["games_outside_design"] = int(pm.loc[game < 0, "game_id"].nunique())
    counts["rows_outside_design"] = int((game < 0).sum())
    stint = pm["on_pitch_gf"].notna().to_numpy()
    m = pm["minutes"].to_numpy(np.float64)
    gd = (pm["on_pitch_gf"] - pm["on_pitch_ga"]).to_numpy(np.float64)
    values = np.column_stack(
        [m, *(pm[c].to_numpy(np.float64) for c in BOX), np.where(stint, gd, 0.0),
         np.where(stint, m, 0.0), m * pm["team_elo"].to_numpy(np.float64)]
    )  # fmt: skip
    return Panel(frame, np.asarray(unit), game, values, counts)


def player_season(weights: np.ndarray) -> tuple[pd.DataFrame, int]:
    """M5's player_season from panel row weights (all 1 = the original data), and the
    number of rows whose on_pitch_gd is the group-season mean."""
    p = panel()
    n = len(p.keys)
    s = np.column_stack(
        [np.bincount(p.unit, weights=p.values[:, j] * weights, minlength=n)
         for j in range(len(SUM_COLUMNS))]
    )  # fmt: skip
    keep = s[:, 0] > 0
    out = p.keys[keep].reset_index(drop=True)
    s = s[keep]
    out["minutes"] = s[:, 0]
    for j, c in enumerate(BOX, start=1):
        out[c] = 90.0 * s[:, j] / s[:, 0]
    out["team_elo"] = s[:, 6] / s[:, 0]
    gd, sm = pd.Series(s[:, 4]), pd.Series(s[:, 5])
    cell = out["position_group"] + " " + out["season"]
    mean = 90.0 * gd.groupby(cell).transform("sum") / sm.groupby(cell).transform("sum")
    has = (sm > 0).to_numpy()
    out["on_pitch_gd"] = np.where(has, 90.0 * gd / sm.where(has, 1.0), mean)
    if out["on_pitch_gd"].isna().any():
        raise ValueError("a group-season has no stint minutes")
    return out, int((~has).sum())


def impact_season(
    ps: pd.DataFrame, season_est: pd.DataFrame, pooled_est: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int]]:
    """M5's impact_season: per-season RAPM (index player_id, season) and pooled RAPM
    (index player_id), both with offence and defence; fills as the module doc says."""
    pairs = ps[["player_id", "season"]].drop_duplicates().reset_index(drop=True)
    est = season_est.reindex(pd.MultiIndex.from_frame(pairs))[["offence", "defence"]]
    has = est["offence"].notna().to_numpy()
    own = pairs["player_id"].isin(pooled_est.index.drop(REPLACEMENT)).to_numpy()
    prior = pooled_est.reindex(pairs["player_id"])[["offence", "defence"]].to_numpy()
    repl_key = pd.MultiIndex.from_arrays(
        [np.full(len(pairs), REPLACEMENT), pairs["season"].to_numpy()]
    )
    repl = season_est.reindex(repl_key)[["offence", "defence"]].to_numpy()
    values = np.where(has[:, None], est.to_numpy(), np.where(own[:, None], prior, repl))
    total = pooled_est["offence"] + pooled_est["defence"]
    total_pooled = np.where(
        own, total.reindex(pairs["player_id"]).to_numpy(), float(total.loc[REPLACEMENT])
    )
    out = pairs.assign(
        rapm_offence=values[:, 0], rapm_defence=values[:, 1], rapm_total_pooled=total_pooled
    )
    if out[["rapm_offence", "rapm_defence", "rapm_total_pooled"]].isna().to_numpy().any():
        raise ValueError("impact_season has missing values")
    counts = {
        "impact_from_pooled_prior": int((~has & own).sum()),
        "impact_replacement": int((~has & ~own).sum()),
    }
    return out, counts


def point_rapm() -> tuple[pd.DataFrame, pd.DataFrame]:
    season = pd.read_csv(RAPM_DIR / "rapm_season.csv").set_index(["player_id", "season"])
    pooled = pd.read_csv(RAPM_DIR / "rapm_pooled.csv").set_index("player_id")
    return season[["offence", "defence"]], pooled[["offence", "defence"]]


@functools.cache
def boot_arrays() -> dict[str, Any]:
    zs = np.load(RAPM_DIR / "bootstrap_season.npz")
    zp = np.load(RAPM_DIR / "bootstrap_pooled.npz")
    return {
        "season_keys": pd.MultiIndex.from_arrays(
            [zs["player_id"], zs["season"].astype(str)], names=["player_id", "season"]
        ),
        "season_offence": zs["offence"], "season_defence": zs["defence"],
        "players": pd.Index(zp["players"], name="player_id"),
        "pooled_offence": zp["offence"], "pooled_defence": zp["defence"],
        "counts": zp["counts"],
    }  # fmt: skip


def replicate_inputs(b: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    z, p = boot_arrays(), panel()
    w = np.where(p.game >= 0, z["counts"][b][np.maximum(p.game, 0)], 1).astype(np.float64)
    ps, _ = player_season(w)
    season = pd.DataFrame(
        {"offence": z["season_offence"][b].astype(np.float64),
         "defence": z["season_defence"][b].astype(np.float64)}, index=z["season_keys"],
    )  # fmt: skip
    pooled = pd.DataFrame(
        {"offence": z["pooled_offence"][b].astype(np.float64),
         "defence": z["pooled_defence"][b].astype(np.float64)}, index=z["players"],
    )  # fmt: skip
    imp, _ = impact_season(ps, season, pooled)
    return ps, imp


def point_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    ps, filled = player_season(np.ones(len(panel().unit)))
    imp, counts = impact_season(ps, *point_rapm())
    return ps, imp, panel().counts | counts | {"on_pitch_gd_group_season_fill": filled}


@functools.cache
def stabilisation() -> dict[str, dict[str, float]]:
    """A16 (i): k per (metric, position group); GK box metrics GK_BOX_K."""
    by = read_json(REL_DIR / "stabilisation_k.json")["by_position_group"]
    out: dict[str, dict[str, float]] = {}
    for metric in STAT_METRICS:
        out[metric] = {}
        for group, ks in by.items():
            if metric not in ks and not (group == "GK" and metric in BOX):
                raise ValueError(f"no stabilisation constant for {metric} in {group}")
            out[metric][group] = float(ks.get(metric, GK_BOX_K))
    return out


def stage_inputs() -> None:
    ps, imp, counts = point_inputs()
    check = pd.read_csv(PROCESSED / "panel_player_season.csv.gz", low_memory=False)
    check = check[check["scope"].isin(SCOPES) & (check["player_id"] >= 0)]
    check = check[["player_id", "season", "scope", "games", "minutes", "minutes_stints",
                   "on_pitch_gf", "on_pitch_ga", *(f"{c}_p90" for c in BOX)]]  # fmt: skip
    both = ps.merge(check, on=["player_id", "season", "scope"], how="left", validate="1:1")
    if both["games"].isna().any():
        raise ValueError("player_season rows missing from panel_player_season")
    np.testing.assert_allclose(both["minutes_x"], both["minutes_y"])
    for c in BOX:
        np.testing.assert_allclose(both[c], both[f"{c}_p90"], rtol=1e-9, atol=1e-12)
    has = both["minutes_stints"] > 0
    gd = 90 * (both["on_pitch_gf"] - both["on_pitch_ga"]) / both["minutes_stints"]
    np.testing.assert_allclose(both.loc[has, "on_pitch_gd"], gd[has], rtol=1e-9, atol=1e-12)
    to_csv(ps, M5_DIR / "inputs" / "player_season.csv.gz", exact=True)
    to_csv(imp, M5_DIR / "inputs" / "impact_season.csv.gz", exact=True)
    write_json(M5_DIR / "inputs" / "stabilisation.json", stabilisation())
    q = ps.groupby(["player_id", "season"])["minutes"].sum()
    in_pool = ps["player_id"].isin(pool_ids())
    print(f"player_season: {len(ps)} rows, {ps['player_id'].nunique()} players; "
          f"counts {counts}", flush=True)  # fmt: skip
    record(M5_DIR, {
        "n_player_season_rows": (len(ps), "M5 player_season rows (player, season, scope in "
                                 "league/ucl/intl), original data"),
        "n_player_seasons_900": (int((q >= 900).sum()), "Player-seasons with >= 900 minutes "
                                 "over league, ucl and intl"),
        "n_players": (int(ps["player_id"].nunique()), "Players in M5's population"),
        "n_pool_rows": (int(in_pool.sum()), "Rows of pool players"),
        **{f"count_{k}": (v, f"Input count: {k}") for k, v in counts.items()},
        "gk_box_k": (GK_BOX_K, "Stabilisation constant (minutes) used for goalkeeper goals, "
                     "np goals and assists: M1 has none, so full shrinkage (post hoc)"),
    })  # fmt: skip


# --- M5 -------------------------------------------------------------------------------


def m5_scores(result: Multiverse, ids: np.ndarray) -> np.ndarray:
    r = result.ranks
    if not (r["player_id"].to_numpy()[: ids.size] == ids).all():
        raise ValueError("unexpected order of M5's ranks")
    return r["score"].to_numpy(np.float64).reshape(-1, ids.size)


def m5_chunk(reps: list[int]) -> list[tuple[int, float]]:
    ids = pool_ids()

    def inputs() -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        for b in reps:
            yield replicate_inputs(b)

    times = []
    t0 = time.perf_counter()
    runs = multiverse_runs(inputs(), stabilisation(), list(ids), rng("multiverse"))  # type: ignore[arg-type]
    for b, result in zip(reps, runs, strict=True):
        np.save(REP_DIR / f"b{b:03d}.npy", m5_scores(result, ids).astype(np.float32))
        t1 = time.perf_counter()
        times.append((b, t1 - t0))
        t0 = t1
    return times


def stage_multiverse() -> None:
    ids = pool_ids()
    ps, imp, _ = point_inputs()
    t0 = time.perf_counter()
    runs = multiverse_runs([(ps, imp)], stabilisation(), list(ids), rng("multiverse"))  # type: ignore[arg-type]
    point = next(runs)
    t_point = time.perf_counter() - t0
    print(f"point M5 run: {t_point:.1f} s", flush=True)
    ranks = point.ranks
    grid = spec_grid()
    to_csv(grid, M5_DIR / "spec_grid.csv")
    to_csv(ranks[["spec_id", "player_id", "segment", "score", "rank"]],
           M5_DIR / "point_ranks.csv.gz", exact=True)  # fmt: skip
    np.save(M5_DIR / "point_scores.npy", m5_scores(point, ids))
    to_csv(point.weights, M5_DIR / "weights.csv.gz", exact=True)
    write_json(M5_DIR / "fallbacks.json", point.fallbacks)
    probs = rank_probabilities(ranks)
    probs.insert(1, "player", probs["player_id"].map(names()))
    to_csv(probs.sort_values(["segment", "p_first"], ascending=[True, False]),
           M5_DIR / "rank_probabilities.csv")  # fmt: skip
    indices = variance_indices(ranks)
    to_csv(indices, M5_DIR / "variance_indices.csv")
    curve = specification_curve(ranks)
    to_csv(curve[["player_id", "curve_position", "spec_id", "rank"]],
           M5_DIR / "specification_curve.csv.gz")  # fmt: skip
    distinct = distinct_rankings(ranks)
    out = probs[probs["segment"] == "outfield"].set_index("player_id")
    top = int(out["p_first"].idxmax())

    # replicates (A13 iii), resumable: one file per replicate, stacked at the end
    REP_DIR.mkdir(parents=True, exist_ok=True)
    todo = [b for b in range(N_BOOT) if not (REP_DIR / f"b{b:03d}.npy").exists()]
    chunks = [todo[i::N_M5_WORKERS] for i in range(N_M5_WORKERS) if todo[i::N_M5_WORKERS]]
    t0 = time.perf_counter()
    times: list[tuple[int, float]] = []
    if chunks:
        with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
            for part in ex.map(m5_chunk, chunks):
                times.extend(part)
    wall = time.perf_counter() - t0
    stacked = np.lib.format.open_memmap(
        M5_DIR / "replicate_scores.npy", mode="w+", dtype=np.float32,
        shape=(N_BOOT, len(grid), ids.size),
    )  # fmt: skip
    for b in range(N_BOOT):
        stacked[b] = np.load(REP_DIR / f"b{b:03d}.npy")
    stacked.flush()
    del stacked
    shutil.rmtree(REP_DIR)
    old = read_json(M5_DIR / "replicate_times.json").get("times", {})
    old.update({str(b): s for b, s in times})
    write_json(M5_DIR / "replicate_times.json", {"times": old})
    per_run = np.array(list(old.values()), dtype=np.float64)
    print(f"replicates: {len(times)} runs in {wall:.0f} s wall; "
          f"median {np.median(per_run):.1f} s per run", flush=True)  # fmt: skip

    record(M5_DIR, {
        "NDistinctRankingsOutfield": (distinct.get("outfield"), "Distinct outfield pool "
                                      "rankings among the 5,760 specifications"),
        "NDistinctRankingsGoalkeeper": (distinct.get("goalkeeper"), "Distinct goalkeeper "
                                        "pool rankings"),
        "MultiverseTopName": (names()[top], "Outfield pool player ranked #1 by the largest "
                              "share of specifications, original data"),
        "MultiverseTopPFirst": (out.loc[top, "p_first"], "That share of specifications"),
        "NPlayersEverFirst": (int((out["p_first"] > 0).sum()), "Outfield pool players "
                              "ranked #1 by at least one specification"),
        "n_specs": (len(grid), "Specifications in the full factorial"),
        "n_replicates": (N_BOOT, "Match-bootstrap replicates rerun through M5 (A13 iii)"),
        "seconds_point_run": (t_point, "Wall time of the M5 run on the original data"),
        "seconds_per_replicate_median": (float(np.median(per_run)), "Median wall time of one "
                                         "replicate (inputs + M5) inside a worker"),
        "seconds_replicates_wall": (wall, f"Wall time of this call's replicates "
                                    f"({len(times)} runs, {len(chunks)} workers)"),
        **{f"fallback_{k}": (v, f"M5 fallback count: {k}") for k, v in point.fallbacks.items()},
    })  # fmt: skip


# --- M8 -------------------------------------------------------------------------------


def point_ranks() -> pd.DataFrame:
    ranks = pd.read_csv(M5_DIR / "point_ranks.csv.gz")
    grid = spec_grid()
    ranks = ranks.merge(grid, on="spec_id", how="left")
    ranks["segment"] = pd.Categorical(ranks["segment"], categories=["outfield", "goalkeeper"])
    return ranks


def replicate_frames() -> Iterator[pd.DataFrame]:
    arr = np.load(M5_DIR / "replicate_scores.npy", mmap_mode="r")
    ids = pool_ids()
    spec_ids = spec_grid()["spec_id"].to_numpy(np.int64)
    spec_col, player_col = np.repeat(spec_ids, ids.size), np.tile(ids, spec_ids.size)
    for b in range(arr.shape[0]):
        yield pd.DataFrame(
            {"spec_id": spec_col, "player_id": player_col,
             "score": np.asarray(arr[b], dtype=np.float64).ravel()}
        )  # fmt: skip


def groups() -> pd.Series:
    keys = panel().keys
    return keys.drop_duplicates("player_id").set_index("player_id")["position_group"]


def m4_shares() -> dict[str, float]:
    s = pd.read_csv(GAP_DIR / "shares.csv")
    s = s[(s["component"] == "total") & (s["estimator"] == "cross") & (s["mode"] == "player")]
    s = s[~s["group"].str.contains("+", regex=False)]  # per group, not the H3 pools
    return {str(g): float(v) for g, v in zip(s["group"], s["share_clipped"], strict=True)}


def bdo_points() -> pd.DataFrame:
    b = pd.read_csv(PAPER / "data" / "ballon_dor.csv")
    b["player_id"] = b["player"].map(pool().set_index("player")["player_id"])
    return b


def run_synthesis(share: dict[str, float], out_dir: Path, task: str, label: str) -> None:
    ranks = point_ranks()
    weights = pd.read_csv(M5_DIR / "weights.csv.gz")
    omega = stats_weight_share(weights)
    t0 = time.perf_counter()
    result = synthesise(ranks, replicate_frames(), share, groups(), rng(task), omega=omega)
    seconds = time.perf_counter() - t0
    print(f"synthesise: {seconds:.0f} s", flush=True)
    indices = variance_indices(ranks)
    verdict = verdicts(result, indices)
    print(verdict.to_string(), flush=True)

    players = result.players.copy()
    players.insert(1, "player", players["player_id"].map(names()))
    players.insert(3, "position_group", players["player_id"].map(groups()))
    for seg, p in result.pairwise.items():
        mean = 1.0 + np.nansum(1.0 - p.to_numpy(), axis=1)
        in_seg = players["segment"] == seg
        players.loc[in_seg, "mean_rank"] = players.loc[in_seg, "player_id"].map(
            pd.Series(mean, index=p.index)
        )
        to_csv(p.reset_index(), out_dir / f"pairwise_{seg}.csv")
    sets = result.rank_sets.copy()
    sets.insert(1, "player", sets["player_id"].map(names()))
    players = players.sort_values(["segment", "p_first"], ascending=[False, False])
    to_csv(players, out_dir / "players.csv")
    to_csv(sets, out_dir / "rank_sets.csv")
    to_csv(verdict, out_dir / "verdicts.csv")

    # Kendall's W (A13 v): BdO points summed over 2021-2026, 0 if unranked
    b = bdo_points()
    pts = b[b["edition"].between(2021, 2026)].groupby("player_id")["points"].sum()
    recog = pts.reindex(pool_ids(), fill_value=0.0).astype(np.float64)
    w = view_agreement(result.players, recog)

    # exploratory: V1's order against M8, V1's outfield players
    v1 = pd.read_csv(PAPER / "data" / "v1_ranking.csv")
    v1["player_id"] = v1["player"].map(pool().set_index("player")["player_id"])
    of = players[players["segment"] == "outfield"].set_index("player_id")
    v1 = v1[v1["player_id"].isin(of.index)]
    rho_median = float(spearmanr(v1["rank"], of.loc[v1["player_id"], "median_rank"])[0])
    rho_mean = float(spearmanr(v1["rank"], of.loc[v1["player_id"], "mean_rank"])[0])

    view = of.sort_values("p_first", ascending=False)
    leader, second = view.index[0], view.index[1]
    top_set = sets[(sets["segment"] == "outfield")].set_index("player_id")
    pair = result.pairwise["outfield"].to_numpy()
    upper = pair[np.triu_indices(len(pair), 1)]
    decided = int(((upper >= DECIDED) | (upper <= 1 - DECIDED)).sum())
    top10 = top_set.index[top_set["point_rank"] <= 10]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (of.loc[top10, "var_specification"] / of.loc[top10, "var_sampling"]).to_numpy()
    n_h4 = int((~np.isnan(ratio)).sum())
    gk = players[players["segment"] == "goalkeeper"].sort_values("p_first", ascending=False)
    spec = spec_grid().set_index("spec_id").loc[result.fixed_spec]
    spec_label = ", ".join(LEVEL_LABELS[d][spec[d]] for d in DIMENSIONS)
    total = indices[indices["player_id"] == leader].set_index("dimension")["total_order"]
    v = verdict.set_index("hypothesis")

    def says(h: str) -> str:
        return "supported" if bool(v.loc[h, "supported"]) else "refuted"

    shown = ["player", "p_first", "p_top3", "median_rank", "rank_lo", "rank_hi",
             "share_specification", "share_sampling", "share_unmeasured"]  # fmt: skip
    print(f"top 10 by P(#1):\n{view.head(10)[shown].round(3).to_string()}")
    print(f"decided {decided}/{upper.size}; W = {w:.3f}; V1 rho median {rho_median:.3f}, "
          f"mean {rho_mean:.3f}", flush=True)  # fmt: skip
    entries: dict[str, tuple[Any, str]] = {
        "TopPlayerName": (names()[leader], "Outfield pool player with the highest P(#1)"),
        "TopPlayerPFirst": (of.loc[leader, "p_first"], "That player's P(#1); H2's statistic"),
        "TopPlayerPTopThree": (of.loc[leader, "p_top3"], "That player's P(top-3)"),
        "TopPlayerPTopTen": (of.loc[leader, "p_top10"], "That player's P(top-10)"),
        "TopPlayerMedianRank": (of.loc[leader, "median_rank"], "That player's median rank"),
        "TopPlayerRankLo": (of.loc[leader, "rank_lo"], "Lower end, 90% rank interval"),
        "TopPlayerRankHi": (of.loc[leader, "rank_hi"], "Upper end, 90% rank interval"),
        "TopPlayerMarginalLo": (top_set.loc[leader, "marginal_lo"], "Lower end, marginal 90% "
                                "rank confidence set at the median specification"),
        "TopPlayerMarginalHi": (top_set.loc[leader, "marginal_hi"], "Upper end of the same"),
        "TopPlayerVarSpec": (of.loc[leader, "share_specification"], "Specification share of "
                             "the top player's rank variance"),
        "TopPlayerVarSamp": (of.loc[leader, "share_sampling"], "Sampling share"),
        "TopPlayerVarUnm": (of.loc[leader, "share_unmeasured"], "Unmeasured share"),
        "SecondPlayerName": (names()[second], "Outfield player with the second-highest P(#1)"),
        "SecondPlayerPFirst": (of.loc[second, "p_first"], "That player's P(#1)"),
        "NPairsOutfield": (int(upper.size), "Unordered outfield pool pairs"),
        "NPairsDecided": (decided, "Pairs with P >= 0.9 or <= 0.1"),
        "SharePairsDecided": (v.loc["H1", "statistic"], "H1 statistic: share decided"),
        "HFourRatio": (v.loc["H4", "statistic"], "H4: mean Var_spec / Var_samp, median "
                       "specification's top 10"),
        "HFourNPlayers": (n_h4, "Players behind H4's mean"),
        "MedianSpecLabel": (spec_label, f"Median specification (spec_id {result.fixed_spec})"),
        "median_spec_id": (result.fixed_spec, "spec_id of the median specification"),
        "KendallW": (w, "Kendall's W of stats, impact and recognition views (A13 v)"),
        "KendallMeanSpearman": ((3 * w - 1) / 2, "Mean pairwise Spearman of the three views"),
        "TopGoalkeeperName": (gk["player"].iloc[0], "Goalkeeper with the highest P(#1)"),
        "TopGoalkeeperPFirst": (gk["p_first"].iloc[0], "That goalkeeper's P(#1)"),
        "VOneSpearman": (rho_median, "Exploratory: Spearman of V1's rank and M8's median rank, "
                         f"{len(v1)} V1 outfield players"),
        "v1_spearman_mean_rank": (rho_mean, "Same with M8's mean rank"),
        "VerdictHOne": (says("H1"), "H1 verdict"),
        "VerdictHTwo": (says("H2"), "H2 verdict"),
        "VerdictHFour": (says("H4"), "H4 verdict"),
        "VerdictHFive": (says("H5"), "H5 verdict"),
        "h5_largest_dimension": (DIM_NAMES[str(total.idxmax())], "H5: dimension with the "
                                 "largest total-order index for the P(#1) leader"),
        "n_replicates": (result.n_replicates, "Replicates in the mixture"),
        "n_unmeasured": (result.n_unmeasured, "U draws per (s, b)"),
        "seconds_synthesise": (seconds, "Wall time of synthesis.synthesise"),
        **{f"sigma_{g}": (s, "Unmeasured share used for group " + g) for g, s in share.items()},
    }  # fmt: skip
    record(out_dir, entries)
    if out_dir == M8_DIR:
        record(M5_DIR, {
            "HFiveAggTotal": (total.get("aggregation"), "H5: total-order index of aggregation "
                              "for the rank of the M8 P(#1) leader"),
            "HFiveMaxDim": (DIM_NAMES[str(total.idxmax())], "H5: dimension with the largest "
                            "total-order index for that player"),
            "HFiveMaxTotal": (total.max(), "H5: that largest total-order index"),
        })  # fmt: skip
    print(f"{label}: H1 {says('H1')}, H2 {says('H2')}, H4 {says('H4')}, H5 {says('H5')}")


def stage_synthesis() -> None:
    run_synthesis(m4_shares(), M8_DIR, "synthesis", "M8")


def stage_synthesis_bias() -> None:
    path = GAP_DIR / "assortative" / "summary.json"
    if not path.exists():
        print(f"synthesis_bias: {path} does not exist yet; stage skipped (run it after "
              "run_impact.py --stage h3_assortative).")  # fmt: skip
        return
    summary = read_json(path)
    share = {g: float(summary[f"share_bias_corrected_{g}"]["value"]) for g in m4_shares()
             if f"share_bias_corrected_{g}" in summary}  # fmt: skip
    missing = sorted(set(groups().loc[pool_ids()]) - set(share))
    if missing:
        raise ValueError(f"no bias-corrected share for groups {missing}")
    M8_BIAS_DIR.mkdir(parents=True, exist_ok=True)
    run_synthesis(share, M8_BIAS_DIR, "synthesis_bias", "M8, bias-corrected shares (post hoc)")


# --- M6 -------------------------------------------------------------------------------


def shootout_winner(date: str, home: str, away: str) -> str | None:
    """Winner of a drawn final's shoot-out from the StatsBomb events, if StatsBomb has it."""
    m = pd.read_csv(PROCESSED / "sb_matches.csv.gz")
    hit = m[(m["date"] == date) & (m["stage"] == "Final")]
    if len(hit) != 1:
        return None
    with gzip.open(RAW / "statsbomb" / "events" / f"{hit['sb_match_id'].iloc[0]}.json.gz") as f:
        events = json.loads(f.read())
    goals: dict[str, int] = {}
    for e in events:
        if e.get("period") == 5 and e["type"]["name"] == "Shot":
            team = e["team"]["name"]
            goals[team] = goals.get(team, 0) + int(e["shot"]["outcome"]["name"] == "Goal")
    if set(goals) != {home, away} or goals[home] == goals[away]:
        return None
    return home if goals[home] > goals[away] else away


def finals(games: pd.DataFrame) -> pd.DataFrame:
    """UCL and national-team finals: competition_id, tm season, winner and loser club ids."""
    f = games[
        (games["round"] == "Final")
        & ((games["competition_id"] == "CL")
           | (games["competition_type"] == "national_team_competition"))
    ].copy()  # fmt: skip
    rows = []
    for g in f.itertuples():
        if g.home_goals != g.away_goals:
            home_won: bool | None = g.home_goals > g.away_goals
            how = "score"
        else:
            name = shootout_winner(g.date, g.home_club_name, g.away_club_name)
            home_won = None if name is None else name == g.home_club_name
            how = "StatsBomb shoot-out" if name else "undecided"
        win, lose = (g.home_club_id, g.away_club_id)
        if not home_won:
            win, lose = lose, win
        rows.append({
            "game_id": g.game_id, "competition_id": g.competition_id, "tm_season": g.season,
            "date": g.date, "home": g.home_club_name, "away": g.away_club_name,
            "score": f"{g.home_goals}-{g.away_goals}", "decided_by": how,
            "winner_id": win if home_won is not None else np.nan,
            "loser_id": lose if home_won is not None else np.nan,
            "winner": (g.home_club_name if home_won else g.away_club_name)
            if home_won is not None else None,
        })  # fmt: skip
    return pd.DataFrame(rows)


def league_champions(games: pd.DataFrame) -> pd.DataFrame:
    """First in the computed table (3/1/0, then goal difference, then goals) per
    domestic-league competition-season; complete flag as the module doc says."""
    lg = games[games["competition_type"] == "domestic_league"]
    home = lg.assign(club=lg["home_club_id"], name=lg["home_club_name"],
                     gf=lg["home_goals"], ga=lg["away_goals"])  # fmt: skip
    away = lg.assign(club=lg["away_club_id"], name=lg["away_club_name"],
                     gf=lg["away_goals"], ga=lg["home_goals"])  # fmt: skip
    t = pd.concat([home, away])
    t["pts"] = np.select([t["gf"] > t["ga"], t["gf"] == t["ga"]], [3, 1], 0)
    table = t.groupby(["competition_id", "season", "club", "name"], as_index=False).agg(
        pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"), played=("pts", "size")
    )
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(["competition_id", "season", "pts", "gd", "gf"],
                              ascending=[True, True, False, False, False])  # fmt: skip
    first = table.groupby(["competition_id", "season"]).head(2)
    rows = []
    for (comp, season), part in first.groupby(["competition_id", "season"]):
        a = part.iloc[0]
        tie = len(part) > 1 and (part.iloc[1][["pts", "gd", "gf"]] == a[["pts", "gd", "gf"]]).all()
        rows.append({"competition_id": comp, "tm_season": season, "champion_id": a["club"],
                     "champion": a["name"], "pts": a["pts"], "gd": a["gd"],
                     "runner_up": part.iloc[1]["name"], "runner_up_pts": part.iloc[1]["pts"],
                     "tied_on_all_criteria": bool(tie)})  # fmt: skip
    champ = pd.DataFrame(rows)
    span = lg.groupby(["competition_id", "season"])["date"].agg(["min", "max"])
    ref = span.xs("2024/25", level="season")
    calendar = set(ref.index[ref["min"].str[:4] == ref["max"].str[:4]])
    unfinished = champ["competition_id"].isin(calendar) & (champ["tm_season"] == "2025/26")
    champ["complete"] = ~unfinished & (champ["tm_season"] != "2019/20")
    return champ


def recognition_frame() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(PROCESSED / "tm_games.csv.gz")
    cols = ["game_id", "player_id", "season", "competition_id", "scope", "club_id", "minutes"]
    pm = pd.read_csv(PROCESSED / "panel_player_match.csv.gz", usecols=cols)
    pm = pm[pm["minutes"] > 0]
    champ = league_champions(games)
    fin = finals(games)
    # player-season indicators
    lgc = champ[champ["complete"]][["competition_id", "tm_season", "champion_id"]]
    league = pm[pm["scope"] == "league"].merge(
        lgc, left_on=["competition_id", "season", "club_id"],
        right_on=["competition_id", "tm_season", "champion_id"],
    )  # fmt: skip
    title = set(zip(league["player_id"], league["season"], strict=True))
    fin_season = pm.drop_duplicates("game_id").set_index("game_id")["season"]
    fin["season"] = fin["game_id"].map(fin_season)
    fin["season"] = fin["season"].fillna(fin["tm_season"])  # a final without panel rows

    def played_for(rows: pd.DataFrame, club: str) -> set[tuple[int, str]]:
        hit = pm.merge(rows[["competition_id", "season", club]].dropna(),
                       left_on=["competition_id", "season", "club_id"],
                       right_on=["competition_id", "season", club])  # fmt: skip
        return set(zip(hit["player_id"], hit["season"], strict=True))

    ucl = fin[fin["competition_id"] == "CL"]
    intl = fin[fin["competition_id"] != "CL"]
    won, lost = played_for(ucl, "winner_id"), played_for(ucl, "loser_id")
    nat = played_for(intl, "winner_id")

    ps = pd.read_csv(PROCESSED / "panel_player_season.csv.gz", low_memory=False,
                     usecols=["player_id", "season", "scope", "minutes", "goals", "assists",
                              "goals_p90", "assists_p90", "np_goals_p90"])  # fmt: skip
    ps = ps[ps["scope"] == "all"].set_index(["player_id", "season"])
    rapm = pd.read_csv(RAPM_DIR / "rapm_season.csv").set_index(["player_id", "season"])
    val = pd.read_csv(PROCESSED / "tm_valuations.csv.gz").sort_values("date")
    b = bdo_points()
    outfield = pool()[~pool()["is_goalkeeper"]]
    counts: dict[str, Any] = {}
    rows = []
    for e, season in EDITION_SEASON.items():
        ed = b[b["edition"] == e].set_index("player_id")
        prev = b[b["edition"] == PREVIOUS_EDITION[e]].groupby("player")["points"].sum()
        cut = f"{e}-06-30"
        last_val = val[val["date"] <= cut].groupby("player_id")["market_value_eur"].last()
        ranked_out = 0
        for pid, name in zip(outfield["player_id"], outfield["player"], strict=True):
            key = (pid, season)
            if key not in ps.index or ps.loc[key, "minutes"] <= 0:
                ranked_out += int(pid in ed.index)
                continue
            s = ps.loc[key]
            rows.append({
                "edition": e, "player": pid, "name": name, "season": season,
                "points": float(ed.loc[pid, "points"]) if pid in ed.index else 0.0,
                "rank": float(ed.loc[pid, "rank"]) if pid in ed.index else np.nan,
                "goals": s["goals"], "assists": s["assists"], "minutes": s["minutes"],
                "np_goals_p90": s["np_goals_p90"], "goals_p90": s["goals_p90"],
                "assists_p90": s["assists_p90"],
                "rapm_offence": rapm["offence"].get(key, np.nan),
                "rapm_defence": rapm["defence"].get(key, np.nan),
                "league_title": float(key in title), "ucl_winner": float(key in won),
                "ucl_finalist": float(key in lost), "intl_winner": float(key in nat),
                "log_prev_points": math.log1p(float(prev.get(name, 0.0))),
                "market_value": last_val.get(pid, np.nan),
            })  # fmt: skip
        counts[f"ranked_outfield_without_minutes_{e}"] = ranked_out
    frame = pd.DataFrame(rows)
    counts["n_candidate_editions_before_drops"] = len(frame)
    no_rapm = frame["rapm_offence"].isna()
    no_value = frame["market_value"].isna() | (frame["market_value"] <= 0)
    counts["NBdoDropped"] = int(no_rapm.sum())
    counts["dropped_no_market_value"] = int((no_value & ~no_rapm).sum())
    counts["dropped_ranked"] = int(frame.loc[no_rapm | no_value, "rank"].notna().sum())
    frame = frame[~no_rapm & ~no_value].reset_index(drop=True)
    frame["log_market_value"] = np.log(frame["market_value"].astype(np.float64))
    return frame, counts, champ, fin


def stage_recognition() -> None:
    frame, counts, champ, fin = recognition_frame()
    to_csv(champ, M6_DIR / "league_champions.csv")
    to_csv(fin, M6_DIR / "finals.csv")
    to_csv(frame, M6_DIR / "universe.csv")
    print(fin.to_string(), flush=True)
    print(frame.groupby("edition")[["league_title", "ucl_winner", "ucl_finalist",
                                     "intl_winner"]].sum().to_string())  # fmt: skip
    t0 = time.perf_counter()
    fit = fit_recognition(frame, BLOCKS)
    alt = fit_recognition(frame, BLOCKS_PER90)
    boot = bootstrap_shares(frame, BLOCKS, rng("recognition/bootstrap"), n_boot=N_BOOT)
    seconds = time.perf_counter() - t0
    to_csv(boot, M6_DIR / "bootstrap_shares.csv")
    res = fit.residuals.copy()
    res.insert(2, "name", res["player"].map(names()))
    res = res.sort_values("ols_residual", ascending=False)
    to_csv(res, M6_DIR / "residuals.csv")
    q = [(1 - LEVEL) / 2, (1 + LEVEL) / 2]
    ci = {(m, blk): boot.loc[boot["model"] == m, blk].quantile(q).to_numpy()
          for m in ("ols", "pl") for blk in BLOCKS}  # fmt: skip
    ols_boot = boot[boot["model"] == "ols"]
    above = float((ols_boot["team"] > ols_boot["stats"]).mean())
    h6 = fit.ols_shares["team"] > fit.ols_shares["stats"]
    pl_same = (fit.pl_shares["team"] > fit.pl_shares["stats"]) == h6
    table = pd.DataFrame(
        {
            "ols": fit.ols_shares,
            "pl": fit.pl_shares,
            "ols_per90_reading": alt.ols_shares,
            "pl_per90_reading": alt.pl_shares,
        }
    )
    table["ols_lo"] = [ci[("ols", k)][0] for k in table.index]
    table["ols_hi"] = [ci[("ols", k)][1] for k in table.index]
    table["pl_lo"] = [ci[("pl", k)][0] for k in table.index]
    table["pl_hi"] = [ci[("pl", k)][1] for k in table.index]
    to_csv(table.rename_axis("block").reset_index(), M6_DIR / "shares.csv")
    betas = pd.DataFrame({"ols": fit.ols_beta, "pl": fit.pl_beta})
    to_csv(betas.rename_axis("feature").reset_index(), M6_DIR / "coefficients.csv")
    print(table.round(4).to_string())
    print(
        f"OLS R2 {fit.ols_r2:.3f}, LOEO {fit.ols_cv_r2:.3f}, PL {fit.pl_pseudo_r2:.3f}; "
        f"H6 {'supported' if h6 else 'refuted'}; PL {'agrees' if pl_same else 'disagrees'}"
    )
    shown = ["edition", "name", "points", "rank", "ols_residual", "pl_residual"]
    print(f"most over-rewarded (OLS):\n{res.head(10)[shown].to_string()}")
    print(f"most under-rewarded (OLS):\n{res.tail(10)[shown].to_string()}")
    entries: dict[str, tuple[Any, str]] = {
        "NBdoRows": (len(frame), "(edition, candidate) rows after drops (A14)"),
        "NBdoCandidates": (int(frame["player"].nunique()), "Distinct players in the universe"),
        "NBdoDropped": (counts["NBdoDropped"], "Candidate-editions dropped: no per-season RAPM"),
        "OlsRTwo": (fit.ols_r2, "Within-edition R2, full OLS model"),
        "OlsRTwoLoeo": (fit.ols_cv_r2, "Leave-one-edition-out R2"),
        "PlPseudoRTwo": (fit.pl_pseudo_r2, "McFadden pseudo-R2, full Plackett-Luce model"),
        "ShapleyStatsOls": (fit.ols_shares["stats"], "OLS Shapley share, statistics"),
        "ShapleyImpactOls": (fit.ols_shares["impact"], "OLS Shapley share, impact"),
        "ShapleyTeamOls": (fit.ols_shares["team"], "OLS Shapley share, team success"),
        "ShapleyRepOls": (fit.ols_shares["reputation"], "OLS Shapley share, reputation"),
        "ShapleyTeamOlsLo": (ci[("ols", "team")][0], "90% player-cluster bootstrap, OLS team"),
        "ShapleyTeamOlsHi": (ci[("ols", "team")][1], "Upper bound of the same"),
        "ShapleyStatsOlsLo": (ci[("ols", "stats")][0], "90% bootstrap, OLS statistics"),
        "ShapleyStatsOlsHi": (ci[("ols", "stats")][1], "Upper bound of the same"),
        "ShapleyTeamAboveStatsBoot": (above, "Share of replicates with OLS team > statistics"),
        "ShapleyStatsPl": (fit.pl_shares["stats"], "PL Shapley share, statistics"),
        "ShapleyImpactPl": (fit.pl_shares["impact"], "PL Shapley share, impact"),
        "ShapleyTeamPl": (fit.pl_shares["team"], "PL Shapley share, team success"),
        "ShapleyRepPl": (fit.pl_shares["reputation"], "PL Shapley share, reputation"),
        "OlsPlAgreement": ("agree" if pl_same else "disagree", "Whether PL orders team and "
                           "statistics as OLS does (A1)"),
        "VerdictHSix": ("supported" if h6 else "refuted", "H6 by the OLS shares (A1)"),
        "pl_converged": (fit.pl_converged, "Every PL fit behind the shares converged"),
        "pl_boot_converged_share": (float(boot.loc[boot["model"] == "pl", "converged"].mean()),
                                    "Share of bootstrap replicates whose PL fits converged"),
        "seconds_fit_and_bootstrap": (seconds, "Wall time, fits and B = 200 bootstrap"),
        **{f"count_{k}": (v, f"Universe count: {k}") for k, v in counts.items()},
        **{f"per90_reading_ols_{k}": (v, "OLS share, alternative reading of the statistics "
           "block (per-90 goals, assists, np goals, plus minutes; post hoc)")
           for k, v in alt.ols_shares.items()},
        **{f"per90_reading_pl_{k}": (v, "PL share, same alternative reading")
           for k, v in alt.pl_shares.items()},
        "per90_reading_ols_r2": (alt.ols_r2, "OLS R2, alternative reading"),
    }  # fmt: skip
    for (m, blk), (lo, hi) in ci.items():
        entries[f"{m}_{blk}_lo"] = (lo, f"90% bootstrap lower bound, {m} {blk}")
        entries[f"{m}_{blk}_hi"] = (hi, f"90% bootstrap upper bound, {m} {blk}")
    record(M6_DIR, entries)


# --- M9 -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    stints: pd.DataFrame
    stint_players: pd.DataFrame
    positions: pd.Series


@functools.cache
def schedule() -> Schedule:
    """M3's primary schedule (A12): every stint, group = competition_id + study season."""

    def read(name: str, **kw: Any) -> pd.DataFrame:
        return pd.read_csv(PROCESSED / f"{name}.csv.gz", low_memory=False, **kw)

    stints = pd.concat([read("tm_stints"), read("sb_stints")], ignore_index=True)
    sp = pd.concat([read("tm_stint_players"), read("sb_stint_players")], ignore_index=True)
    sp["side"] = sp["side"].astype("category")
    pm = read("panel_player_match", usecols=["game_id", "season", "competition_id"])
    games = pm.drop_duplicates("game_id").set_index("game_id").astype(str)
    if not stints["game_id"].isin(games.index).all():
        raise ValueError("stints of games without panel rows")
    stints = stints.join(games, on="game_id")
    stints["group"] = stints["competition_id"] + " " + stints["season"]
    ids = pd.Index(np.sort(sp["player_id"].unique()))
    pos = read("player_positions").set_index("player_id")["position_group"]
    positions = pos.reindex(ids).astype(object).where(lambda s: s.notna(), UNKNOWN)
    return Schedule(stints, sp, positions)


def a9_params() -> SimParams:
    p = read_json(GAP_DIR / "h3_params.json")
    return SimParams(**{f.name: p[f.name] for f in fields(SimParams)})


def recovery_chunk(task: tuple[np.random.SeedSequence, int, list[int]]) -> pd.DataFrame:
    s, n_sims, ids = task
    sc = schedule()
    lam = float(read_json(RAPM_DIR / "cv.json")["lam"])
    out = rank_recovery(
        sc.stints, sc.stint_players, sc.positions, ids, a9_params(), np.random.default_rng(s),
        lam=lam, min_minutes=MIN_MINUTES, n_sims=n_sims, n_bins=1, group="group",
    )  # fmt: skip
    return out.sims


def stage_recovery() -> None:
    sc = schedule()
    sp = sc.stint_players.merge(sc.stints[["stint_id", "minutes"]], on="stint_id")
    played = sp.groupby("player_id")["minutes"].sum()
    outfield = pool()[~pool()["is_goalkeeper"]]["player_id"]
    ok = outfield[outfield.map(played).fillna(0) >= MIN_MINUTES]
    dropped = sorted(set(outfield) - set(ok))
    print(f"recovery pool: {len(ok)} outfield players; dropped {dropped}", flush=True)
    del sc, sp
    schedule.cache_clear()  # the workers load their own copy
    sizes = [N_RECOVERY_SIMS // N_SIM_WORKERS + (i < N_RECOVERY_SIMS % N_SIM_WORKERS)
             for i in range(N_SIM_WORKERS)]  # fmt: skip
    seeds = seed("recovery").spawn(N_SIM_WORKERS)
    tasks = [(s, n, [int(i) for i in ok]) for s, n in zip(seeds, sizes, strict=True)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_SIM_WORKERS) as ex:
        sims = pd.concat(list(ex.map(recovery_chunk, tasks)), ignore_index=True)
    seconds = time.perf_counter() - t0
    bins = pd.qcut(sims["gap"].rank(method="first"), RECOVERY_BINS, labels=False)
    by_gap = sims.groupby(bins).agg(
        n=("gap", "size"), gap_low=("gap", "min"), gap_high=("gap", "max"),
        gap_mean=("gap", "mean"), p_top_rapm=("top_rapm", "mean"), p_top_box=("top_box", "mean"),
        mae_rapm=("mae_rapm", "mean"), mae_box=("mae_box", "mean"),
    )  # fmt: skip
    by_gap.index.name = "gap_bin"
    to_csv(sims, M9_DIR / "sims.csv")
    to_csv(by_gap.reset_index(), M9_DIR / "by_gap.csv")
    print(by_gap.round(3).to_string())
    print(f"recovery: {seconds:.0f} s", flush=True)
    entries: dict[str, tuple[Any, str]] = {
        "NRecoverySims": (len(sims), "Simulated leagues in the rank-recovery study"),
        "RecoveryRapmFirst": (sims["top_rapm"].mean(), "Share in which pooled RAPM total ranks "
                              "the planted #1 of the pool first"),
        "RecoveryBoxFirst": (sims["top_box"].mean(), "Same for the equal-z within-group box "
                             "composite"),
        "RecoveryRapmFirstSmallGap": (by_gap["p_top_rapm"].iloc[0], "RAPM's recovery share, "
                                      "smallest-gap bin"),
        "RecoveryRapmFirstLargeGap": (by_gap["p_top_rapm"].iloc[-1], "RAPM's recovery share, "
                                      "largest-gap bin"),
        "RecoveryRankErrorRapm": (sims["mae_rapm"].mean(), "Mean absolute rank error of RAPM "
                                  "total within the pool"),
        "RecoveryRankErrorBox": (sims["mae_box"].mean(), "Same for the box composite"),
        "recovery_box_first_small_gap": (by_gap["p_top_box"].iloc[0], "Box recovery share, "
                                         "smallest-gap bin"),
        "recovery_box_first_large_gap": (by_gap["p_top_box"].iloc[-1], "Box recovery share, "
                                         "largest-gap bin"),
        "n_pool_outfield": (len(ok), "Outfield pool players in the recovery pool"),
        "n_pool_dropped": (len(dropped), "Outfield pool players dropped: < 900 stint minutes"),
        "recovery_chance_level": (1 / len(ok), "Chance of ranking the planted #1 first at "
                                  "random"),
        "seconds_recovery": (seconds, f"Wall time, {N_SIM_WORKERS} worker processes"),
    }  # fmt: skip
    record(M9_DIR, entries)


# --- main -----------------------------------------------------------------------------

STAGES: dict[str, tuple[Callable[[], None], Path]] = {
    "inputs": (stage_inputs, M5_DIR),
    "multiverse": (stage_multiverse, M5_DIR),
    "synthesis": (stage_synthesis, M8_DIR),
    "synthesis_bias": (stage_synthesis_bias, M8_DIR),
    "recognition": (stage_recognition, M6_DIR),
    "recovery": (stage_recovery, M9_DIR),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--stage", choices=list(STAGES))
    which.add_argument("--from", dest="start", choices=list(STAGES))
    args = parser.parse_args()
    names_ = list(STAGES)
    if args.stage:
        names_ = [args.stage]
    elif args.start:
        names_ = names_[names_.index(args.start) :]
    for name in names_:
        run, folder = STAGES[name]
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        run()
        seconds = time.perf_counter() - t0
        timings = read_json(folder / "timings.json")
        timings[name] = {"seconds": seconds, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(folder / "timings.json", timings)
        print(f"=== {name}: {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
