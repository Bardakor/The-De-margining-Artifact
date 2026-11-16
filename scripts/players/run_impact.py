"""M3 (RAPM) and M4 (unmeasured share) on the real data, and the H3 decision.

Design §5 (M3, M4, M9); pre-registration A7, A9-A13. Stages, in order:

- design: stints of the panel (Transfermarkt, plus StatsBomb for WC 2022 and Euros
  2020/2024), group = competition_id + season (the panel's study season, joined by
  game_id), players with >= 900 minutes get their own columns (A12). Checks only.
- cv: lambda by 5-fold game-grouped CV (A12).
- fit: pooled RAPM at that lambda; home term and intercept per competition-season.
- season: per-season RAPM shrunk toward the pooled fit; lambda_season by 5-fold
  game-grouped CV (each fold's prior is the pooled fit on its training games).
- bootstrap: B = 200 game bootstrap of the pooled fit, and per-season replicates
  with the same game counts and the replicate's pooled fit as prior (A13 iii).
- halves: 20 random game splits at lambda; split-half reliability per group.
- attribution: attribution factors (full and half data) per group and H3 pool.
- m4: M4's unmeasured share per group and component, cross (primary, A10) and
  within estimators, player bootstrap and the A7 joint bootstrap; H3 contrast.
- h3: the A9 parametric bootstrap on the real schedule, calibrated as A13 (iv).
- robustness: the with-Elo fit (A12) against the primary.
- h3_assortative (post hoc, exploratory): the A9 bootstrap with the real pooled RAPM
  as planted effects (rescaled to the calibrated SDs), so the planted quality keeps the
  real who-plays-with-whom structure; outputs in results/players/construct_gap/assortative/.

Every stage reads what it needs from the files of earlier stages, so it can rerun
alone. Outputs: results/players/rapm/ and results/players/construct_gap/, each with a
summary.json of headline numbers ({"value", "description"}); stage timings in
results/players/rapm/timings.json.

Bootstrap files (results/players/rapm/), estimates as float32, counts as uint8:
bootstrap_pooled.npz (game_ids, counts[B, games], players, offence[B, players],
defence[B, players]; total = offence + defence), bootstrap_season.npz (player_id,
season, minutes, offence, defence; one column per player-season), bootstrap_halves.npz
(the two halves of each replicate's games, as halves.npz).

Seeds: master 20261114 (A12); each task draws from
SeedSequence(20261114, spawn_key=(crc32(task name),)), so a stage's numbers do not
depend on which stages ran before it.

Usage:
    uv run python scripts/players/run_impact.py               # every stage
    uv run python scripts/players/run_impact.py --stage m4    # one stage
    uv run python scripts/players/run_impact.py --from halves # that stage and later
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import time
import zlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from footy.players.construct_gap import Splits, position_contrast, split_stats, unmeasured_share
from footy.players.paths import PAPER, PROCESSED, RESULTS
from footy.players.rapm import (
    COMPONENTS,
    REPLACEMENT,
    Design,
    Halves,
    RapmFit,
    attribution_factor,
    bootstrap_rapm,
    build_design,
    cv_lambda,
    fit_rapm,
    fit_rapm_by_season,
    split_half_reliability,
    split_halves,
)
from footy.players.simulation import (
    STATS,
    SimParams,
    parametric_bootstrap_contrast,
    simulate_on_schedule,
)

MASTER_SEED = 20261114
MIN_MINUTES = 900.0
N_FOLDS = 5
LAMS = [float(x) for x in 10 ** np.arange(2.5, 5.51, 0.25)]
LAMS_SEASON = [float(x) for x in 10 ** np.arange(3.0, 7.01, 0.5)]
MAX_EXTENSIONS = 6  # grid steps added at an edge optimum before giving up
N_BOOT = 200
N_SPLITS = 20  # real-data game splits (rapm.split_halves default)
REL_HALF_MINUTES = 450.0  # split-half reliability: minutes needed in each half
N_ATTRIBUTION_SIMS = 5  # rapm.attribution_factor default
M4_STATS = ("np_goals", "penalty_goals", "assists", "yellow", "red")
POOLS = {"CB": "CB+DM", "DM": "CB+DM", "CF": "CF+W", "W": "CF+W"}
GROUPS = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "CF")
UNKNOWN = "unknown"  # the few stint players without any position group
ELO_SCALE = 100.0  # covariate in units of 100 Elo points
# H3 (A9, A13 iv)
N_SIMS = 200
N_REPEATS = 5
CALIB_ROUNDS = 5  # evaluations; the SDs are updated after each but the last
CALIB_SIMS = 2
SD_START = 0.1
CONVERSION = 0.11
SHOT_SD_FLOOR = 0.01
N_WORKERS = 3  # parallel chunks of the parametric bootstrap, ~3 GB each
LEVEL = 0.9

RAPM_DIR = RESULTS / "rapm"
GAP_DIR = RESULTS / "construct_gap"
ASSORT_DIR = GAP_DIR / "assortative"


# --- data -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Data:
    stints: pd.DataFrame  # + competition_id, competition_type, season, group, elo_diff
    stint_players: pd.DataFrame
    player_match: pd.DataFrame  # panel rows of games with stints, M4's columns
    positions: pd.Series  # player_id -> position group, every stint player


@functools.cache
def data() -> Data:
    def read(name: str, **kw: Any) -> pd.DataFrame:
        return pd.read_csv(PROCESSED / f"{name}.csv.gz", low_memory=False, **kw)

    stints = pd.concat([read("tm_stints"), read("sb_stints")], ignore_index=True)
    stint_players = pd.concat(
        [read("tm_stint_players"), read("sb_stint_players")], ignore_index=True
    )
    stint_players["side"] = stint_players["side"].astype("category")  # memory, 6M rows
    columns = ["game_id", "player_id", "season", "competition_id", "competition_type"]
    pm = read("panel_player_match", usecols=[*columns, "minutes", *M4_STATS])
    games = pm.drop_duplicates("game_id").set_index("game_id")[columns[2:]].astype(str)
    if not stints["game_id"].isin(games.index).all():
        raise ValueError("stints of games without panel rows")
    stints = stints.join(games, on="game_id")
    stints["group"] = stints["competition_id"] + " " + stints["season"]
    elo = read("team_elo").set_index("game_id")
    diff = stints["game_id"].map(elo["home_elo_pre"]) - stints["game_id"].map(elo["away_elo_pre"])
    stints["elo_diff"] = diff / ELO_SCALE
    pm = pm[pm["game_id"].isin(stints["game_id"].unique())]
    pm = pm[["game_id", "player_id", "minutes", *M4_STATS]].reset_index(drop=True)
    ids = pd.Index(np.sort(stint_players["player_id"].unique()))
    groups = read("player_positions").set_index("player_id")["position_group"]
    positions = groups.reindex(ids).astype(object).where(lambda s: s.notna(), UNKNOWN)
    return Data(stints, stint_players, pm, positions)


@functools.cache
def design(covariate: str | None = None) -> Design:
    d = data()
    return build_design(
        d.stints, d.stint_players, group="group", covariate=covariate, min_minutes=MIN_MINUTES
    )


def season_of_stints() -> pd.Series:
    return data().stints.set_index("stint_id")["season"]


def rng(task: str) -> np.random.Generator:
    return np.random.default_rng(seed(task))


def seed(task: str) -> np.random.SeedSequence:
    return np.random.SeedSequence(MASTER_SEED, spawn_key=(zlib.crc32(task.encode()),))


# --- files ----------------------------------------------------------------------------


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


def to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, float_format="%.6g")


def lam() -> float:
    return float(read_json(RAPM_DIR / "cv.json")["lam"])


def lam_season() -> float:
    return float(read_json(RAPM_DIR / "cv_season.json")["lam_season"])


def pooled_fit() -> pd.DataFrame:
    return pd.read_csv(RAPM_DIR / "fit_pooled.csv").set_index("player_id")


def save_halves(path: Path, h: Halves, dtype: type = np.float64) -> None:
    arrays = {f"{s}_{c}": getattr(h, s)[c].to_numpy(dtype) for s in "ab" for c in COMPONENTS[:2]}
    np.savez_compressed(
        path,
        players=h.minutes_a.columns.to_numpy(np.int64),
        game_ids=h.games_a.columns.to_numpy(np.int64),
        minutes_a=h.minutes_a.to_numpy(dtype),
        minutes_b=h.minutes_b.to_numpy(dtype),
        games_a=h.games_a.to_numpy(np.uint8),
        games_b=h.games_b.to_numpy(np.uint8),
        **arrays,
    )


def load_halves(path: Path) -> Halves:
    z = np.load(path)
    players, games = pd.Index(z["players"]), pd.Index(z["game_ids"])

    def side(s: str) -> dict[str, pd.DataFrame]:
        off, dfn = (z[f"{s}_{c}"].astype(np.float64) for c in COMPONENTS[:2])
        return {
            c: pd.DataFrame(v, columns=players)
            for c, v in zip(COMPONENTS, (off, dfn, off + dfn), strict=True)
        }

    def frame(key: str, columns: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(z[key].astype(np.float64), columns=columns)

    return Halves(
        a=side("a"),
        b=side("b"),
        minutes_a=frame("minutes_a", players),
        minutes_b=frame("minutes_b", players),
        games_a=frame("games_a", games),
        games_b=frame("games_b", games),
    )


def pooled_replicates() -> dict[str, pd.DataFrame]:
    z = np.load(RAPM_DIR / "bootstrap_pooled.npz")
    off, dfn = z["offence"].astype(np.float64), z["defence"].astype(np.float64)
    cols = pd.Index(z["players"])
    return {
        c: pd.DataFrame(v, columns=cols)
        for c, v in zip(COMPONENTS, (off, dfn, off + dfn), strict=True)
    }


def attribution_table() -> pd.DataFrame:
    return pd.read_csv(RAPM_DIR / "attribution.csv")


def attribution_map(kind: str, halves: bool, comp: str) -> dict[str, float]:
    a = attribution_table()
    a = a[(a["kind"] == kind) & (a["halves"] == halves)]
    return {str(g): float(v) for g, v in zip(a["group"], a[comp], strict=True)}


# --- stages ---------------------------------------------------------------------------


def stage_design() -> None:
    dd, d = data(), design()
    own = d.players[d.players != REPLACEMENT]
    minutes = pd.Series(d.player_minutes(), index=d.players)
    share_repl = float(minutes.get(REPLACEMENT, 0.0) / minutes.sum())
    st = dd.stints
    by_comp = (
        st.groupby("competition_id")
        .agg(
            competition_type=("competition_type", "first"),
            n_stints=("stint_id", "size"),
            n_games=("game_id", "nunique"),
            n_seasons=("season", "nunique"),
            minutes=("minutes", "sum"),
        )
        .sort_values("n_stints", ascending=False)
    )
    to_csv(by_comp, RAPM_DIR / "stints_by_competition.csv", index=True)
    own_by_group = dd.positions.reindex(own).value_counts().reindex([*GROUPS, UNKNOWN]).fillna(0)
    n_players = int(dd.stint_players["player_id"].nunique())
    missing_elo = int(st["elo_diff"].isna().sum())
    unknown = dd.positions[dd.positions == UNKNOWN].index
    on = dd.stint_players.merge(st[["stint_id", "minutes"]], on="stint_id")
    played = on.groupby("player_id")["minutes"].sum()
    print(f"stints {len(st):,} in {st['game_id'].nunique():,} games, {by_comp.shape[0]} "
          f"competitions, {len(d.groups)} competition-seasons", flush=True)  # fmt: skip
    print(f"players {n_players:,}; own columns (>= {MIN_MINUTES:.0f} min) {len(own):,}; "
          f"replacement share of player-minutes {share_repl:.4f}")  # fmt: skip
    print("own columns by group:", own_by_group.astype(int).to_dict())
    print(
        f"players without position group: {len(unknown)} "
        f"(minutes {played.reindex(unknown).to_dict()}, "
        f"own columns: {int(unknown.isin(own).sum())}); stints without Elo: {missing_elo}"
    )
    print(by_comp.head(15).to_string())
    entries: dict[str, tuple[Any, str]] = {
        "n_stints": (len(st), "Stints in the RAPM design (Transfermarkt + StatsBomb tournaments)"),
        "n_games": (int(st["game_id"].nunique()), "Games with stints"),
        "n_competitions": (by_comp.shape[0], "Competitions with stints"),
        "n_groups": (len(d.groups), "Competition-seasons (intercept and home term each)"),
        "n_players": (n_players, "Players on the pitch in at least one stint"),
        "n_players_own": (len(own), "Players with own RAPM columns (>= 900 window minutes)"),
        "minutes_share_replacement": (
            share_repl,
            "Share of player-minutes played by replacement-column players",
        ),
        "n_players_no_position": (len(unknown), "Stint players without a position group"),
        "n_stints_missing_elo": (missing_elo, "Stints without an Elo difference"),
    }
    for g, n in own_by_group.items():
        entries[f"n_players_own_{g}"] = (int(n), f"Players with own columns, position group {g}")
    record(RAPM_DIR, entries)


def stage_cv() -> None:
    d = design()
    grid = list(LAMS)
    for _ in range(MAX_EXTENSIONS + 1):
        best, curve = cv_lambda(d, grid, rng("cv"), n_folds=N_FOLDS)  # same folds every time
        step = grid[1] / grid[0]
        if best == grid[0]:
            grid.insert(0, grid[0] / step)
        elif best == grid[-1]:
            grid.append(grid[-1] * step)
        else:
            break
    to_csv(curve.reset_index(), RAPM_DIR / "cv_curve.csv")
    write_json(RAPM_DIR / "cv.json", {"lam": best, "grid": grid, "n_folds": N_FOLDS})
    print(curve.to_string(), f"\nlambda = {best:.6g}")
    record(RAPM_DIR, {
        "lambda": (best, "Ridge penalty of pooled RAPM, 5-fold game-grouped CV (A12)"),
        "lambda_grid_low": (grid[0], "Smallest lambda in the CV grid"),
        "lambda_grid_high": (grid[-1], "Largest lambda in the CV grid"),
    })  # fmt: skip


def stage_fit() -> None:
    dd, d = data(), design()
    fit = fit_rapm(d, lam())
    players = fit.players.copy()
    players.insert(1, "position_group", dd.positions.reindex(players.index).fillna("replacement"))
    to_csv(players.rename_axis("player_id").reset_index(), RAPM_DIR / "fit_pooled.csv")

    st = dd.stints
    raw = st.groupby("group").agg(
        competition_id=("competition_id", "first"),
        competition_type=("competition_type", "first"),
        season=("season", "first"),
        n_games=("game_id", "nunique"),
        n_stints=("stint_id", "size"),
        minutes=("minutes", "sum"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
    )
    terms = raw.assign(
        home=fit.home.reindex(raw.index),
        intercept=fit.intercepts.reindex(raw.index),
        raw_home_minus_away_per90=(raw["home_goals"] - raw["away_goals"]) * 90 / raw["minutes"],
    )
    terms["base_rate"] = terms["intercept"] + terms["home"] / 2
    to_csv(terms.rename_axis("group").reset_index(), RAPM_DIR / "group_terms.csv")

    league = terms[terms["competition_type"] == "domestic_league"]
    by_season = league.groupby("season").apply(
        lambda t: pd.Series({
            "home_weighted": np.average(t["home"], weights=t["minutes"]),
            "raw_weighted": np.average(t["raw_home_minus_away_per90"], weights=t["minutes"]),
            "n_groups": len(t),
            "minutes": t["minutes"].sum(),
        }),
        include_groups=False,
    )  # fmt: skip
    top5 = terms[terms["competition_id"].isin(["GB1", "ES1", "L1", "IT1", "FR1"])]
    top5_table = top5.pivot(index="season", columns="competition_id", values="home")
    to_csv(by_season.reset_index(), RAPM_DIR / "home_by_season.csv")
    to_csv(top5_table.reset_index(), RAPM_DIR / "home_top5.csv")
    empty = league["season"].isin(["2019/20", "2020/21"])

    def wmean(t: pd.DataFrame) -> float:
        return float(np.average(t["home"], weights=t["minutes"]))

    q = np.quantile(terms["home"], [0.05, 0.25, 0.5, 0.75, 0.95])
    wq = weighted_quantiles(terms["home"].to_numpy(), terms["minutes"].to_numpy(), [0.1, 0.5, 0.9])
    print("home term, all groups: quantiles 5/25/50/75/95%", np.round(q, 3))
    print("home term, minutes-weighted 10/50/90%", np.round(wq, 3))
    print("domestic leagues, minutes-weighted home term by season:\n", by_season.round(3))
    print("top-5 leagues home term:\n", top5_table.round(3))
    own = players.drop(index=REPLACEMENT, errors="ignore")
    print("own players RAPM SD:", own[list(COMPONENTS)].std().round(4).to_dict())
    record(RAPM_DIR, {
        "home_median": (q[2], "Median home term over competition-seasons (goals per 90)"),
        "home_weighted_mean": (wmean(terms), "Minutes-weighted mean home term, all groups"),
        "home_weighted_p10": (wq[0], "Minutes-weighted 10th percentile of the home term"),
        "home_weighted_p90": (wq[2], "Minutes-weighted 90th percentile of the home term"),
        "home_league_empty": (
            wmean(league[empty]),
            "Minutes-weighted home term, domestic leagues, 2019/20-2020/21 (empty stadiums)",
        ),
        "home_league_other": (
            wmean(league[~empty]), "Minutes-weighted home term, domestic leagues, 2021/22-2025/26"
        ),
        "sd_offence_own": (own["offence"].std(), "SD of pooled RAPM offence, own-column players"),
        "sd_defence_own": (own["defence"].std(), "SD of pooled RAPM defence, own-column players"),
        "sd_total_own": (own["total"].std(), "SD of pooled RAPM total, own-column players"),
        "replacement_total": (
            float(players.loc[REPLACEMENT, "total"]), "RAPM total of the replacement column"
        ),
    })  # fmt: skip


def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs: list[float]) -> list[float]:
    order = np.argsort(x)
    cum = np.cumsum(w[order]) / w.sum()
    return [float(x[order][np.searchsorted(cum, q)]) for q in qs]


def season_player_part(d: Design, est: pd.DataFrame, row_season: np.ndarray) -> np.ndarray:
    """sum(offence on the attacking side) - sum(defence of the defenders) per row,
    each player at his estimate for the row's season."""
    p = d.n_players
    out = np.zeros(d.x.shape[0])
    for s in np.unique(row_season):
        rows = np.flatnonzero(row_season == s)
        e = est.xs(s, level="season")[["offence", "defence"]].reindex(d.players).fillna(0.0)
        out[rows] = (
            d.x[rows, :p] @ e["offence"].to_numpy() + d.x[rows, p : 2 * p] @ e["defence"].to_numpy()
        )
    return out


def with_unpenalised(d: Design, part: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Add the home and intercept terms that go with ``part``: their weighted least
    squares on the training rows (the normal equations the joint ridge solution
    satisfies for these unpenalised columns)."""
    xh = d.x[:, 2 * d.n_players :]
    w = d.w * train
    a = (xh.T @ sp.diags(w) @ xh).toarray()
    h = np.linalg.lstsq(a, xh.T @ (w * (d.y - part)), rcond=None)[0]
    return np.asarray(part + xh @ h)


def stage_season() -> None:
    d, lam_ = design(), lam()
    season = season_of_stints()
    row_season = np.tile(season.reindex(d.stint_ids).to_numpy().astype(str), 2)
    game_fold = rng("season/cv").permutation(len(d.game_ids)) % N_FOLDS
    fold = game_fold[d.game]
    priors: dict[int, RapmFit] = {}
    err: dict[float, float] = {}
    pooled_only = 0.0
    for k in range(N_FOLDS):
        train = (fold != k).astype(np.float64)
        priors[k] = fit_rapm(replace(d, w=d.w * train), lam_)
        part = d.x[:, : 2 * d.n_players] @ np.concatenate(
            [priors[k].players[c].reindex(d.players).to_numpy() for c in COMPONENTS[:2]]
        )
        r = d.y - with_unpenalised(d, part, train)
        pooled_only += float(d.w[train == 0] @ r[train == 0] ** 2)

    def evaluate(lams: list[float]) -> None:
        for ls in lams:
            total = 0.0
            for k in range(N_FOLDS):
                train = (fold != k).astype(np.float64)
                game_w = (game_fold != k).astype(np.float64)
                est = fit_rapm_by_season(d, priors[k], season, ls, weights=game_w)
                pred = with_unpenalised(d, season_player_part(d, est, row_season), train)
                r = d.y - pred
                total += float(d.w[train == 0] @ r[train == 0] ** 2)
            err[ls] = total
            print(f"  lambda_season {ls:.4g}: cv_mse {total / d.w.sum():.6f}", flush=True)

    grid = list(LAMS_SEASON)
    evaluate(grid)
    for _ in range(MAX_EXTENSIONS):
        best = min(grid, key=lambda x: err[x])
        step = grid[1] / grid[0]
        if best == grid[0]:
            grid.insert(0, grid[0] / step)
            evaluate(grid[:1])
        elif best == grid[-1]:
            grid.append(grid[-1] * step)
            evaluate(grid[-1:])
        else:
            break
    best = min(grid, key=lambda x: err[x])
    curve = pd.DataFrame({"lam_season": grid, "cv_mse": [err[x] / d.w.sum() for x in grid]})
    to_csv(curve, RAPM_DIR / "cv_season_curve.csv")
    write_json(RAPM_DIR / "cv_season.json", {
        "lam_season": best, "grid": grid, "pooled_only_cv_mse": pooled_only / d.w.sum(),
        "n_folds": N_FOLDS,
    })  # fmt: skip
    print(curve.to_string(), f"\npooled only: {pooled_only / d.w.sum():.6f}; "
          f"lambda_season = {best:.6g}")  # fmt: skip

    by_season = fit_rapm_by_season(d, fit_rapm(d, lam_), season, best)
    out = by_season.reset_index()
    out.insert(2, "position_group", data().positions.reindex(out["player_id"]).to_numpy())
    out["position_group"] = out["position_group"].fillna("replacement")
    to_csv(out, RAPM_DIR / "rapm_season.csv")
    own = out[out["player_id"] != REPLACEMENT]
    record(RAPM_DIR, {
        "lambda_season": (best, "Penalty on per-season deviations from pooled RAPM, 5-fold "
                          "game-grouped CV (each fold's prior refitted on its training games)"),
        "cv_mse_season_best": (err[best] / d.w.sum(), "CV MSE at lambda_season"),
        "cv_mse_pooled_only": (pooled_only / d.w.sum(),
                               "CV MSE of the pooled fit alone (no season deviations)"),
        "n_player_seasons": (len(own), "Player-seasons with a per-season RAPM estimate"),
    })  # fmt: skip


def stage_bootstrap() -> None:
    d, lam_, ls = design(), lam(), lam_season()
    t0 = time.perf_counter()
    boot = bootstrap_rapm(d, lam_, rng("bootstrap"), n_boot=N_BOOT)
    print(f"pooled bootstrap: {time.perf_counter() - t0:.0f} s", flush=True)
    if boot.counts.max() > np.iinfo(np.uint8).max:
        raise ValueError("a game drawn more than 255 times")
    np.savez_compressed(
        RAPM_DIR / "bootstrap_pooled.npz",
        game_ids=d.game_ids.to_numpy(np.int64),
        counts=boot.counts.astype(np.uint8),
        players=d.players.to_numpy(np.int64),
        offence=boot.replicates["offence"].to_numpy(np.float32),
        defence=boot.replicates["defence"].to_numpy(np.float32),
        lam=lam_,
    )
    save_halves(RAPM_DIR / "bootstrap_halves.npz", boot.halves, np.float32)

    season = season_of_stints()
    keys: pd.MultiIndex | None = None
    arrays: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for b in range(N_BOOT):
        prior = RapmFit(
            players=pd.DataFrame({c: boot.replicates[c].iloc[b] for c in COMPONENTS[:2]}),
            home=pd.Series(dtype=float), intercepts=pd.Series(dtype=float), covariate=None,
            lam=lam_,
        )  # fmt: skip
        est = fit_rapm_by_season(d, prior, season, ls, weights=boot.counts[b])
        if keys is None:
            keys = pd.MultiIndex.from_frame(est.index.to_frame(index=False))
            arrays = {
                c: np.empty((N_BOOT, len(est)), np.float32) for c in ("minutes", *COMPONENTS[:2])
            }
        elif not est.index.equals(keys):
            raise ValueError("per-season keys differ between replicates")
        for c in arrays:
            arrays[c][b] = est[c].to_numpy(np.float32)
        if (b + 1) % 20 == 0:
            print(f"  season replicate {b + 1}/{N_BOOT} ({time.perf_counter() - t0:.0f} s)",
                  flush=True)  # fmt: skip
    assert keys is not None
    np.savez_compressed(
        RAPM_DIR / "bootstrap_season.npz",
        player_id=keys.get_level_values("player_id").to_numpy(np.int64),
        season=keys.get_level_values("season").to_numpy().astype(str),
        lam_season=ls,
        **arrays,
    )

    fit = pooled_fit()
    se = boot.se.add_prefix("se_")
    out = fit.join(se)
    to_csv(out.reset_index(), RAPM_DIR / "rapm_pooled.csv")
    write_pool(out, boot.replicates)
    own = out.drop(index=REPLACEMENT)
    record(RAPM_DIR, {
        "n_boot": (N_BOOT, "Game bootstrap replicates (A12)"),
        "se_total_median": (own["se_total"].median(), "Median bootstrap SE of RAPM total, "
                            "own-column players"),
        "se_offence_median": (own["se_offence"].median(), "Median bootstrap SE of RAPM offence"),
        "se_defence_median": (own["se_defence"].median(), "Median bootstrap SE of RAPM defence"),
    })  # fmt: skip


def write_pool(rapm: pd.DataFrame, reps: dict[str, pd.DataFrame]) -> None:
    pool = pd.read_csv(PAPER / "data" / "pool.csv")
    ids = pool["tm_player_id"].astype(np.int64)
    own = ids.isin(rapm.index.drop(REPLACEMENT))
    out = pool[["player", "tm_player_id", "is_goalkeeper"]].copy()
    out["position_group"] = data().positions.reindex(ids).to_numpy()
    out["own_columns"] = own.to_numpy()
    cols = ["minutes", *COMPONENTS, *(f"se_{c}" for c in COMPONENTS)]
    out[cols] = rapm.reindex(ids)[cols].to_numpy()
    lo, hi = (1 - LEVEL) / 2, (1 + LEVEL) / 2
    for c in COMPONENTS:
        r = reps[c].reindex(columns=ids)
        out[f"{c}_ci_low"] = r.quantile(lo).to_numpy()
        out[f"{c}_ci_high"] = r.quantile(hi).to_numpy()
    out["total_excludes_zero"] = (out["total_ci_low"] > 0) | (out["total_ci_high"] < 0)
    outfield = (~out["is_goalkeeper"]) & out["own_columns"]
    out["rank_outfield"] = out["total"].where(outfield).rank(ascending=False, method="min")
    out = out.sort_values(["is_goalkeeper", "rank_outfield"])
    to_csv(out, RAPM_DIR / "pool_rapm.csv")
    show = out[outfield.reindex(out.index)].head(20)
    pd.set_option("display.width", 200)
    print("top 20 pool outfield players by RAPM total:")
    cols = ["rank_outfield", "player", "position_group", "minutes", "offence", "defence",
            "total", "se_total", "total_ci_low", "total_ci_high"]  # fmt: skip
    print(show[cols].round(3).to_string(index=False))
    ex_all = float(out.loc[out["own_columns"], "total_excludes_zero"].mean())
    ex_out = float(out.loc[outfield, "total_excludes_zero"].mean())
    print(
        f"pool players whose 90% interval of total excludes 0: {ex_all:.3f} (outfield {ex_out:.3f})"
    )
    top = show.iloc[0]
    record(RAPM_DIR, {
        "pool_n": (len(out), "Pool players"),
        "pool_n_own": (int(out["own_columns"].sum()), "Pool players with own RAPM columns"),
        "pool_excludes_zero": (ex_all, "Share of pool players whose 90% bootstrap interval of "
                               "RAPM total excludes 0"),
        "pool_excludes_zero_outfield": (ex_out, "Same, outfield pool players"),
        "pool_top_player": (str(top["player"]), "Pool outfield player with the highest RAPM total"),
        "pool_top_total": (float(top["total"]), "His RAPM total (goals per 90)"),
        "pool_top_se": (float(top["se_total"]), "Its bootstrap SE"),
    })  # fmt: skip


def stage_halves() -> None:
    d = design()
    h = split_halves(d, lam(), rng("halves"), n_repeats=N_SPLITS)
    save_halves(RAPM_DIR / "halves.npz", h)
    rel = split_half_reliability(h, min_minutes=REL_HALF_MINUTES, groups=data().positions)
    rel_all = split_half_reliability(h, min_minutes=REL_HALF_MINUTES)
    rel = pd.concat([rel, rel_all.rename(index={"all": "ALL"})])
    to_csv(rel.reset_index(), RAPM_DIR / "reliability.csv")
    print(rel.round(3).to_string())
    entries: dict[str, tuple[Any, str]] = {
        "n_splits": (N_SPLITS, "Random game splits for split-half reliability"),
        "rel_half_minutes": (REL_HALF_MINUTES, "Minutes a player needs in each half to enter "
                             "split-half reliability"),
    }  # fmt: skip
    for g, row in rel.iterrows():
        for c in COMPONENTS:
            entries[f"rel_{c}_{g}"] = (row[c], f"Split-half reliability (Spearman-Brown) of "
                                       f"RAPM {c}, group {g}")  # fmt: skip
    record(RAPM_DIR, entries)


def stage_attribution() -> None:
    d, lam_ = design(), lam()
    pos = data().positions
    frames = []
    every = pd.Series("ALL", index=pos.index)
    for kind, groups in (("groups", pos), ("pools", pos.map(POOLS)), ("all", every)):
        for halves in (False, True):
            a = attribution_factor(
                d, lam_, rng(f"attribution/{kind}/{halves}"), min_minutes=MIN_MINUTES,
                n_sims=N_ATTRIBUTION_SIMS, groups=groups, halves=halves,
            )  # fmt: skip
            frames.append(a.reset_index().assign(kind=kind, halves=halves))
            print(f"{kind} halves={halves}:\n{a.round(3).to_string()}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    to_csv(out, RAPM_DIR / "attribution.csv")
    entries: dict[str, tuple[Any, str]] = {}
    for _, row in out.iterrows():
        which = "half-data" if row["halves"] else "full-data"
        sfx = "half" if row["halves"] else "full"
        for c in COMPONENTS:
            entries[f"attribution_{sfx}_{c}_{row['group']}"] = (
                row[c],
                f"Attribution factor ({which}) of RAPM {c}, group {row['group']}",
            )
    record(RAPM_DIR, entries)


def m4_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """M4's players: own RAPM columns, a position group, >= 900 panel minutes in the
    games with stints. Returns (pooled RAPM, per-90 statistics, groups, player rows)."""
    dd = data()
    fit = pooled_fit()
    pm = dd.player_match
    sums = pm.groupby("player_id")[["minutes", *M4_STATS]].sum()
    own = fit.index[fit.index != REPLACEMENT]
    ok = sums.index[sums["minutes"] >= MIN_MINUTES]
    ids = own[dd.positions.reindex(own).ne(UNKNOWN).to_numpy()].intersection(ok)
    stats = sums.loc[ids, list(M4_STATS)].div(sums.loc[ids, "minutes"], axis=0) * 90
    groups = dd.positions.loc[ids].astype(str)
    rows = sums.loc[ids].join(stats.add_suffix("_p90")).join(fit[list(COMPONENTS)])
    rows.insert(0, "position_group", groups)
    return fit, stats, groups, rows


def stage_m4() -> None:
    fit, stats, groups, rows = m4_inputs()
    to_csv(rows.reset_index(), GAP_DIR / "m4_players.csv")
    pm = data().player_match
    pm = pm[pm["player_id"].isin(stats.index)]
    halves = load_halves(RAPM_DIR / "halves.npz")
    t0 = time.perf_counter()
    split = split_stats(pm, halves, M4_STATS)
    boot_halves = load_halves(RAPM_DIR / "bootstrap_halves.npz")
    split_boot = split_stats(pm, boot_halves, M4_STATS)
    reps = pooled_replicates()
    print(f"split statistics: {time.perf_counter() - t0:.0f} s; M4 players {len(stats)}:",
          groups.value_counts().to_dict(), flush=True)  # fmt: skip

    pools = groups.map(POOLS).dropna()
    frames = []
    for comp in COMPONENTS:
        base = Splits(halves.a[comp], halves.b[comp], split)
        joint = (reps[comp], Splits(boot_halves.a[comp], boot_halves.b[comp], split_boot))
        for est in ("cross", "within"):
            for kind, labels in (("groups", groups), ("pools", pools)):
                att = attribution_map(kind, est == "cross", comp)
                for mode, replicates in (("player", None), ("a7", joint)):
                    out = unmeasured_share(
                        fit[comp], stats, base, att, labels,
                        rng(f"m4/share/{comp}/{est}/{kind}/{mode}"), estimator=est,
                        n_boot=N_BOOT, replicates=replicates, level=LEVEL,
                    )  # fmt: skip
                    frames.append(out.rename_axis("group").reset_index().assign(
                        component=comp, estimator=est, mode=mode))  # fmt: skip
        print(f"  shares {comp} done ({time.perf_counter() - t0:.0f} s)", flush=True)
    shares = pd.concat(frames, ignore_index=True)
    lead = ["component", "estimator", "mode", "group"]
    shares = shares[lead + [c for c in shares.columns if c not in lead]]
    to_csv(shares, GAP_DIR / "shares.csv")

    base = Splits(halves.a["total"], halves.b["total"], split)
    joint = (reps["total"], Splits(boot_halves.a["total"], boot_halves.b["total"], split_boot))
    contrasts = []
    for est in ("cross", "within"):
        att = attribution_map("pools", est == "cross", "total")
        for mode, replicates in (("player", None), ("a7", joint)):
            c = position_contrast(
                fit["total"], stats, base, att, groups, rng(f"m4/contrast/{est}/{mode}"),
                estimator=est, n_boot=N_BOOT, replicates=replicates, level=LEVEL,
            )  # fmt: skip
            contrasts.append(c.to_dict() | {"estimator": est, "mode": mode})
    contrast = pd.DataFrame(contrasts)
    contrast = contrast[["estimator", "mode", *contrast.columns.drop(["estimator", "mode"])]]
    to_csv(contrast, GAP_DIR / "contrast.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    view = shares[(shares["mode"] == "player")]
    cols = ["n", "r2_obs", "rel_rapm", "rel_pred", "share", "share_clipped", "ci_low", "ci_high"]
    for est in ("cross", "within"):
        print(f"\n{est} estimator, player bootstrap:")
        print(view[view["estimator"] == est].set_index(["component", "group"])[cols]
              .round(3).to_string())  # fmt: skip
    print("\nA7 joint bootstrap intervals (cross):")
    a7 = shares[(shares["mode"] == "a7") & (shares["estimator"] == "cross")]
    print(a7.set_index(["component", "group"])[["n", "share", "ci_low", "ci_high"]].round(3)
          .to_string())  # fmt: skip
    print("\nH3 contrast (total):\n", contrast.round(3).to_string())
    record_m4(shares, contrast, len(stats))


def record_m4(shares: pd.DataFrame, contrast: pd.DataFrame, n: int) -> None:
    entries: dict[str, tuple[Any, str]] = {
        "m4_n_players": (n, "Players in M4 (own RAPM columns, >= 900 minutes, known group)"),
        "m4_stats": (", ".join(M4_STATS), "Per-90 statistics in M4 (A11)"),
        "m4_n_boot": (N_BOOT, "Player bootstrap draws in M4"),
    }
    gk = " GK flagged: these statistics say nothing about goalkeeping."
    for _, r in shares.iterrows():
        key = f"m4_{r['estimator']}_{r['component']}_{r['group']}"
        what = (f"M4 {r['estimator']} estimator, RAPM {r['component']}, group {r['group']}"
                + (gk if r["group"] == "GK" else ""))  # fmt: skip
        if r["mode"] == "player":
            entries[f"{key}_n"] = (r["n"], f"Players: {what}")
            for col in ("share", "share_clipped", "share_naive", "r2_obs", "r2_true",
                        "rel_rapm", "rel_pred", "ci_low", "ci_high", "ci_low_clipped",
                        "ci_high_clipped"):  # fmt: skip
                label = col.replace("_clipped", " (clipped to [0, 1])")
                entries[f"{key}_{col}"] = (r[col], f"{label}, 90% player bootstrap: {what}")
        else:
            for col in ("ci_low", "ci_high", "ci_low_clipped", "ci_high_clipped"):
                entries[f"{key}_a7_{col}"] = (r[col], f"{col}, A7 joint bootstrap: {what}")
    for _, r in contrast.iterrows():
        key = f"h3_contrast_{r['estimator']}_{r['mode']}"
        what = (f"H3 contrast CB+DM minus CF+W (total), {r['estimator']} estimator, "
                f"{'player' if r['mode'] == 'player' else 'A7 joint'} bootstrap")  # fmt: skip
        for col in ("diff", "ci_low", "ci_high", "diff_clipped", "ci_low_clipped",
                    "ci_high_clipped"):  # fmt: skip
            entries[f"{key}_{col}"] = (r[col], f"{col}: {what}")
    record(GAP_DIR, entries)


def split_cov(h: Halves, labels: pd.Series) -> pd.DataFrame:
    """Mean over splits of the covariance between the halves' estimates, per group
    (rows) and component (offence, defence)."""
    players = h.minutes_a.columns
    lab = labels.reindex(players).to_numpy()
    out = {}
    for c in COMPONENTS[:2]:
        a, b = h.a[c].to_numpy(), h.b[c].to_numpy()
        col = {}
        for g in GROUPS:
            idx = np.flatnonzero(lab == g)
            ac, bc = a[:, idx], b[:, idx]
            ac = ac - ac.mean(axis=1, keepdims=True)
            bc = bc - bc.mean(axis=1, keepdims=True)
            col[g] = float(((ac * bc).sum(axis=1) / (len(idx) - 1)).mean())
        out[c] = col
    return pd.DataFrame(out)


def h3_params(sd: pd.DataFrame, fixed: dict[str, Any]) -> SimParams:
    off = {g: float(sd.loc[g, "offence"]) for g in GROUPS} | {UNKNOWN: 0.0}
    dfn = {g: float(sd.loc[g, "defence"]) for g in GROUPS} | {UNKNOWN: 0.0}
    return SimParams(offence_sd=off, defence_sd=dfn, **fixed)


def simulate_cov(
    params: SimParams, lam_: float, labels: pd.Series, g: np.random.Generator
) -> pd.DataFrame:
    dd = data()
    sim = simulate_on_schedule(dd.stints, dd.stint_players, dd.positions, g, params, group="group")
    d = build_design(sim.stints, dd.stint_players, group="group", min_minutes=MIN_MINUTES)
    return split_cov(split_halves(d, lam_, g, n_repeats=N_REPEATS), labels)


def h3_chunk(
    task: tuple[np.random.SeedSequence, int, SimParams, pd.Series, dict[str, float], float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s, n_sims, params, real, attribution, lam_ = task
    dd = data()
    out = parametric_bootstrap_contrast(
        dd.stints, dd.stint_players, dd.positions, real, attribution, params,
        np.random.default_rng(s), lam=lam_, min_minutes=MIN_MINUTES, n_sims=n_sims,
        n_repeats=N_REPEATS, group="group", estimator="cross", n_boot=N_BOOT, level=LEVEL,
    )  # fmt: skip
    return out.sims, out.summary


def contrast_summary(sims: pd.DataFrame, real: pd.Series, planted: float) -> pd.DataFrame:
    """``simulation.parametric_bootstrap_contrast``'s summary, recomputed on the pooled
    simulations of several chunks (checked against each chunk's own summary)."""
    rows = {}
    for kind, sfx in (("raw", ""), ("clipped", "_clipped")):
        err, t = sims[f"error{sfx}"].to_numpy(), float(real[f"diff{sfx}"])
        q = [(1 - LEVEL) / 2, (1 + LEVEL) / 2]
        q_lo, q_hi = np.quantile(err, q)
        loo = [np.quantile(np.delete(err, s), q) for s in range(len(err))]
        covered = [lo <= e <= hi for (lo, hi), e in zip(loo, err, strict=True)]
        m4 = (sims[f"ci_low{sfx}"] <= planted) & (planted <= sims[f"ci_high{sfx}"])
        rows[kind] = {
            "estimate": t, "planted": planted, "mean_error": float(err.mean()),
            "sd_error": float(err.std(ddof=1)), "ci_low": t - q_hi, "ci_high": t - q_lo,
            "coverage_m4": float(m4.mean()), "coverage_loo": float(np.mean(covered)),
        }  # fmt: skip
    return pd.DataFrame.from_dict(rows, orient="index")


def stage_h3() -> None:
    dd, d, lam_ = data(), design(), lam()
    own = d.players[d.players != REPLACEMENT]
    labels = dd.positions.reindex(own)
    terms = pd.read_csv(RAPM_DIR / "group_terms.csv").set_index("group")
    shares = pd.read_csv(GAP_DIR / "shares.csv")
    shares = shares[(shares["mode"] == "player") & (shares["component"] == "total")]
    cross = shares[shares["estimator"] == "cross"].set_index("group")
    within = shares[shares["estimator"] == "within"].set_index("group")
    m4p = pd.read_csv(GAP_DIR / "m4_players.csv")
    m4p["box_p90"] = m4p["np_goals_p90"] + m4p["assists_p90"]
    contrast = pd.read_csv(GAP_DIR / "contrast.csv")
    real = contrast[(contrast["estimator"] == "cross") & (contrast["mode"] == "player")].iloc[0]
    real = real.drop(["estimator", "mode"]).astype(float)
    attribution = attribution_map("pools", True, "total")

    # Box score (A13 iv): mean = real group mean of np goals + assists per 90; SD such
    # that sd^2 / (sd^2 + mean * 90 / T) equals M4's full-window rel_pred (within
    # estimator, total), T the group's median minutes.
    box = m4p.groupby("position_group").agg(mean=("box_p90", "mean"), t=("minutes", "median"))
    box["rel_pred_full"] = within["rel_pred"].reindex(box.index)
    noise = box["mean"] * 90 / box["t"]
    rel = box["rel_pred_full"]
    box["sd_solved"] = np.sqrt(np.where(rel > 0, rel * noise / (1 - rel), 0.0))
    box["floor_binds"] = box["sd_solved"] < SHOT_SD_FLOOR
    box["sd"] = box["sd_solved"].clip(lower=SHOT_SD_FLOOR)
    box["measurable_share"] = cross["r2_true_clipped"].reindex(box.index)
    mean_all = float(m4p["box_p90"].mean())
    fixed: dict[str, Any] = {
        "measurable_share": box["measurable_share"].to_dict() | {UNKNOWN: 0.0},
        "component": "total",
        "shot_rate_mean": box["mean"].to_dict() | {UNKNOWN: mean_all},
        "shot_rate_sd": box["sd"].to_dict() | {UNKNOWN: SHOT_SD_FLOOR},
        "conversion": CONVERSION,
        "base_rate": terms["base_rate"].to_dict(),
        "home_advantage": terms["home"].to_dict(),
        "match_shock_sd": 0.0,
    }
    print("box-score calibration:\n", box.round(4).to_string(), flush=True)

    # Effect SDs (A13 iv): match the real split-half covariance per group.
    cov_real = split_cov(load_halves(RAPM_DIR / "halves.npz"), labels)
    sd = pd.DataFrame(SD_START, index=list(GROUPS), columns=list(COMPONENTS[:2]))
    sd = sd.where(cov_real > 0, 0.0)  # no positive covariance to match: no planted effect
    log = []
    g = rng("h3/calibration")
    for r in range(CALIB_ROUNDS):
        t0 = time.perf_counter()
        params = h3_params(sd, fixed)
        cov_sim = sum(simulate_cov(params, lam_, labels, g) for _ in range(CALIB_SIMS)) / CALIB_SIMS
        assert isinstance(cov_sim, pd.DataFrame)
        ratio = cov_real / cov_sim
        for grp in GROUPS:
            for c in COMPONENTS[:2]:
                log.append(
                    {"round": r + 1, "group": grp, "component": c, "sd": sd.loc[grp, c],
                     "cov_real": cov_real.loc[grp, c], "cov_sim": cov_sim.loc[grp, c],
                     "ratio": ratio.loc[grp, c]}
                )  # fmt: skip
        print(f"calibration round {r + 1} ({time.perf_counter() - t0:.0f} s): ratio real/sim\n"
              f"{ratio.round(3).to_string()}", flush=True)  # fmt: skip
        if r < CALIB_ROUNDS - 1:
            update = (sd > 0) & (ratio > 0)
            sd = sd.where(~update, sd * np.sqrt(ratio.where(update, 1.0)))
    calib = pd.DataFrame(log)
    to_csv(calib, GAP_DIR / "h3_calibration.csv")
    last = calib[calib["round"] == CALIB_ROUNDS]
    mismatch = float((last["ratio"] - 1).abs().where(last["sd"] > 0).max())
    params = h3_params(sd, fixed)
    write_json(GAP_DIR / "h3_params.json", {
        "offence_sd": params.offence_sd, "defence_sd": params.defence_sd, **fixed,
        "lam": lam_, "min_minutes": MIN_MINUTES, "n_sims": N_SIMS, "n_repeats": N_REPEATS,
        "attribution": attribution, "calibration_rounds": CALIB_ROUNDS,
        "calibration_sims_per_round": CALIB_SIMS, "final_max_abs_ratio_minus_1": mismatch,
    })  # fmt: skip

    # Parametric bootstrap (A9), in N_WORKERS chunks with independent seeds.
    sizes = [N_SIMS // N_WORKERS + (i < N_SIMS % N_WORKERS) for i in range(N_WORKERS)]
    seeds = seed("h3/bootstrap").spawn(N_WORKERS)
    tasks = [(s, n, params, real, attribution, lam_) for s, n in zip(seeds, sizes, strict=True)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool_:
        results = list(pool_.map(h3_chunk, tasks))
    print(f"parametric bootstrap: {time.perf_counter() - t0:.0f} s", flush=True)
    planted = float(results[0][1].loc["raw", "planted"])
    for sims_k, summary_k in results:
        check = contrast_summary(sims_k, real, planted)
        pd.testing.assert_frame_equal(check, summary_k[check.columns], check_dtype=False)
    sims = pd.concat([r[0] for r in results], ignore_index=True)
    summary = contrast_summary(sims, real, planted)
    to_csv(sims, GAP_DIR / "h3_sims.csv")
    to_csv(summary.rename_axis("kind").reset_index(), GAP_DIR / "h3_summary.csv")
    print(summary.round(4).to_string())

    def verdict(kind: str) -> str:
        row = summary.loc[kind]
        return "supported" if row["estimate"] > 0 and row["ci_low"] > 0 else "refuted"

    entries: dict[str, tuple[Any, str]] = {
        "h3_verdict": (verdict("clipped"), "H3 by A9's parametric interval of the clipped "
                       "contrast (A7 clips R2_true): supported iff it excludes 0 and the "
                       "difference is positive"),
        "h3_verdict_raw": (verdict("raw"), "Same decision on the raw (unclipped) contrast"),
        "h3_n_sims": (len(sims), "Parametric bootstrap simulations (A13 iv)"),
        "h3_calibration_max_mismatch": (mismatch, "Largest |cov_real / cov_sim - 1| over "
                                        "groups and components in the last calibration round"),
    }  # fmt: skip
    for kind, row in summary.iterrows():
        for col, v in row.items():
            entries[f"h3_param_{kind}_{col}"] = (v, f"Parametric bootstrap ({kind}): {col}")
    for grp in GROUPS:
        for c in COMPONENTS[:2]:
            entries[f"h3_sd_{c}_{grp}"] = (sd.loc[grp, c], f"Planted {c} SD, group {grp}")
        entries[f"h3_shot_sd_{grp}"] = (box.loc[grp, "sd"], f"Simulated box-score SD, {grp}")
        entries[f"h3_shot_floor_binds_{grp}"] = (bool(box.loc[grp, "floor_binds"]),
                                                 f"Shot-rate SD floor binds, {grp}")  # fmt: skip
    record(GAP_DIR, entries)
    print("H3:", entries["h3_verdict"][0], "(clipped);", entries["h3_verdict_raw"][0], "(raw)")


# --- post hoc: assortative planting ---------------------------------------------------


def a9_params() -> SimParams:
    p = read_json(GAP_DIR / "h3_params.json")
    keys = [f.name for f in fields(SimParams)]
    return SimParams(**{k: p[k] for k in keys})


def assortative_effects(params: SimParams) -> pd.DataFrame:
    """Real pooled RAPM of own-column players, centred and rescaled per group and
    component to the A9 SDs; every other stint player gets the replacement column's
    offence and defence, unscaled."""
    dd, fit = data(), pooled_fit()
    own = fit.drop(index=REPLACEMENT)
    grp = dd.positions.reindex(own.index)
    out = pd.DataFrame(index=own.index)
    for c, sds in (("offence", params.offence_sd), ("defence", params.defence_sd)):
        x = own[c].groupby(grp)
        assert isinstance(sds, dict)
        out[c] = (own[c] - x.transform("mean")) / x.transform("std", ddof=0) * grp.map(sds)
    eff = out.reindex(dd.positions.index)
    for c in ("offence", "defence"):
        eff[c] = eff[c].fillna(float(fit.loc[REPLACEMENT, c]))
    return eff.rename_axis("player_id").reset_index()


def group_corr(a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = {}
    for g in sorted({x for x in labels if isinstance(x, str)}):
        ok = labels == g
        rows[g] = [float(np.corrcoef(a[ok, k], b[ok, k])[0, 1]) for k in range(3)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(COMPONENTS))


def attribution_given(effects: pd.DataFrame, lam_: float, g: np.random.Generator) -> pd.DataFrame:
    """``rapm.attribution_factor`` with the planted effects given instead of drawn:
    noiseless fit at lam; full data corr(est, beta)^2, half data the mean over
    N_ATTRIBUTION_SIMS random game splits of corr(a, beta) corr(b, beta) / corr(a, b)."""
    d = design()
    p = d.n_players
    eff = effects.set_index("player_id")
    eff.loc[REPLACEMENT] = pooled_fit().loc[REPLACEMENT, ["offence", "defence"]].astype(float)
    eff = eff.reindex(d.players)
    beta = np.concatenate([eff["offence"].to_numpy(), eff["defence"].to_numpy()])
    planted = np.column_stack([beta[:p], beta[p:], beta[:p] + beta[p:]])
    dn = replace(d, y=np.asarray(d.x[:, : 2 * p] @ beta, dtype=np.float64))

    def est(w: np.ndarray | None) -> np.ndarray:
        dw = dn if w is None else replace(dn, w=dn.w * w[d.game])
        return fit_rapm(dw, lam_).players[list(COMPONENTS)].to_numpy()

    full = est(None)
    n_games = len(d.game_ids)
    halves = []
    for _ in range(N_ATTRIBUTION_SIMS):
        in_a = (g.permutation(n_games) < n_games // 2).astype(np.float64)
        halves.append((est(in_a), est(1 - in_a)))
    pos = data().positions.reindex(d.players).to_numpy(dtype=object).copy()
    pos[d.players == REPLACEMENT] = None
    frames = []
    for kind, labels in (("groups", pos), ("pools", pd.Series(pos).map(POOLS).to_numpy())):
        frames.append(group_corr(full, planted, labels).pow(2).assign(kind=kind, halves=False))
        ratios = [
            group_corr(a, planted, labels)
            * group_corr(b, planted, labels)
            / group_corr(a, b, labels)
            for a, b in halves
        ]
        frames.append((sum(ratios) / len(ratios)).assign(kind=kind, halves=True))
    return pd.concat(frames).rename_axis("group").reset_index()


def per90(pm: pd.DataFrame) -> pd.DataFrame:
    g = pm.groupby("player_id")[["minutes", *STATS]].sum()
    return g[list(STATS)].div(g["minutes"], axis=0) * 90


def h3a_chunk(
    task: tuple[np.random.SeedSequence, int, SimParams, pd.DataFrame, dict[str, float],
                dict[str, float], float],
) -> list[dict[str, float]]:  # fmt: skip
    s, n_sims, params, effects, att_groups, att_pools, lam_ = task
    dd, g = data(), np.random.default_rng(s)
    groups = dd.positions[dd.positions.isin(GROUPS)]
    rows = []
    for _ in range(n_sims):
        sim = simulate_on_schedule(
            dd.stints, dd.stint_players, dd.positions, g, params, group="group", effects=effects
        )
        d = build_design(sim.stints, dd.stint_players, group="group", min_minutes=MIN_MINUTES)
        rapm = fit_rapm(d, lam_).players["total"]
        halves = split_halves(d, lam_, g, n_repeats=N_REPEATS)
        splits = Splits(
            halves.a["total"], halves.b["total"], split_stats(sim.player_match, halves, STATS)
        )
        stats = per90(sim.player_match)
        row = position_contrast(
            rapm, stats, splits, att_pools, dd.positions, g, n_boot=N_BOOT, level=LEVEL
        ).to_dict()
        share = unmeasured_share(rapm, stats, splits, att_groups, groups, g, n_boot=1)
        for grp, r in share.iterrows():
            row[f"share_{grp}"], row[f"r2_true_{grp}"] = float(r["share"]), float(r["r2_true"])
        rows.append(row)
    return rows


def stage_h3_assortative() -> None:
    lam_, params = lam(), a9_params()
    effects = assortative_effects(params)
    to_csv(effects, ASSORT_DIR / "effects.csv")

    # (i) attribution under the planted (real-structure) effects vs independent effects
    t0 = time.perf_counter()
    att = attribution_given(effects, lam_, rng("h3_assortative/attribution"))
    indep = attribution_table()
    both = att.merge(indep, on=["kind", "halves", "group"], suffixes=("", "_independent"))
    to_csv(both, ASSORT_DIR / "attribution.csv")
    print(f"attribution ({time.perf_counter() - t0:.0f} s):\n"
          f"{both.drop(columns='n_players').round(3).to_string()}", flush=True)  # fmt: skip

    # planted shares: with the effects fixed, R2(t, z) in a group is m r / (m r + 1 - m),
    # r = Var(t) / sd_t^2 (r = 1 when offence and defence are independent, as in A9)
    own = design().players
    own = own[own != REPLACEMENT]
    eff = effects.set_index("player_id").loc[own]
    grp = data().positions.reindex(own)
    t = eff["offence"] + eff["defence"]
    planted = pd.DataFrame({"n": grp.value_counts(), "var_t": t.groupby(grp).var(ddof=0)})
    assert isinstance(params.offence_sd, dict) and isinstance(params.defence_sd, dict)
    assert isinstance(params.measurable_share, dict)
    planted["sd_t2"] = [
        params.offence_sd[g] ** 2 + params.defence_sd[g] ** 2 for g in planted.index
    ]
    planted["m"] = [params.measurable_share[g] for g in planted.index]
    planted["r"] = planted["var_t"] / planted["sd_t2"]
    mr = planted["m"] * planted["r"]
    planted["r2_planted"] = mr / (mr + 1 - planted["m"])
    planted["share_planted"] = 1 - planted["r2_planted"]
    planted["share_planted_a9_definition"] = 1 - planted["m"]

    def pool_share(pool: tuple[str, ...]) -> float:
        q = planted.loc[list(pool)]
        v = q["n"] * q["var_t"]
        return float(1 - (v * q["r2_planted"]).sum() / v.sum())

    planted_contrast = pool_share(("CB", "DM")) - pool_share(("CF", "W"))

    # (ii) S = 200 simulations with these effects, everything else as the A9 run
    contrast = pd.read_csv(GAP_DIR / "contrast.csv")
    real = contrast[(contrast["estimator"] == "cross") & (contrast["mode"] == "player")].iloc[0]
    real = real.drop(["estimator", "mode"]).astype(float)
    att_groups = attribution_map("groups", True, "total")
    att_pools = attribution_map("pools", True, "total")
    sizes = [N_SIMS // N_WORKERS + (i < N_SIMS % N_WORKERS) for i in range(N_WORKERS)]
    seeds = seed("h3_assortative/bootstrap").spawn(N_WORKERS)
    tasks = [(s, n, params, effects, att_groups, att_pools, lam_)
             for s, n in zip(seeds, sizes, strict=True)]  # fmt: skip
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool_:
        sims = pd.DataFrame([row for rows in pool_.map(h3a_chunk, tasks) for row in rows])
    print(f"assortative bootstrap: {time.perf_counter() - t0:.0f} s", flush=True)
    sims["error"] = sims["diff"] - planted_contrast
    sims["error_clipped"] = sims["diff_clipped"] - planted_contrast
    to_csv(sims, ASSORT_DIR / "sims.csv")
    summary = contrast_summary(sims, real, planted_contrast)
    to_csv(summary.rename_axis("kind").reset_index(), ASSORT_DIR / "contrast_summary.csv")

    shares = pd.read_csv(GAP_DIR / "shares.csv")
    shares = shares[(shares["component"] == "total") & (shares["estimator"] == "cross")
                    & (shares["mode"] == "player")].set_index("group")  # fmt: skip
    per = planted.copy()
    for g in per.index:
        err = sims[f"share_{g}"] - per.loc[g, "share_planted"]
        per.loc[g, "mean_error"], per.loc[g, "sd_error"] = err.mean(), err.std(ddof=1)
        per.loc[g, "p_r2_true_gt1"] = float((sims[f"r2_true_{g}"] > 1).mean())
        per.loc[g, "share_sim_mean"] = sims[f"share_{g}"].mean()
    per["share_real_raw"] = shares["share"].reindex(per.index)
    per["r2_true_real_raw"] = shares["r2_true"].reindex(per.index)
    per["share_bias_corrected"] = (per["share_real_raw"] - per["mean_error"]).clip(0, 1)
    per = per.reindex(list(GROUPS))
    to_csv(per.rename_axis("group").reset_index(), ASSORT_DIR / "groups.csv")
    pd.set_option("display.width", 250)
    print(per.round(3).to_string())
    print(summary.round(4).to_string())

    def verdict(kind: str) -> str:
        row = summary.loc[kind]
        return "supported" if row["estimate"] > 0 and row["ci_low"] > 0 else "refuted"

    note = " (post hoc, exploratory: planted effects = real pooled RAPM rescaled to A9 SDs)"
    entries: dict[str, tuple[Any, str]] = {
        "verdict_clipped": (verdict("clipped"), "H3 under assortative planting, clipped" + note),
        "verdict_raw": (verdict("raw"), "H3 under assortative planting, raw" + note),
        "n_sims": (len(sims), "Simulations" + note),
        "planted_contrast": (planted_contrast, "Planted CB+DM minus CF+W share" + note),
    }
    for kind, row in summary.iterrows():
        for col, v in row.items():
            entries[f"contrast_{kind}_{col}"] = (v, f"H3 contrast ({kind}): {col}" + note)
    for g, row in per.iterrows():
        for col in ("share_planted", "mean_error", "sd_error", "p_r2_true_gt1", "r",
                    "share_bias_corrected", "share_real_raw"):  # fmt: skip
            entries[f"{col}_{g}"] = (row[col], f"{col}, group {g}, total component" + note)
    for _, row in both.iterrows():
        sfx = "half" if row["halves"] else "full"
        entries[f"attribution_{sfx}_total_{row['group']}"] = (
            row["total"],
            f"Attribution factor ({sfx} data, total), planted real structure" + note,
        )
        entries[f"attribution_{sfx}_total_{row['group']}_independent"] = (
            row["total_independent"],
            "Same with independent planted effects (A9)",
        )
    record(ASSORT_DIR, entries)
    print("H3 (assortative):", verdict("clipped"), "(clipped);", verdict("raw"), "(raw)")


def stage_robustness() -> None:
    lam_ = lam()
    fit = pooled_fit()
    de = design("elo_diff")
    elo = fit_rapm(de, lam_)
    both = fit[list(COMPONENTS)].join(elo.players[list(COMPONENTS)], rsuffix="_elo")
    both = both.drop(index=REPLACEMENT)
    to_csv(both.rename_axis("player_id").reset_index(), RAPM_DIR / "rapm_elo.csv")
    pool = pd.read_csv(PAPER / "data" / "pool.csv")["tm_player_id"]
    entries: dict[str, tuple[Any, str]] = {
        "elo_coef_per100": (elo.covariate, "With-Elo fit: goals per 90 per 100 Elo points of "
                            "attacking minus defending side"),
    }  # fmt: skip
    for c in COMPONENTS:
        for name, sub in (("all", both), ("pool", both[both.index.isin(pool)])):
            r = float(np.corrcoef(sub[c], sub[f"{c}_elo"])[0, 1])
            rho = float(sub[c].rank().corr(sub[f"{c}_elo"].rank()))
            entries[f"elo_corr_{c}_{name}"] = (
                r,
                f"Pearson correlation of RAPM {c} with and "
                f"without the Elo covariate, {name} players",
            )
            entries[f"elo_spearman_{c}_{name}"] = (rho, f"Spearman correlation, same, {name}")
            print(f"{c} {name}: r = {r:.3f}, rho = {rho:.3f}")
    print(f"Elo coefficient per 100 points: {elo.covariate:.4f}")
    record(RAPM_DIR, entries)


STAGES: dict[str, Callable[[], None]] = {
    "design": stage_design,
    "cv": stage_cv,
    "fit": stage_fit,
    "season": stage_season,
    "bootstrap": stage_bootstrap,
    "halves": stage_halves,
    "attribution": stage_attribution,
    "m4": stage_m4,
    "h3": stage_h3,
    "robustness": stage_robustness,
    "h3_assortative": stage_h3_assortative,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--stage", choices=list(STAGES))
    which.add_argument("--from", dest="start", choices=list(STAGES))
    args = parser.parse_args()
    names = list(STAGES)
    if args.stage:
        names = [args.stage]
    elif args.start:
        names = names[names.index(args.start) :]
    RAPM_DIR.mkdir(parents=True, exist_ok=True)
    GAP_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        STAGES[name]()
        seconds = time.perf_counter() - t0
        timings = read_json(RAPM_DIR / "timings.json")
        timings[name] = {"seconds": seconds, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(RAPM_DIR / "timings.json", timings)
        print(f"=== {name}: {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
