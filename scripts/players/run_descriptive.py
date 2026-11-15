"""Descriptive and trait analyses of paper 2 on the real panel: M1, M2, M7(b) / H7, coverage.

Stages, run in this order by default, each resumable (a stage whose summary.json exists
is skipped unless --force; the M1 tasks are checkpointed one by one under parts/):

  reliability  M1 sampling noise                -> results/players/reliability/
  structure    M2 dependence structure          -> results/players/structure/
  clutch       M7(b), H7 with the A5 power check -> results/players/clutch/
  coverage     pool coverage flags              -> results/players/coverage/pool_coverage.csv

Every summary.json maps a key to {"value": ..., "description": ...} for the paper's macros.
Master seed 20261114 (A12); each task draws from its own generator, seeded by the master
seed and the task's name, so a resumed run gives the same numbers as a single one.

Choices the pre-registration does not fix (post-hoc, Claude's calls; flagged in the report):

- Rows: Transfermarkt player ids only (the 66 rows of StatsBomb pseudo ids, players without
  a Transfermarkt id, are dropped) and a known position group (3 rows dropped). Primary
  metrics use Transfermarkt-derived columns only (A11); ESPN columns are never read.
- M1 unit: player-season over all competitions (scope "all", national-team matches
  included). The pre-registered 900-minute floor is applied to the unit's total minutes
  before both the split-half and the stabilisation fit, and the split-half additionally
  requires 450 minutes in each half (the module's default). on_pitch_gd = on_pitch_gf -
  on_pitch_ga uses only the rows of matches with stints (minutes in stint games), so its
  halves need 450 stint minutes each. "OUTFIELD" pools every group but GK; GK is its own
  group, for cards and on_pitch_gd only. Match bootstraps of the pool's window rates use
  all window matches (no floor), B = 200, 90% percentile intervals (pre-registered).
- M1 exploratory StatsBomb metrics (sb_np_xg, sb_xa per 90, StatsBomb-linked matches
  only, outfield pooled): split-half with the same 450-per-half rule; stabilisation on
  every StatsBomb player-season (no floor) at levels 90-450 per half, because only 69
  player-seasons reach 900 StatsBomb minutes and almost all are from four club teams.
- M2 sample: scope "all" rows of panel_player_season (the sum over every competition)
  with minutes >= 900, outfield only. team_elo = minutes-weighted mean of the player's
  club-match team Elo that season (national-team Elo is a separate pool, so national-team
  matches are excluded). league_season = main competition + season, the main competition
  being the domestic league (scope "league") where he played most minutes that season, or,
  without league minutes, the club competition with most minutes. main club = the club
  with most club-match minutes that season (a mid-season transfer counts for his main
  club). goals_p90 = np_goals_p90 + penalty_goals_p90 exactly, so PCA, parallel analysis,
  factor analysis and effective numbers use the six non-redundant metrics (goals dropped);
  correlations, clustering, ICC and the composite keep all seven. Clustering uses the
  Spearman matrix. A partial correlation that also has position group as a categorical
  is reported beside the pre-specified one. Distance correlation: 2,000 player-seasons
  drawn at random, 999 permutations (rows treated as exchangeable although some players
  appear in several seasons). Team ICC: 1,000 club bootstraps, 90% interval.
  V1-like composite: A13 (i) weights restricted to goals, np_goals, assists, on_pitch_gd
  (0.30, 0.15, 0.25, 0.20, renormalised to sum to 1; the diagnostics are scale-free, so
  renormalising changes nothing but the printed weights).
- M7(b): outfield players only; every competition, with the panel's knockout flag (any
  round that is neither a league matchday nor a group, so domestic cups and European
  qualifiers count). Player bootstrap 2,000, permutations 2,000 (module defaults), 90%
  interval. Placebo: 100 independent relabellings, each tested like the real split.
  Exploratory sensitivity: only knockouts of CL, EL, Conference League, Club World Cup and
  national-team tournaments; the other knockout rows are dropped.
- A5: within-period split-half reliability of the over-performance z, players of the H7
  sample only: in each of 200 random splits, each player's knockout and non-knockout
  matches of the period are dealt alternately (in random order) into two halves; z is
  recomputed per half with the module's formula (``clutch._overperformance``); the
  correlation of the halves across players with a defined z in both is stepped up with
  Spearman-Brown, and the reliabilities are averaged over splits. Power: 20,000
  simulated samples of the observed n from a latent-plus-noise model, true scores
  bivariate normal with correlation 0.3 and each observed score's reliability equal to
  its period's estimate (clipped to [0, 1]); a sample "detects" when the one-sided
  Pearson test at alpha = 0.05 rejects (the t test, which the module's permutation test
  approximates). The share of samples with r >= 0.2 (H7 refuted) is reported too.
  Verdict: refuted if r >= 0.2; confirmed if r < 0.2 and power >= 0.8; otherwise
  inconclusive (underpowered).
- Coverage: a pool player's club league in a season comes from his Transfermarkt
  valuations dated in that season (1 July rule; for 2019/20 only those from 2020-01-01),
  or, with none, the last earlier one, evaluated at the season's (or window's) start. A
  valuation without a league takes the league of the same club id's other valuations; a
  club never given one plays outside the snapshot's leagues. A league counts as covered at
  a date within 60 days of the first-to-last game span of one of its competition-seasons
  in results/players/coverage/tm_coverage.csv. Continental national-team championships
  held in the window but absent from the data are listed for the player's confederation
  (from his first Transfermarkt citizenship); whether he was selected is not checked.

Usage:
    uv run python scripts/players/run_descriptive.py [--stage NAME] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections.abc import Callable
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from footy.players import clutch
from footy.players import reliability as rel
from footy.players import structure as st
from footy.players.paths import PAPER, PROCESSED, RAW, RESULTS
from footy.players.window import season_label

SEED = 20261114
N_SPLITS = 200
N_BOOT = 200
LEVEL = 0.9
MIN_MINUTES = 900.0
MIN_HALF = 450.0

M1_METRICS = ("goals", "np_goals", "assists", "np_ga", "penalty_goals", "yellow", "red",
              "on_pitch_gd")  # fmt: skip
SB_METRICS = ("sb_np_xg", "sb_xa")
SB_LEVELS = (90.0, 180.0, 270.0, 360.0, 450.0)
GROUPS = ("OUTFIELD", "CB", "FB", "DM", "CM", "AM", "W", "CF", "GK")
GK_METRICS = ("yellow", "red", "on_pitch_gd")  # a goalkeeper's goals and assists are ~0

M2_METRICS = ("goals", "np_goals", "penalty_goals", "assists", "yellow", "red", "on_pitch_gd")
M2_BASIS = tuple(m for m in M2_METRICS if m != "goals")
V1_WEIGHTS = {"goals": 0.30, "np_goals": 0.15, "assists": 0.25, "on_pitch_gd": 0.20}
DCOR_ROWS = 2000
DCOR_PERMUTATIONS = 999
ICC_BOOT = 1000

H7_METRIC = "np_ga"
H7_MIN_KO = 360.0
N_PLACEBO = 100
N_POWER = 20_000
TRUE_R = 0.3
ALPHA = 0.05
SB_H7_MIN_PLAYERS = 10
MAJOR_KNOCKOUT = ("CL", "EL", "UCOL", "KLUB")  # exploratory H7 sensitivity, with intl

COVERAGE_SLACK_DAYS = 60
WINDOW_START = pd.Timestamp("2020-01-01")
OUTSIDE = "outside_snapshot"
CONFEDERATION = {
    **dict.fromkeys(
        ("Spain", "France", "England", "Portugal", "Germany", "Italy", "Belgium", "Norway",
         "Croatia", "Netherlands", "Poland", "Georgia", "Denmark", "Serbia", "Switzerland",
         "Türkiye", "Ukraine", "Sweden", "Scotland"), "UEFA"),
    **dict.fromkeys(("Brazil", "Argentina", "Uruguay", "Ecuador", "Colombia"), "CONMEBOL"),
    **dict.fromkeys(
        ("Morocco", "Nigeria", "Egypt", "Senegal", "Algeria", "Cote d'Ivoire", "Cameroon",
         "Guinea"), "CAF"),
    "Korea, South": "AFC",
    "Mexico": "CONCACAF",
}  # fmt: skip
# Continental championships played in the window that the snapshot lacks, by the season
# label of their first match (the panel's rule for national-team tournaments).
ABSENT_TOURNAMENTS = {
    "CONMEBOL": {"2020/21": "Copa America 2021 (13 Jun-10 Jul 2021)"},
    "CAF": {
        "2021/22": "AFCON 2021 (9 Jan-6 Feb 2022)",
        "2023/24": "AFCON 2023 (13 Jan-11 Feb 2024)",
    },
    "CONCACAF": {
        "2021/22": "Gold Cup 2021 (10 Jul-1 Aug 2021)",
        "2022/23": "Gold Cup 2023 (24 Jun-16 Jul 2023)",
        "2024/25": "Gold Cup 2025 (14 Jun-6 Jul 2025)",
    },
}

PM_COLUMNS = ["game_id", "player_id", "date", "season", "competition_id", "scope", "club_id",
              "position_group", "minutes", "goals", "np_goals", "assists", "penalty_goals",
              "yellow", "red", "on_pitch_gf", "on_pitch_ga", "team_elo", "knockout",
              "sb_linked", "sb_np_xg", "sb_xa"]  # fmt: skip

Summary = dict[str, dict[str, Any]]


# --- shared helpers ----------------------------------------------------------------------


def rng_for(*names: str) -> np.random.Generator:
    """The task's own generator: master seed plus a hash of its name."""
    return np.random.default_rng([SEED, *(zlib.crc32(n.encode()) for n in names)])


def clean(value: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + "\n")


def entry(value: Any, description: str) -> dict[str, Any]:
    return {"value": value, "description": description}


@cache
def match_panel() -> pd.DataFrame:
    pm = pd.read_csv(PROCESSED / "panel_player_match.csv.gz", usecols=PM_COLUMNS)
    pm = pm[(pm["player_id"] > 0) & pm["position_group"].notna()].copy()
    pm["np_ga"] = pm["np_goals"] + pm["assists"]
    pm["on_pitch_gd"] = pm["on_pitch_gf"] - pm["on_pitch_ga"]
    return pm


def pool() -> pd.DataFrame:
    return pd.read_csv(PAPER / "data" / "pool.csv")


# --- M1 reliability ----------------------------------------------------------------------


def m1_rows(pm: pd.DataFrame, metric: str, group: str) -> pd.DataFrame:
    rows = pm if metric != "on_pitch_gd" else pm[pm["on_pitch_gf"].notna()]
    if group == "OUTFIELD":
        return rows[rows["position_group"] != "GK"]
    return rows[rows["position_group"] == group]


def reliability_task(
    rows: pd.DataFrame, metric: str, name: str, levels: tuple[float, ...] | None
) -> dict[str, Any]:
    split = rel.split_half(
        rows, metric, rng_for("split", name), n_splits=N_SPLITS, min_half_minutes=MIN_HALF
    )
    kwargs = {} if levels is None else {"levels": levels}
    try:
        stab = rel.stabilisation(rows, metric, rng_for("stab", name), n_splits=N_SPLITS, **kwargs)
        k, curve, fit_error = stab.k, stab.curve, None
    except (RuntimeError, ValueError, TypeError) as error:  # a metric with no signal: no finite k
        k, curve, fit_error = float("nan"), pd.DataFrame(), str(error)
    unit_minutes = rows.groupby(["player_id", "season"])["minutes"].sum()
    harmonic = float(1 / (1 / unit_minutes[unit_minutes >= MIN_MINUTES]).mean())
    return {
        "reliability": split.estimate,
        "reliability_sd": split.sd,
        "n_units": split.n_units,
        "harmonic_minutes": harmonic,
        "k": k,
        "minutes_05": k,
        "minutes_07": 7 * k / 3,
        "implied_reliability": harmonic / (harmonic + k),
        "fit_error": fit_error,
        "curve": curve.to_dict(orient="list"),
    }


def run_reliability(out: Path) -> Summary:
    pm = match_panel()
    units = pm.groupby(["player_id", "season"])["minutes"].transform("sum")
    keep = ["player_id", "season", "minutes", "position_group", "on_pitch_gf", *M1_METRICS]
    eligible = pm.loc[units >= MIN_MINUTES, keep]
    parts = out / "parts"
    tasks: list[tuple[str, str, Callable[[], pd.DataFrame], tuple[float, ...] | None]] = [
        (metric, group, (lambda m=metric, g=group: m1_rows(eligible, m, g)), None)
        for metric in M1_METRICS
        for group in GROUPS
        if group != "GK" or metric in GK_METRICS
    ]
    sb = pm[pm["sb_linked"] & (pm["position_group"] != "GK")]
    tasks += [
        (metric, "SB_OUTFIELD", (lambda m=metric: sb[sb[m].notna()]), SB_LEVELS)
        for metric in SB_METRICS
    ]
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for metric, group, rows_of, levels in tasks:
        path = parts / f"{metric}__{group}.json"
        if not path.exists():
            t0 = time.time()
            write_json(path, reliability_task(rows_of(), metric, f"{metric}/{group}", levels))
            print(f"  M1 {metric:13s} {group:11s} {time.time() - t0:6.1f}s", flush=True)
        results[metric, group] = json.loads(path.read_text())

    table = pd.DataFrame(
        [
            {"metric": m, "group": g, **{k: v for k, v in r.items() if k != "curve"}}
            for (m, g), r in results.items()
        ]
    )
    table["exploratory"] = table["group"] == "SB_OUTFIELD"
    table.to_csv(out / "reliability.csv", index=False, float_format="%.5g")
    curves = pd.concat(
        [pd.DataFrame(r["curve"]).assign(metric=m, group=g) for (m, g), r in results.items()],
        ignore_index=True,
    )
    curves.to_csv(out / "stabilisation_curves.csv", index=False, float_format="%.5g")

    k_by = {(m, g): r["k"] for (m, g), r in results.items()}
    stabilisation_k = {
        **{m: k_by[m, "OUTFIELD"] for m in M1_METRICS},
        "by_position_group": {
            g: {m: k_by[m, g] for m in M1_METRICS if (m, g) in k_by} for g in GROUPS[1:]
        },
        "description": "Stabilisation constant k (minutes) of r(m) = m / (m + k) per metric, "
        "per-90 rates, player-seasons >= 900 minutes, all competitions; top-level keys are "
        "outfield pooled (GK excluded), by_position_group within each group. Pooled k counts "
        "between-group differences as signal; for shrinkage toward a group mean the "
        "within-group k is the consistent constant.",
    }
    write_json(out / "stabilisation_k.json", stabilisation_k)

    pool_rates = pool_bootstrap(pm)
    pool_rates.to_csv(out / "pool_window_rates.csv", index=False, float_format="%.5g")

    summary: Summary = {}
    for (m, g), r in results.items():
        tag = f"{m}_{g.lower()}"
        what = f"{m} per 90, {g.lower().replace('_', ' ')}"
        if g == "SB_OUTFIELD":
            what += " (exploratory, StatsBomb-linked matches)"
        summary[f"rel_{tag}"] = entry(
            r["reliability"],
            f"Split-half reliability (Spearman-Brown, mean of {N_SPLITS} splits), {what}",
        )
        summary[f"n_units_{tag}"] = entry(r["n_units"], f"Mean player-seasons per split, {what}")
        summary[f"k_{tag}"] = entry(r["k"], f"Stabilisation k = minutes to reliability 0.5, {what}")
        summary[f"minutes07_{tag}"] = entry(r["minutes_07"], f"Minutes to reliability 0.7, {what}")
        summary[f"harmonic_minutes_{tag}"] = entry(
            r["harmonic_minutes"], f"Harmonic mean minutes of eligible player-seasons, {what}"
        )
    pool_out = pool_rates[pool_rates["position_group"] != "GK"]
    width = pool_out.assign(w=pool_out["hi"] - pool_out["lo"]).groupby("metric")["w"]
    for metric, w in width.median().items():
        summary[f"pool_ci_width_{metric}"] = entry(
            w,
            f"Median width of the 90% match-bootstrap interval of a pool outfield player's "
            f"window {metric} per 90",
        )
    summary["n_eligible_player_seasons"] = entry(
        int(eligible.groupby(["player_id", "season"]).ngroups),
        "Player-seasons with >= 900 minutes (all competitions), M1 sample incl. GK",
    )
    return summary


def pool_bootstrap(pm: pd.DataFrame) -> pd.DataFrame:
    players = pool()
    rows = pm[pm["player_id"].isin(players["tm_player_id"])]
    frames = []
    for metric in M1_METRICS:
        sub = rows if metric != "on_pitch_gd" else rows[rows["on_pitch_gf"].notna()]
        boot = rel.bootstrap_rates(
            sub,
            metric,
            rng_for("pool_boot", metric),
            keys=("player_id",),
            n_boot=N_BOOT,
            level=LEVEL,
        )
        frames.append(boot.assign(metric=metric))
    out = pd.concat(frames, ignore_index=True)
    names = players.set_index("tm_player_id")[["player", "position_group"]]
    out = out.join(names, on="player_id")
    return out[
        [
            "player",
            "player_id",
            "position_group",
            "metric",
            "matches",
            "minutes",
            "rate",
            "se",
            "lo",
            "hi",
        ]
    ]


# --- M2 structure ------------------------------------------------------------------------


def season_panel(pm: pd.DataFrame) -> pd.DataFrame:
    ps = pd.read_csv(PROCESSED / "panel_player_season.csv.gz")
    ps = ps[(ps["scope"] == "all") & (ps["minutes"] >= MIN_MINUTES) & (ps["player_id"] > 0)]
    ps = ps.assign(on_pitch_gd_p90=ps["on_pitch_gf_p90"] - ps["on_pitch_ga_p90"])
    frame = ps[["player_id", "season", "minutes", "minutes_stints"]].copy()
    for m in M2_METRICS:
        frame[m] = ps[f"{m}_p90"].to_numpy()
    group = pm.groupby("player_id")["position_group"].first()
    frame["position_group"] = frame["player_id"].map(group)

    club = pm[pm["scope"] != "intl"]
    keys = ["player_id", "season"]
    by_comp = club.groupby([*keys, "competition_id", "scope"])["minutes"].sum().reset_index()
    by_comp["is_league"] = by_comp["scope"] == "league"
    main = by_comp.sort_values(["is_league", "minutes"], ascending=False, kind="stable")
    main = main.drop_duplicates(keys).set_index(keys)["competition_id"]
    by_club = club.groupby([*keys, "club_id"])["minutes"].sum().reset_index()
    main_club = by_club.sort_values("minutes", ascending=False, kind="stable")
    main_club = main_club.drop_duplicates(keys).set_index(keys)["club_id"]
    elo = club[club["team_elo"].notna()]
    elo_mean = (elo["team_elo"] * elo["minutes"]).groupby([elo[k] for k in keys]).sum() / (
        elo.groupby(keys)["minutes"].sum()
    )
    frame = frame.join(main.rename("main_competition"), on=keys)
    frame = frame.join(main_club.rename("main_club"), on=keys)
    frame = frame.join(elo_mean.rename("team_elo"), on=keys)
    frame["league_season"] = frame["main_competition"] + " " + frame["season"]
    return frame


def run_structure(out: Path) -> Summary:
    pm = match_panel()
    full = season_panel(pm)
    outfield = full[full["position_group"] != "GK"]
    needed = [*M2_METRICS, "team_elo", "league_season", "main_club"]
    complete = outfield.dropna(subset=needed)
    complete = complete[np.isfinite(complete[list(M2_METRICS)]).all(axis=1)].reset_index(drop=True)
    x7, x6 = complete[list(M2_METRICS)], complete[list(M2_BASIS)]
    summary: Summary = {
        "n_player_seasons": entry(len(complete), "Outfield player-seasons >= 900 minutes, M2"),
        "n_dropped_incomplete": entry(
            len(outfield) - len(complete),
            "Outfield player-seasons >= 900 minutes dropped for a missing covariate or metric",
        ),
        "n_players": entry(int(complete["player_id"].nunique()), "Distinct players, M2"),
        "n_league_seasons": entry(int(complete["league_season"].nunique()), "League-seasons, M2"),
        "n_clubs": entry(int(complete["main_club"].nunique()), "Distinct main clubs, M2"),
    }
    complete.to_csv(out / "sample.csv.gz", index=False, float_format="%.6g")

    matrices = {
        "pearson": st.correlation(x7, "pearson"),
        "spearman": st.correlation(x7, "spearman"),
    }
    for method in ("pearson", "spearman"):
        matrices[f"partial_{method}"] = st.partial_correlation(
            complete, M2_METRICS, ["team_elo"], ["league_season"], method
        )
        matrices[f"partial_position_{method}"] = st.partial_correlation(
            complete, M2_METRICS, ["team_elo"], ["league_season", "position_group"], method
        )
    long = pd.concat(
        [m.stack().rename("value").reset_index().assign(kind=k) for k, m in matrices.items()]
    )
    long.columns = ["metric_a", "metric_b", "value", "kind"]
    long[["kind", "metric_a", "metric_b", "value"]].to_csv(
        out / "correlations.csv", index=False, float_format="%.5f"
    )
    pairs = [
        ("goals", "np_goals"),
        ("np_goals", "assists"),
        ("np_goals", "on_pitch_gd"),
        ("assists", "on_pitch_gd"),
        ("yellow", "red"),
        ("np_goals", "penalty_goals"),
    ]
    for kind in ("spearman", "partial_spearman", "pearson", "partial_pearson"):
        for a, b in pairs:
            summary[f"{kind}_{a}_{b}"] = entry(
                float(matrices[kind].loc[a, b]),
                f"{kind.replace('_', ' ')} correlation of {a} and {b} per 90"
                + (" (team Elo, league-season fixed effects)" if "partial" in kind else ""),
            )

    tree, order = st.cluster_metrics(matrices["spearman"])
    write_json(
        out / "clustering.json",
        {
            "order": order,
            "linkage": tree.tolist(),
            "labels": list(M2_METRICS),
            "matrix": "spearman",
        },
    )
    summary["cluster_order"] = entry(", ".join(order), "Leaf order of average-linkage tree")

    pc = st.pca(x6)
    pa = st.parallel_analysis(x6, rng_for("parallel"), n_repeats=N_SPLITS)
    partial6 = st.residualise(complete, M2_BASIS, ["team_elo"], ["league_season"])
    pc_partial = st.pca(partial6)
    pca_table = pc.components.copy()
    pca_table.loc["eigenvalue"] = pc.eigenvalues
    pca_table.loc["pa_threshold"] = pa.threshold
    pca_table.loc["eigenvalue_partial"] = pc_partial.eigenvalues
    pca_table.to_csv(out / "pca.csv", float_format="%.5f")
    summary["pa_n_components"] = entry(pa.n_components, "Components retained, parallel analysis")
    for i, (lam, thr) in enumerate(zip(pc.eigenvalues, pa.threshold, strict=True), start=1):
        summary[f"eigenvalue_{i}"] = entry(float(lam), f"Eigenvalue {i} of the 6-metric R")
        summary[f"pa_threshold_{i}"] = entry(float(thr), f"Parallel-analysis 95% null, rank {i}")
    for tag, lam in (("", pc.eigenvalues), ("_partial", pc_partial.eigenvalues)):
        what = "6 non-redundant metrics" + (", partial (Elo, league-season)" if tag else "")
        summary[f"participation_ratio{tag}"] = entry(
            st.participation_ratio(lam), f"Participation ratio, {what}"
        )
        summary[f"entropy_number{tag}"] = entry(
            st.entropy_effective_number(lam), f"Entropy effective number, {what}"
        )
    v1_eig = np.linalg.eigvalsh(matrices["pearson"].loc[list(V1_WEIGHTS), list(V1_WEIGHTS)])
    summary["participation_ratio_v1_metrics"] = entry(
        st.participation_ratio(v1_eig), "Participation ratio of the 4 V1-composite metrics"
    )

    p = len(M2_BASIS)
    max_k = max(k for k in range(1, p) if (p - k) ** 2 > p + k)  # factor_analysis needs dof > 0
    fa_k = min(max(pa.n_components, 1), max_k)
    fa = st.factor_analysis(x6, fa_k)
    fa_table = fa.loadings.assign(uniqueness=fa.uniquenesses)
    fa_table.to_csv(out / "factor_analysis.csv", float_format="%.5f")
    summary |= {
        "fa_k": entry(fa_k, f"Factors fitted (parallel analysis k, at most {max_k} with dof > 0)"),
        "fa_chi2": entry(fa.chi2, "ML factor analysis Bartlett chi-square"),
        "fa_dof": entry(fa.dof, "ML factor analysis degrees of freedom"),
        "fa_p_value": entry(fa.p_value, "ML factor analysis p-value"),
        "fa_rmsea": entry(fa.rmsea, "ML factor analysis RMSEA"),
        "fa_converged": entry(fa.converged, "Optimiser converged"),
    }

    rng = rng_for("dcor")
    sub = complete.iloc[rng.choice(len(complete), size=DCOR_ROWS, replace=False)]
    dcor_rows = []
    for i, a in enumerate(M2_BASIS):
        for b in M2_BASIS[i + 1 :]:
            xa, xb = sub[a].to_numpy(dtype=np.float64), sub[b].to_numpy(dtype=np.float64)
            dcor, p = st.distance_correlation_test(xa, xb, rng, DCOR_PERMUTATIONS)
            dcor_rows.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "dcor": dcor,
                    "p_value": p,
                    "pearson": np.corrcoef(xa, xb)[0, 1],
                    "spearman": stats.spearmanr(xa, xb).statistic,
                }
            )
    dcor_table = pd.DataFrame(dcor_rows)
    dcor_table.to_csv(out / "distance_correlation.csv", index=False, float_format="%.5f")
    top = dcor_table.loc[dcor_table["dcor"].idxmax()]
    summary["dcor_max"] = entry(
        float(top["dcor"]), f"Largest distance correlation ({top['metric_a']}, {top['metric_b']})"
    )
    summary["dcor_n_significant"] = entry(
        int((dcor_table["p_value"] < 0.05).sum()), f"Pairs (of {len(dcor_table)}) with p < 0.05"
    )

    icc = st.team_icc(
        complete,
        M2_METRICS,
        "main_club",
        "league_season",
        rng_for("icc"),
        n_boot=ICC_BOOT,
        level=LEVEL,
    )
    icc.to_csv(out / "team_icc.csv", index_label="metric", float_format="%.5f")
    for metric, row in icc.iterrows():
        for col in ("icc", "lo", "hi"):
            summary[f"icc_{metric}{'' if col == 'icc' else '_' + col}"] = entry(
                float(row[col]),
                f"Team ICC(1) of {metric} per 90 within league-season"
                + (
                    ""
                    if col == "icc"
                    else f", 90% club-bootstrap {col} (percentile; the club bootstrap sits "
                    "about 0.01 below the estimate, duplicated clubs shrinking the "
                    "between-club variance, so the interval is shifted low)"
                ),
            )

    weights = pd.Series(V1_WEIGHTS) / sum(V1_WEIGHTS.values())
    diag = st.composite_diagnostics(matrices["pearson"], weights)
    shares = diag.shares.assign(weight=weights)
    shares.to_csv(out / "composite_v1.csv", index_label="metric", float_format="%.5f")
    summary |= {
        "v1_inflation": entry(
            diag.inflation,
            "Var(C) / sum w^2 of the V1-like composite (goals, np_goals, assists, on_pitch_gd)",
        ),
        "v1_pc1_share": entry(
            diag.pc1_share,
            "Share of the V1-like composite's variance on "
            "the first principal axis of its four metrics",
        ),
        "v1_goals_claimed": entry(
            float(shares.loc[["goals", "np_goals"], "claimed"].sum()),
            "Claimed variance share of goals + np_goals in the composite",
        ),
        "v1_goals_effective": entry(
            float(shares.loc[["goals", "np_goals"], "effective"].sum()),
            "Effective variance share of goals + np_goals",
        ),
    }
    for metric, row in shares.iterrows():
        summary[f"v1_weight_{metric}"] = entry(
            float(row["weight"]), f"Renormalised weight, {metric}"
        )
        summary[f"v1_effective_{metric}"] = entry(
            float(row["effective"]), f"Effective variance share of {metric} in the composite"
        )
    return summary


# --- M7(b) clutch ------------------------------------------------------------------------


def period_reliability(
    rows: pd.DataFrame, metric: str, start: str, end: str, rng: np.random.Generator
) -> dict[str, float]:
    """Split-half reliability of the period's over-performance z (see module docstring)."""
    dates = pd.to_datetime(rows["date"])
    rows = rows[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    rows = rows[["player_id", "knockout", "minutes", metric, "date"]].reset_index(drop=True)
    cells = rows.groupby(["player_id", "knockout"]).ngroup().to_numpy()
    stepped, used = np.empty(N_SPLITS), np.empty(N_SPLITS)
    for s in range(N_SPLITS):
        order = np.lexsort((rng.random(len(rows)), cells))
        sorted_cells = cells[order]
        rank = np.empty(len(rows), dtype=np.int64)
        rank[order] = np.arange(len(rows)) - np.searchsorted(sorted_cells, sorted_cells)
        pseudo = rows.assign(player_id=2 * rows["player_id"] + rank % 2)
        z = clutch._overperformance(pseudo, metric, start, end)["z"]
        halves = pd.DataFrame({"z": z, "player": z.index // 2, "half": z.index % 2})
        wide = halves.pivot(index="player", columns="half", values="z").dropna()
        r = float(np.corrcoef(wide[0], wide[1])[0, 1])
        stepped[s], used[s] = 2 * r / (1 + r), len(wide)
    return {
        "reliability": float(stepped.mean()),
        "sd": float(stepped.std(ddof=1)),
        "n_players": float(used.mean()),
    }


def power(n: int, rel_a: float, rel_b: float, rng: np.random.Generator) -> dict[str, float]:
    """Monte-Carlo power of the one-sided Pearson test for a true correlation TRUE_R."""
    attenuation = float(np.sqrt(np.clip(rel_a, 0, 1) * np.clip(rel_b, 0, 1)))
    rho = TRUE_R * attenuation
    hits = refutes = 0
    for start in range(0, N_POWER, 1000):
        size = min(1000, N_POWER - start)
        x = rng.standard_normal((size, n))
        y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal((size, n))
        r = clutch._rowcorr(x, y)
        t = r * np.sqrt((n - 2) / (1 - r**2))
        hits += int((stats.t.sf(t, n - 2) < ALPHA).sum())
        refutes += int((r >= 0.2).sum())
    fisher = float(stats.norm.sf(stats.norm.isf(ALPHA) - np.arctanh(rho) * np.sqrt(n - 3)))
    return {
        "attenuation": attenuation,
        "expected_r": rho,
        "power": hits / N_POWER,
        "power_fisher": fisher,
        "p_r_ge_02": refutes / N_POWER,
    }


def run_clutch(out: Path) -> Summary:
    pm = match_panel()
    outfield = pm[pm["position_group"] != "GK"]
    res = clutch.knockout_persistence(
        outfield, H7_METRIC, rng_for("h7"), min_knockout_minutes=H7_MIN_KO, level=LEVEL
    )
    res.players.to_csv(out / "h7_players.csv", float_format="%.5f")

    rng = rng_for("placebo")
    placebo = []
    for _ in range(N_PLACEBO):
        fake = clutch.knockout_persistence(
            clutch.placebo_knockout(outfield, rng),
            H7_METRIC,
            rng,
            min_knockout_minutes=H7_MIN_KO,
            n_boot=200,
            n_perm=200,
            level=LEVEL,
        )
        placebo.append({"r": fake.r, "p_value": fake.p_value, "n_players": fake.n_players})
    placebo_table = pd.DataFrame(placebo)
    placebo_table.to_csv(out / "placebo.csv", index=False, float_format="%.5f")

    sample = outfield[outfield["player_id"].isin(res.players.index)]
    rels = {
        tag: period_reliability(sample, H7_METRIC, a, b, rng_for("a5", tag))
        for tag, (a, b) in zip("ab", clutch.PERIODS, strict=True)
    }
    pw = power(res.n_players, rels["a"]["reliability"], rels["b"]["reliability"], rng_for("power"))
    if res.r >= 0.2:
        verdict = "refuted"
    elif pw["power"] >= 0.8:
        verdict = "confirmed"
    else:
        verdict = "inconclusive (underpowered)"

    # Exploratory, post-hoc: only knockouts of continental club competitions and
    # national-team tournaments; the other knockout rows (domestic cups, super cups,
    # qualifiers, often against weaker sides) are dropped, from the baseline too.
    minor_ko = outfield["knockout"] & ~(
        outfield["competition_id"].isin(MAJOR_KNOCKOUT) | (outfield["scope"] == "intl")
    )
    major = clutch.knockout_persistence(
        outfield[~minor_ko],
        H7_METRIC,
        rng_for("h7_major"),
        min_knockout_minutes=H7_MIN_KO,
        level=LEVEL,
    )
    major.players.to_csv(out / "h7_major_players.csv", float_format="%.5f")

    sb = outfield[outfield["sb_linked"] & (outfield["scope"] == "intl")].copy()
    sb["sb_np_xg_xa"] = sb["sb_np_xg"] + sb["sb_xa"]
    sb_ko = sb[sb["knockout"]].assign(late=pd.to_datetime(sb["date"]) >= "2023-01-01")
    sb_ko = sb_ko.groupby(["player_id", "late"])["minutes"].sum().unstack(fill_value=0)
    sb_n = int((sb_ko >= H7_MIN_KO).all(axis=1).sum()) if sb_ko.shape[1] == 2 else 0
    sb_result = None
    if sb_n >= SB_H7_MIN_PLAYERS:
        fit = clutch.knockout_persistence(
            sb, "sb_np_xg_xa", rng_for("h7_sb"), min_knockout_minutes=H7_MIN_KO, level=LEVEL
        )
        sb_result = {
            "r": fit.r,
            "ci_low": fit.ci_low,
            "ci_high": fit.ci_high,
            "p_value": fit.p_value,
            "n_players": fit.n_players,
        }

    q = placebo_table["r"].quantile([0.05, 0.95])
    periods = " vs ".join(f"{a} to {b}" for a, b in clutch.PERIODS)
    summary: Summary = {
        "h7_r": entry(res.r, f"Correlation of knockout over-performance z, {periods}"),
        "h7_ci_low": entry(res.ci_low, "90% player-bootstrap interval, low"),
        "h7_ci_high": entry(res.ci_high, "90% player-bootstrap interval, high"),
        "h7_p_value": entry(res.p_value, "One-sided permutation p-value (2,000 permutations)"),
        "h7_n": entry(res.n_players, "Outfield players with >= 360 knockout minutes per period"),
        "h7_ko_minutes_median_a": entry(
            float(res.players["knockout_minutes_a"].median()), "Median knockout minutes, period A"
        ),
        "h7_ko_minutes_median_b": entry(
            float(res.players["knockout_minutes_b"].median()), "Median knockout minutes, period B"
        ),
        "placebo_r_mean": entry(float(placebo_table["r"].mean()), "Mean r, 100 placebo splits"),
        "placebo_r_q05": entry(float(q[0.05]), "5th percentile of placebo r"),
        "placebo_r_q95": entry(float(q[0.95]), "95th percentile of placebo r"),
        "placebo_n_mean": entry(
            float(placebo_table["n_players"].mean()), "Mean players per placebo split"
        ),
        "rel_a": entry(rels["a"]["reliability"], "Split-half reliability of z, period A"),
        "rel_b": entry(rels["b"]["reliability"], "Split-half reliability of z, period B"),
        "rel_a_sd": entry(rels["a"]["sd"], "SD over 200 splits, period A"),
        "rel_b_sd": entry(rels["b"]["sd"], "SD over 200 splits, period B"),
        "attenuation": entry(
            pw["attenuation"], "sqrt(rel_a rel_b), reliabilities clipped to [0, 1]"
        ),
        "h7_r_disattenuated": entry(
            res.r / pw["attenuation"] if pw["attenuation"] > 0 else float("nan"),
            "Observed r divided by the attenuation sqrt(rel_a rel_b) (not clipped to 1)",
        ),
        "expected_r_at_03": entry(pw["expected_r"], "Expected observed r for a true r of 0.3"),
        "power_at_03": entry(pw["power"], "Power, one-sided alpha 0.05, true r 0.3 (simulated)"),
        "power_at_03_fisher": entry(pw["power_fisher"], "Same, Fisher-z approximation"),
        "p_refute_at_03": entry(pw["p_r_ge_02"], "P(observed r >= 0.2) for a true r of 0.3"),
        "h7_verdict": entry(verdict, "H7 verdict under A5"),
        "h7_sb_n": entry(
            sb_n,
            "Players with >= 360 StatsBomb tournament knockout minutes in both "
            "periods (secondary, sb_np_xg + sb_xa)",
        ),
    }
    what = "exploratory H7, knockouts of CL, EL, UCOL, KLUB and national-team tournaments only"
    summary |= {
        "h7_major_r": entry(major.r, f"r, {what}"),
        "h7_major_ci_low": entry(major.ci_low, f"90% interval low, {what}"),
        "h7_major_ci_high": entry(major.ci_high, f"90% interval high, {what}"),
        "h7_major_p_value": entry(major.p_value, f"One-sided permutation p, {what}"),
        "h7_major_n": entry(major.n_players, f"Players, {what}"),
    }
    if sb_result is not None:
        summary |= {
            f"h7_sb_{k}": entry(v, f"Secondary H7 on StatsBomb, {k}") for k, v in sb_result.items()
        }
    write_json(out / "a5_reliability.json", rels)
    return summary


# --- coverage ----------------------------------------------------------------------------


def run_coverage(out: Path) -> Summary:
    minutes = pd.read_csv(RESULTS / "linking" / "pool_season_minutes.csv").set_index("player")
    players = pool().set_index("player")
    seasons = list(minutes.columns)
    profile = pd.read_csv(PROCESSED / "tm_players.csv.gz").set_index("player_id")
    citizenship = players["tm_player_id"].map(profile["citizenship"])
    unmapped = set(citizenship.dropna()) - set(CONFEDERATION)
    if unmapped:
        raise ValueError(f"no confederation for {unmapped}")

    values = pd.read_csv(RAW / "transfermarkt" / "player_valuations.csv.gz")
    # A missing league is filled from the same club id's other valuations; a club never
    # given one plays in a league outside the snapshot (e.g. the English Championship).
    league = values["player_club_domestic_competition_id"]
    by_id = league.groupby(values["current_club_id"]).agg(
        lambda c: c.mode().iloc[0] if c.notna().any() else None
    )
    values["league"] = league.fillna(values["current_club_id"].map(by_id)).fillna(OUTSIDE)
    values = values[values["player_id"].isin(players["tm_player_id"])].copy()
    values["day"] = pd.to_datetime(values["date"])
    values["season"] = [season_label(d.date()) for d in values["day"]]
    spans = pd.read_csv(
        RESULTS / "coverage" / "tm_coverage.csv", parse_dates=["first_date", "last_date"]
    )
    slack = pd.Timedelta(days=COVERAGE_SLACK_DAYS)

    def covered(league: str, day: pd.Timestamp) -> bool:
        s = spans[spans["competition_id"] == league]
        return bool(((s["first_date"] - slack <= day) & (day <= s["last_date"] + slack)).any())

    intl = match_panel()
    wc26 = set(
        intl.loc[(intl["competition_id"] == "FIWC") & (intl["season"] == "2025/26"), "player_id"]
    )
    rows = []
    for player, info in players.iterrows():
        mine = values[values["player_id"] == info["tm_player_id"]].sort_values("day")
        confed = CONFEDERATION.get(str(citizenship[player]))
        for season in seasons:
            start = max(pd.Timestamp(date(int(season[:4]), 7, 1)), WINDOW_START)
            here = mine[(mine["season"] == season) & (mine["day"] >= WINDOW_START)]
            if here.empty:
                here = mine[mine["day"] < start].tail(1)
            # a valuation before the season (or the window) stands for the season's start
            days = here["day"].clip(lower=start)
            leagues = list(zip(here["league"], days, strict=True))
            flags = [covered(c, d) for c, d in leagues]
            absent = sorted({c for (c, _), ok in zip(leagues, flags, strict=True) if not ok})
            mins = int(minutes.loc[player, season])
            if not leagues:
                status = "no_valuation_yet"
            elif not any(flags):
                status = "structural_hole"
            elif not all(flags):
                status = "partial_structural_hole"
            elif mins < MIN_MINUTES:
                status = "below_floor_league_covered"
            else:
                status = "ok"
            holes = [ABSENT_TOURNAMENTS.get(confed or "", {}).get(season)]
            if season == "2025/26" and info["tm_player_id"] in wc26:
                holes.append("World Cup 2026 knockout rounds (the snapshot has its 72 group games)")
            rows.append(
                {
                    "player": player,
                    "tm_player_id": info["tm_player_id"],
                    "position_group": info["position_group"],
                    "season": season,
                    "minutes": mins,
                    "below_floor": mins < MIN_MINUTES,
                    "club_leagues": ";".join(dict.fromkeys(c for c, _ in leagues)),
                    "absent_club_leagues": ";".join(absent),
                    "club_status": status,
                    "confederation": confed,
                    "absent_intl_tournaments": "; ".join(h for h in holes if h),
                    "window_truncated": season == "2019/20",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "pool_coverage.csv", index=False)
    counts = table["club_status"].value_counts()
    outfield = table[table["position_group"] != "GK"]
    return {
        f"n_{status}": entry(int(n), f"Pool player-seasons with club status {status}")
        for status, n in counts.items()
    } | {
        "n_player_seasons": entry(len(table), "Pool player-seasons (2019/20 to 2025/26)"),
        "n_players_with_structural_hole": entry(
            int(table.loc[table["club_status"].str.contains("structural"), "player"].nunique()),
            "Pool players with at least one season whose club league is (partly) absent",
        ),
        "n_outfield_qualifying_seasons": entry(
            int((~outfield["below_floor"]).sum()), "Outfield pool player-seasons >= 900 minutes"
        ),
        "n_intl_hole_seasons": entry(
            int((table["absent_intl_tournaments"] != "").sum()),
            "Pool player-seasons with an absent continental championship or WC 2026 knockout",
        ),
    }


# --- driver ------------------------------------------------------------------------------

STAGES: dict[str, tuple[str, Callable[[Path], Summary]]] = {
    "reliability": ("reliability", run_reliability),
    "structure": ("structure", run_structure),
    "clutch": ("clutch", run_clutch),
    "coverage": ("coverage", run_coverage),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="recompute a finished stage")
    args = parser.parse_args()
    for name in STAGES if args.stage == "all" else [args.stage]:
        folder, run = STAGES[name]
        out = RESULTS / folder
        target = out / ("pool_coverage_summary.json" if name == "coverage" else "summary.json")
        if target.exists() and not args.force:
            print(f"{name}: done already ({target}); --force to recompute")
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        print(f"{name}: running", flush=True)
        summary = run(out)
        summary["runtime_seconds"] = entry(round(time.time() - t0, 1), f"Wall time of {name}")
        write_json(target, summary)
        print(f"{name}: wrote {target} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
