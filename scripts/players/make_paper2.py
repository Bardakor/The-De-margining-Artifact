"""Write paper 2's numbers, tables and figures out of the run outputs.

Nothing in ``paper-unmeasured/main.tex`` carries a number typed by hand. This
script reads ``results/players/`` (and, for data counts,
``data/players/processed/`` and ``paper-unmeasured/data/pool.csv``) and writes
``paper-unmeasured/generated/``:

- ``macros.tex``: ``\\renewcommand`` for every macro declared in
  ``macros-defaults.tex`` whose source exists (paper 1's convention: provide,
  then renew);
- ``tab-*.tex``: booktabs tabulars; main.tex supplies the float and caption;
- ``fig-*.pdf``: vector figures drawn at the width main.tex includes them at,
  so their text is set at its true size.

It runs at any stage of the study. A macro, table or figure whose source file
does not exist yet is skipped and listed as pending at the end; the paper
renders it as [pending]. The ranking stage (M5, M6, M8, M9) writes summary.json
files keyed by the macro names themselves, which are picked up as they appear.

Formats: counts with thousands separators, shares, reliabilities and
correlations to two decimals, p-values to three, minutes as whole numbers,
negative numbers in math mode, intervals as [lo, hi].

Usage:
    uv run python scripts/players/make_paper2.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("pdf")  # vector backend; no display needed

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "players"
PROCESSED = ROOT / "data" / "players" / "processed"
PAPER = ROOT / "paper-unmeasured"
GENERATED = PAPER / "generated"
POOL = PAPER / "data" / "pool.csv"

LINEWIDTH_IN = 6.27  # \linewidth of A4 with 1in margins (main.tex geometry)

GROUPS = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "CF")
OUTFIELD = GROUPS[1:]
POOLS = ("CB+DM", "CF+W")

METRIC_LABEL = {
    "goals": "Goals",
    "np_goals": "Non-penalty goals",
    "penalty_goals": "Penalty goals",
    "assists": "Assists",
    "np_ga": "Non-penalty goals + assists",
    "yellow": "Yellow cards",
    "red": "Red cards",
    "on_pitch_gd": "On-pitch goal difference",
    "sb_np_xg": "StatsBomb non-penalty xG",
    "sb_xa": "StatsBomb xA",
}
TARGET_LABEL = {
    "xt_added": "xT added",
    "progressive_passes": "Progressive passes",
    "progressive_carries": "Progressive carries",
    "passes_into_box": "Passes into the box",
    "key_passes": "Key passes",
    "sb_xa": "StatsBomb xA",
    "pressures": "Pressures",
    "tackles_interceptions": "Tackles + interceptions",
}

# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def num(x: object, d: int = 2) -> str:
    """A decimal in text mode; a negative one in math mode, never "-0.00"."""
    v = round(float(x), d)  # type: ignore[arg-type]
    if not np.isfinite(v):
        return "{--}"
    if v == 0:
        v = 0.0
    body = f"{abs(v):.{d}f}"
    return f"$-{body}$" if v < 0 else body


def count(n: object) -> str:
    return f"{round(float(n)):,}".replace(",", "{,}")  # type: ignore[arg-type]


def interval(lo: object, hi: object, d: int = 2) -> str:
    return f"[{num(lo, d)}, {num(hi, d)}]"


def tex(text: object) -> str:
    """Escape LaTeX specials; UTF-8 letters pass through (inputenc utf8, T1)."""
    out = str(text)
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        out = out.replace(char, repl)
    return out


def row(cells: list[str]) -> str:
    # A row opening with "[" would be read as the previous \\'s optional argument.
    first = "{}" if cells and cells[0].startswith("[") else ""
    return "  " + first + " & ".join(cells) + r" \\"


def tabular(colspec: str, header: list[str], body: list[str], *, align: str = "") -> str:
    """A booktabs tabular; ``header`` and ``body`` are finished LaTeX lines;
    ``align="[t]"`` top-aligns panels set side by side."""
    return "\n".join(
        [
            rf"\begin{{tabular}}{align}{{{colspec}}}",
            r"  \toprule",
            *header,
            r"  \midrule",
            *body,
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )


def sort_key(name: str) -> str:
    """Accent-blind sort key for player names."""
    folded = unicodedata.normalize("NFKD", name)
    return "".join(c for c in folded if not unicodedata.combining(c)).casefold()


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def json_values(path: Path) -> dict[str, object]:
    """A results JSON as key -> value, with or without the value/description wrapper."""
    payload = json.loads(path.read_text())
    return {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in payload.items()
    }


def summary(module: str) -> dict[str, object]:
    """A module's summary.json as key -> value (raises if the run has not happened)."""
    return json_values(RESULTS / module / "summary.json")


def descriptions(module: str) -> dict[str, str]:
    payload = json.loads((RESULTS / module / "summary.json").read_text())
    return {k: str(v.get("description", "")) for k, v in payload.items() if isinstance(v, dict)}


def csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)  # type: ignore[arg-type]


def declared_macros() -> list[str]:
    text = (PAPER / "macros-defaults.tex").read_text()
    return re.findall(r"\\providecommand\{\\(\w+)\}", text)


def slots() -> dict[str, float | None]:
    """Every generated/ slot in main.tex -> its width as a share of \\linewidth
    (None for a table)."""
    text = (PAPER / "main.tex").read_text()
    out: dict[str, float | None] = dict.fromkeys(re.findall(r"generated/(tab-[\w-]+)\.tex", text))
    for width, name in re.findall(
        r"\\includegraphics\[width=([\d.]*)\\linewidth\]\{generated/(fig-[\w-]+)\.pdf\}", text
    ):
        out[name] = float(width) if width else 1.0
    return out


def m4_total() -> pd.DataFrame:
    """M4 cross-half estimates of RAPM total, player bootstrap (the primary rows)."""
    shares = csv(RESULTS / "construct_gap" / "shares.csv")
    keep = (
        (shares["component"] == "total")
        & (shares["estimator"] == "cross")
        & (shares["mode"] == "player")
    )
    out = shares[keep].set_index("group")
    return out.assign(r2_full=1.0 - out["share_naive"])  # Equation r2full


def contrast() -> pd.DataFrame:
    return csv(RESULTS / "construct_gap" / "contrast.csv").set_index(["estimator", "mode"])


# --------------------------------------------------------------------------
# macros
# --------------------------------------------------------------------------


def macros_data() -> dict[str, str]:
    """Section 4: panel counts from the coverage table and the processed tables."""
    cov = csv(RESULTS / "coverage" / "tm_coverage.csv")
    tm_stints = csv(PROCESSED / "tm_stints.csv.gz", usecols=["game_id"])
    sb_stints = csv(PROCESSED / "sb_stints.csv.gz", usecols=["game_id"])
    panel = csv(
        PROCESSED / "panel_player_match.csv.gz", usecols=["player_id", "competition_id", "scope"]
    )
    seasons = csv(PROCESSED / "panel_player_season.csv.gz", usecols=["scope", "minutes"])
    with_stints = set(tm_stints["game_id"]) | set(sb_stints["game_id"])
    qualifying = (seasons["scope"] == "all") & (seasons["minutes"] >= 900)
    elo = csv(RESULTS / "elo" / "report.csv").set_index("pool")
    leagues = set(panel.loc[panel["scope"] == "league", "competition_id"])
    in_leagues = cov[cov["competition_id"].isin(leagues)]
    first_season = in_leagues.groupby("competition_id")["season"].min()
    return {
        "NLeaguesPanel": count(len(first_season)),
        "NLeaguesLateStart": count((first_season >= "2024/25").sum()),
        "EloKClub": count(elo.loc["club", "k"]),
        "EloHClub": count(elo.loc["club", "home_advantage"]),
        "EloKNational": count(elo.loc["national", "k"]),
        "EloHNational": count(elo.loc["national", "home_advantage"]),
        "NGames": count(cov["games"].sum()),
        "NGamesWithStints": count(len(with_stints)),
        "NPlayersPanel": count(panel["player_id"].nunique()),
        "NPlayerMatches": count(len(panel)),
        "NPlayerSeasonsQualifying": count(qualifying.sum()),
        "NSbStintGames": count(sb_stints["game_id"].nunique()),
        "CovLastGameDate": tex(cov["last_date"].max()),
        "CovSaudiFirstSeason": tex(cov.loc[cov["competition_id"] == "SA1", "season"].min()),
        "CovMlsFirstSeason": tex(cov.loc[cov["competition_id"] == "MLS1", "season"].min()),
    }


def macros_linking() -> dict[str, str]:
    rates = csv(RESULTS / "linking" / "game_link_rates.csv")
    links = csv(RESULTS / "linking" / "player_links.csv")
    return {
        "NSbMatches": count(rates["matches"].sum() - rates["before_window"].sum()),
        "NSbMatchesLinked": count(rates["linked"].sum()),
        "SbPlayerLinkRate": num(links["tm_player_id"].notna().mean()),
    }


def macros_pool() -> dict[str, str]:
    pool = csv(POOL)
    matching = csv(RESULTS / "linking" / "pool_matching.csv")
    return {
        "NPool": count(len(pool)),
        "NPoolOutfield": count((~pool["is_goalkeeper"]).sum()),
        "NPoolGoalkeepers": count(pool["is_goalkeeper"].sum()),
        "NPoolBdoOnly": count((~pool["in_v1"]).sum()),
        "NPoolMediumConfidence": count((pool["confidence"] == "medium").sum()),
        "NPoolUnmapped": count(matching["tm_player_id"].isna().sum()),
    }


def macros_m1() -> dict[str, str]:
    s = summary("reliability")
    return {
        "RelGoals": num(s["rel_goals_outfield"]),
        "RelNpGoals": num(s["rel_np_goals_outfield"]),
        "RelAssists": num(s["rel_assists_outfield"]),
        "RelNpga": num(s["rel_np_ga_outfield"]),
        "RelOnPitchGd": num(s["rel_on_pitch_gd_outfield"]),
        "NReliabilityUnits": count(s["n_units_np_ga_outfield"]),
        "StabKNpga": count(s["k_np_ga_outfield"]),
        "StabSevenNpga": count(s["minutes07_np_ga_outfield"]),
        "StabKOnPitchGd": count(s["k_on_pitch_gd_outfield"]),
    }


def macros_m2() -> dict[str, str]:
    s = summary("structure")
    team_icc = csv(RESULTS / "structure" / "team_icc.csv").set_index("metric")["icc"]
    out = {
        # The metrics of the eigen-analysis that the paragraph reports on.
        "NMetricsStructure": count(sum(1 for k in s if re.fullmatch(r"eigenvalue_\d+", k))),
        "NComponentsRetained": count(s["pa_n_components"]),
        "ParticipationRatio": num(s["participation_ratio"]),
        "EntropyEffective": num(s["entropy_number"]),
        "CorrGoalsNpGoals": num(s["spearman_goals_np_goals"]),
        "IccNpGoals": num(team_icc["np_goals"]),
        "IccAssists": num(team_icc["assists"]),
        "IccOnPitchGd": num(team_icc["on_pitch_gd"]),
        "VOneInflation": num(s["v1_inflation"]),
        "VOnePcOneShare": num(s["v1_pc1_share"]),
        "VOneGoalsClaimed": num(s["v1_goals_claimed"]),
        "VOneGoalsEffective": num(s["v1_goals_effective"]),
    }
    return out


def macros_m3() -> dict[str, str]:
    s = summary("rapm")
    pool = csv(RESULTS / "rapm" / "pool_rapm.csv")
    outfield = pool[~pool["is_goalkeeper"]]
    out = {
        "NStints": count(s["n_stints"]),
        "NCompetitionSeasons": count(s["n_groups"]),
        "RapmLambda": count(s["lambda"]),
        "NPlayersOwnColumns": count(s["n_players_own"]),
        "RelRapmOffence": num(s["rel_offence_ALL"]),
        "RelRapmDefence": num(s["rel_defence_ALL"]),
        "RelRapmTotal": num(s["rel_total_ALL"]),
        "NPoolRapmExcludesZero": count(outfield["total_excludes_zero"].sum()),
        "RapmEloSpearman": num(s["elo_spearman_total_all"]),
        # Domestic leagues: the run's own empty-stadium comparison.
        "HomeTermEmpty": num(s["home_league_empty"]),
        "HomeTermOther": num(s["home_league_other"]),
        "PoolTopRapm": num(s["pool_top_total"]),
        "PoolTopRapmSe": num(s["pool_top_se"]),
    }
    # The all-player attribution factor is not among M3's outputs yet.
    if "attribution_full_total_ALL" in s:
        a_full = float(s["attribution_full_total_ALL"])  # type: ignore[arg-type]
        out["AttrFullTotal"] = num(a_full)
        out["RelRapmTotalAttributed"] = num(float(s["rel_total_ALL"]) * a_full)  # type: ignore[arg-type]
    if "attribution_half_total_ALL" in s:
        out["AttrHalfTotal"] = num(s["attribution_half_total_ALL"])
    return out


def macros_m4() -> dict[str, str]:
    shares = m4_total()
    con = contrast()
    h3_kinds = csv(RESULTS / "construct_gap" / "h3_summary.csv").set_index("kind")
    h3 = h3_kinds.loc["clipped"]
    sims = csv(RESULTS / "construct_gap" / "h3_sims.csv")
    s = summary("construct_gap")
    groups = shares.loc[list(GROUPS)]
    outside = (groups["r2_true"] < 0) | (groups["r2_true"] > 1)
    r2_full = shares.loc[list(OUTFIELD), "r2_full"]
    return {
        "HThreeSimClipped": num((sims["diff"] != sims["diff_clipped"]).mean()),
        "HThreeSimReachReal": num((sims["diff"] >= h3_kinds.loc["raw", "estimate"]).mean()),
        "RTwoFullMin": num(r2_full.min()),
        "RTwoFullMax": num(r2_full.max()),
        "ShareHigh": num(shares.loc["CB+DM", "share_clipped"]),
        "ShareLow": num(shares.loc["CF+W", "share_clipped"]),
        "ShareContrast": num(con.loc[("cross", "player"), "diff_clipped"]),
        "ShareContrastRaw": num(con.loc[("cross", "player"), "diff"]),
        "ShareContrastParamLo": num(h3["ci_low"]),
        "ShareContrastParamHi": num(h3["ci_high"]),
        "ShareContrastPlayerLo": num(con.loc[("cross", "player"), "ci_low_clipped"]),
        "ShareContrastPlayerHi": num(con.loc[("cross", "player"), "ci_high_clipped"]),
        "ShareContrastJointLo": num(con.loc[("cross", "a7"), "ci_low_clipped"]),
        "ShareContrastJointHi": num(con.loc[("cross", "a7"), "ci_high_clipped"]),
        "ShareContrastWithin": num(con.loc[("within", "player"), "diff_clipped"]),
        "NGroupsClipped": count(outside.sum()),
        "ParamCoverageLoo": num(h3["coverage_loo"]),
        "ParamCoverageMFour": num(h3["coverage_m4"]),
        "VerdictHThree": tex(s["h3_verdict"]),
    }


def macros_m7a() -> dict[str, str]:
    v = csv(RESULTS / "tournaments" / "visibility.csv")
    xt = v[(v["scope"] == "all") & (v["group"] == "ALL")].set_index("target").loc["xt_added"]
    return {
        "NSbUnitsVisibility": count(xt["n_units"]),
        "NSbPlayersVisibility": count(xt["n_players"]),
        "VisXtCross": num(xt["r2_cross"]),
        "VisXtCrossLo": num(xt["r2_cross_lo"]),
        "VisXtCrossHi": num(xt["r2_cross_hi"]),
        "VisXtDisatt": num(xt["r2_disatt"]),
    }


def macros_m7b() -> dict[str, str]:
    s = summary("clutch")
    out = {
        "NPlayersClutch": count(s["h7_n"]),
        "ClutchR": num(s["h7_r"]),
        "ClutchRLo": num(s["h7_ci_low"]),
        "ClutchRHi": num(s["h7_ci_high"]),
        "ClutchP": num(s["h7_p_value"], 3),
        "ClutchRelFirst": num(s["rel_a"]),
        "ClutchRelSecond": num(s["rel_b"]),
        "ClutchPower": num(s["power_at_03"]),
        "ClutchPlaceboR": num(s["placebo_r_mean"]),
        "VerdictHSeven": tex(s["h7_verdict"]),
        "ClutchAttenuation": num(s["attenuation"]),
        "ClutchRDisatt": num(s["h7_r_disattenuated"]),
        "ClutchRDisattLo": num(float(s["h7_ci_low"]) / float(s["attenuation"])),  # type: ignore[arg-type]
        "ClutchRDisattHi": num(float(s["h7_ci_high"]) / float(s["attenuation"])),  # type: ignore[arg-type]
        # Post hoc (d): knockouts of continental and international competitions only.
        "NPlayersClutchMajor": count(s["h7_major_n"]),
        "ClutchMajorR": num(s["h7_major_r"]),
        "ClutchMajorRLo": num(s["h7_major_ci_low"]),
        "ClutchMajorRHi": num(s["h7_major_ci_high"]),
        "ClutchMajorP": num(s["h7_major_p_value"], 3),
    }
    # The run states its test level only in the description of the power.
    level = re.search(r"alpha (\d*\.\d+)", descriptions("clutch")["power_at_03"])
    if level:
        out["HSevenAlpha"] = num(float(level.group(1)))
    return out


def _and_list(names: list[str]) -> str:
    """'A, B and C'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def macros_m8_views() -> dict[str, str]:
    """Names and shares the results prose quotes from M5 and M8 outputs."""
    firsts = csv(RESULTS / "synthesis" / "firsts_by_dimension.csv")
    probs = csv(RESULTS / "multiverse" / "rank_probabilities.csv").set_index("player_id")
    players = synthesis_players("outfield")
    pool = csv(POOL)
    leader = firsts.loc[firsts["m8_p_first"].idxmax(), "player_id"]
    lead = firsts[(firsts["player_id"] == leader) & (firsts["aggregation"] == "all")]
    lead = lead.set_index("family")["share_first"]
    quinones = pool.loc[pool["player"] == "Julián Quiñones", "tm_player_id"].item()
    recognition = recognition_view()
    leaders = synthesis_leaders()
    spec_largest = leaders["share_specification"] >= leaders[
        ["share_sampling", "share_unmeasured"]
    ].max(axis=1)
    return {
        "TopFirstImpact": num(lead["impact"]),
        "TopFirstBox": num(lead["box"]),
        "TopFirstHybrid": num(lead["hybrid"]),
        "NLeaders": count(len(leaders)),
        "NLeadersSpecLargest": count(spec_largest.sum()),
        "QuinonesSpecShare": num(probs.loc[quinones, "p_first"]),
        "StatsViewTopName": tex(players.loc[players["stats_view"].idxmin(), "player"]),
        "ImpactViewTopName": tex(players.loc[players["impact_view"].idxmin(), "player"]),
        "RecognitionTopName": tex(recognition.loc[recognition["share_sum"].idxmax(), "player"]),
        "RecognitionTopNameRaw": tex(recognition.loc[recognition["raw_sum"].idxmax(), "player"]),
    }


def macros_m6_extra() -> dict[str, str]:
    s = summary("recognition")
    return {
        "NBdoDroppedValuation": count(s["count_dropped_no_market_value"]),
        "PerNinetyStatsOls": num(s["per90_reading_ols_stats"]),
        "PerNinetyTeamOls": num(s["per90_reading_ols_team"]),
    }


def macros_assortative() -> dict[str, str]:
    """Post hoc (a) and the bias-corrected shares of (b)."""
    a = summary("construct_gap/assortative")
    attr = csv(RESULTS / "construct_gap" / "assortative" / "attribution.csv")
    attr = attr[(attr["kind"] == "groups") & ~attr["halves"] & attr["group"].isin(OUTFIELD)]
    sims = csv(RESULTS / "construct_gap" / "assortative" / "sims.csv")
    real = csv(RESULTS / "construct_gap" / "assortative" / "contrast_summary.csv")
    real_raw = real.set_index("kind").loc["raw", "estimate"]
    return {
        "AssortSimClipped": num((sims["diff"] != sims["diff_clipped"]).mean()),
        "AssortSimReachReal": num((sims["diff"] >= real_raw).mean()),
        "AssortCoverageRaw": num(a["contrast_raw_coverage_m4"]),
        "AssortAttrLo": num(attr["total"].min()),
        "AssortAttrHi": num(attr["total"].max()),
        "AssortPlanted": num(a["contrast_raw_planted"]),
        "AssortMeanErrorRaw": num(a["contrast_raw_mean_error"]),
        "AssortRawLo": num(a["contrast_raw_ci_low"]),
        "AssortRawHi": num(a["contrast_raw_ci_high"]),
        "AssortClippedLo": num(a["contrast_clipped_ci_low"]),
        "AssortClippedHi": num(a["contrast_clipped_ci_high"]),
        "AssortCoverageClipped": num(a["contrast_clipped_coverage_m4"]),
        "BiasShareCF": num(a["share_bias_corrected_CF"]),
        "BiasShareOtherLo": num(
            min(float(a[f"share_bias_corrected_{g}"]) for g in OUTFIELD if g != "CF")  # type: ignore[arg-type]
        ),
    }


def macros_bias_corrected() -> dict[str, str]:
    s = summary("synthesis/bias_corrected")
    return {
        "BiasSharePairsDecided": num(s["SharePairsDecided"]),
        "BiasTopPlayerVarUnm": num(s["TopPlayerVarUnm"]),
    }


def macros_europe() -> dict[str, str]:
    s = summary("synthesis/europe")
    lost = csv(RESULTS / "synthesis" / "europe" / "lost_seasons.csv")
    names = lost.loc[lost["lost_qualifying_900"], "player"].drop_duplicates().tolist()
    return {
        "NPoolLostSeasons": count(s["n_pool_players_lost_qualifying_900"]),
        "EuropeLostPlayers": _and_list([tex(n) for n in names]),
    }


TEAM_INDICATORS = ("league_title", "ucl_winner", "ucl_finalist", "intl_winner")


def curated_changed_editions() -> int:
    """Candidate-editions with at least one team-success indicator changed by the
    curated honours (the summary's changed_total counts indicators instead)."""
    keys = ["edition", "player"]
    primary = csv(RESULTS / "recognition" / "universe.csv")[[*keys, *TEAM_INDICATORS]]
    curated = csv(RESULTS / "recognition" / "curated" / "universe.csv")[[*keys, *TEAM_INDICATORS]]
    both = primary.merge(curated, on=keys, suffixes=("", "_curated"))
    changed = np.zeros(len(both), dtype=bool)
    for c in TEAM_INDICATORS:
        changed |= (both[c] != both[f"{c}_curated"]).to_numpy()
    return int(changed.sum())


def macros_curated() -> dict[str, str]:
    """Post hoc (e): M6 refitted with curated team honours."""
    s = summary("recognition/curated")
    return {
        # Indicators, as the paragraph counts them (not candidate-editions).
        "CuratedNChanged": count(s["changed_total"]),
        "CuratedShapleyTeamOls": num(s["ShapleyTeamOls"]),
        "CuratedShapleyStatsOls": num(s["ShapleyStatsOls"]),
        "CuratedVerdictHSix": tex(s["VerdictHSix"]),
        "CuratedTeamAboveStatsBoot": num(s["ShapleyTeamAboveStatsBoot"]),
        "CuratedPerNinetyStatsOls": num(s["per90_reading_ols_stats"]),
        "CuratedPerNinetyTeamOls": num(s["per90_reading_ols_team"]),
    }


def macros_rapm_seasons() -> dict[str, str]:
    """Spread of per-season RAPM total within and between players with own columns
    and at least two seasons, as the macro comments define it."""
    own = csv(RESULTS / "rapm" / "rapm_pooled.csv", usecols=["player_id"])["player_id"]
    season = csv(RESULTS / "rapm" / "rapm_season.csv", usecols=["player_id", "total"])
    season = season[season["player_id"].isin(own[own != -1])]
    by_player = season.groupby("player_id")["total"]
    several = by_player.size() >= 2
    return {
        # Three decimals: the within-player SD is a few thousandths of a goal.
        "RapmSeasonWithinSd": num(np.sqrt(by_player.var(ddof=0)[several].mean()), 3),
        "RapmSeasonBetweenSd": num(by_player.mean()[several].std(ddof=1), 3),
    }


def macros_point_ranks() -> dict[str, str]:
    """M5 and H4 counts that need the rank of every pool player under every specification."""
    ranks = csv(RESULTS / "multiverse" / "point_ranks.csv.gz")
    ranks = ranks[ranks["segment"] == "outfield"]
    grid = csv(RESULTS / "multiverse" / "spec_grid.csv").set_index("spec_id")
    impact = ranks[ranks["spec_id"].map(grid["family"]) == "impact"]
    vectors = impact.pivot(index="spec_id", columns="player_id", values="rank").to_numpy()
    median_spec = summary("synthesis")["median_spec_id"]
    top = ranks[(ranks["spec_id"] == median_spec) & (ranks["rank"] <= 10)]["player_id"]
    players = synthesis_players("outfield").set_index("player_id").loc[top]
    ratio = players["var_specification"] / players["var_sampling"]
    distinct = RESULTS / "synthesis" / "primary_distinct" / "summary.json"
    if distinct.exists():
        weighted = json_values(distinct)["n_distinct_outfield_impact"]
        if weighted != len(np.unique(vectors, axis=0)):
            CHECKS.append(f"primary_distinct counts {weighted} distinct impact rankings")
    return {
        "NDistinctRankingsImpact": count(len(np.unique(vectors, axis=0))),
        "HFourNSpecAbove": count((players["var_specification"] > players["var_sampling"]).sum()),
        "HFourMedianRatio": num(ratio.median()),
    }


def macros_leakage() -> dict[str, str]:
    """Post hoc (f): the leakage ratios and the registered leader's P(#1)."""
    out: dict[str, str] = {}
    for run, name in (("leakage_independent", "Indep"), ("leakage_assortative", "Assort")):
        s = summary(f"synthesis/{run}")
        ratios = [float(s[f"lambda_{g}"]) for g in OUTFIELD]  # type: ignore[arg-type]
        out[f"LeakLambda{name}Lo"] = num(min(ratios))
        out[f"LeakLambda{name}Hi"] = num(max(ratios))
    leader = summary("synthesis")["TopPlayerName"]
    players = csv(RESULTS / "synthesis" / "leakage_independent" / "players.csv")
    out["LeakIndepLeaderPFirst"] = num(players.set_index("player").loc[leader, "p_first"])
    return out


LEAGUE_NAME = {
    "GB1": "Premier League", "L1": "Bundesliga", "IT1": "Serie A", "ES1": "La Liga",
    "FR1": "Ligue 1", "NL1": "Eredivisie", "MLS1": "MLS", "BRA1": "Brazilian league",
    "NO1": "Norwegian league", "SA1": "Saudi league", "PO1": "Portuguese league",
    "SE1": "Swedish league", "DK1": "Danish league", "RU1": "Russian league",
    "BE1": "Belgian league", "TS1": "Czech league", "PL1": "Polish league",
    "TR1": "Turkish league", "C1": "Swiss league", "SC1": "Scottish league",
    "A1": "Austrian league", "GR1": "Greek league", "UKR1": "Ukrainian league",
    "RO1": "Romanian league", "RSK1": "South Korean league", "ARG1": "Argentine league",
    "KR1": "Croatian league", "JAP1": "Japanese league", "SER1": "Serbian league",
    "COL1": "Colombian league", "AUS1": "Australian league", "MEX1": "Mexican league",
}  # fmt: skip
NO_ARTICLE = {"Serie A", "La Liga", "Ligue 1", "MLS"}


def league_name(code: str, *, article: bool = False) -> str:
    name = LEAGUE_NAME.get(code, code)
    return f"the {name}" if article and name not in NO_ARTICLE else name


PORTUGUESE_CLUBS = ("Porto", "Benfica", "Sporting CP")


def macros_league_rapm() -> dict[str, str]:
    """Section 8.1--8.3: club means and league means of the RAPM with league means."""
    s = summary("rapm_league_men")
    as_written = csv(RESULTS / "rapm_league" / "league_means.csv")
    above = as_written.loc[as_written["mu"] > 0, "column"].nunique()
    out = {
        "NLeaguesRest": count(len(s["leagues_rest"])),  # type: ignore[arg-type]
        "NLeaguesAboveAsWritten": count(above),
        "LeagueGapPortugal": num(22 * abs(float(s["mu_PO1"]))),  # type: ignore[arg-type]
    }
    for fit, name in (("primary", "Primary"), ("league", "League")):
        values = [float(s[f"club_total_{fit}_{club}"]) for club in PORTUGUESE_CLUBS]  # type: ignore[arg-type]
        out[f"ClubRapm{name}PortugalLo"] = num(min(values), 3)
        out[f"ClubRapm{name}PortugalHi"] = num(max(values), 3)
    # A22's two checks of the design with the man-advantage column.
    if "noise_free_bias_with_men" in s:
        means_only = abs(float(s["noise_free_bias_means_only_with_men"]))  # type: ignore[arg-type]
        if means_only >= 0.0005:
            CHECKS.append(f"planted league means are not recovered exactly: bias {means_only:.4f}")
        shifts = [abs(float(s[f"lambda_{side}_mean_shift_with_men"])) for side in ("low", "high")]  # type: ignore[arg-type]
        largest = max(float(s[f"lambda_{side}_max_shift_with_men"]) for side in ("low", "high"))  # type: ignore[arg-type]
        out |= {
            "LeagueNoiseFreeBiasMen": num(abs(float(s["noise_free_bias_with_men"])), 3),  # type: ignore[arg-type]
            "LeagueLambdaShiftMeanLo": num(min(shifts), 3),
            "LeagueLambdaShiftMeanHi": num(max(shifts), 3),
            "LeagueLambdaShiftMax": num(largest, 3),
            "LeagueLambdaShiftAsWritten": num(
                abs(float(s["lambda_low_mean_shift_as_written"])),  # type: ignore[arg-type]
                3,
            ),
        }
    return out


def macros_league_validation() -> dict[str, str]:
    """Section 8.4: the leagues the text singles out in the validation of the levels."""
    s = summary("league/validation")
    levels = csv(RESULTS / "league" / "validation" / "levels.csv").set_index("league")
    worst = s["disagreements_mu_club"][0]["league"]  # type: ignore[index]
    return {
        "ValMaxGapLeague": tex(league_name(worst, article=True)),
        "ValMaxGapRapm": num(abs(levels.loc[worst, "mu22"])),
        "ValMaxGapClub": num(abs(levels.loc[worst, "club_level"])),
        "ValGapSpainRapm": num(abs(levels.loc["ES1", "mu22"])),
        "ValGapSpainClub": num(abs(levels.loc["ES1", "club_level"])),
    }


# The seven league-adjusted runs of the macro comments (A24's club-results variant included).
LEAGUE_RUNS = (
    "league_adjusted", "league_plain", "league_uefa", "league_mls", "league_club",
    "league_leakage_independent", "league_leakage_assortative",
)  # fmt: skip
CHECKS: list[str] = []  # claims of the text that the results contradict


def league_comparison() -> pd.DataFrame:
    return csv(RESULTS / "synthesis" / "league_comparison.csv").set_index("player")


def macros_league_ranking() -> dict[str, str]:
    """The league-adjusted runs beside the registered mixture."""
    primary = summary("synthesis")
    runs = {run: summary(f"synthesis/{run}") for run in LEAGUE_RUNS}
    variants = sorted((RESULTS / "synthesis").glob("*/summary.json"))
    every = [primary, *(json_values(path) for path in variants)]
    table = league_comparison()
    leader = str(primary["TopPlayerName"])
    adjusted = "league_adjusted"

    def best(column: str, n: int, *, ascending: bool = True) -> str:
        order = table.sort_values([column, f"{adjusted}_p_first"], ascending=[ascending, False])
        return _and_list([tex(name) for name in order.index[:n]])

    leaders = {str(s["TopPlayerName"]) for s in runs.values()}
    if len(leaders) > 1:
        CHECKS.append(f"the league runs do not share one leader: {sorted(leaders)}")
    if runs[adjusted]["SecondPlayerName"] != leader:
        CHECKS.append(
            f"{leader} is not second in league_adjusted: {runs[adjusted]['SecondPlayerName']} is"
        )
    p_first = [float(s["TopPlayerPFirst"]) for s in runs.values()]  # type: ignore[arg-type]
    decided = [float(s["SharePairsDecided"]) for s in runs.values()]  # type: ignore[arg-type]
    return {
        "LeagueVariantsPFirstLo": num(min(p_first)),
        "LeagueVariantsPFirstHi": num(max(p_first)),
        "LeagueVariantsDecidedLo": num(min(decided)),
        "LeagueVariantsDecidedHi": num(max(decided)),
        "AllVariantsMaxPFirst": num(max(float(s["TopPlayerPFirst"]) for s in every)),  # type: ignore[arg-type]
        "AllVariantsMaxDecided": num(max(float(s["SharePairsDecided"]) for s in every)),  # type: ignore[arg-type]
        "LeagueSecondPFirst": num(table.loc[leader, f"{adjusted}_p_first"]),
        "LeagueMedianTopThree": best(f"{adjusted}_median_rank", 3),
        "LeagueQuinonesPFirst": num(table.loc["Julián Quiñones", f"{adjusted}_p_first"]),
        "MessiMedianRank": count(table.loc["Lionel Messi", "primary_median_rank"]),
        "LeagueMessiMedianRank": count(table.loc["Lionel Messi", f"{adjusted}_median_rank"]),
        "LeagueMlsMessiMedianRank": count(table.loc["Lionel Messi", "league_mls_median_rank"]),
        "LeagueLeakIndepLeaderPFirst": num(table.loc[leader, "league_leakage_independent_p_first"]),
        "LeagueLeakAssortLeaderPFirst": num(
            table.loc[leader, "league_leakage_assortative_p_first"]
        ),
    }


def macros_league_club() -> dict[str, str]:
    """A24: the league-adjusted run with league levels from club results, beside the
    league-adjusted run with RAPM's league means."""
    club = summary("synthesis/league_club")
    adjusted_leader = summary("synthesis/league_adjusted")["TopPlayerName"]
    if club["TopPlayerName"] != adjusted_leader:
        CHECKS.append(f"league_club is led by {club['TopPlayerName']}, not {adjusted_leader}")
    spain = csv(RESULTS / "league" / "validation" / "levels.csv").set_index("league").loc["ES1"]
    table = league_comparison()
    orders = [
        table.sort_values(f"{run}_p_first", ascending=False).index
        for run in ("league_adjusted", "league_club")
    ]
    same = 0
    while same < len(table) and orders[0][same] == orders[1][same]:
        same += 1
    shares = csv(RESULTS / "rapm_league_men" / "league_shares.csv")
    la_liga = shares.loc[(shares["league"] == "ES1") & (shares["share"] > 0.5), "player_id"]
    in_la_liga = table["player_id"].isin(la_liga)
    gain = table["league_adjusted_median_rank"] - table["league_club_median_rank"]
    if gain[in_la_liga].min() < 1:
        CHECKS.append(f"a La Liga player does not gain in league_club: {gain[in_la_liga].min()}")
    rho = table["league_adjusted_median_rank"].corr(
        table["league_club_median_rank"], method="spearman"
    )
    return {
        "LeagueClubSpainGapZ": num(
            (spain["club_level"] - spain["mu22"]) / spain["club_level_minus_mu22_se"], 1
        ),
        "LeagueClubNLeagues": count(len(club["club_level_leagues"])),  # type: ignore[arg-type]
        "LeagueClubTopPFirst": num(club["TopPlayerPFirst"]),
        "LeagueClubTopSameOrder": count(same),
        "LeagueClubNLaLiga": count(in_la_liga.sum()),
        "LeagueClubLaLigaGainLo": count(gain[in_la_liga].min()),
        "LeagueClubLaLigaGainHi": count(gain[in_la_liga].max()),
        "LeagueClubOthersMaxMove": count(gain[~in_la_liga].abs().max()),
        "LeagueClubSpearmanMedianRank": num(rho, 3),
        "LeagueClubSharePairsDecided": num(club["SharePairsDecided"]),
    }


# The seven runs of the corrected block's conventions, in the column order of
# tab-corrected-sensitivity; the first four are the corrected mixture and its sensitivities.
PLACE_RUNS = (
    "corrected", "corrected_equal_weights", "corrected_no_contamination",
    "corrected_assortative", "league_adjusted", "primary_distinct", "primary",
)  # fmt: skip
CORRECTED_RUNS = PLACE_RUNS[:4]
ATTACKING_GROUPS = ("CF", "W", "AM")
FAMILIES = {"box": "Box", "box_gd": "Box + GD", "hybrid": "Hybrid", "impact": "Impact"}


def run_summary(run: str) -> dict[str, object]:
    return summary("synthesis" if run == "primary" else f"synthesis/{run}")


def corrected_comparison() -> pd.DataFrame:
    """The outfield pool under the seven runs, one row per player in the order of the
    corrected run, with his place in each run: the position when the players are
    sorted by median rank, then mean rank."""
    table = csv(RESULTS / "synthesis" / "corrected_comparison.csv").set_index("player")
    for run in PLACE_RUNS:
        order = table.sort_values([f"{run}_median_rank", f"{run}_mean_rank"]).index
        table[f"{run}_place"] = pd.Series(range(1, len(order) + 1), index=order)
    return table.sort_values("corrected_place")


def macros_corrected() -> dict[str, str]:
    """Sections 8.3 and 9 (A25): the corrected mixture, its weights and its sensitivities."""
    table = corrected_comparison()
    s = summary("synthesis/corrected")
    distinct = summary("synthesis/primary_distinct")
    registered = summary("synthesis")
    runs = {run: summary(f"synthesis/{run}") for run in CORRECTED_RUNS}
    leader = str(registered["TopPlayerName"])
    top5 = list(table.index[:5])
    registered_top5 = list(table.sort_values("primary_place").index[:5])
    new = [name for name in top5 if name not in registered_top5]
    always = [
        name for name in top5 if all(table.loc[name, f"{run}_place"] <= 5 for run in PLACE_RUNS)
    ]
    pairwise = csv(
        RESULTS / "synthesis" / "corrected" / "pairwise_outfield.csv", index_col="player_id"
    )
    pairwise.columns = pairwise.columns.astype(int)
    ids = table["player_id"]
    above = pairwise.loc[ids.iloc[0], ids.iloc[1:5]]
    other = table[~table["position_group"].isin(ATTACKING_GROUPS)].iloc[0]
    rodri = table.loc["Rodri"]
    grid = csv(RESULTS / "multiverse" / "spec_grid.csv")
    per_family = (grid["family"] == "impact").sum()
    p_first = {run: float(r["TopPlayerPFirst"]) for run, r in runs.items()}  # type: ignore[arg-type]
    decided = {run: float(r["SharePairsDecided"]) for run, r in runs.items()}  # type: ignore[arg-type]
    shares = [num(s[f"weight_share_outfield_{family}"]) for family in ("box", "box_gd", "hybrid")]

    def place(name: str, run: str = "corrected") -> str:
        return count(table.loc[name, f"{run}_place"])

    def first_place(group: str) -> str:
        return count(table.loc[table["position_group"] == group, "corrected_place"].min())

    def rho(a: str, b: str) -> str:
        return num(table[a].corr(table[b], method="spearman"))

    best_dm = table.loc[table["position_group"] == "DM", "corrected_place"].min()
    same_five = all(
        set(table.sort_values(f"{run}_place").index[:5]) == set(top5) for run in CORRECTED_RUNS
    )
    claims = (
        (distinct["TopPlayerName"] == registered["SecondPlayerName"],
         f"primary_distinct is led by {distinct['TopPlayerName']}"),
        (len(set(shares)) == 1, f"the box, box+GD and hybrid weight shares differ: {shares}"),
        (s["TopPlayerName"] == table.index[0],
         f"{s['TopPlayerName']} has the highest P(#1) but place 1 is {table.index[0]}"),
        (len(new) == 1, f"{5 - len(new)} of the corrected first five are in the registered five"),
        (len({str(r["TopPlayerName"]) for r in runs.values()}) == 1,
         "the four corrected runs do not share one leader"),
        (same_five, "the first five differ between the corrected run and its sensitivities"),
        (min(decided, key=decided.__getitem__) == "corrected_equal_weights",
         "the equal-weights run does not have the smallest share of decided pairs"),
        (max(decided, key=decided.__getitem__) == "corrected_no_contamination",
         "the run without contamination does not have the largest share of decided pairs"),
        (rodri["corrected_place"] == best_dm, "Rodri is not the best-placed DM, corrected run"),
        (other["corrected_place"] > 20,
         f"{other.name} ({other['position_group']}) is in the first twenty places"),
    )  # fmt: skip
    CHECKS.extend(message for holds, message in claims if not holds)

    out = {
        # 8.3, repetition in the registered factorial
        "ImpactRepeatMean": num(per_family / float(distinct["n_distinct_outfield_impact"]), 1),  # type: ignore[arg-type]
        "PrimaryDistinctWeightImpact": num(distinct["weight_share_outfield_impact"]),
        "PrimaryDistinctTopName": tex(distinct["TopPlayerName"]),
        "PrimaryDistinctTopPFirst": num(distinct["TopPlayerPFirst"]),
        "PrimaryDistinctLeaderPFirst": num(table.loc[leader, "primary_distinct_p_first"]),
        # 9.1, weights of the corrected mixture
        "CorrectedNDistinctOutfield": count(s["n_distinct_outfield"]),
        "CorrectedNDistinctImpact": count(s["n_distinct_outfield_impact"]),
        "CorrectedWeightImpact": num(s["weight_share_outfield_impact"]),
        "CorrectedWeightOtherFamily": shares[0],
        "CorrectedWeightBoxFamilies": num(1 - float(s["weight_share_outfield_impact"])),  # type: ignore[arg-type]
        "CorrectedNDistinctGoalkeeper": count(s["n_distinct_goalkeeper"]),
        "CorrectedWeightGkImpact": num(s["weight_share_goalkeeper_impact"]),
        # 9.4, the corrected ranking
        "CorrectedTopFive": _and_list([tex(name) for name in top5]),
        "CorrectedNextFive": _and_list([tex(name) for name in table.index[5:10]]),
        "CorrectedTopFiveMedianLo": count(table["corrected_median_rank"].iloc[:5].min()),
        "CorrectedTopFiveMedianHi": count(table["corrected_median_rank"].iloc[:5].max()),
        "CorrectedTopName": tex(s["TopPlayerName"]),
        "CorrectedTopPFirst": num(s["TopPlayerPFirst"]),
        "CorrectedTopPTopThree": num(s["TopPlayerPTopThree"]),
        "CorrectedTopPTopTen": num(s["TopPlayerPTopTen"]),
        "CorrectedTopMedianRank": count(s["TopPlayerMedianRank"]),
        "CorrectedTopRankLo": count(s["TopPlayerRankLo"]),
        "CorrectedTopRankHi": count(s["TopPlayerRankHi"]),
        "CorrectedSecondName": tex(s["SecondPlayerName"]),
        "CorrectedSecondPFirst": num(s["SecondPlayerPFirst"]),
        "CorrectedTopFivePFirstSum": num(table["corrected_p_first"].iloc[:5].sum()),
        "CorrectedTopPairLo": num(above.min()),
        "CorrectedTopPairHi": num(above.max()),
        "CorrectedNPairsDecided": count(s["NPairsDecided"]),
        "CorrectedSharePairsDecided": num(s["SharePairsDecided"]),
        "CorrectedTopVarSpec": num(s["TopPlayerVarSpec"]),
        "CorrectedTopVarSamp": num(s["TopPlayerVarSamp"]),
        "CorrectedTopVarUnm": num(s["TopPlayerVarUnm"]),
        "CorrectedHFourRatio": num(s["HFourRatio"]),
        "CorrectedRegLeaderPlace": place(leader),
        "CorrectedRegLeaderPFirst": num(table.loc[leader, "corrected_p_first"]),
        "CorrectedQuinonesPlace": place("Julián Quiñones"),
        "CorrectedGyokeresPlace": place("Viktor Gyökeres"),
        "CorrectedMessiPlace": place("Lionel Messi"),
        "RegisteredTopFive": _and_list([tex(name) for name in registered_top5]),
        "CorrectedTopFiveInRegistered": count(5 - len(new)),
        "CorrectedSpearmanRegistered": rho("corrected_median_rank", "primary_median_rank"),
        "RegisteredLeaderPlace": place(leader, "primary"),
        "RegisteredQuinonesPlace": place("Julián Quiñones", "primary"),
        "CorrectedTopGoalkeeperName": tex(s["TopGoalkeeperName"]),
        "CorrectedTopGoalkeeperPFirst": num(s["TopGoalkeeperPFirst"]),
        # 9.5, sensitivities
        "CorrectedSensPFirstLo": num(min(p_first.values())),
        "CorrectedSensPFirstHi": num(max(p_first.values())),
        "CorrectedSensDecidedLo": num(min(decided.values())),
        "CorrectedSensDecidedHi": num(max(decided.values())),
        "CorrectedAlwaysTopFive": _and_list([tex(name) for name in always]),
        "CorrectedRodriPlace": place("Rodri"),
        "CorrectedEqualRodriPlace": place("Rodri", "corrected_equal_weights"),
        "CorrectedEqualRegLeaderPlace": place(leader, "corrected_equal_weights"),
        "LeagueRegLeaderPlace": place(leader, "league_adjusted"),
        # 9.6, scope of the corrected mixture
        "CorrectedSpearmanStatsView": rho("corrected_median_rank", "corrected_stats_view"),
        "CorrectedSpearmanImpactView": rho("corrected_median_rank", "corrected_impact_view"),
        "RegisteredSpearmanStatsView": rho("primary_median_rank", "primary_stats_view"),
        "RegisteredSpearmanImpactView": rho("primary_median_rank", "primary_impact_view"),
        "CorrectedFirstOtherName": tex(other.name),
        "CorrectedFirstOtherPlace": count(other["corrected_place"]),
        "CorrectedFirstFBPlace": first_place("FB"),
        "CorrectedFirstCBPlace": first_place("CB"),
        "CorrectedRodriRankLo": count(rodri["corrected_rank_lo"]),
        "CorrectedRodriRankHi": count(rodri["corrected_rank_hi"]),
        "CorrectedRodriPFirst": num(rodri["corrected_p_first"]),
        # 11, comparison with V1
        "CorrectedVOneSpearman": num(s["VOneSpearman"]),
    }
    if len(new) == 1:
        out["CorrectedTopFiveNewName"] = tex(new[0])
        out["CorrectedTopFiveNewRegisteredPlace"] = place(new[0], "primary")
    return out


def on_off() -> pd.DataFrame:
    """Pooled on/off figures of the outfield pool players, one row per player."""
    table = csv(RESULTS / "rapm" / "pool_on_off.csv")
    return table[table["club"] == "all clubs"].set_index("player")


def macros_on_off() -> dict[str, str]:
    table = on_off()
    leader = table.loc[str(summary("synthesis")["TopPlayerName"])]
    return {
        "OnOffLeaderOn": num(leader["gd_on_p90"]),
        "OnOffLeaderOff": num(leader["gd_off_p90"]),
        "OnOffLeaderDiff": num(leader["on_minus_off"]),
        "OnOffLeaderShare": num(leader["minutes_share"]),
        "OnOffSalahDiff": num(table.loc["Mohamed Salah", "on_minus_off"]),
        "OnOffSalahShare": num(table.loc["Mohamed Salah", "minutes_share"]),
        "OnOffMessiDiff": num(table.loc["Lionel Messi", "on_minus_off"]),
    }


def num3(x: object) -> str:
    return num(x, 3)


def percent(x: object) -> str:
    """A share in [0, 1] as a percentage with one decimal."""
    return num(100 * float(x), 1) + r"\%"  # type: ignore[arg-type]


# Macros that are one key of one results JSON: file -> (macro, key, format). A file or
# key that does not exist yet leaves its macros pending.
KEYED: dict[str, tuple[tuple[str, str, Callable[[object], str]], ...]] = {
    "rapm/cv_season.json": (("RapmLambdaSeason", "lam_season", count),),
    "coverage/pool_coverage_summary.json": (
        ("NPoolStructuralHole", "n_players_with_structural_hole", count),
        ("NPoolIntlHoleSeasons", "n_intl_hole_seasons", count),
    ),
    "construct_gap/summary.json": (
        ("ShareGK", "m4_cross_total_GK_share_clipped", num),
        ("ShareGKLo", "m4_cross_total_GK_ci_low_clipped", num),
        ("ShareGKHi", "m4_cross_total_GK_ci_high_clipped", num),
        ("NHThreeHigh", "m4_cross_total_CB+DM_n", count),
        ("NHThreeHighJoint", "m4_cross_total_CB+DM_a7_n", count),
        ("ShareContrastParamRawLo", "h3_param_raw_ci_low", num),
        ("ShareContrastParamRawHi", "h3_param_raw_ci_high", num),
        ("HThreePlanted", "h3_param_raw_planted", num),
        ("HThreePlantedWeighted", "h3_param_raw_planted_weighted", num),
        ("HThreeMeanError", "h3_param_raw_mean_error", num),
    ),
    "synthesis/summary.json": (("KendallWRaw", "KendallWRawPoints", num),),
    "rapm/summary.json": (
        ("OnOffSpearmanDiff", "on_off_spearman_rapm_on_minus_off", num),
        ("OnOffSpearmanShare", "on_off_spearman_rapm_minutes_share", num),
    ),
    "synthesis/leakage_independent/summary.json": (
        ("LeakIndepTopName", "TopPlayerName", tex),
        ("LeakIndepTopPFirst", "TopPlayerPFirst", num),
        ("LeakIndepSharePairsDecided", "SharePairsDecided", num),
    ),
    "synthesis/leakage_assortative/summary.json": (
        ("LeakAssortTopName", "TopPlayerName", tex),
        ("LeakAssortTopPFirst", "TopPlayerPFirst", num),
        ("LeakAssortSharePairsDecided", "SharePairsDecided", num),
    ),
    "rapm_league_men/summary.json": (
        ("ClubRapmPrimaryRealMadrid", "club_total_primary_Real Madrid", num3),
        ("ClubRapmLeagueRealMadrid", "club_total_league_Real Madrid", num3),
        ("NLeaguesOwnMean", "n_leagues_own", count),
        ("UnequalMinutesShare", "unequal_minutes_share", percent),
        ("LeagueNoiseFreeBias", "noise_free_bias_as_written", num3),
        ("LeagueMen", "men", num),
        ("LeagueMenSe", "men_se", num),
        ("LeagueMuGermany", "mu_L1", num3),
        ("LeagueMuPortugal", "mu_PO1", num3),
        ("LeagueRapmSpearmanAll", "spearman_total_all", num),
        ("LeagueRapmSpearmanPool", "spearman_total_pool", num),
    ),
    "league/summary.json": (
        ("NLeaguesOwnFactor", "factors_leagues_own", count),
        ("NLeaguesPooledFactor", "factors_leagues_pooled", count),
        ("FactorShiftDummyMaxSe", "factors_context_goals_shift_dummy_max_se", num),
        ("FactorShiftDroppedMaxSe", "factors_context_goals_shift_dropped_max_se", num),
        ("FactorGoalsSpain", "factors_context_goals_multiplier_ES1", num),
        ("FactorGoalsGermany", "factors_context_goals_multiplier_L1", num),
        ("FactorGoalsPortugal", "factors_context_goals_multiplier_PO1", num),
        ("FactorGoalsMls", "factors_context_goals_multiplier_MLS1", num),
        ("FactorGoalsSaudi", "factors_context_goals_multiplier_SA1", num),
        ("FactorGoalsPortugalPlain", "factors_plain_goals_multiplier_PO1", num),
        ("FactorTeamEloGoals", "factors_context_goals_team_elo", num3),
        ("FactorAssistsSpain", "factors_context_assists_multiplier_ES1", num),
        ("FactorAssistsItaly", "factors_context_assists_multiplier_IT1", num),
        ("FactorAssistsGermany", "factors_context_assists_multiplier_L1", num),
    ),
    "league/validation/summary.json": (
        ("ValNLeaguesClub", "n_mu_club", count),
        ("ValPearsonClub", "pearson_mu_club", num),
        ("ValSpearmanClub", "spearman_mu_club", num),
        ("ValSlopeClub", "slope_mu_club", num),
        ("ValNTwoSeClub", "n_2se_mu_club", count),
        ("ValNLeaguesUefa", "n_mu_uefa", count),
        ("ValSpearmanUefa", "spearman_mu_uefa", num),
        ("ValSpearmanClubUefa", "spearman_club_uefa", num),
        ("ValNFactorGoals", "n_multiplier_goals", count),
        ("ValPearsonFactorGoals", "pearson_multiplier_goals", num),
        ("ValSpearmanFactorGoals", "spearman_multiplier_goals", num),
        ("ValNTwoSeFactorGoals", "n_2se_multiplier_goals", count),
        ("ValNTwoSeFactorNpGoals", "n_2se_multiplier_np_goals", count),
    ),
    "synthesis/league_mls/summary.json": (
        ("NMlsFillRows", "count_mls_fill_rows", count),
        ("NMlsFillPlayers", "count_mls_fill_players", count),
    ),
    "synthesis/league_adjusted/summary.json": (
        ("LeagueTopName", "TopPlayerName", tex),
        ("LeagueTopPFirst", "TopPlayerPFirst", num),
        ("LeagueTopMedianRank", "TopPlayerMedianRank", count),
        ("LeagueTopRankLo", "TopPlayerRankLo", count),
        ("LeagueTopRankHi", "TopPlayerRankHi", count),
        ("LeagueSharePairsDecided", "SharePairsDecided", num),
    ),
    "synthesis/league_leakage_independent/summary.json": (
        ("LeagueLeakIndepTopName", "TopPlayerName", tex),
        ("LeagueLeakIndepTopPFirst", "TopPlayerPFirst", num),
        ("LeagueLeakIndepSecondName", "SecondPlayerName", tex),
        ("LeagueLeakIndepSecondPFirst", "SecondPlayerPFirst", num),
        ("LeagueLeakIndepSharePairsDecided", "SharePairsDecided", num),
    ),
    "synthesis/league_leakage_assortative/summary.json": (
        ("LeagueLeakAssortTopPFirst", "TopPlayerPFirst", num),
        ("LeagueLeakAssortSharePairsDecided", "SharePairsDecided", num),
    ),
}


# The ranking stage's primary runs. Robustness runs (subfolders, rapm_league*, league)
# reuse the same key names and are never read into a primary macro.
NAMED_MODULES = ("multiverse", "recognition", "synthesis", "recovery")


def macros_named(declared: set[str], done: set[str]) -> dict[str, str]:
    """Keys of the ranking stage's summary.json files that are macro names themselves."""
    out: dict[str, str] = {}
    for module in NAMED_MODULES:
        if not (RESULTS / module / "summary.json").exists():
            continue
        for key, value in summary(module).items():
            if key not in declared or key in done or key in out:
                continue
            if isinstance(value, str):
                out[key] = tex(value)
            elif isinstance(value, bool):
                out[key] = "yes" if value else "no"
            elif isinstance(value, int | float):
                is_count = key.startswith("N") or "Rank" in key or "Marginal" in key
                if isinstance(value, int) or (is_count and float(value).is_integer()):
                    out[key] = count(value)
                else:
                    out[key] = num(value)
    return out


def macro_block(macros: dict[str, str]) -> str:
    declare = "".join(f"\\providecommand{{\\{k}}}{{}}\n" for k in sorted(macros))
    define = "".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
    return declare + "% Generated by scripts/players/make_paper2.py --- do not edit.\n" + define


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _season_short(season: str) -> str:
    return season[2:]  # "2019/20" -> "19/20"


def table_coverage() -> str:
    """Per competition (the 323 competition-seasons do not fit one float): games,
    games with stints (Transfermarkt and StatsBomb), player-games of the panel and
    player-games known only from Transfermarkt stints."""
    cov = csv(RESULTS / "coverage" / "tm_coverage.csv")
    relabels = csv(RESULTS / "linking" / "season_relabels.csv")
    relabel = dict(
        zip(
            zip(relabels["competition_id"], relabels["season"], strict=True),
            relabels["study_season"],
            strict=True,
        )
    )
    games = csv(PROCESSED / "tm_games.csv.gz", usecols=["game_id", "competition_id"])
    sb = csv(PROCESSED / "sb_stints.csv.gz", usecols=["game_id"]).drop_duplicates()
    panel = csv(PROCESSED / "panel_player_match.csv.gz", usecols=["competition_id"])

    cov["study_season"] = [
        relabel.get((c, s), s) for c, s in zip(cov["competition_id"], cov["season"], strict=True)
    ]
    sb_games = sb.merge(games, on="game_id").groupby("competition_id").size()
    per = cov.groupby("competition_id").agg(
        first=("study_season", "min"),
        last=("study_season", "max"),
        n_seasons=("study_season", "nunique"),
        games=("games", "sum"),
        tm_stints=("games_with_stints", "sum"),
        only=("player_games_only_from_stints", "sum"),
    )
    per["sb_stints"] = sb_games.reindex(per.index).fillna(0).astype(int)
    per["with_stints"] = per["tm_stints"] + per["sb_stints"]
    per["player_games"] = panel.groupby("competition_id").size().reindex(per.index).fillna(0)
    per = per.sort_index()

    def line(code: str, r: pd.Series) -> str:
        span = _season_short(r["first"])
        if r["last"] != r["first"]:
            span += "--" + _season_short(r["last"])
        mark = r"$^\dagger$" if r["sb_stints"] > 0 else ""
        return row(
            [
                tex(code) + mark,
                f"{span} ({r['n_seasons']})",
                count(r["games"]),
                count(r["with_stints"]),
                count(r["player_games"]),
                count(r["only"]),
            ]
        )

    header = [
        row(["", "Seasons", "Games", "With", "Player-", "Only"]),
        row(["Comp.", "(number)", "", "stints", "games", "stints"]),
    ]
    lines = [line(code, r) for code, r in per.iterrows()]
    half = (len(lines) + 2) // 2
    total = row(
        [
            r"\textit{All}",
            "",
            count(per["games"].sum()),
            count(per["with_stints"].sum()),
            count(per["player_games"].sum()),
            count(per["only"].sum()),
        ]
    )
    colspec = "@{}lrrrrr@{}"
    left = tabular(colspec, header, lines[:half], align="[t]")
    right = tabular(colspec, header, [*lines[half:], r"  \midrule", total], align="[t]")
    note = (
        r"\par\smallskip\raggedright Transfermarkt competition codes; seasons as study "
        r"seasons. $^\dagger$~Includes games whose stints come from StatsBomb ("
        + count(per["sb_stints"].sum())
        + r" in all). Player-games: panel rows with at least one minute. Only stints: "
        r"Transfermarkt player-games known only from stints."
    )
    return (
        "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        + left
        + "\\hfill\n"
        + right
        + note
        + "\\par}\n"
    )


def table_reliability() -> str:
    rel = csv(RESULTS / "reliability" / "reliability.csv")
    primary = rel[(rel["group"] == "OUTFIELD") & ~rel["exploratory"]]
    explor = rel[rel["exploratory"]]

    def line(r: pd.Series) -> str:
        return row(
            [
                tex(METRIC_LABEL.get(r["metric"], r["metric"])),
                num(r["reliability"]),
                num(r["reliability_sd"], 3),
                count(r["n_units"]),
                count(r["k"]),
                count(r["minutes_07"]),
            ]
        )

    header = [
        row(["", "", "", "Player-", r"$k$ (min.\ to", "Min.\\ to"]),
        row(["Metric", "Reliability", "SD", "seasons", "0.5)", "0.7"]),
    ]
    body = [line(r) for _, r in primary.iterrows()]
    if len(explor):
        body += [
            r"  \midrule",
            r"  \multicolumn{6}{@{}l}{\textit{StatsBomb-linked matches (exploratory)}} \\",
        ]
        body += [line(r) for _, r in explor.iterrows()]
    return tabular("@{}lrrrrr@{}", header, body) + "\n"


def table_composite() -> str:
    comp = csv(RESULTS / "structure" / "composite_v1.csv").set_index("metric")
    s = summary("structure")
    body = [
        row(
            [
                tex(METRIC_LABEL[m]),
                num(r["weight"]),
                num(r["claimed"]),
                num(r["effective"]),
            ]
        )
        for m, r in comp.iterrows()
    ]
    body += [
        r"  \midrule",
        row(
            [
                "Goals and non-penalty goals",
                num(comp.loc[["goals", "np_goals"], "weight"].sum()),
                num(s["v1_goals_claimed"]),
                num(s["v1_goals_effective"]),
            ]
        ),
        r"  \midrule",
        row([r"Variance inflation $w^\top R w / w^\top w$", "", "", num(s["v1_inflation"])]),
        row(["First principal component's share", "", "", num(s["v1_pc1_share"])]),
    ]
    header = [row(["Metric", "Weight", "Claimed $c_k$", "Effective $e_k$"])]
    note = (
        r"\par\smallskip{\footnotesize Weights renormalised over the four box-score "
        r"metrics of the composite.\par}"
    )
    return tabular("@{}lrrr@{}", header, body) + note + "\n"


def table_icc() -> str:
    icc = csv(RESULTS / "structure" / "team_icc.csv")
    level = re.search(r"(\d+)% club-bootstrap", descriptions("structure")["icc_goals_lo"])
    label = f"{level.group(1)}\\% interval" if level else "Interval"
    body = [
        row([tex(METRIC_LABEL[r["metric"]]), num(r["icc"]), interval(r["lo"], r["hi"])])
        for _, r in icc.iterrows()
    ]
    return tabular("@{}lrr@{}", [row(["Metric", "ICC(1)", label])], body) + "\n"


def table_rapm_reliability() -> str:
    rel = csv(RESULTS / "rapm" / "reliability.csv").set_index("group")
    attr = csv(RESULTS / "rapm" / "attribution.csv")
    attr = attr[attr["kind"] == "groups"]
    full = attr[~attr["halves"]].set_index("group")
    half = attr[attr["halves"]].set_index("group")
    s = summary("rapm")
    body = []
    for g in GROUPS:
        a_full = full.loc[g, "total"]
        body.append(
            row(
                [
                    g,
                    count(full.loc[g, "n_players"]),
                    count(rel.loc[g, "n_players"]),
                    num(rel.loc[g, "offence"]),
                    num(rel.loc[g, "defence"]),
                    num(rel.loc[g, "total"]),
                    num(a_full),
                    num(half.loc[g, "total"]),
                    num(rel.loc[g, "total"] * a_full),
                ]
            )
        )
    a_all = s.get("attribution_full_total_ALL")
    h_all = s.get("attribution_half_total_ALL")
    body += [
        r"  \midrule",
        row(
            [
                "All",
                count(s["n_players_own"]),
                count(rel.loc["ALL", "n_players"]),
                num(rel.loc["ALL", "offence"]),
                num(rel.loc["ALL", "defence"]),
                num(rel.loc["ALL", "total"]),
                num(a_all) if a_all is not None else "{--}",
                num(h_all) if h_all is not None else "{--}",
                num(rel.loc["ALL", "total"] * float(a_all)) if a_all is not None else "{--}",  # type: ignore[arg-type]
            ]
        ),
    ]
    header = [
        r"  & \multicolumn{2}{c}{Players} & \multicolumn{3}{c}{Split-half reliability}"
        r" & \multicolumn{2}{c}{Attribution (total)} & Attributed \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
        row(
            [
                "Group",
                "Own col.",
                "Per split",
                "Offence",
                "Defence",
                "Total",
                "$A_g$",
                "$A^{1/2}_g$",
                "reliability",
            ]
        ),
    ]
    note = (
        r"\par\smallskip{\footnotesize Per split: mean number of players with at least "
        + count(s["rel_half_minutes"])
        + r" minutes in each half. Attributed reliability: split-half reliability of "
        r"the total times $A_g$.\par}"
    )
    return tabular("@{}lrrrrrrrr@{}", header, body) + note + "\n"


def table_unmeasured() -> str:
    shares = m4_total()

    def line(g: str) -> str:
        r = shares.loc[g]
        return row(
            [
                tex(g),
                count(r["n"]),
                num(r["r2_full"]),
                num(r["r2_obs"]),
                num(r["rel_pred"]),
                num(r["rel_rapm"]),
                num(r["r2_true"]),
                num(r["r2_true_clipped"]),
                num(r["share_clipped"]),
                interval(r["ci_low_clipped"], r["ci_high_clipped"]),
            ]
        )

    header = [
        r"  & & & & & & \multicolumn{2}{c}{$R^2_{\text{true}}$} & & \\",
        r"  \cmidrule(lr){7-8}",
        row(
            [
                "Group",
                "Players",
                r"$R^2_{\text{full}}$",
                r"$R^2_{\text{obs}}$",
                r"$\rho_P$",
                r"$\rho_\beta$",
                "raw",
                "clipped",
                r"$\sigma^{\text{clip}}_g$",
                r"90\% interval",
            ]
        ),
    ]
    body = [line(g) for g in GROUPS] + [r"  \midrule"] + [line(g) for g in POOLS]
    return (
        "{\\setlength{\\tabcolsep}{4pt}\n" + tabular("@{}lrrrrrrrrc@{}", header, body) + "\\par}\n"
    )


def table_h3() -> str:
    con = contrast()
    h3 = csv(RESULTS / "construct_gap" / "h3_summary.csv").set_index("kind")
    s = summary("construct_gap")
    raw, clip = h3.loc["raw"], h3.loc["clipped"]
    body = [
        row(
            [
                r"Parametric basic interval (A9; decides H3)",
                num(raw["estimate"]),
                interval(raw["ci_low"], raw["ci_high"]),
                num(clip["estimate"]),
                interval(clip["ci_low"], clip["ci_high"]),
            ]
        )
    ]
    for label, key in (
        ("Player bootstrap", ("cross", "player")),
        ("A7 joint bootstrap", ("cross", "a7")),
        (r"Within-sample estimator, player bootstrap (A10)", ("within", "player")),
    ):
        r = con.loc[key]
        body.append(
            row(
                [
                    label,
                    num(r["diff"]),
                    interval(r["ci_low"], r["ci_high"]),
                    num(r["diff_clipped"]),
                    interval(r["ci_low_clipped"], r["ci_high_clipped"]),
                ]
            )
        )
    n_sims = count(s["h3_n_sims"])
    body += [
        r"  \midrule",
        rf"  \multicolumn{{5}}{{@{{}}l}}{{\textit{{Parametric bootstrap, $S$ = {n_sims}"
        r" simulations}} \\",
        row(
            [
                "Planted contrast, mean over simulations (A18)",
                num(raw["planted"]),
                "",
                num(clip["planted"]),
                "",
            ]
        ),
        row(
            [
                "Planted contrast, earlier variance-weighted definition",
                num(raw["planted_weighted"]),
                "",
                num(clip["planted_weighted"]),
                "",
            ]
        ),
        row(
            [
                "Mean error against the A18 planted value",
                num(raw["mean_error"]),
                "",
                num(clip["mean_error"]),
                "",
            ]
        ),
        row(["SD of the error", num(raw["sd_error"]), "", num(clip["sd_error"]), ""]),
        row(
            [
                "Leave-one-out coverage of the parametric interval",
                num(raw["coverage_loo"]),
                "",
                num(clip["coverage_loo"]),
                "",
            ]
        ),
        row(
            [
                "Coverage of the planted contrast by the player-bootstrap interval",
                num(raw["coverage_m4"]),
                "",
                num(clip["coverage_m4"]),
                "",
            ]
        ),
    ]
    header = [
        r"  & \multicolumn{2}{c}{Raw} & \multicolumn{2}{c}{Clipped} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        row(["", "$D$", r"90\% interval", "$D$", r"90\% interval"]),
    ]
    label = "@{}>{\\raggedright\\arraybackslash}p{6.4cm}"
    return tabular(label + "rcrc@{}", header, body) + "\n"


def table_visibility() -> str:
    v = csv(RESULTS / "tournaments" / "visibility.csv")
    v = v[(v["scope"] == "all") & (v["group"] == "ALL")]
    body = [
        row(
            [
                tex(TARGET_LABEL.get(r["target"], r["target"])),
                count(r["n_units"]),
                num(r["r2_cv"]),
                num(r["rel_target"]),
                num(r["rel_box"]),
                num(r["r2_disatt"]),
                interval(r["r2_disatt_lo"], r["r2_disatt_hi"]),
                num(r["r2_cross"]),
                interval(r["r2_cross_lo"], r["r2_cross_hi"]),
            ]
        )
        for _, r in v.iterrows()
    ]
    header = [
        r"  & & & \multicolumn{2}{c}{Reliability} & \multicolumn{2}{c}{$r^2_{\text{disatt}}$}"
        r" & \multicolumn{2}{c}{$r^2_{\text{cross}}$} \\",
        r"  \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        row(
            [
                "Target",
                "Units",
                r"$R^2_{\text{cv}}$",
                "target",
                "box",
                "",
                r"90\% interval",
                "",
                r"90\% interval",
            ]
        ),
    ]
    return (
        "{\\setlength{\\tabcolsep}{4pt}\n" + tabular("@{}lrrrrrcrc@{}", header, body) + "\\par}\n"
    )


def table_clutch() -> str:
    s = summary("clutch")
    body = [
        row([r"Players with $\ge 360$ knockout minutes in each period", count(s["h7_n"])]),
        row(
            [
                r"Persistence $r$ [90\% player-bootstrap interval]",
                num(s["h7_r"]) + " " + interval(s["h7_ci_low"], s["h7_ci_high"]),
            ]
        ),
        row(["One-sided permutation p-value", num(s["h7_p_value"], 3)]),
        row(["Within-period reliability of $Z$, 2020--2022", num(s["rel_a"])]),
        row(["Within-period reliability of $Z$, 2023--2026", num(s["rel_b"])]),
        row(["Power against a true correlation of 0.3", num(s["power_at_03"])]),
        row(
            [
                r"Placebo knockout labels: mean $r$ [5th, 95th percentile]",
                num(s["placebo_r_mean"]) + " " + interval(s["placebo_r_q05"], s["placebo_r_q95"]),
            ]
        ),
    ]
    return tabular("@{}lr@{}", [row(["", "Value"])], body) + "\n"


def table_verdicts() -> str:
    """Statistic and threshold are the pre-registered text (tab:hypotheses, as
    amended); observed values and verdicts are the macros, filled from results."""
    rows = [
        ("H1", r"Share of outfield pool pairs decided at 0.9 (M8)", r"share $\ge 0.5$",
         r"\SharePairsDecided{}", r"\VerdictHOne{}"),
        ("H2", r"Highest outfield $P(\#1)$ (M8)", r"some $P(\#1) \ge 0.5$",
         r"\TopPlayerPFirst{} (\TopPlayerName{})", r"\VerdictHTwo{}"),
        ("H3", r"Clipped cross-half share, CB+DM minus CF+W, with its 90\% parametric "
         r"basic interval (M4)", r"interval includes 0, or the difference is negative",
         r"\ShareContrast{} [\ShareContrastParamLo, \ShareContrastParamHi]",
         r"\VerdictHThree{}"),
        ("H4", r"Mean ratio of specification to sampling variance of rank, top ten of the "
         r"median specification (M8)", r"mean ratio $\le 1$", r"\HFourRatio{}",
         r"\VerdictHFour{}"),
        ("H5", r"Total-order index of time aggregation, player with the highest "
         r"$P(\#1)$ (M5)", r"another dimension has a larger index",
         r"\HFiveAggTotal{} (largest: \HFiveMaxDim{}, \HFiveMaxTotal{})",
         r"\VerdictHFive{}"),
        ("H6", r"OLS Shapley shares of team success and individual statistics (M6)",
         r"team share does not exceed statistics share",
         r"\ShapleyTeamOls{} vs \ShapleyStatsOls{}", r"\VerdictHSix{}"),
        ("H7", r"Persistence $r$ of knockout over-performance, with power against 0.3 (M7b)",
         r"$r \ge 0.2$; inconclusive if power $< 0.8$",
         r"$r$ = \ClutchR{}, power \ClutchPower{}", r"\VerdictHSeven{}"),
        ("H8", r"Correlation of non-penalty xG per shot, zone-based vs StatsBomb",
         r"correlation $< 0.8$", "{--}", r"not tested (A6, A11, A15)"),
    ]  # fmt: skip
    body = [row(list(r)) for r in rows]
    header = [row(["", "Statistic", "Refuted if", "Observed", "Verdict"])]
    return (
        "{\\setlength{\\tabcolsep}{4pt}\n"
        r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}p{4.4cm}YY"
        r">{\raggedright\arraybackslash}p{2.9cm}@{}}"
        "\n  \\toprule\n"
        + "\n".join(header)
        + "\n  \\midrule\n"
        + "\n".join(body)
        + "\n  \\bottomrule\n\\end{tabularx}\\par}\n"
    )


DIMENSION_LABEL = {
    "family": "Family",
    "weights": "Weights",
    "position": "Position",
    "context": "Context",
    "aggregation": "Aggregation",
    "competition": "Competition",
    "min_minutes": "Minimum minutes",
    "shrinkage": "Shrinkage",
}
BLOCK_LABEL = {
    "stats": "Individual statistics",
    "impact": "Impact (RAPM)",
    "team": "Team success",
    "reputation": "Reputation",
}


def multiverse_leaders() -> pd.DataFrame:
    """M5's leading outfield players: those ranked first by at least one
    specification on the original data, by P(#1), then P(top-3), P(top-10)."""
    probs = csv(RESULTS / "multiverse" / "rank_probabilities.csv")
    out = probs[(probs["segment"] == "outfield") & (probs["p_first"] > 0)]
    return out.sort_values(["p_first", "p_top3", "p_top10"], ascending=False)


def _share3(x: float) -> str:
    """Three decimals: one specification of 5,760 is 0.0002, and a leader by
    definition has P(#1) > 0, which two decimals would print as 0.00."""
    return "$<0.001$" if 0 < x < 0.0005 else num(x, 3)


def table_multiverse() -> str:
    body = [
        row([tex(r["player"]), _share3(r["p_first"]), _share3(r["p_top3"]),
             _share3(r["p_top10"])])
        for _, r in multiverse_leaders().iterrows()
    ]  # fmt: skip
    header = [row(["Player", r"$P(\#1)$", r"$P(\text{top-3})$", r"$P(\text{top-10})$"])]
    return tabular("@{}lrrr@{}", header, body) + "\n"


def table_recognition() -> str:
    shares = csv(RESULTS / "recognition" / "shares.csv").set_index("block")
    s = summary("recognition")
    body = [
        row(
            [
                BLOCK_LABEL[b],
                num(r["ols"]),
                interval(r["ols_lo"], r["ols_hi"]),
                num(r["pl"]),
                interval(r["pl_lo"], r["pl_hi"]),
            ]
        )
        for b, r in shares.iterrows()
    ]
    body += [
        r"  \midrule",
        row([r"Fit: within-edition $R^2$; McFadden pseudo-$R^2$", num(s["OlsRTwo"]), "",
             num(s["PlPseudoRTwo"]), ""]),
        row(["Leave-one-edition-out $R^2$", num(s["OlsRTwoLoeo"]), "", "", ""]),
        row(["Candidate-editions", count(s["NBdoRows"]), "", count(s["NBdoRows"]), ""]),
    ]  # fmt: skip
    header = [
        r"  & \multicolumn{2}{c}{OLS, $\log(1 + \text{points})$}"
        r" & \multicolumn{2}{c}{Plackett--Luce} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        row(["Block", "Share", r"90\% interval", "Share", r"90\% interval"]),
    ]
    return tabular("@{}lrcrc@{}", header, body) + "\n"


LEADING_P_FIRST = 0.01  # M8's leading players: at least a 1% chance of being #1


def synthesis_players(segment: str) -> pd.DataFrame:
    players = csv(RESULTS / "synthesis" / "players.csv")
    sets = csv(RESULTS / "synthesis" / "rank_sets.csv")
    out = players[players["segment"] == segment].merge(
        sets[["player_id", "marginal_lo", "marginal_hi"]], on="player_id", how="left"
    )
    return out.sort_values(["p_first", "median_rank"], ascending=[False, True])


def synthesis_leaders() -> pd.DataFrame:
    players = synthesis_players("outfield")
    return players[players["p_first"] >= LEADING_P_FIRST]


def _rank_set(lo: object, hi: object) -> str:
    if not (np.isfinite(float(lo)) and np.isfinite(float(hi))):  # type: ignore[arg-type]
        return "{--}"
    return f"[{count(lo)}, {count(hi)}]"


def _synthesis_table(players: pd.DataFrame, note: str, *, top10: bool = True) -> str:
    """``top10=False`` drops P(top-10), which is 1 for every one of the eight goalkeepers."""
    probabilities = ["p_first", "p_top3", *(["p_top10"] if top10 else [])]
    labels = {"p_first": r"$P(\#1)$", "p_top3": r"$P(\text{top-3})$",
              "p_top10": r"$P(\text{top-10})$"}  # fmt: skip
    body = [
        row(
            [
                tex(r["player"]),
                *(num(r[p]) for p in probabilities),
                count(r["median_rank"]) + " " + _rank_set(r["rank_lo"], r["rank_hi"]),
                _rank_set(r["marginal_lo"], r["marginal_hi"]),
                num(r["share_specification"]),
                num(r["share_sampling"]),
                num(r["share_unmeasured"]),
            ]
        )
        for _, r in players.iterrows()
    ]
    k = 1 + len(probabilities)
    header = [
        "  " + "& " * k + r"\multicolumn{2}{c}{Rank} & \multicolumn{3}{c}{Share of rank variance}"
        r" \\",
        rf"  \cmidrule(lr){{{k + 1}-{k + 2}}}\cmidrule(lr){{{k + 3}-{k + 5}}}",
        row(
            [
                "Player",
                *(labels[p] for p in probabilities),
                r"median [90\%]",
                "marginal set",
                "spec.",
                "samp.",
                "unm.",
            ]
        ),
    ]
    return (
        "{\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
        + tabular("@{}l" + "r" * len(probabilities) + "rcrrr@{}", header, body)
        + note
        + "\\par}\n"
    )


def table_synthesis() -> str:
    """The leading outfield players: the 98 rows of the pool do not fit one float."""
    leaders = synthesis_leaders()
    n = (synthesis_players("outfield")["p_first"] > 0).sum()
    note = (
        r"\par\smallskip{\footnotesize The "
        + count(len(leaders))
        + r" outfield players with $P(\#1) \ge "
        + num(LEADING_P_FIRST)
        + r"$ ("
        + count(n)
        + r" have $P(\#1) > 0$). Marginal set: 90\% rank confidence set at the median "
        r"specification.\par}"
    )
    return _synthesis_table(leaders, note)


def table_goalkeepers() -> str:
    note = (
        r"\par\smallskip{\footnotesize Marginal set: 90\% rank confidence set at the median "
        r"specification. $P(\text{top-10})$ is 1 for each of the eight goalkeepers and is "
        r"left out.\par}"
    )
    return _synthesis_table(synthesis_players("goalkeeper"), note, top10=False)


def recognition_view() -> pd.DataFrame:
    """The recognition view of the outfield pool: share_sum is the sum over the
    2021--2026 editions of the player's share of each edition's Ballon d'Or points
    (A19); raw_sum the earlier sum of points (A13 (v))."""
    view = csv(RESULTS / "synthesis" / "recognition_view.csv")
    return view[view["segment"] == "outfield"].reset_index(drop=True)


def table_views() -> str:
    players = synthesis_players("outfield").merge(
        recognition_view()[["player_id", "share_sum", "rank_share_sum"]], on="player_id"
    )
    players = players.sort_values(["rank_share_sum", "stats_view"])
    lines = [
        row(
            [
                tex(r["player"]),
                num(r["stats_view"], 1),
                num(r["impact_view"], 1),
                num(r["share_sum"], 3),
                count(r["rank_share_sum"]),
            ]
        )
        for _, r in players.iterrows()
    ]
    half = (len(lines) + 1) // 2
    header = [
        r"  & \multicolumn{2}{c}{Mean rank} & \multicolumn{2}{c}{Ballon d'Or} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        row(["Player", "stats", "impact", "shares", "rank"]),
    ]
    colspec = "@{}lrrrr@{}"
    note = (
        r"\par\smallskip\raggedright Ballon d'Or shares: sum over the 2021--2026 editions "
        r"of the player's share of each edition's points (A19); ordered by its rank, ties "
        r"sharing the best rank. Kendall's $W$ = \KendallW{} (mean pairwise Spearman "
        r"correlation \KendallMeanSpearman{})."
    )
    return (
        "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        + tabular(colspec, header, lines[:half], align="[t]")
        + "\\hfill\n"
        + tabular(colspec, header, lines[half:], align="[t]")
        + note
        + "\\par}\n"
    )


def table_v1() -> str:
    v1 = csv(PAPER / "data" / "v1_ranking.csv")
    players = synthesis_players("outfield").set_index("player")
    v1 = v1[v1["player"].isin(players.index)]
    body = []
    for _, r in v1.iterrows():
        p = players.loc[r["player"]]
        body.append(
            row(
                [
                    count(r["rank"]),
                    tex(r["player"]),
                    count(p["median_rank"]),
                    _rank_set(p["rank_lo"], p["rank_hi"]),
                    num(p["p_first"]),
                ]
            )
        )
    header = [row(["V1 rank", "Player", "Median rank", r"90\% rank interval", r"$P(\#1)$"])]
    return "{\\footnotesize\n" + tabular("@{}rlrcr@{}", header, body) + "\\par}\n"


def table_recovery() -> str:
    gaps = csv(RESULTS / "recovery" / "by_gap.csv")
    s = summary("recovery")
    body = [
        row(
            [
                f"[{num(r['gap_low'], 3)}, {num(r['gap_high'], 3)}]",
                num(r["gap_mean"], 3),
                count(r["n"]),
                num(r["p_top_rapm"]),
                num(r["p_top_box"]),
                num(r["mae_rapm"]),
                num(r["mae_box"]),
            ]
        )
        for _, r in gaps.iterrows()
    ]
    body += [
        r"  \midrule",
        row(
            [
                "All",
                "",
                count(s["NRecoverySims"]),
                num(s["RecoveryRapmFirst"]),
                num(s["RecoveryBoxFirst"]),
                num(s["RecoveryRankErrorRapm"]),
                num(s["RecoveryRankErrorBox"]),
            ]
        ),
    ]
    header = [
        r"  \multicolumn{2}{c}{Planted top-two gap} & & \multicolumn{2}{c}{Ranked first}"
        r" & \multicolumn{2}{c}{Mean abs.\ rank error} \\",
        r"  \cmidrule(lr){1-2}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        row(["range", "mean", "Simulations", "RAPM", "box", "RAPM", "box"]),
    ]
    note = r"\par\smallskip{\footnotesize Gaps in goals per 90; equal-count bins.\par}"
    return tabular("@{}crrrrrr@{}", header, body) + note + "\n"


def _editions(text: object) -> str:
    """'2021;2022;2023;2026' -> '21--23, 26'."""
    if not isinstance(text, str) or not text:
        return ""
    years = sorted(int(y) % 100 for y in text.split(";"))
    runs: list[list[int]] = []
    for y in years:
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    return ", ".join(f"{r[0]}--{r[-1]}" if len(r) > 2 else ", ".join(map(str, r)) for r in runs)


# Mapping methods other than the default (name and Ballon d'Or club) are marked.
METHOD_MARK = {"manual": "m", "name_unique": "u"}


def table_pool() -> str:
    pool = csv(POOL)
    pool = pool.iloc[sorted(range(len(pool)), key=lambda i: sort_key(pool["player"].iloc[i]))]
    default = pool["match_method"].mode().iloc[0]

    def line(r: pd.Series) -> str:
        marks = METHOD_MARK.get(r["match_method"], "") if r["match_method"] != default else ""
        if r["confidence"] != "high":
            marks += r"\ast"
        mark = rf"$^{{{marks}}}$" if marks else ""
        return row(
            [
                tex(r["player"]),
                tex(r["position_group"]),
                r"\checkmark" if r["in_v1"] else "",
                _editions(r["bdo_editions"]),
                str(int(r["tm_player_id"])) + mark,
                count(r["minutes_in_window"]),
            ]
        )

    lines = [line(r) for _, r in pool.iterrows()]
    half = (len(lines) + 1) // 2
    header = [row(["Player", "Grp", "V1", "BdO", "TM id", "Min."])]
    colspec = "@{}llcllr@{}"
    note = (
        r"\par\smallskip\raggedright BdO: Ballon d'Or editions 2021--2026 in which he was "
        r"ranked. TM id: Transfermarkt identifier, mapped by name and a club the Ballon d'Or "
        r"lists unless marked: $^{\mathrm m}$~listed manual decision, $^{\mathrm u}$~unique "
        r"name candidate; $^\ast$~medium confidence. Min.: minutes in the window."
    )
    return (
        "{\\scriptsize\\setlength{\\tabcolsep}{1.6pt}\n"
        + tabular(colspec, header, lines[:half], align="[t]")
        + "\\hfill\n"
        + tabular(colspec, header, lines[half:], align="[t]")
        + note
        + "\\par}\n"
    )


# --------------------------------------------------------------------------
# figures --- palette from the dataviz skill's reference instance (light mode),
# the one paper 1 uses. Identity never rests on hue alone: every multi-series
# figure also differs in marker or fill, and carries a legend.
# --------------------------------------------------------------------------

BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
AQUA = "#1baf7a"  # categorical slot 3
RED = "#e34948"  # diverging pole paired with BLUE
BLUE_DARK = "#0d366b"  # blue ramp step 700
RED_DARK = "#9c2a29"
MID_GREY = "#f0efec"  # diverging midpoint

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Seven groups exceed what hue can separate all-pairs (dataviz: three slots), so
# the pool figure colours the three lines and gives each group its own marker.
LINE_OF = {"CB": "Defence", "FB": "Defence", "DM": "Midfield", "CM": "Midfield",
           "AM": "Midfield", "W": "Attack", "CF": "Attack"}  # fmt: skip
LINE_COLOR = {"Defence": BLUE, "Midfield": ORANGE, "Attack": AQUA}
GROUP_MARKER = {"CB": "o", "FB": "s", "DM": "o", "CM": "s", "AM": "D", "W": "o", "CF": "s"}

DIVERGING = LinearSegmentedColormap.from_list(
    "paper2_diverging", [RED_DARK, RED, MID_GREY, BLUE, BLUE_DARK]
)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def figure_reliability(width: float) -> Figure:
    curves = csv(RESULTS / "reliability" / "stabilisation_curves.csv")
    rel = csv(RESULTS / "reliability" / "reliability.csv")
    rel = rel[(rel["group"] == "OUTFIELD") & ~rel["exploratory"]].set_index("metric")
    curves = curves[curves["group"] == "OUTFIELD"]
    metrics = [m for m in METRIC_LABEL if m in rel.index]
    x_max = 2000.0
    size_scale = 40.0 / float(curves["n_units"].max())

    fig, axes = plt.subplots(2, 4, figsize=(width, 3.6), sharex=True, sharey=True)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        _clean(ax)
        ax.grid(axis="y", zorder=0)
        part = curves[curves["metric"] == metric]
        k = float(rel.loc[metric, "k"])
        d_last = float(part["minutes"].max())
        inside = np.linspace(0.0, d_last, 100)
        beyond = np.linspace(d_last, x_max, 60)
        ax.plot(inside, inside / (inside + k), color=BLUE, lw=1.4, zorder=2)
        ax.plot(beyond, beyond / (beyond + k), color=BLUE, lw=1.0, ls=(0, (3, 2)), zorder=2)
        ax.scatter(
            part["minutes"],
            part["reliability"],
            s=part["n_units"] * size_scale,
            color=BLUE,
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
        )
        for level in (0.5, 0.7):
            ax.axhline(level, color=INK_MUTED, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
        title = "Non-penalty G + A" if metric == "np_ga" else METRIC_LABEL[metric]
        ax.set_title(title, fontsize=7.5, color=INK, loc="left", pad=3)
        ax.text(0.97, 0.05, f"k = {k:,.0f}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.5, color=INK_SECONDARY)  # fmt: skip
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    axes[0, 0].set_xlim(0, x_max)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_yticks([0, 0.5, 0.7, 1.0])
    for ax in axes[1]:
        ax.set_xlabel("Half length (minutes)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Reliability")
    handles = [
        Line2D([], [], color=BLUE, lw=1.4, label="fitted d/(d+k)"),
        Line2D([], [], color=BLUE, lw=1.0, ls=(0, (3, 2)), label="extrapolated"),
        Line2D([], [], color=INK_MUTED, lw=0.7, ls=(0, (1, 1.5)), label="0.5 and 0.7"),
    ]
    for n in (10_000, 30_000):
        handles.append(
            Line2D([], [], ls="", marker="o", color=BLUE, markeredgecolor=SURFACE,
                   markersize=float(np.sqrt(n * size_scale)), label=f"{n:,} player-seasons")
        )  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.0, w_pad=0.6)
    return fig


def figure_scree(width: float) -> Figure:
    pca = csv(RESULTS / "structure" / "pca.csv", index_col=0)
    eig = pca.loc["eigenvalue"].to_numpy(float)
    null = pca.loc["pa_threshold"].to_numpy(float)
    j = np.arange(1, eig.size + 1)
    fig, ax = plt.subplots(figsize=(width, 2.6))
    _clean(ax)
    ax.grid(axis="y", zorder=0)
    ax.plot(j, null, color=ORANGE, lw=1.6, ls=(0, (4, 2)), zorder=2,
            label="parallel analysis, 95th percentile of 200 permutations")  # fmt: skip
    ax.scatter(j, eig, s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=3,
               label="eigenvalue of the correlation matrix")  # fmt: skip
    ax.set_xticks(j)
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_ylim(0, max(eig.max(), null.max()) * 1.12)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def figure_corr_heatmap(width: float) -> Figure:
    corr = csv(RESULTS / "structure" / "correlations.csv")
    order = json.loads((RESULTS / "structure" / "clustering.json").read_text())["order"]
    labels = [METRIC_LABEL[m] for m in order]
    fig, axes = plt.subplots(1, 2, figsize=(width, 3.5), constrained_layout=True)
    image = None
    for ax, kind, title in zip(
        axes,
        ("spearman", "partial_spearman"),
        ("Spearman", "Partial (team Elo, league-season)"),
        strict=True,
    ):
        part = corr[corr["kind"] == kind]
        m = part.pivot(index="metric_a", columns="metric_b", values="value").loc[order, order]
        values = m.to_numpy(float)
        image = ax.imshow(values, cmap=DIVERGING, vmin=-1, vmax=1)
        for (i, jj), v in np.ndenumerate(values):
            ax.text(jj, i, f"{v:.2f}".replace("-", "\u2212"), ha="center", va="center",
                    fontsize=6, color=SURFACE if abs(v) > 0.6 else INK)  # fmt: skip
        ax.set_xticks(range(len(order)), labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(order)), labels if ax is axes[0] else [])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    bar = fig.colorbar(image, ax=axes, shrink=0.7, pad=0.02)
    bar.set_label("Correlation")
    bar.outline.set_visible(False)
    return fig


def figure_rapm_pool(width: float) -> Figure:
    pool = csv(RESULTS / "rapm" / "pool_rapm.csv")
    pool = pool[~pool["is_goalkeeper"]].sort_values("total", ascending=False)
    pool = pool.reset_index(drop=True)
    half = (len(pool) + 1) // 2
    lo = float(pool["total_ci_low"].min())
    hi = float(pool["total_ci_high"].max())
    pad = 0.05 * (hi - lo)
    fig, axes = plt.subplots(1, 2, figsize=(width, 7.6), sharex=True)
    for ax, part in zip(axes, (pool.iloc[:half], pool.iloc[half:]), strict=True):
        _clean(ax)
        ax.grid(axis="x", zorder=0)
        y = np.arange(len(part))[::-1] + (half - len(part))
        ax.axvline(0.0, color=INK_MUTED, lw=0.8, zorder=1)
        for yy, (_, r) in zip(y, part.iterrows(), strict=True):
            color = LINE_COLOR[LINE_OF[r["position_group"]]]
            ax.plot([r["total_ci_low"], r["total_ci_high"]], [yy, yy], color=color, lw=1.0,
                    solid_capstyle="round", zorder=2)  # fmt: skip
            ax.plot(r["total"], yy, marker=GROUP_MARKER[r["position_group"]], ms=3.6,
                    color=color, mec=SURFACE, mew=0.5, zorder=3)  # fmt: skip
        ax.set_yticks(
            y,
            [f"{i + 1}. {n} ({g})" for i, n, g in zip(part.index, part["player"],
             part["position_group"], strict=True)],
        )  # fmt: skip
        ax.tick_params(axis="y", length=0, labelsize=6.2)
        ax.set_ylim(-0.7, half - 0.3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("RAPM total (goals per 90)")
    handles = []
    for g in OUTFIELD:
        handles.append(
            Line2D([], [], color=LINE_COLOR[LINE_OF[g]], marker=GROUP_MARKER[g], ms=4,
                   markeredgecolor=SURFACE, lw=1.0, label=g)
        )  # fmt: skip
    handles.append(Line2D([], [], color=INK_SECONDARY, lw=1.0, label="90% interval"))
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.025, 1, 1), w_pad=1.5)
    return fig


def figure_unmeasured_share(width: float) -> Figure:
    shares = m4_total()
    order = [*GROUPS, *POOLS]
    y = np.arange(len(order))[::-1].astype(float)
    y[-len(POOLS) :] -= 0.6  # gap before the H3 pools
    fig, ax = plt.subplots(figsize=(width, 3.3))
    _clean(ax)
    ax.grid(axis="x", zorder=0)
    for edge in (0.0, 1.0):
        ax.axvline(edge, color=INK_MUTED, lw=0.8, zorder=1)
    for yy, g in zip(y, order, strict=True):
        r = shares.loc[g]
        color = ORANGE if g in POOLS else BLUE
        ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], color=color, lw=1.4,
                solid_capstyle="round", zorder=2)  # fmt: skip
        ax.plot(r["share"], yy, "o", ms=6, mfc=SURFACE, mec=color, mew=1.3, zorder=3)
        ax.plot(r["share_clipped"], yy, "o", ms=4.6, color=color, mec=SURFACE, mew=0.6,
                zorder=4)  # fmt: skip
    ax.set_yticks(y, order)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Unmeasured share of RAPM total (cross-half)")
    handles = [
        Line2D([], [], ls="", marker="o", ms=4.6, color=INK_SECONDARY, mec=SURFACE,
               label="clipped to [0, 1]"),
        Line2D([], [], ls="", marker="o", ms=6, mfc=SURFACE, mec=INK_SECONDARY, mew=1.3,
               label="raw"),
        Line2D([], [], color=INK_SECONDARY, lw=1.4, label="90% player-bootstrap interval (raw)"),
        Line2D([], [], color=BLUE, lw=3, label="position group"),
        Line2D([], [], color=ORANGE, lw=3, label="H3 pool"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.tight_layout()
    return fig


def figure_h3_parametric(width: float) -> Figure:
    sims = csv(RESULTS / "construct_gap" / "h3_sims.csv")
    h3 = csv(RESULTS / "construct_gap" / "h3_summary.csv").set_index("kind").loc["clipped"]
    values = sims["diff_clipped"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(width, 3.3))
    _clean(ax)
    ax.grid(axis="y", zorder=0)
    ax.axvspan(h3["ci_low"], h3["ci_high"], color=BLUE, alpha=0.12, lw=0, zorder=0,
               label=f"90% basic interval [{h3['ci_low']:.2f}, {h3['ci_high']:.2f}]")  # fmt: skip
    ax.hist(values, bins=24, color=INK_MUTED, edgecolor=SURFACE, linewidth=0.6, zorder=2,
            label=f"simulated estimates (S = {values.size})")  # fmt: skip
    ax.axvline(h3["planted"], color=INK, lw=1.3, ls=(0, (4, 2)), zorder=3,
               label=f"planted contrast, mean over simulations {h3['planted']:.2f}")  # fmt: skip
    ax.axvline(h3["estimate"], color=BLUE, lw=1.8, zorder=3,
               label=f"real estimate {h3['estimate']:.2f}")  # fmt: skip
    ax.set_xlabel("H3 contrast, clipped share CB+DM minus CF+W")
    ax.set_ylabel("Simulations")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.5)
    fig.tight_layout()
    return fig


def figure_visibility(width: float) -> Figure:
    v = csv(RESULTS / "tournaments" / "visibility.csv")
    v = v[(v["scope"] == "all") & (v["group"] == "ALL")].reset_index(drop=True)
    y = np.arange(len(v))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(width, 3.1))
    _clean(ax)
    ax.grid(axis="x", zorder=0)
    ax.axvline(0.0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.axvline(1.0, color=INK_MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
    for key, offset, color, filled, label in (
        ("r2_cross", 0.14, BLUE, True, r"cross-half $r^2_{\mathrm{cross}}$"),
        ("r2_disatt", -0.14, ORANGE, False, r"within-sample $r^2_{\mathrm{disatt}}$"),
    ):
        yy = y + offset
        ax.hlines(yy, v[f"{key}_lo"], v[f"{key}_hi"], color=color, lw=1.3, zorder=2)
        ax.plot(v[key], yy, ls="", marker="o" if filled else "s", ms=4.8,
                mfc=color if filled else SURFACE, mec=color, mew=1.2, zorder=3,
                label=label)  # fmt: skip
    ax.set_yticks(y, [TARGET_LABEL.get(t, t) for t in v["target"]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of the target's variance visible to the box score")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK_SECONDARY, lw=1.3))
    labels.append("90% player-bootstrap interval")
    ax.legend(handles, labels, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    return fig


def figure_knockout(width: float) -> Figure:
    players = csv(RESULTS / "clutch" / "h7_players.csv")
    fig, ax = plt.subplots(figsize=(width, width * 0.9))
    _clean(ax)
    ax.grid(zorder=0)
    ax.axhline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.scatter(players["z_a"], players["z_b"], s=7, color=BLUE, alpha=0.4, edgecolor="none",
               zorder=2)  # fmt: skip
    ax.set_xlabel("Knockout over-performance Z, 2020–2022 (SD units)")
    ax.set_ylabel("Knockout over-performance Z, 2023–2026 (SD units)")
    lim = float(np.nanmax(np.abs(players[["z_a", "z_b"]].to_numpy()))) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


SPEC_CURVE_PLAYERS = 4  # columns that fit \linewidth with readable level labels
SPEC_CURVE_BINS = 120  # curve positions per bin: 5,760 / 120 = 48


def figure_spec_curve(width: float) -> Figure:
    """Top: rank under every specification, best to worst. Bottom: per bin of
    curve positions, the share of its specifications at each dimension level."""
    leaders = multiverse_leaders().head(SPEC_CURVE_PLAYERS)
    curve = csv(RESULTS / "multiverse" / "specification_curve.csv.gz")
    grid = csv(RESULTS / "multiverse" / "spec_grid.csv").set_index("spec_id")
    levels = [(d, lv) for d in DIMENSION_LABEL for lv in grid[d].drop_duplicates()]
    blue = LinearSegmentedColormap.from_list("paper2_blue", [SURFACE, BLUE, BLUE_DARK])

    fig, axes = plt.subplots(
        2,
        len(leaders),
        figsize=(width, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 2.4]},
        squeeze=False,
        layout="constrained",
    )
    image = None
    for col, (_, leader) in enumerate(leaders.iterrows()):
        part = curve[curve["player_id"] == leader["player_id"]].sort_values("curve_position")
        top, bottom = axes[0, col], axes[1, col]
        _clean(top)
        top.grid(axis="y", zorder=0)
        top.plot(part["curve_position"], part["rank"], color=BLUE, lw=1.2,
                 drawstyle="steps-post")  # fmt: skip
        top.set_ylim(part["rank"].max() + 1, 0)
        top.set_title(leader["player"], fontsize=7.5, color=INK, loc="left", pad=3)
        n = len(part)
        edges = np.linspace(0, n, SPEC_CURVE_BINS + 1)
        bins = np.minimum(np.searchsorted(edges, part["curve_position"], side="right") - 1,
                          SPEC_CURVE_BINS - 1)  # fmt: skip
        specs = grid.loc[part["spec_id"].to_numpy()]
        share = np.array(
            [
                np.bincount(bins, weights=(specs[d].to_numpy() == lv), minlength=SPEC_CURVE_BINS)
                / np.bincount(bins, minlength=SPEC_CURVE_BINS)
                for d, lv in levels
            ]
        )
        image = bottom.pcolormesh(edges, np.arange(len(levels) + 1), share, cmap=blue,
                                  vmin=0, vmax=1, rasterized=False)  # fmt: skip
        bottom.set_ylim(len(levels), 0)
        start = 0
        for d in DIMENSION_LABEL:
            start += int(grid[d].nunique())
            if start < len(levels):
                bottom.axhline(start, color=INK_SECONDARY, lw=0.6)
        bottom.set_yticks(np.arange(len(levels)) + 0.5)
        bottom.set_yticklabels(
            [f"{DIMENSION_LABEL[d]}: {lv}".replace("_", " ") for d, lv in levels]
            if col == 0
            else []
        )
        bottom.tick_params(axis="y", length=0, labelsize=5.6)
        bottom.set_xlim(0, n)
        bottom.set_xlabel("Specification")
        for spine in bottom.spines.values():
            spine.set_visible(False)
    axes[0, 0].set_ylabel("Rank")
    bar = fig.colorbar(image, ax=axes[1, :].tolist(), location="bottom", shrink=0.45,
                       aspect=40, pad=0.02)  # fmt: skip
    bar.set_label(f"Share of the specifications in each bin of {n // SPEC_CURVE_BINS} "
                  "at that level", fontsize=6.5)  # fmt: skip
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6)
    return fig


def figure_total_order(width: float) -> Figure:
    leaders = multiverse_leaders()
    indices = csv(RESULTS / "multiverse" / "variance_indices.csv")
    table = indices.pivot(index="player_id", columns="dimension", values="total_order")
    table = table.reindex(index=leaders["player_id"], columns=list(DIMENSION_LABEL))
    values = table.to_numpy(float)
    blue = LinearSegmentedColormap.from_list("paper2_blue", [SURFACE, BLUE, BLUE_DARK])
    fig, ax = plt.subplots(figsize=(width, 0.17 * len(leaders) + 1.2))
    image = ax.imshow(values, cmap=blue, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color=SURFACE if v > 0.55 else INK)  # fmt: skip
    ax.set_xticks(range(values.shape[1]), [DIMENSION_LABEL[d] for d in table.columns],
                  rotation=35, ha="left", rotation_mode="anchor")  # fmt: skip
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(leaders)), leaders["player"].tolist())
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    bar.set_label("Total-order index")
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def figure_shapley(width: float) -> Figure:
    shares = csv(RESULTS / "recognition" / "shares.csv").set_index("block")
    y = np.arange(len(shares))[::-1].astype(float)
    fig, ax = plt.subplots(figsize=(width, 2.8))
    _clean(ax)
    ax.grid(axis="x", zorder=0)
    for model, offset, color, label in (
        ("ols", 0.18, BLUE, "OLS, log(1 + points)"),
        ("pl", -0.18, ORANGE, "Plackett–Luce"),
    ):
        ax.barh(y + offset, shares[model], height=0.32, color=color, zorder=2, label=label)
        ax.hlines(y + offset, shares[f"{model}_lo"], shares[f"{model}_hi"], color=INK,
                  lw=1.0, zorder=3)  # fmt: skip
    ax.set_yticks(y, [BLOCK_LABEL[b] for b in shares.index])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, None)
    ax.set_xlabel("Shapley share (OLS: R² units; Plackett–Luce: pseudo-R² units)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, lw=1.0))
    labels.append("90% player-cluster bootstrap interval")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3)
    fig.tight_layout()
    return fig


def figure_pfirst(width: float) -> Figure:
    """P(#1) and rank are on different scales: two panels sharing the players."""
    leaders = synthesis_leaders().reset_index(drop=True)
    y = np.arange(len(leaders))[::-1]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(width, 0.19 * len(leaders) + 0.9), sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.3]},
    )  # fmt: skip
    for ax in (left, right):
        _clean(ax)
        ax.grid(axis="x", zorder=0)
    left.barh(y, leaders["p_first"], height=0.62, color=BLUE, zorder=2)
    left.set_xlabel("P(#1)")
    left.set_yticks(y, leaders["player"])
    left.tick_params(axis="y", length=0)
    right.hlines(y, leaders["rank_lo"], leaders["rank_hi"], color=INK_SECONDARY, lw=1.1, zorder=2)
    right.plot(leaders["median_rank"], y, "o", ms=4.5, color=BLUE, mec=SURFACE, mew=0.6,
               zorder=3)  # fmt: skip
    right.set_xlim(0, float(leaders["rank_hi"].max()) + 2)
    right.set_xlabel("Rank (1 = best): median and 90% interval")
    right.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.2)
    return fig


def figure_variance_decomposition(width: float) -> Figure:
    leaders = synthesis_leaders().reset_index(drop=True)
    y = np.arange(len(leaders))[::-1]
    fig, ax = plt.subplots(figsize=(width, 0.19 * len(leaders) + 1.0))
    _clean(ax)
    left = np.zeros(len(leaders))
    for key, color, label in (
        ("share_specification", BLUE, "specification"),
        ("share_sampling", ORANGE, "sampling"),
        ("share_unmeasured", AQUA, "unmeasured"),
    ):
        ax.barh(y, leaders[key], left=left, height=0.66, color=color, edgecolor=SURFACE,
                linewidth=1.0, label=label, zorder=2)  # fmt: skip
        left = left + leaders[key].to_numpy(float)
    ax.set_yticks(y, leaders["player"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the rank's variance")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 * 20 / len(leaders)), ncol=3)
    fig.tight_layout()
    return fig


def figure_decided_pairs(width: float) -> Figure:
    players = synthesis_players("outfield").sort_values(["median_rank", "mean_rank"])
    pair = csv(RESULTS / "synthesis" / "pairwise_outfield.csv", index_col="player_id")
    pair.columns = pair.columns.astype(int)
    order = players["player_id"].to_numpy()
    values = pair.loc[order, order].to_numpy(float)
    n = len(order)
    # Taller than wide, so that 98 row names stay legible at print size.
    fig, ax = plt.subplots(figsize=(width, 8.0), layout="constrained")
    image = ax.imshow(values, cmap=DIVERGING, vmin=0, vmax=1, interpolation="nearest",
                      aspect="auto")  # fmt: skip
    decided = np.argwhere((values >= 0.9) | (values <= 0.1))
    ax.scatter(decided[:, 1], decided[:, 0], s=1.2, color=INK, linewidths=0, zorder=3)
    ax.set_yticks(range(n), [f"{i + 1}. {p}" for i, p in enumerate(players["player"])])
    ax.tick_params(axis="y", labelsize=5, length=0, pad=1)
    ticks = list(range(0, n, 10))
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xlabel("Column player, in the order of the rows")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, location="bottom", shrink=0.5, aspect=40, pad=0.01)
    bar.set_label("P(row player above column player); black dots: decided pairs "
                  "(≥ 0.9 or ≤ 0.1)", fontsize=6.5)  # fmt: skip
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=6)
    return fig


def figure_recovery(width: float) -> Figure:
    gaps = csv(RESULTS / "recovery" / "by_gap.csv")
    s = summary("recovery")
    fig, ax = plt.subplots(figsize=(width, 2.9))
    _clean(ax)
    ax.grid(axis="y", zorder=0)
    ax.axhline(float(s["recovery_chance_level"]), color=INK_MUTED, lw=0.9, ls=(0, (1, 1.5)),
               zorder=1, label="chance (1 / pool size)")  # fmt: skip
    for key, color, marker, label in (
        ("p_top_rapm", BLUE, "o", "RAPM total"),
        ("p_top_box", ORANGE, "s", "box composite"),
    ):
        ax.hlines(gaps[key], gaps["gap_low"], gaps["gap_high"], color=color, lw=0.8,
                  alpha=0.6, zorder=2)  # fmt: skip
        ax.plot(gaps["gap_mean"], gaps[key], marker=marker, ms=5, color=color, mec=SURFACE,
                mew=0.8, lw=1.4, zorder=3, label=label)  # fmt: skip
    ax.set_ylim(0, 1)
    ax.set_xlim(0, None)
    ax.set_xlabel("Planted top-two gap (goals per 90): bin mean, with the bin's range")
    ax.set_ylabel("Share ranking the planted #1 first")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def table_robustness() -> str:
    """Post-hoc robustness: the M8 headline numbers under each alternative synthesis
    next to the primary; H3 under assortative planting, raw and clipped; M6 with
    curated honours."""
    variants = {"Registered": RESULTS / "synthesis"}
    for name, label in (
        ("bias_corrected", r"Bias-corrected $\sigma_g$"),
        ("europe", "European leagues only"),
    ):
        if (RESULTS / "synthesis" / name / "summary.json").exists():
            variants[label] = RESULTS / "synthesis" / name
    if len(variants) == 1:
        raise FileNotFoundError(RESULTS / "synthesis" / "bias_corrected" / "summary.json")
    sums = {label: summary(str(path.relative_to(RESULTS))) for label, path in variants.items()}
    rows = [
        ("Player with the highest $P(\\#1)$", lambda s: tex(s["TopPlayerName"])),
        ("His $P(\\#1)$", lambda s: num(s["TopPlayerPFirst"])),
        ("His shares of rank variance: spec., samp., unm.",
         lambda s: ", ".join(num(s[k]) for k in ("TopPlayerVarSpec", "TopPlayerVarSamp",
                                                  "TopPlayerVarUnm"))),
        ("Share of outfield pairs decided (H1)", lambda s: num(s["SharePairsDecided"])),
        ("Mean specification / sampling variance, top ten (H4)", lambda s: num(s["HFourRatio"])),
        ("Kendall's $W$ of the three views", lambda s: num(s["KendallW"])),
        ("Verdicts H1, H2, H4, H5",
         lambda s: ", ".join(tex(s[k]) for k in ("VerdictHOne", "VerdictHTwo", "VerdictHFour",
                                                  "VerdictHFive"))),
    ]  # fmt: skip
    body = [row([label, *(f(s) for s in sums.values())]) for label, f in rows]
    labels = list(sums)
    out = "{\\small\\setlength{\\tabcolsep}{4pt}\n" + tabular(
        "@{}>{\\raggedright\\arraybackslash}p{5.2cm}"
        + ">{\\raggedright\\arraybackslash}p{2.6cm}" * len(labels)
        + "@{}",
        [row(["M8 synthesis", *labels])],
        body,
    )
    blocks: list[tuple[str, list[tuple[str, str]]]] = []
    if (RESULTS / "construct_gap" / "assortative" / "summary.json").exists():
        a = summary("construct_gap/assortative")

        def both(key: str) -> str:
            return num(a[f"contrast_raw_{key}"]) + "; " + num(a[f"contrast_clipped_{key}"])

        h3 = [
            ("Planted contrast, mean over simulations (A18)", num(a["contrast_raw_planted"])),
            (r"Raw contrast and its 90\% basic interval",
             num(a["contrast_raw_estimate"]) + " "
             + interval(a["contrast_raw_ci_low"], a["contrast_raw_ci_high"])),
            (r"Clipped contrast and its 90\% basic interval",
             num(a["contrast_clipped_estimate"]) + " "
             + interval(a["contrast_clipped_ci_low"], a["contrast_clipped_ci_high"])),
            ("Mean error: raw; clipped", both("mean_error")),
            ("Coverage of the planted contrast by the player-bootstrap interval: raw; clipped",
             both("coverage_m4")),
            ("H3 verdict: raw; clipped", tex(a["verdict_raw"]) + "; " + tex(a["verdict_clipped"])),
        ]  # fmt: skip
        blocks.append((r"H3 with planted effects on the real teammate structure (assortative)", h3))
    if (RESULTS / "recognition" / "curated" / "summary.json").exists():
        c = summary("recognition/curated")
        m6 = [
            ("OLS Shapley shares, team success vs statistics",
             num(c["ShapleyTeamOls"]) + " vs " + num(c["ShapleyStatsOls"])),
            ("H6 verdict", tex(c["VerdictHSix"])),
        ]  # fmt: skip
        blocks.append(("M6 with curated team honours", m6))
    for title, lines in blocks:
        out += "\\par\\medskip\n" + tabular(
            "@{}>{\\raggedright\\arraybackslash}p{7.8cm}l@{}",
            [row([title, ""])],
            [row([label, value]) for label, value in lines],
        )
    return out + "\\par}\n"


def table_recognition_curated() -> str:
    """Post hoc (e): M6 with curated team honours beside the primary fit."""
    shares = {
        "Registered": csv(RESULTS / "recognition" / "shares.csv").set_index("block"),
        "Curated honours": csv(RESULTS / "recognition" / "curated" / "shares.csv").set_index(
            "block"
        ),
    }
    sums = {
        "Registered": summary("recognition"),
        "Curated honours": summary("recognition/curated"),
    }
    body: list[str] = []
    for model, label in (("ols", r"OLS, $\log(1 + \text{points})$"), ("pl", "Plackett--Luce")):
        body.append(rf"  \multicolumn{{3}}{{@{{}}l}}{{\textit{{{label}: Shapley share "
                    r"[90\% interval]}} \\")  # fmt: skip
        for block in shares["Registered"].index:
            body.append(
                row(
                    [
                        r"\quad " + BLOCK_LABEL[block],
                        *(
                            num(t.loc[block, model])
                            + " "
                            + interval(t.loc[block, f"{model}_lo"], t.loc[block, f"{model}_hi"])
                            for t in shares.values()
                        ),
                    ]
                )
            )
    body.append(r"  \midrule")
    for label, key, fmt in (
        (r"OLS within-edition $R^2$", "OlsRTwo", num),
        (r"OLS leave-one-edition-out $R^2$", "OlsRTwoLoeo", num),
        (r"Plackett--Luce McFadden pseudo-$R^2$", "PlPseudoRTwo", num),
        ("Bootstrap replicates with team share above statistics share (OLS)",
         "ShapleyTeamAboveStatsBoot", num),
        ("H6 verdict", "VerdictHSix", tex),
    ):  # fmt: skip
        body.append(row([label, *(fmt(s[key]) for s in sums.values())]))
    c = sums["Curated honours"]
    by_type = ", ".join(
        f"{label} {count(c[f'changed_{key}'])}"
        for key, label in zip(
            TEAM_INDICATORS,
            ("league title", "Champions League winner", "Champions League finalist",
             "international title"),
            strict=True,
        )
    )  # fmt: skip
    note = (
        r"\par\smallskip{\footnotesize\raggedright The curated honours change "
        + count(c["changed_total"])
        + " team-success indicators in "
        + count(curated_changed_editions())
        + " candidate-editions: "
        + by_type
        + r".\par}"
    )
    return (
        "{\\setlength{\\tabcolsep}{4pt}\n"
        + tabular(
            "@{}>{\\raggedright\\arraybackslash}p{6.6cm}cc@{}",
            [row(["", *shares])],
            body,
        )
        + note
        + "\\par}\n"
    )


def table_firsts_by_dimension() -> str:
    """Post hoc: for M8's leaders, the share of M5's specifications ranking each
    first, by family and by time aggregation."""
    frame = csv(RESULTS / "synthesis" / "firsts_by_dimension.csv")
    grid = csv(RESULTS / "multiverse" / "spec_grid.csv")
    families = grid["family"].drop_duplicates().tolist()
    aggregations = grid["aggregation"].drop_duplicates().tolist()
    share = frame.set_index(["player_id", "family", "aggregation"])["share_first"]
    leaders = frame.drop_duplicates("player_id").sort_values("m8_p_first", ascending=False)
    body = [
        row(
            [
                tex(r["player"]),
                num(r["m8_p_first"]),
                num(share[(r["player_id"], "all", "all")]),
                *(num(share[(r["player_id"], f, "all")]) for f in families),
                *(num(share[(r["player_id"], "all", a)]) for a in aggregations),
            ]
        )
        for _, r in leaders.iterrows()
    ]
    n_fam, n_agg = len(families), len(aggregations)
    header = [
        rf"  & M8 & \multicolumn{{{1 + n_fam + n_agg}}}{{c}}{{Share of M5 specifications "
        r"ranking him first} \\",
        rf"  \cmidrule(lr){{3-{3 + n_fam + n_agg}}}",
        rf"  & & & \multicolumn{{{n_fam}}}{{c}}{{family}} & \multicolumn{{{n_agg}}}{{c}}"
        r"{aggregation} \\",
        rf"  \cmidrule(lr){{4-{3 + n_fam}}}\cmidrule(lr){{{4 + n_fam}-{3 + n_fam + n_agg}}}",
        row(["Player", r"$P(\#1)$", "all", *(tex(f.replace("_", " ")) for f in families),
             *(tex(a) for a in aggregations)]),
    ]  # fmt: skip
    return (
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
        + tabular("@{}l" + "r" * (2 + n_fam + n_agg) + "@{}", header, body)
        + "\\par}\n"
    )


def table_club_rapm() -> str:
    s = summary("rapm_league_men")
    prefix = "club_total_primary_"
    clubs = sorted((k.removeprefix(prefix) for k in s if k.startswith(prefix)),
                   key=lambda club: -float(s[prefix + club]))  # type: ignore[arg-type]  # fmt: skip
    body = [
        row([tex(club), num(s[prefix + club], 3), num(s[f"club_total_league_{club}"], 3)])
        for club in clubs
    ]
    header = [row(["Club", "Registered fit", "Fit with league means"])]
    return tabular("@{}lrr@{}", header, body) + "\n"


def league_levels() -> pd.DataFrame:
    """The A22 fit's league means, best first, joined with the validation columns."""
    means = csv(RESULTS / "rapm_league_men" / "league_means.csv")
    levels = csv(RESULTS / "league" / "validation" / "levels.csv")
    keep = ["league", "mu22", "club_level", "club_level_se", "h2h_games", "h2h_gd", "uefa_mean"]
    return means.merge(levels[keep], on="league", how="left").sort_values("mu", ascending=False)


def _or_dash(value: object, text: str) -> str:
    return text if pd.notna(value) else "{--}"


def table_league_levels() -> str:
    def line(r: pd.Series) -> str:
        reference = r["se"] == 0
        if r["league"] == "OTHER":
            name = "No covered league"
        else:
            name = tex(league_name(r["league"])) + (r"$^\dagger$" if r["column"] == "REST" else "")
        played = pd.notna(r["h2h_gd"])
        return row(
            [
                name,
                "0" if reference else num(r["mu"], 3),
                "{--}" if reference else num(r["se"], 3),
                "{--}" if reference else interval(r["ci_low"], r["ci_high"], 3),
                "0" if reference else num(r["mu22"]),
                _or_dash(r["games_other_leagues"], count(np.nan_to_num(r["games_other_leagues"]))),
                _or_dash(r["linking_players"], count(np.nan_to_num(r["linking_players"]))),
                "0"
                if reference
                else _or_dash(
                    r["club_level"], f"{num(r['club_level'])} ({num(r['club_level_se'])})"
                ),
                f"{num(r['h2h_gd'])} ({count(r['h2h_games'])})" if played else "{--}",
                num(r["uefa_mean"], 1),
            ]
        )

    header = [
        r"  & \multicolumn{4}{c}{League mean (goals per 90)} & \multicolumn{2}{c}{Links}"
        r" & \multicolumn{3}{c}{Validation} \\",
        r"  \cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
        row(
            [
                "League",
                r"$\hat\mu_L$",
                "SE",
                r"90\% interval",
                r"$22\,\hat\mu_L$",
                "games",
                "players",
                "clubs (SE)",
                "H2H (games)",
                "UEFA",
            ]
        ),
    ]
    note = (
        r"\par\smallskip\raggedright $^\dagger$~Leagues sharing one mean. No covered league: "
        r"stint players without minutes in a covered league. Links: the two counts of the "
        r"pooling rule of Section~\ref{sec:league-method}. Clubs: the club-results level, in "
        r"goals per game relative to the Premier League; H2H: goal difference per game "
        r"against Premier League clubs; UEFA: mean association coefficient."
    )
    body = [line(r) for _, r in league_levels().iterrows()]
    return (
        "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
        + tabular("@{}lrrcrrrccr@{}", header, body)
        + note
        + "\\par}\n"
    )


FACTOR_METRICS = ("goals", "np_goals", "assists")


def table_league_factors() -> str:
    factors = csv(RESULTS / "league" / "factors.csv")
    wide = factors.pivot(index="league", columns=["metric", "model"], values=["multiplier", "se"])
    info = factors.drop_duplicates("league").set_index("league")[["n_linking", "status"]]
    order = wide[("multiplier", "goals", "context")].sort_values(ascending=False).index

    def cells(league: str) -> list[str]:
        return [
            num(wide.loc[league, ("multiplier", metric, model)])
            + " ("
            + num(wide.loc[league, ("se", metric, model)])
            + ")"
            for metric in FACTOR_METRICS
            for model in ("context", "plain")
        ]

    reference = info.index[info["status"] == "reference"][0]
    pooled = [league for league in order if info.loc[league, "status"] == "pooled"]
    body = [
        row([tex(league_name(reference)), count(info.loc[reference, "n_linking"]), *["1"] * 6]),
        *(
            row([tex(league_name(league)), count(info.loc[league, "n_linking"]), *cells(league)])
            for league in order
            if info.loc[league, "status"] == "own"
        ),
        r"  \midrule",
        row([f"{count(len(pooled))} other leagues$^\\dagger$", "{--}", *cells(pooled[0])]),
    ]
    header = [
        "  & & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{METRIC_LABEL[m]}}}" for m in FACTOR_METRICS)
        + r" \\",
        r"  \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        row(["League", "Players", *["team term", "none"] * 3]),
    ]
    note = (
        r"\par\smallskip\raggedright Factor $\exp(-\hat\gamma_L)$ with the standard error of "
        r"$\hat\gamma_L$ in parentheses, with the team term (A21) and without it. Players: "
        r"players observed in the league and in another. $^\dagger$~One factor shared by: "
        + ", ".join(tex(league_name(league)) for league in sorted(pooled, key=league_name))
        + "."
    )
    return (
        "{\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        + tabular("@{}lr" + "c" * 6 + "@{}", header, body)
        + note
        + "\\par}\n"
    )


# Column groups of tab-league-ranking, in the caption's order; a run that has not
# happened (no columns in league_comparison.csv) is left out.
RANKING_VARIANTS = (
    ("primary", "Registered"),
    ("league_adjusted", "League-adjusted (A21)"),
    ("league_plain", "No team term"),
    ("league_uefa", "UEFA leagues only"),
    ("league_mls", "MLS fill"),
    ("league_leakage_independent", r"\shortstack{Contamination,\\independent $A_g$}"),
    ("league_leakage_assortative", r"\shortstack{Contamination,\\assortative $A_g$}"),
    ("league_club", r"\shortstack{Club-results\\levels (A24)}"),
)
RANKING_PER_PANEL = 4  # variants side by side that fit \linewidth


def table_league_ranking() -> str:
    """Players in the top ten by median rank or by P(#1) under the registered or the
    league-adjusted run, by registered P(#1); the variants in stacked panels."""
    table = league_comparison()
    chosen: set[str] = set()
    for run in ("primary", "league_adjusted"):
        chosen |= set(table.nlargest(10, f"{run}_p_first").index)
        by_rank = table.sort_values([f"{run}_median_rank", f"{run}_p_first"],
                                    ascending=[True, False])  # fmt: skip
        chosen |= set(by_rank.index[:10])
    players = table.loc[sorted(chosen, key=lambda name: -table.loc[name, "primary_p_first"])]
    variants = [(k, label) for k, label in RANKING_VARIANTS if f"{k}_p_first" in table.columns]
    panels = []
    for start in range(0, len(variants), RANKING_PER_PANEL):
        part = variants[start : start + RANKING_PER_PANEL]
        header = [
            "  & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{label}}}" for _, label in part)
            + r" \\",
            "  " + "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(part))),
            row(["Player", *[r"\#1", "top 3", "rank"] * len(part)]),
        ]  # fmt: skip
        body = [
            row(
                [
                    tex(name),
                    *(
                        cell
                        for k, _ in part
                        for cell in (
                            num(r[f"{k}_p_first"]),
                            num(r[f"{k}_p_top3"]),
                            count(r[f"{k}_median_rank"])
                            + " "
                            + _rank_set(r[f"{k}_rank_lo"], r[f"{k}_rank_hi"]),
                        )
                    ),
                ]
            )
            for name, r in players.iterrows()
        ]
        # A wider gap before each variant keeps its three columns together.
        panels.append(tabular("@{}l" + r"@{\hspace{10pt}}rrr" * len(part) + "@{}", header, body))
    note = (
        r"\par\smallskip\raggedright \#1 and top 3: $P(\#1)$ and $P(\text{top-3})$; rank: "
        r"median rank [90\% rank interval]. Players in the top ten by median rank or by "
        r"$P(\#1)$ under the registered or the league-adjusted mixture."
    )
    return (
        "{\\scriptsize\\setlength{\\tabcolsep}{2pt}\n"
        + "\\par\\medskip\n".join(panels)
        + note
        + "\\par}\n"
    )


def table_on_off() -> str:
    """The run's own list (rapm summary on_off_players): the impact view's top ten,
    then the other players it names."""
    listed = [str(r["player"]) for r in summary("rapm")["on_off_players"]]  # type: ignore[union-attr]
    ten = set(synthesis_players("outfield").nsmallest(10, "impact_view")["player"])
    table = on_off()

    def line(name: str) -> str:
        r = table.loc[name]
        return row(
            [
                tex(name),
                num(r["gd_on_p90"]),
                num(r["gd_off_p90"]),
                num(r["on_minus_off"]),
                num(r["minutes_share"]),
            ]
        )

    header = [
        r"  & \multicolumn{3}{c}{Goal difference per 90} & Share of his \\",
        r"  \cmidrule(lr){2-4}",
        row(["Player", "on the pitch", "off it", "on minus off", "clubs' minutes"]),
    ]
    body = [
        *(line(name) for name in listed if name in ten),
        r"  \midrule",
        *(line(name) for name in listed if name not in ten),
    ]
    return tabular("@{}lrrrr@{}", header, body) + "\n"


def table_leakage() -> str:
    runs = {
        "Registered": summary("synthesis"),
        r"Independent-planting $A_g$": summary("synthesis/leakage_independent"),
        r"Assortative-planting $A_g$": summary("synthesis/leakage_assortative"),
    }
    lines = (
        (r"Player with the highest $P(\#1)$", lambda s: tex(s["TopPlayerName"])),
        (r"His $P(\#1)$", lambda s: num(s["TopPlayerPFirst"])),
        ("Share of outfield pairs decided", lambda s: num(s["SharePairsDecided"])),
        ("Verdicts H1, H2, H4, H5",
         lambda s: ", ".join(tex(s[k]) for k in ("VerdictHOne", "VerdictHTwo", "VerdictHFour",
                                                  "VerdictHFive"))),
    )  # fmt: skip
    body = [row([label, *(f(s) for s in runs.values())]) for label, f in lines]
    return (
        "{\\setlength{\\tabcolsep}{4pt}\n"
        + tabular(
            "@{}>{\\raggedright\\arraybackslash}p{4.6cm}"
            + ">{\\raggedright\\arraybackslash}p{3.1cm}" * len(runs)
            + "@{}",
            [row(["", *runs])],
            body,
        )
        + "\\par}\n"
    )


def figure_league_levels(width: float) -> Figure:
    levels = league_levels()
    levels = levels[levels["own_mean"] & (levels["league"] != "OTHER")].reset_index(drop=True)
    y = np.arange(len(levels))[::-1]
    reference = (levels["se"] == 0).to_numpy()
    fig, ax = plt.subplots(figsize=(width, 0.19 * len(levels) + 0.8))
    _clean(ax)
    ax.grid(axis="x", zorder=0)
    ax.axvline(0.0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.hlines(y[~reference], levels["ci_low"][~reference], levels["ci_high"][~reference],
              color=BLUE, lw=1.3, zorder=2)  # fmt: skip
    ax.plot(levels["mu"][~reference], y[~reference], "o", ms=4.6, color=BLUE, mec=SURFACE,
            mew=0.6, zorder=3, label="league mean")  # fmt: skip
    ax.plot(levels["mu"][reference], y[reference], "o", ms=5, mfc=SURFACE, mec=INK_SECONDARY,
            mew=1.2, zorder=3, label="reference (fixed at 0)")  # fmt: skip
    ax.set_yticks(y, [league_name(code) for code in levels["league"]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("League mean (goals per 90 per player, Premier League = 0)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=BLUE, lw=1.3))
    labels.append("90% bootstrap interval")
    ax.legend(handles, labels, loc="lower right")
    fig.tight_layout()
    return fig


def table_distinct_rankings() -> str:
    """Distinct rankings per family and the family's weight share when each distinct
    ranking counts once: registered and corrected factorial, outfield and goalkeepers."""
    runs = (summary("synthesis/primary_distinct"), summary("synthesis/corrected"))

    def cells(family: str | None) -> list[str]:
        out = []
        for segment in ("outfield", "goalkeeper"):
            for s in runs:
                if family is None:
                    n = s[f"n_distinct_{segment}"]
                    share = sum(float(s[f"weight_share_{segment}_{f}"]) for f in FAMILIES)  # type: ignore[arg-type]
                else:
                    n = s[f"n_distinct_{segment}_{family}"]
                    share = float(s[f"weight_share_{segment}_{family}"])  # type: ignore[arg-type]
                out += [count(n), num(share)]
        return out

    factorials = ["Registered", "Corrected"] * 2
    header = [
        r"  & \multicolumn{4}{c}{Outfield pool} & \multicolumn{4}{c}{Goalkeepers} \\",
        r"  \cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        "  & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{f}}}" for f in factorials) + r" \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        row(["Family", *["rankings", "weight"] * 4]),
    ]  # fmt: skip
    body = [
        *(row([label, *cells(family)]) for family, label in FAMILIES.items()),
        r"  \midrule",
        row(["All", *cells(None)]),
    ]
    return tabular("@{}l" + "rr" * 4 + "@{}", header, body) + "\n"


CORRECTED_ROWS = 20  # places shown in the main table and figure


def table_corrected_ranking() -> str:
    table = corrected_comparison().iloc[:CORRECTED_ROWS]
    body = [
        row(
            [
                count(r["corrected_place"]),
                tex(name),
                tex(r["position_group"]),
                count(r["corrected_median_rank"]),
                _rank_set(r["corrected_rank_lo"], r["corrected_rank_hi"]),
                num(r["corrected_p_first"]),
                num(r["corrected_p_top3"]),
                count(r["primary_place"]),
                count(r["primary_median_rank"]),
                num(r["primary_p_first"]),
            ]
        )
        for name, r in table.iterrows()
    ]
    header = [
        r"  & & & \multicolumn{4}{c}{Corrected mixture}"
        r" & \multicolumn{3}{c}{Registered mixture} \\",
        r"  \cmidrule(lr){4-7}\cmidrule(lr){8-10}",
        row(
            [
                "Place",
                "Player",
                "Group",
                r"\shortstack{median\\rank}",
                r"\shortstack{90\%\\interval}",
                r"$P(\#1)$",
                r"$P(\text{top-3})$",
                "place",
                r"\shortstack{median\\rank}",
                r"$P(\#1)$",
            ]
        ),
    ]
    return (
        "{\\setlength{\\tabcolsep}{3pt}\n" + tabular("@{}rllrcrrrrr@{}", header, body) + "\\par}\n"
    )


def _goalkeeper_places(path: Path) -> pd.DataFrame:
    players = csv(path)
    keepers = players[players["segment"] == "goalkeeper"].sort_values(["median_rank", "mean_rank"])
    return keepers.assign(place=range(1, len(keepers) + 1)).set_index("player")


def table_corrected_goalkeepers() -> str:
    corrected = _goalkeeper_places(RESULTS / "synthesis" / "corrected" / "players.csv")
    registered = _goalkeeper_places(RESULTS / "synthesis" / "players.csv")["place"]
    body = [
        row(
            [
                tex(name),
                count(r["median_rank"]),
                _rank_set(r["rank_lo"], r["rank_hi"]),
                num(r["p_first"]),
                num(r["p_top3"]),
                count(registered[name]),
            ]
        )
        for name, r in corrected.iterrows()
    ]
    header = [
        r"  & \multicolumn{4}{c}{Corrected mixture} & Registered \\",
        r"  \cmidrule(lr){2-5}",
        row(["Goalkeeper", "median rank", r"90\% interval", r"$P(\#1)$", r"$P(\text{top-3})$",
             "place"]),
    ]  # fmt: skip
    return tabular("@{}lrcrrr@{}", header, body) + "\n"


SENSITIVITY_LABEL = {
    "corrected": "as run",
    "corrected_equal_weights": r"\shortstack{equal\\weights}",
    "corrected_no_contamination": r"\shortstack{no conta-\\mination}",
    "corrected_assortative": r"\shortstack{real-struc-\\ture $A_g$}",
    "league_adjusted": r"\shortstack{League-\\adjusted}",
    "primary_distinct": r"\shortstack{distinct\\weights}",
    "primary": "as run",
}


def table_corrected_sensitivity() -> str:
    table = corrected_comparison()
    places = table[[f"{run}_place" for run in PLACE_RUNS]]
    shown = table[(places <= 10).any(axis=1)]
    sums = [run_summary(run) for run in PLACE_RUNS]
    # First and last name on two lines, so a name is no wider than a place cell.
    leaders = [r"\shortstack{" + tex(s["TopPlayerName"]).replace(" ", r"\\", 1) + "}" for s in sums]
    body = [
        row(
            [
                tex(name),
                *(
                    f"{count(r[f'{run}_place'])} ({count(r[f'{run}_median_rank'])})"
                    for run in PLACE_RUNS
                ),
            ]
        )
        for name, r in shown.iterrows()
    ]
    body += [
        r"  \midrule",
        row([r"Highest $P(\#1)$", *leaders]),
        row([r"His $P(\#1)$", *(num(s["TopPlayerPFirst"]) for s in sums)]),
        row(["Share of pairs decided", *(num(s["SharePairsDecided"]) for s in sums)]),
    ]
    header = [
        r"  & \multicolumn{4}{c}{Corrected} & & \multicolumn{2}{c}{Registered} \\",
        r"  \cmidrule(lr){2-5}\cmidrule(lr){7-8}",
        row(["Player", *(SENSITIVITY_LABEL[run] for run in PLACE_RUNS)]),
    ]
    return (
        "{\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
        + tabular("@{}l" + "c" * len(PLACE_RUNS) + "@{}", header, body)
        + "\\par}\n"
    )


def figure_corrected_ranking(width: float) -> Figure:
    full = corrected_comparison()
    table = full.iloc[:CORRECTED_ROWS]
    y = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=(width, 0.2 * len(table) + 0.8))
    _clean(ax)
    ax.grid(axis="x", zorder=0)
    ax.hlines(y, table["corrected_rank_lo"], table["corrected_rank_hi"], color=INK_SECONDARY,
              lw=1.2, zorder=2, label="90% rank interval")  # fmt: skip
    ax.plot(table["corrected_median_rank"], y, "o", ms=5, color=BLUE, mec=SURFACE, mew=0.7,
            zorder=3, label="median rank")  # fmt: skip
    ax.set_yticks(y, [f"{place}. {name}" for place, name in
                      zip(table["corrected_place"], table.index, strict=True)])  # fmt: skip
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(1, len(full))
    ax.set_xticks([1, *range(10, len(full) + 1, 10)])
    ax.set_xlabel(f"Rank in the corrected mixture (1 = best of {len(full)} outfield players)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


TABLES: dict[str, Callable[[], str]] = {
    "tab-coverage": table_coverage,
    "tab-reliability": table_reliability,
    "tab-composite": table_composite,
    "tab-icc": table_icc,
    "tab-rapm-reliability": table_rapm_reliability,
    "tab-unmeasured": table_unmeasured,
    "tab-h3": table_h3,
    "tab-multiverse": table_multiverse,
    "tab-recognition": table_recognition,
    "tab-visibility": table_visibility,
    "tab-clutch": table_clutch,
    "tab-synthesis": table_synthesis,
    "tab-goalkeepers": table_goalkeepers,
    "tab-views": table_views,
    "tab-recovery": table_recovery,
    "tab-v1": table_v1,
    "tab-verdicts": table_verdicts,
    "tab-robustness": table_robustness,
    "tab-firsts-by-dimension": table_firsts_by_dimension,
    "tab-recognition-curated": table_recognition_curated,
    "tab-leakage": table_leakage,
    "tab-club-rapm": table_club_rapm,
    "tab-league-levels": table_league_levels,
    "tab-league-factors": table_league_factors,
    "tab-league-ranking": table_league_ranking,
    "tab-on-off": table_on_off,
    "tab-distinct-rankings": table_distinct_rankings,
    "tab-corrected-ranking": table_corrected_ranking,
    "tab-corrected-goalkeepers": table_corrected_goalkeepers,
    "tab-corrected-sensitivity": table_corrected_sensitivity,
    "tab-pool": table_pool,
}
FIGURES: dict[str, Callable[[float], Figure]] = {
    "fig-reliability": figure_reliability,
    "fig-scree": figure_scree,
    "fig-corr-heatmap": figure_corr_heatmap,
    "fig-rapm-pool": figure_rapm_pool,
    "fig-unmeasured-share": figure_unmeasured_share,
    "fig-h3-parametric": figure_h3_parametric,
    "fig-spec-curve": figure_spec_curve,
    "fig-total-order": figure_total_order,
    "fig-shapley": figure_shapley,
    "fig-visibility": figure_visibility,
    "fig-knockout": figure_knockout,
    "fig-pfirst": figure_pfirst,
    "fig-variance-decomposition": figure_variance_decomposition,
    "fig-decided-pairs": figure_decided_pairs,
    "fig-recovery": figure_recovery,
    "fig-league-levels": figure_league_levels,
    "fig-corrected-ranking": figure_corrected_ranking,
}
MACRO_SOURCES: tuple[Callable[[], dict[str, str]], ...] = (
    macros_data,
    macros_linking,
    macros_pool,
    macros_m1,
    macros_m2,
    macros_m3,
    macros_m4,
    macros_m7a,
    macros_m7b,
    macros_m8_views,
    macros_rapm_seasons,
    macros_point_ranks,
    macros_m6_extra,
    macros_assortative,
    macros_bias_corrected,
    macros_europe,
    macros_curated,
    macros_leakage,
    macros_league_rapm,
    macros_league_validation,
    macros_league_ranking,
    macros_league_club,
    macros_corrected,
    macros_on_off,
)


def _missing(error: FileNotFoundError) -> str:
    path = Path(str(error.filename)) if error.filename else None
    if path is not None and path.is_absolute():
        return f"missing {path.relative_to(ROOT)}"
    return f"missing {error}"


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    declared = declared_macros()
    macros: dict[str, str] = {}
    skipped: list[str] = []
    for source in MACRO_SOURCES:
        try:
            macros.update(source())
        except FileNotFoundError as error:
            skipped.append(f"{source.__name__}: {_missing(error)}")
    for path, entries in KEYED.items():
        try:
            values = json_values(RESULTS / path)
        except FileNotFoundError as error:
            skipped.append(f"{path}: {_missing(error)}")
            continue
        macros.update({macro: fmt(values[key]) for macro, key, fmt in entries if key in values})
    macros.update(macros_named(set(declared), set(macros)))
    stray = sorted(set(macros) - set(declared))
    if stray:
        raise SystemExit(f"macros not declared in macros-defaults.tex: {stray}")
    (GENERATED / "macros.tex").write_text(macro_block(macros))
    print(f"wrote generated/macros.tex ({len(macros)} of {len(declared)} macros)")

    pending: dict[str, str] = {}
    _style()
    for name, width in slots().items():
        try:
            if name in TABLES:
                (GENERATED / f"{name}.tex").write_text(TABLES[name]())
            elif name in FIGURES and width is not None:
                fig = FIGURES[name](width * LINEWIDTH_IN)
                fig.savefig(GENERATED / f"{name}.pdf", format="pdf")
                plt.close(fig)
            else:
                pending[name] = "no renderer for this slot"
                continue
            print(f"wrote generated/{name}")
        except FileNotFoundError as error:
            pending[name] = _missing(error)

    missing = [m for m in declared if m not in macros]
    print(f"\npending macros ({len(missing)}): {', '.join(missing) or 'none'}")
    for line in skipped:
        print(f"  skipped {line}")
    print(f"pending tables and figures ({len(pending)}):")
    for name, reason in pending.items():
        print(f"  {name}: {reason}")
    for claim in CHECKS:
        print(f"text check failed: {claim}")


if __name__ == "__main__":
    main()
