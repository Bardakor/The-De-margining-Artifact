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


def summary(module: str) -> dict[str, object]:
    """A module's summary.json as key -> value (raises if the run has not happened)."""
    payload = json.loads((RESULTS / module / "summary.json").read_text())
    return {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in payload.items()
    }


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
    panel = csv(PROCESSED / "panel_player_match.csv.gz", usecols=["player_id"])
    seasons = csv(PROCESSED / "panel_player_season.csv.gz", usecols=["scope", "minutes"])
    with_stints = set(tm_stints["game_id"]) | set(sb_stints["game_id"])
    qualifying = (seasons["scope"] == "all") & (seasons["minutes"] >= 900)
    return {
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
    h3 = csv(RESULTS / "construct_gap" / "h3_summary.csv").set_index("kind").loc["clipped"]
    s = summary("construct_gap")
    groups = shares.loc[list(GROUPS)]
    outside = (groups["r2_true"] < 0) | (groups["r2_true"] > 1)
    r2_full = shares.loc[list(OUTFIELD), "r2_full"]
    return {
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
    }
    # The run states its test level only in the description of the power.
    level = re.search(r"alpha (\d*\.\d+)", descriptions("clutch")["power_at_03"])
    if level:
        out["HSevenAlpha"] = num(float(level.group(1)))
    return out


def macros_named(declared: set[str], done: set[str]) -> dict[str, str]:
    """Keys of any summary.json that are macro names themselves (ranking stage)."""
    out: dict[str, str] = {}
    for path in sorted(RESULTS.glob("*/summary.json")):
        for key, value in summary(path.parent.name).items():
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
        r"900 minutes in both halves. Attributed reliability: split-half reliability of "
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
        row(["Planted contrast $D_0$", num(raw["planted"]), "", num(clip["planted"]), ""]),
        row(["Mean error", num(raw["mean_error"]), "", num(clip["mean_error"]), ""]),
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
                "Coverage of $D_0$ by the player-bootstrap interval",
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
    return tabular("@{}lrcrc@{}", header, body) + "\n"


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


def _synthesis_table(players: pd.DataFrame, note: str) -> str:
    body = [
        row(
            [
                tex(r["player"]),
                num(r["p_first"]),
                num(r["p_top3"]),
                num(r["p_top10"]),
                count(r["median_rank"]) + " " + _rank_set(r["rank_lo"], r["rank_hi"]),
                _rank_set(r["marginal_lo"], r["marginal_hi"]),
                num(r["share_specification"]),
                num(r["share_sampling"]),
                num(r["share_unmeasured"]),
            ]
        )
        for _, r in players.iterrows()
    ]
    header = [
        r"  & & & & \multicolumn{2}{c}{Rank} & \multicolumn{3}{c}{Share of rank variance} \\",
        r"  \cmidrule(lr){5-6}\cmidrule(lr){7-9}",
        row(
            [
                "Player",
                r"$P(\#1)$",
                r"$P(\text{top-3})$",
                r"$P(\text{top-10})$",
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
        + tabular("@{}lrrrrcrrr@{}", header, body)
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
        r"specification.\par}"
    )
    return _synthesis_table(synthesis_players("goalkeeper"), note)


def recognition_view() -> pd.Series:
    """Ballon d'Or points summed over 2021--2026 per pool player, 0 if unranked
    (A13 (v), as run_ranking.py computes it for Kendall's W)."""
    pool = csv(POOL).set_index("player")["tm_player_id"]
    bdo = csv(PAPER / "data" / "ballon_dor.csv")
    bdo = bdo[bdo["edition"].between(2021, 2026)]
    points = bdo.groupby(bdo["player"].map(pool))["points"].sum()
    return points.reindex(pool.to_numpy(), fill_value=0).astype(int)


def table_views() -> str:
    players = synthesis_players("outfield").set_index("player_id")
    points = recognition_view().reindex(players.index)
    players = players.assign(points=points, recognition=points.rank(ascending=False, method="min"))
    players = players.sort_values(["recognition", "stats_view"])
    lines = [
        row(
            [
                tex(r["player"]),
                num(r["stats_view"], 1),
                num(r["impact_view"], 1),
                count(r["points"]),
                count(r["recognition"]),
            ]
        )
        for _, r in players.iterrows()
    ]
    half = (len(lines) + 1) // 2
    header = [
        r"  & \multicolumn{2}{c}{Mean rank} & \multicolumn{2}{c}{Ballon d'Or} \\",
        r"  \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        row(["Player", "stats", "impact", "points", "rank"]),
    ]
    colspec = "@{}lrrrr@{}"
    note = (
        r"\par\smallskip\raggedright Ordered by Ballon d'Or rank (points summed over "
        r"2021--2026; ties share the best rank). Kendall's $W$ = \KendallW{} (mean pairwise "
        r"Spearman correlation \KendallMeanSpearman{})."
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
    fig, ax = plt.subplots(figsize=(width, 2.9))
    _clean(ax)
    ax.grid(axis="y", zorder=0)
    ax.axvspan(h3["ci_low"], h3["ci_high"], color=BLUE, alpha=0.12, lw=0, zorder=0,
               label=f"90% basic interval [{h3['ci_low']:.2f}, {h3['ci_high']:.2f}]")  # fmt: skip
    ax.hist(values, bins=24, color=INK_MUTED, edgecolor=SURFACE, linewidth=0.6, zorder=2,
            label=f"simulated estimates (S = {values.size})")  # fmt: skip
    ax.axvline(h3["planted"], color=INK, lw=1.3, ls=(0, (4, 2)), zorder=3,
               label=f"planted contrast {h3['planted']:.2f}")  # fmt: skip
    ax.axvline(h3["estimate"], color=BLUE, lw=1.8, zorder=3,
               label=f"real estimate {h3['estimate']:.2f}")  # fmt: skip
    ax.set_xlabel("H3 contrast, clipped share CB+DM minus CF+W")
    ax.set_ylabel("Simulations")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
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
    """Post-hoc robustness, not a main.tex slot: the M8 headline numbers under each
    alternative synthesis next to the primary, and H3 under assortative planting."""
    variants = {"Primary": RESULTS / "synthesis"}
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
    path = RESULTS / "construct_gap" / "assortative" / "summary.json"
    if path.exists():
        a = summary("construct_gap/assortative")
        h3 = [
            ("Planted contrast $D_0$", num(a["contrast_clipped_planted"])),
            (r"Clipped estimate and 90\% basic interval",
             num(a["contrast_clipped_estimate"]) + " "
             + interval(a["contrast_clipped_ci_low"], a["contrast_clipped_ci_high"])),
            ("Leave-one-out coverage; player-bootstrap coverage of $D_0$",
             num(a["contrast_clipped_coverage_loo"]) + "; "
             + num(a["contrast_clipped_coverage_m4"])),
            ("H3 verdict (clipped)", tex(a["verdict_clipped"])),
        ]  # fmt: skip
        out += "\\par\\medskip\n" + tabular(
            "@{}>{\\raggedright\\arraybackslash}p{7.8cm}l@{}",
            [row([r"H3 with planted effects on the real teammate structure (assortative)", ""])],
            [row([label, value]) for label, value in h3],
        )
    return out + "\\par}\n"


def table_firsts_by_dimension() -> str:
    """Post hoc, not a main.tex slot: for M8's leaders, the share of M5's
    specifications ranking each first, by family and by time aggregation."""
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


# Post-hoc robustness tables: written when their source exists, not yet main.tex slots.
EXTRA_TABLES: dict[str, Callable[[], str]] = {
    "tab-robustness": table_robustness,
    "tab-firsts-by-dimension": table_firsts_by_dimension,
}

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
                pending[name] = "no renderer yet (ranking-stage result files)"
                continue
            print(f"wrote generated/{name}")
        except FileNotFoundError as error:
            pending[name] = _missing(error)
    for name, render in EXTRA_TABLES.items():
        try:
            (GENERATED / f"{name}.tex").write_text(render())
            print(f"wrote generated/{name} (post hoc; main.tex has no slot for it)")
        except FileNotFoundError as error:
            pending[name] = f"post hoc, {_missing(error)}"

    missing = [m for m in declared if m not in macros]
    print(f"\npending macros ({len(missing)}): {', '.join(missing) or 'none'}")
    for line in skipped:
        print(f"  skipped {line}")
    print(f"pending tables and figures ({len(pending)}):")
    for name, reason in pending.items():
        print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
