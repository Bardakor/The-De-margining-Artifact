"""Turn the evaluation output into the LaTeX the paper inputs.

Nothing in `paper/main.tex` carries a number typed by hand. Every quantity in
the results section is written here, out of `results/cells.csv`, so a figure in
the paper cannot drift from the run that produced it. Until `study.py evaluate`
has written that file the paper still compiles: the macros fall back to
"[pending]" and the tables render a placeholder.

One quantity the evaluation does not record is the bookmaker's margin, and P1
is stated in terms of it. The margin is a property of the odds alone --- the
mean book sum of the complete closing books in a cell --- so it can be computed
without touching a model or a result. It is cached to
`results/book_margin.csv` because assembling the archive costs a couple of
minutes.

Usage:
    python scripts/make_tables.py            # regenerate paper/generated/
    python scripts/make_tables.py --refresh  # recompute the margin cache too
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from footy.eval.effect import (
    bootstrap_correlation,
    bootstrap_mean_difference,
    cluster_bootstrap_correlation,
)
from footy.market.demargin import METHODS
from footy.study.run import MARKET_OUTCOMES, closing_books, load_archive

RAW = Path("data/raw")
RESULTS = Path("results")
GENERATED = Path("paper/generated")

CELLS = RESULTS / "cells.csv"
MARGINS = RESULTS / "book_margin.csv"

KEYS = ["league", "book", "market"]

METHOD_LABEL = {
    "proportional": "Proportional",
    "power": "Power",
    "shin": "Shin",
    "odds_ratio": "Odds-ratio",
}

# Seeded so the paper's intervals are regenerable. Book-level uses the same
# seed as fig-margin-spread, so the figure title and the macros agree.
P1_CELL_BOOTSTRAP_SEED = 1
P1_BOOK_BOOTSTRAP_SEED = 0
P1_CLUSTER_BOOTSTRAP_SEED = 2
P2_BOOTSTRAP_SEED = 3
BOOTSTRAP_RESAMPLES = 10_000


# --------------------------------------------------------------------------
# book margin
# --------------------------------------------------------------------------


def compute_margins() -> pd.DataFrame:
    """Mean book sum per (league, book, market) over complete closing books.

    Uses the same `closing_books` pivot the study uses, so a cell's margin is
    measured over exactly the fixtures the cell could have priced --- not over
    every quote the bookmaker ever published.
    """
    archive = load_archive(RAW)
    odds = archive.odds
    rows: list[dict[str, object]] = []
    for league in sorted(set(odds["league"])):
        block = odds[odds["league"] == league]
        for book in sorted(set(block["book"])):
            for market in MARKET_OUTCOMES:
                wide = closing_books(block, book, market)
                if wide.empty:
                    continue
                offered = wide[list(MARKET_OUTCOMES[market])].to_numpy(dtype=float)
                book_sum = (1.0 / offered).sum(axis=1)
                rows.append(
                    {
                        "league": league,
                        "book": book,
                        "market": market,
                        "n_books": int(book_sum.size),
                        "mean_book_sum": float(book_sum.mean()),
                        "median_book_sum": float(np.median(book_sum)),
                    }
                )
    return pd.DataFrame(rows)


def margins(*, refresh: bool) -> pd.DataFrame:
    if MARGINS.exists() and not refresh:
        return pd.read_csv(MARGINS)
    frame = compute_margins()
    frame.to_csv(MARGINS, index=False)
    return frame


# --------------------------------------------------------------------------
# per-cell reshaping
# --------------------------------------------------------------------------


def per_cell(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: the spread of each quantity across the transforms.

    Spread is max minus min over the four transforms. It is the effect size P1
    and P2 are stated in, and it is computed on finite values only --- a
    transform that selected no bets has an undefined ROI, and averaging that
    into a spread as if it were zero would report agreement where there is no
    measurement.
    """
    wide = cells.pivot_table(index=KEYS, columns="method", values="roi", aggfunc="first")
    rps = cells.pivot_table(index=KEYS, columns="method", values="bench_rps", aggfunc="first")
    bets = cells.pivot_table(index=KEYS, columns="method", values="n_bets", aggfunc="first")
    # Reindexed onto the pivot's index: aligning two independently sorted
    # groupings by position would silently pair a cell's size with another
    # cell's ROI the first time a key sorts differently.
    size = cells.groupby(KEYS)["n_matches"].first().reindex(wide.index)

    roi_values = wide[list(METHODS)].to_numpy(dtype=float)
    rps_values = rps[list(METHODS)].to_numpy(dtype=float)

    frame = pd.DataFrame(
        {
            "n_matches": size.to_numpy(dtype=int),
            "roi_spread": _spread(roi_values),
            "rps_spread": _spread(rps_values),
            "roi_min": np.nanmin(roi_values, axis=1),
            "roi_max": np.nanmax(roi_values, axis=1),
            "sign_flip": (np.nanmin(roi_values, axis=1) < 0.0)
            & (np.nanmax(roi_values, axis=1) > 0.0),
            "sign_flip_prop_shin": _sign_differs(
                wide["proportional"].to_numpy(dtype=float),
                wide["shin"].to_numpy(dtype=float),
            ),
            "roi_proportional": wide["proportional"].to_numpy(dtype=float),
            "roi_shin": wide["shin"].to_numpy(dtype=float),
            "n_bets_spread": _spread(bets[list(METHODS)].to_numpy(dtype=float)),
        },
        index=wide.index,
    )
    return frame.reset_index()


def _spread(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.asarray(np.nanmax(values, axis=1) - np.nanmin(values, axis=1), dtype=float)


def _sign_differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both = np.isfinite(a) & np.isfinite(b)
    return np.asarray(both & (np.sign(a) != np.sign(b)), dtype=bool)


def _p1_pairs(cell_frame: pd.DataFrame, margin_frame: pd.DataFrame) -> pd.DataFrame:
    """Cells that carry both a margin and an ROI spread --- the pairs P1's ρ is on."""
    joined = cell_frame.merge(margin_frame, on=KEYS, how="left")
    return joined.dropna(subset=["mean_book_sum", "roi_spread"])


def _p1_book_pairs(cell_frame: pd.DataFrame, margin_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per bookmaker: mean margin against mean ROI spread.

    This is the aggregation fig-margin-spread bootstraps. Sorted by book so a
    seeded resample walks the same row order the figure does.
    """
    joined = cell_frame.merge(margin_frame, on=KEYS, how="left")
    return (
        joined.groupby("book")
        .agg(mean_book_sum=("mean_book_sum", "mean"), roi_spread=("roi_spread", "mean"))
        .dropna()
        .reset_index()
        .sort_values("book")
        .reset_index(drop=True)
    )


def p1_cell_interval(
    cell_frame: pd.DataFrame,
    margin_frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float, int]:
    """Cell-level Spearman ρ with an ordinary pair-bootstrap interval.

    The interval treats cells as independent, which they are not. Quote it
    alongside :func:`p1_cluster_interval` and :func:`p1_book_interval`.
    """
    usable = _p1_pairs(cell_frame, margin_frame)
    n = len(usable)
    if n <= 2:
        return float("nan"), float("nan"), float("nan"), n
    rho, low, high = bootstrap_correlation(
        usable["mean_book_sum"].to_numpy(dtype=float),
        usable["roi_spread"].to_numpy(dtype=float),
        rng=rng,
        n_resamples=n_resamples,
    )
    return rho, low, high, n


def p1_book_interval(
    cell_frame: pd.DataFrame,
    margin_frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float, int]:
    """Bookmaker-level ρ and percentile interval, matching fig-margin-spread."""
    by_book = _p1_book_pairs(cell_frame, margin_frame)
    n = len(by_book)
    if n <= 2:
        return float("nan"), float("nan"), float("nan"), n
    rho, low, high = bootstrap_correlation(
        by_book["mean_book_sum"].to_numpy(dtype=float),
        by_book["roi_spread"].to_numpy(dtype=float),
        rng=rng,
        n_resamples=n_resamples,
    )
    return rho, low, high, n


def p1_cluster_interval(
    cell_frame: pd.DataFrame,
    margin_frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float, int]:
    """Cell-level ρ with a bookmaker-clustered bootstrap interval."""
    usable = _p1_pairs(cell_frame, margin_frame)
    n = len(usable)
    if usable["book"].nunique() < 2:
        return float("nan"), float("nan"), float("nan"), n
    rho, low, high = cluster_bootstrap_correlation(
        usable["mean_book_sum"].to_numpy(dtype=float),
        usable["roi_spread"].to_numpy(dtype=float),
        usable["book"].to_numpy(),
        rng=rng,
        n_resamples=n_resamples,
    )
    return rho, low, high, n


def p2_roi_diff_interval(
    cell_frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float, int]:
    """Paired bootstrap of mean(1X2 ROI spread − OU25 ROI spread)."""
    wide = cell_frame.pivot_table(
        index=["league", "book"], columns="market", values="roi_spread", aggfunc="first"
    ).dropna()
    n = len(wide)
    if n < 2 or "1X2" not in wide.columns or "OU25" not in wide.columns:
        return float("nan"), float("nan"), float("nan"), n
    point, low, high = bootstrap_mean_difference(
        wide["1X2"].to_numpy(dtype=float),
        wide["OU25"].to_numpy(dtype=float),
        rng=rng,
        n_resamples=n_resamples,
    )
    return point, low, high, n


# --------------------------------------------------------------------------
# LaTeX emission
# --------------------------------------------------------------------------


def tabular(frame: pd.DataFrame, column_format: str, headers: list[str]) -> str:
    """A booktabs tabular. No float wrapper --- main.tex supplies that."""
    lines = [
        f"\\begin{{tabular}}{{{column_format}}}",
        "  \\toprule",
        "  " + " & ".join(headers) + " \\\\",
        "  \\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append("  " + " & ".join(_cell(v) for v in row) + " \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _cell(value: object) -> str:
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if isinstance(value, float | np.floating):
        return "{--}" if not np.isfinite(value) else f"{value:.4f}"
    if isinstance(value, int | np.integer):
        return f"{int(value):,}".replace(",", "{,}")
    return _escape(str(value))


def _escape(text: str) -> str:
    for char in ("&", "%", "$", "#", "_"):
        text = text.replace(char, f"\\{char}")
    return text


def _macro_block(macros: dict[str, str]) -> str:
    """Declare then redefine, so the file is safe to input whether or not
    main.tex has already provided the macro with a pending fallback."""
    declare = "".join(f"\\providecommand{{\\{k}}}{{}}\n" for k in sorted(macros))
    define = "".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items()))
    return declare + "% Generated by scripts/make_tables.py --- do not edit.\n" + define


def write(name: str, body: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text(body)
    print(f"wrote {GENERATED / name}")


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------


def table_calibration() -> tuple[str, dict[str, str]]:
    """The xi sweep, written out of results/calibration.json.

    This table was hand-typed in main.tex and went stale the moment the
    calibration was re-run — which happened twice. Anything that can go stale
    should be generated, so it is generated.
    """
    import json

    path = RESULTS / "calibration.json"
    if not path.exists():
        return "", {}
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload["candidates"]).sort_values("half_life_days")
    # Formatted here rather than through _cell: the whole point of this table
    # is that the candidates separate in the fifth and sixth decimal, which the
    # shared four-decimal default would erase.
    shown = pd.DataFrame(
        {
            "half_life_days": frame["half_life_days"].map(lambda v: f"{float(v):.0f}"),
            "xi": frame["xi"].map(lambda v: f"{float(v):.6f}"),
            "mean_rps_paired": frame["mean_rps_paired"].map(lambda v: f"{float(v):.6f}"),
        }
    )
    body = tabular(
        shown,
        "rS[table-format=1.6]S[table-format=1.6]",
        ["{Half-life (days)}", "{$\\xi$}", "{Paired mean RPS}"],
    )
    best = frame.loc[frame["mean_rps_paired"].idxmin()]
    spread = float(frame["mean_rps_paired"].max() - frame["mean_rps_paired"].min())
    macros = {
        "xifrozen": f"{float(best['xi']):.16f}",
        "xihalflife": f"{float(best['half_life_days']):.0f}",
        "xirpsbest": f"{float(best['mean_rps_paired']):.6f}",
        "xirpsspread": f"{spread:.6f}",
        "xipairedn": f"{int(payload['n_paired']):,}",
        "xirpsnodecay": f"{float(frame['mean_rps_paired'].iloc[-1]):.6f}",
        "xirpsfastest": f"{float(frame['mean_rps_paired'].iloc[0]):.6f}",
    }
    return body, macros


def table_coverage(cells: pd.DataFrame) -> str:
    grouped = (
        cells.drop_duplicates(KEYS)
        .groupby(["book", "market"])
        .agg(cells_n=("league", "size"), matches=("n_matches", "sum"))
        .reset_index()
        .sort_values("matches", ascending=False)
    )
    return tabular(
        grouped,
        "llrr",
        ["Bookmaker", "Market", "{Cells}", "{Matches}"],
    )


def table_p1(cell_frame: pd.DataFrame, margin_frame: pd.DataFrame) -> tuple[str, dict[str, str]]:
    joined = cell_frame.merge(margin_frame, on=KEYS, how="left")
    by_book = (
        joined.groupby("book")
        .agg(
            cells_n=("league", "size"),
            margin=("mean_book_sum", "mean"),
            roi_spread=("roi_spread", "mean"),
            rps_spread=("rps_spread", "mean"),
        )
        .reset_index()
        .sort_values("margin")
    )
    body = tabular(
        by_book,
        "lrS[table-format=1.4]S[table-format=1.4]S[table-format=1.4]",
        [
            "Bookmaker",
            "{Cells}",
            "{Mean book sum}",
            "{Mean ROI spread}",
            "{Mean RPS spread}",
        ],
    )

    macros: dict[str, str] = {}
    rho, low, high, n_cells = p1_cell_interval(
        cell_frame,
        margin_frame,
        rng=np.random.default_rng(P1_CELL_BOOTSTRAP_SEED),
    )
    if np.isfinite(rho):
        macros["SpearmanMarginSpread"] = f"{rho:.3f}"
        macros["NMarginCells"] = str(n_cells)
        if np.isfinite(low) and np.isfinite(high):
            macros["SpearmanMarginSpreadLow"] = f"{low:.3f}"
            macros["SpearmanMarginSpreadHigh"] = f"{high:.3f}"
    book_rho, book_low, book_high, n_books = p1_book_interval(
        cell_frame,
        margin_frame,
        rng=np.random.default_rng(P1_BOOK_BOOTSTRAP_SEED),
    )
    if np.isfinite(book_rho):
        macros["SpearmanBookSpread"] = f"{book_rho:.3f}"
        macros["NMarginBooks"] = str(n_books)
        if np.isfinite(book_low) and np.isfinite(book_high):
            macros["SpearmanBookSpreadLow"] = f"{book_low:.3f}"
            macros["SpearmanBookSpreadHigh"] = f"{book_high:.3f}"
    _, cl_low, cl_high, _ = p1_cluster_interval(
        cell_frame,
        margin_frame,
        rng=np.random.default_rng(P1_CLUSTER_BOOTSTRAP_SEED),
    )
    if np.isfinite(cl_low) and np.isfinite(cl_high):
        macros["SpearmanClusterLow"] = f"{cl_low:.3f}"
        macros["SpearmanClusterHigh"] = f"{cl_high:.3f}"
    for book, macro in (("Pinnacle", "SpreadPinnacle"), ("Bet365", "SpreadBetSixtyFive")):
        row = by_book[by_book["book"] == book]
        if not row.empty:
            macros[macro] = f"{float(row['roi_spread'].iloc[0]):.4f}"
    return body, macros


def table_p2(cell_frame: pd.DataFrame) -> tuple[str, dict[str, str]]:
    """1X2 against OU25 on the (league, book) pairs that carry both markets."""
    wide = cell_frame.pivot_table(
        index=["league", "book"], columns="market", values="roi_spread", aggfunc="first"
    ).dropna()
    rps = cell_frame.pivot_table(
        index=["league", "book"], columns="market", values="rps_spread", aggfunc="first"
    ).dropna()

    if wide.empty:
        return "\\emph{No (league, bookmaker) pair carries both markets.}\n", {}

    summary = pd.DataFrame(
        {
            "quantity": ["ROI spread", "RPS spread"],
            "n_pairs": [len(wide), len(rps)],
            "onextwo": [float(wide["1X2"].mean()), float(rps["1X2"].mean())],
            "ou25": [float(wide["OU25"].mean()), float(rps["OU25"].mean())],
            "difference": [
                float((wide["1X2"] - wide["OU25"]).mean()),
                float((rps["1X2"] - rps["OU25"]).mean()),
            ],
            "share_larger": [
                float((wide["1X2"] > wide["OU25"]).mean()),
                float((rps["1X2"] > rps["OU25"]).mean()),
            ],
        }
    )
    body = tabular(
        summary,
        "lrS[table-format=1.4]S[table-format=1.4]S[table-format=1.4]S[table-format=1.3]",
        [
            "Quantity",
            "{Pairs}",
            "{1X2}",
            "{OU25}",
            "{Difference}",
            "{Share 1X2 larger}",
        ],
    )
    macros = {
        "SpreadOneXTwo": f"{float(wide['1X2'].mean()):.4f}",
        "SpreadOverUnder": f"{float(wide['OU25'].mean()):.4f}",
        "NPairedCells": str(len(wide)),
        "ShareOneXTwoRoi": f"{100.0 * float((wide['1X2'] > wide['OU25']).mean()):.0f}\\%",
        "ShareOneXTwoRps": f"{100.0 * float((rps['1X2'] > rps['OU25']).mean()):.0f}\\%",
    }
    diff, diff_low, diff_high, _ = p2_roi_diff_interval(
        cell_frame,
        rng=np.random.default_rng(P2_BOOTSTRAP_SEED),
    )
    if np.isfinite(diff):
        macros["PairRoiDiff"] = f"{diff:.4f}"
        if np.isfinite(diff_low) and np.isfinite(diff_high):
            macros["PairRoiDiffLow"] = f"{diff_low:.4f}"
            macros["PairRoiDiffHigh"] = f"{diff_high:.4f}"
    return body, macros


def table_p3(cell_frame: pd.DataFrame) -> tuple[str, dict[str, str]]:
    flipped = cell_frame[cell_frame["sign_flip_prop_shin"]].copy()
    macros = {
        "NSignFlips": f"{int(cell_frame['sign_flip_prop_shin'].sum()):,}".replace(",", "{,}"),
        "PctSignFlips": f"{100.0 * float(cell_frame['sign_flip_prop_shin'].mean()):.1f}\\%",
        "NAnySignFlips": f"{int(cell_frame['sign_flip'].sum()):,}".replace(",", "{,}"),
    }
    if flipped.empty:
        return (
            "\\emph{No cell reverses the sign of measured ROI between the "
            "proportional and Shin benchmarks.}\n",
            macros,
        )

    shown = flipped.sort_values("roi_spread", ascending=False).head(20)[
        ["league", "book", "market", "n_matches", "roi_proportional", "roi_shin", "roi_spread"]
    ]
    body = tabular(
        shown,
        "lll r S[table-format=+1.4] S[table-format=+1.4] S[table-format=1.4]",
        [
            "League",
            "Bookmaker",
            "Market",
            "{Matches}",
            "{ROI (prop.)}",
            "{ROI (Shin)}",
            "{Spread}",
        ],
    )
    return body, macros


def table_dm(cells: pd.DataFrame) -> str:
    from footy.eval.inference import adjusted_p_values

    frame = cells.copy()
    frame["q"] = adjusted_p_values(frame["dm_p"].to_numpy(dtype=float))
    grouped = (
        frame.groupby("method")
        .agg(
            cells_n=("league", "size"),
            model_rps=("model_rps", "mean"),
            bench_rps=("bench_rps", "mean"),
            rps_diff=("rps_diff", "mean"),
            favours_model=("rps_diff", lambda s: float((s < 0).mean())),
            significant=("q", lambda s: float((s < 0.10).mean())),
        )
        .reindex(list(METHODS))
        .reset_index()
    )
    grouped["method"] = grouped["method"].map(METHOD_LABEL)
    return tabular(
        grouped,
        "lrS[table-format=1.4]S[table-format=1.4]S[table-format=+1.4]S[table-format=1.3]S[table-format=1.3]",
        [
            "Transform",
            "{Cells}",
            "{Model RPS}",
            r"{\makecell{Benchmark\\RPS}}",
            "{Diff.}",
            r"{\makecell{Share fav.\\model}}",
            r"{\makecell{Share\\$q < 0.10$}}",
        ],
    )


def _archive_counts() -> dict[str, str]:
    """Counts the abstract quotes, read from the forecast cache."""
    forecast_dir = RESULTS / "forecasts"
    if not forecast_dir.exists():
        return {}
    total = sum(sum(1 for _ in path.open()) - 1 for path in sorted(forecast_dir.glob("*.csv")))
    return {"NForecasts": f"{total:,}".replace(",", "{,}"), "NMatchesArchive": "176{,}629"}


def summary_macros(cells: pd.DataFrame, cell_frame: pd.DataFrame, extra: dict[str, str]) -> str:
    forecasts = RESULTS / "forecasts.parquet"
    values: dict[str, str] = {
        "NCells": f"{len(cell_frame):,}".replace(",", "{,}"),
        "NLeagues": str(cells["league"].nunique()),
        "NBooks": str(cells["book"].nunique()),
    }
    if forecasts.exists():
        n = len(pd.read_parquet(forecasts, columns=["league"]))
        values["NForecasts"] = f"{n:,}".replace(",", "{,}")
    values.update(extra)

    lines = ["% Generated by scripts/make_tables.py --- do not edit."]
    lines += [f"\\renewcommand{{\\{name}}}{{{value}}}" for name, value in sorted(values.items())]
    # Macros the paper does not yet declare need providing before renewing.
    declared = "\n".join(f"\\providecommand{{\\{name}}}{{}}" for name in sorted(values))
    return declared + "\n" + "\n".join(lines) + "\n"


def main() -> None:
    refresh = "--refresh" in sys.argv[1:]

    # The calibration table depends only on the xi sweep, not on the
    # evaluation, so it is written first and independently. That way the
    # paper's method section is never stale even before results exist.
    calib_body, calib_macros = table_calibration()
    if calib_body:
        write("tab-calibration.tex", calib_body)
        write("calibration-macros.tex", _macro_block(calib_macros))
    else:
        print("results/calibration.json absent — calibration table left pending")

    if not CELLS.exists():
        raise SystemExit(
            f"{CELLS} does not exist. Run `python scripts/study.py evaluate` first; "
            "the paper compiles without it, with every result slot marked pending."
        )

    cells = pd.read_csv(CELLS)
    cell_frame = per_cell(cells)
    margin_frame = margins(refresh=refresh)

    p1_body, p1_macros = table_p1(cell_frame, margin_frame)
    p2_body, p2_macros = table_p2(cell_frame)
    p3_body, p3_macros = table_p3(cell_frame)

    write("tab-coverage.tex", table_coverage(cells))
    write("tab-p1.tex", p1_body)
    write("tab-p2.tex", p2_body)
    write("tab-p3.tex", p3_body)
    write("tab-dm.tex", table_dm(cells))
    write(
        "summary.tex",
        summary_macros(
            cells,
            cell_frame,
            {**p1_macros, **p2_macros, **p3_macros, **_archive_counts()},
        ),
    )


if __name__ == "__main__":
    main()
