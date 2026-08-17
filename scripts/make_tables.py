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


def write(name: str, body: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text(body)
    print(f"wrote {GENERATED / name}")


# --------------------------------------------------------------------------
# the tables
# --------------------------------------------------------------------------


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
        "llS[table-format=3.0]S[table-format=6.0]",
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
        "lS[table-format=3.0]S[table-format=1.4]S[table-format=1.4]S[table-format=1.4]",
        [
            "Bookmaker",
            "{Cells}",
            "{Mean book sum}",
            "{Mean ROI spread}",
            "{Mean RPS spread}",
        ],
    )

    usable = joined.dropna(subset=["mean_book_sum", "roi_spread"])
    macros: dict[str, str] = {}
    if len(usable) > 2:
        rho = float(
            np.corrcoef(
                usable["mean_book_sum"].rank(),
                usable["roi_spread"].rank(),
            )[0, 1]
        )
        macros["SpearmanMarginSpread"] = f"{rho:.3f}"
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
        "lS[table-format=3.0]S[table-format=1.4]S[table-format=1.4]S[table-format=1.4]S[table-format=1.3]",
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
    }
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
        "lll S[table-format=5.0] S[table-format=+1.4] S[table-format=+1.4] S[table-format=1.4]",
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
        "lS[table-format=3.0]S[table-format=1.4]S[table-format=1.4]S[table-format=+1.4]S[table-format=1.3]S[table-format=1.3]",
        [
            "Transform",
            "{Cells}",
            "{Model RPS}",
            "{Benchmark RPS}",
            "{Difference}",
            "{Share favouring model}",
            "{Share $q < 0.10$}",
        ],
    )


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
    write("summary.tex", summary_macros(cells, cell_frame, {**p1_macros, **p2_macros, **p3_macros}))


if __name__ == "__main__":
    main()
