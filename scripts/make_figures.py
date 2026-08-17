"""Turn the evaluation output into the paper's figures.

Five figures, each written to ``paper/generated/fig-*.pdf`` --- vector,
greyscale-safe. Like `make_tables.py`, nothing plotted here is typed by hand:
every quantity comes out of `results/cells.csv`, `results/book_margin.csv` or
`results/calibration.json`, so a figure cannot drift from the run that
produced it.

Each figure is a `render_*` function returning a `Figure`; the numbers that
feed it are prepared by a plain function that returns a DataFrame or array
and can be tested without matplotlib. `main()` writes all five.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("pdf")  # vector backend; no display needed

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from footy.eval.effect import bootstrap_correlation, spearman
from footy.market.demargin import METHODS, demargin_all
from footy.study.run import MARKET_OUTCOMES, closing_books, load_archive

RAW = Path("data/raw")
RESULTS = Path("results")
GENERATED = Path("paper/generated")

CELLS = RESULTS / "cells.csv"
MARGINS = RESULTS / "book_margin.csv"
CALIBRATION = RESULTS / "calibration.json"

KEYS = ["league", "book", "market"]

METHOD_LABEL = {
    "proportional": "Proportional",
    "power": "Power",
    "shin": "Shin",
    "odds_ratio": "Odds-ratio",
}

# --------------------------------------------------------------------------
# palette --- from the dataviz skill's reference/palette.md, light mode.
# Categorical slots are assigned in the documented fixed order (never
# cycled); each transform also carries a distinct marker/linestyle so
# identity survives greyscale print, not just the hue channel.
# --------------------------------------------------------------------------

BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
AQUA = "#1baf7a"  # categorical slot 3
YELLOW = "#eda100"  # categorical slot 4
RED = "#e34948"  # diverging pole (paired with BLUE)

INK = "#0b0b0b"  # primary text
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"  # axis/tick labels
GRID = "#e1e0d9"  # hairline gridlines
AXIS = "#c3c2b7"  # baseline / axis rule
SURFACE = "#fcfcfb"

METHOD_ORDER = list(METHODS)
METHOD_COLOR = {"proportional": BLUE, "power": ORANGE, "shin": AQUA, "odds_ratio": YELLOW}
METHOD_MARKER = {"proportional": "o", "power": "s", "shin": "^", "odds_ratio": "D"}
METHOD_LINESTYLE = {
    "proportional": "solid",
    "power": (0, (5, 1.5)),
    "shin": (0, (1, 1)),
    "odds_ratio": (0, (3, 1, 1, 1)),
}

FONT = "sans-serif"


def _style() -> None:
    """Shared rcParams: sans throughout, hairline recessive grid, no chartjunk."""
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 9,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # gridlines are solid hairlines, never dashed
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _clean_axes(ax: plt.Axes) -> None:
    """Recessive chrome: no top/right spine, hairline left/bottom, light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, length=3)


def write(name: str, fig: Figure) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"wrote {path}")


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run `python scripts/study.py evaluate` and "
            "`python scripts/make_tables.py` first; figures cannot be drawn from "
            "results that were never produced."
        )


# --------------------------------------------------------------------------
# data prep --- pure, testable, no matplotlib
# --------------------------------------------------------------------------


def roi_spread_per_cell(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per (league, book, market): the ROI range across the four
    transforms. Matches `make_tables.per_cell`'s `roi_spread` exactly ---
    max minus min over finite values only, so a transform that placed no
    bets (an undefined ROI) does not get treated as agreement at zero."""
    wide = cells.pivot_table(index=KEYS, columns="method", values="roi", aggfunc="first")
    # Guarantee all four transform columns exist even if one placed no bets
    # in every single cell (pivot_table drops a column that is entirely
    # NaN) --- reindexing only fills a missing column with NaN, it does not
    # touch which (league, book, market) rows are present.
    values = wide.reindex(columns=METHOD_ORDER).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    return wide.reset_index()[KEYS].assign(roi_spread=spread)


def margin_spread_by_book(cells: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    """Mean book sum against mean ROI spread, one row per bookmaker.

    Exactly the aggregation `table_p1` uses for `\\SpearmanMarginSpread`: the
    per-cell frame joined to the margin cache on (league, book, market), then
    grouped by book and averaged. Reusing this grouping (rather than a
    fresh one) is what keeps the figure's rho identical to the paper's macro.
    """
    cell_frame = roi_spread_per_cell(cells)
    joined = cell_frame.merge(margins, on=KEYS, how="left")
    by_book = (
        joined.groupby("book")
        .agg(mean_book_sum=("mean_book_sum", "mean"), roi_spread=("roi_spread", "mean"))
        .dropna()
        .reset_index()
        .sort_values("book")
        .reset_index(drop=True)
    )
    return by_book


def calibration_candidates(payload: dict[str, object]) -> pd.DataFrame:
    """The xi sweep candidates, sorted ascending by half-life.

    Same sort `make_tables.table_calibration` uses: `.iloc[0]` is the
    fastest-decaying candidate, `.iloc[-1]` is the no-decay baseline.
    """
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    return pd.DataFrame(candidates).sort_values("half_life_days").reset_index(drop=True)


def divergence_target(margins: pd.DataFrame) -> tuple[str, str, str]:
    """The (league, book, market) with the single highest mean book sum.

    Pre-specified, not eyeballed: P1 finds transform disagreement scales
    with margin, so the highest-margin book is where it should be largest ---
    picking it by margin alone, before looking at any transform output,
    keeps the example from being cherry-picked for a dramatic-looking result.
    """
    if margins.empty:
        raise ValueError("book_margin frame is empty; cannot pick a divergence target")
    row = margins.loc[margins["mean_book_sum"].idxmax()]
    return (str(row["league"]), str(row["book"]), str(row["market"]))


def worked_book_implied(league: str, book: str, market: str, raw: Path) -> np.ndarray:
    """Raw implied probabilities for the first chronologically available
    complete closing book on this (league, book, market)."""
    archive = load_archive(raw)
    block = archive.odds[archive.odds["league"] == league]
    wide = closing_books(block, book, market)
    if wide.empty:
        raise ValueError(f"no complete closing book for {league}/{book}/{market}")
    outcomes = MARKET_OUTCOMES[market]
    row = wide.iloc[0]
    return np.array([1.0 / float(row[o]) for o in outcomes], dtype=float)


def longshot_index(implied: np.ndarray) -> int:
    """Index of the outcome with the smallest raw implied probability ---
    the longshot, where the transforms are expected to disagree most."""
    if implied.size == 0:
        raise ValueError("implied probabilities array is empty")
    return int(np.argmin(implied))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def render_margin_spread(cells: pd.DataFrame, margins: pd.DataFrame) -> Figure:
    """fig-margin-spread: mean book sum against mean transform ROI spread,
    one point per bookmaker, labelled, with a fitted trend and rho + its
    bootstrap CI. This carries P1, currently only a 14-row table."""
    by_book = margin_spread_by_book(cells, margins)
    if by_book.empty:
        raise ValueError("no bookmakers to plot in fig-margin-spread")

    x = by_book["mean_book_sum"].to_numpy(dtype=float)
    y = by_book["roi_spread"].to_numpy(dtype=float)
    rho = spearman(x, y)
    rng = np.random.default_rng(0)
    _, low, high = bootstrap_correlation(x, y, rng=rng, n_resamples=10_000)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _clean_axes(ax)
    ax.grid(axis="y", zorder=0)

    # Trend line: a reference, not data, so it is drawn muted and dashed to
    # separate it from the single-hue data series.
    coeffs = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        xs, np.polyval(coeffs, xs), color=INK_MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1
    )

    ax.scatter(x, y, s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    # 14 labels crowd where bookmakers cluster in (margin, spread) space.
    # Alternating the label above/below the marker, ordered by y, keeps
    # neighbours in a cluster from writing over each other.
    ordered = by_book.sort_values("roi_spread").reset_index(drop=True)
    for i, row in ordered.iterrows():
        dy = 7 if i % 2 == 0 else -11
        ax.annotate(
            str(row["book"]),
            (row["mean_book_sum"], row["roi_spread"]),
            xytext=(5, dy),
            textcoords="offset points",
            fontsize=7,
            color=INK_SECONDARY,
            va="bottom" if dy > 0 else "top",
        )

    ax.set_xlabel("Mean book sum (bookmaker margin)")
    ax.set_ylabel("Mean ROI spread across transforms")
    ax.set_title(
        f"ρ = {rho:.3f} (95% bootstrap CI [{low:.3f}, {high:.3f}]), n = {len(by_book)} bookmakers",
        fontsize=9,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    return fig


def render_divergence(
    implied: np.ndarray, outcomes: tuple[str, ...], league: str, book: str, market: str
) -> Figure:
    """fig-divergence: for one worked closing book, the four transforms'
    recovered probability against raw implied probability per outcome,
    annotated at the longshot. Shows *why* the transforms differ, which no
    table does."""
    results = demargin_all(implied)
    longshot = longshot_index(implied)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    _clean_axes(ax)
    ax.grid(axis="both", zorder=0)

    lo = min(implied.min(), min(r.probabilities.min() for r in results.values())) - 0.02
    hi = max(implied.max(), max(r.probabilities.max() for r in results.values())) + 0.02
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=INK_MUTED,
        linewidth=1.0,
        linestyle=(0, (1, 1.5)),
        zorder=1,
        label="raw implied (no transform)",
    )

    for method in METHOD_ORDER:
        recovered = results[method].probabilities
        ax.plot(
            implied,
            recovered,
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.0,
            linewidth=1.6,
            linestyle=METHOD_LINESTYLE[method],
            label=METHOD_LABEL[method],
            zorder=3,
        )
        ax.annotate(
            f"{recovered[longshot]:.3f}",
            (implied[longshot], recovered[longshot]),
            xytext=(6, (list(METHOD_ORDER).index(method) - 1.5) * 9),
            textcoords="offset points",
            fontsize=7,
            color=METHOD_COLOR[method],
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Raw implied probability (1 / decimal odds)")
    ax.set_ylabel("Recovered probability")
    ax.set_title(
        f"{book} · {market} · {league} — highest-margin book (longshot {outcomes[longshot]!r})",
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def render_calibration(candidates: pd.DataFrame) -> Figure:
    """fig-calibration: paired mean RPS against half-life (log x), optimum
    marked, no-decay baseline as a horizontal rule. Shows the optimum is
    interior and the curve is flat around it."""
    if candidates.empty:
        raise ValueError("no calibration candidates to plot")

    x = candidates["half_life_days"].to_numpy(dtype=float)
    y = candidates["mean_rps_paired"].to_numpy(dtype=float)
    best_idx = int(np.argmin(y))
    baseline = float(y[-1])  # largest half-life = no-decay, per table_calibration

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    _clean_axes(ax)
    ax.set_xscale("log")
    ax.grid(axis="both", zorder=0)

    # Labelled through the legend, not a floating text annotation: the
    # baseline sits close to the curve for most of its length (they meet
    # exactly at the largest half-life), so any fixed text position would
    # sit on top of the line somewhere.
    ax.axhline(
        baseline,
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=1,
        label="no-decay baseline",
    )

    ax.plot(
        x,
        y,
        color=BLUE,
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        zorder=3,
        label="paired mean RPS",
    )
    ax.scatter(
        [x[best_idx]],
        [y[best_idx]],
        s=90,
        color=RED,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=4,
        label="optimum",
    )
    ax.annotate(
        f"{x[best_idx]:.0f} days",
        (x[best_idx], y[best_idx]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=7,
        color=RED,
    )

    ax.set_xlabel("Half-life (days, log scale)")
    ax.set_ylabel("Paired mean RPS")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def render_roi(cells: pd.DataFrame) -> Figure:
    """fig-roi: distribution of per-cell ROI, one panel per transform, zero
    line drawn on each. The model has no edge --- the finding that governs
    P3."""
    n_cells = int(cells.drop_duplicates(KEYS).shape[0])

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 3.2), sharey=True)
    for ax, method in zip(axes, METHOD_ORDER, strict=True):
        _clean_axes(ax)
        values = cells.loc[cells["method"] == method, "roi"].to_numpy(dtype=float)
        ax.hist(
            values, bins=16, color=METHOD_COLOR[method], edgecolor=SURFACE, linewidth=0.6, zorder=2
        )
        ax.axvline(0.0, color=INK, linewidth=1.2, zorder=3)
        ax.set_title(METHOD_LABEL[method], fontsize=9, color=INK)
        ax.set_xlabel("ROI")
        ax.grid(axis="y", zorder=0)

    axes[0].set_ylabel("Cells")
    fig.suptitle(
        f"Per-cell ROI by transform (n = {n_cells} cells each)",
        fontsize=9,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def render_cell_scatter(cells: pd.DataFrame) -> Figure:
    """fig-cell-scatter: per-cell model RPS against benchmark RPS, one point
    per (cell, method), diagonal drawn. Near-universal loss to the market,
    at a glance."""
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    _clean_axes(ax)
    ax.grid(axis="both", zorder=0)

    x = cells["bench_rps"].to_numpy(dtype=float)
    y = cells["model_rps"].to_numpy(dtype=float)
    lo = min(x.min(), y.min()) - 0.003
    hi = max(x.max(), y.max()) + 0.003
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=INK_MUTED,
        linewidth=1.2,
        linestyle=(0, (4, 3)),
        zorder=1,
        label="model = benchmark",
    )

    # Single hue, not one per transform: the story is the aggregate pattern
    # (most points above the diagonal), not telling the four transforms
    # apart, and colouring 364 points by a 4-way category would only add
    # legend/CVD overhead without carrying more information than density.
    ax.scatter(x, y, s=18, color=BLUE, alpha=0.45, edgecolor="none", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Benchmark RPS (de-margined market), lower is better")
    ax.set_ylabel("Model RPS, lower is better")
    # One decimal, not zero: 363/364 rounds to "100%" at zero decimals, which
    # would overstate a near-universal (not literal) pattern by erasing the
    # one cell that goes the other way.
    share_above = float((y > x).mean())
    ax.set_title(
        f"{share_above:.1%} of points lie above the diagonal",
        fontsize=9,
        color=INK,
        loc="left",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------


def main() -> None:
    _style()
    require(CELLS)
    require(MARGINS)
    require(CALIBRATION)

    cells = pd.read_csv(CELLS)
    margins = pd.read_csv(MARGINS)
    calibration_payload = json.loads(CALIBRATION.read_text())
    candidates = calibration_candidates(calibration_payload)

    write("fig-margin-spread.pdf", render_margin_spread(cells, margins))

    league, book, market = divergence_target(margins)
    implied = worked_book_implied(league, book, market, RAW)
    outcomes = MARKET_OUTCOMES[market]
    print(f"fig-divergence worked example: {league}/{book}/{market}, implied={implied}")
    write("fig-divergence.pdf", render_divergence(implied, outcomes, league, book, market))

    write("fig-calibration.pdf", render_calibration(candidates))
    write("fig-roi.pdf", render_roi(cells))
    write("fig-cell-scatter.pdf", render_cell_scatter(cells))


if __name__ == "__main__":
    sys.exit(main())
