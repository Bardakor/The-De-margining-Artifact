"""Tests for the P1/P2 interval helpers in scripts/make_tables.py.

The numbers that reach the paper are generated here, so the sample they are
computed on --- cells versus bookmakers, and paired 1X2/OU25 cells --- is
asserted on small frames rather than trusted from the prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from make_tables import (  # noqa: E402
    p1_book_interval,
    p1_cell_interval,
    p1_cluster_interval,
    p2_roi_diff_interval,
    per_cell,
)

METHODS = ("proportional", "power", "shin", "odds_ratio")


def _cell_rows(
    league: str, book: str, market: str, rois: tuple[float, ...]
) -> list[dict[str, object]]:
    return [
        {
            "league": league,
            "book": book,
            "market": market,
            "method": method,
            "roi": roi,
            "bench_rps": 0.20,
            "n_bets": 10,
            "n_matches": 200,
        }
        for method, roi in zip(METHODS, rois, strict=True)
    ]


def test_p1_cell_interval_is_computed_on_cells_not_books() -> None:
    """Two books, five cells: the cell-level n is 5, the book-level n is 2."""
    rows: list[dict[str, object]] = []
    rows.extend(_cell_rows("E0", "Pinnacle", "1X2", (0.01, 0.02, 0.00, 0.01)))
    rows.extend(_cell_rows("SP1", "Pinnacle", "1X2", (0.02, 0.03, 0.01, 0.02)))
    rows.extend(_cell_rows("E0", "Bet365", "1X2", (0.10, 0.20, 0.00, 0.08)))
    rows.extend(_cell_rows("SP1", "Bet365", "1X2", (0.12, 0.22, 0.01, 0.09)))
    rows.extend(_cell_rows("I1", "Bet365", "1X2", (0.11, 0.21, 0.02, 0.10)))
    cell_frame = per_cell(pd.DataFrame(rows))
    margins = pd.DataFrame(
        [
            {"league": "E0", "book": "Pinnacle", "market": "1X2", "mean_book_sum": 1.02},
            {"league": "SP1", "book": "Pinnacle", "market": "1X2", "mean_book_sum": 1.03},
            {"league": "E0", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.07},
            {"league": "SP1", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.08},
            {"league": "I1", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.09},
        ]
    )
    rng = np.random.default_rng(0)
    _, _, _, n_cells = p1_cell_interval(cell_frame, margins, rng=rng, n_resamples=200)
    _, _, _, n_books = p1_book_interval(cell_frame, margins, rng=rng, n_resamples=200)
    assert n_cells == 5
    assert n_books == 2


def test_p2_interval_uses_the_paired_cells_only() -> None:
    rows: list[dict[str, object]] = []
    # Three paired (league, book) cells, plus an unpaired 1X2 cell that must
    # not enter the P2 difference.
    for league, roi_1x2, roi_ou in (("E0", 0.04, 0.01), ("SP1", 0.05, 0.02), ("I1", 0.03, 0.02)):
        rows.extend(_cell_rows(league, "Bet365", "1X2", (0.0, roi_1x2, 0.0, 0.0)))
        rows.extend(_cell_rows(league, "Bet365", "OU25", (0.0, roi_ou, 0.0, 0.0)))
    rows.extend(_cell_rows("D1", "Pinnacle", "1X2", (0.0, 0.50, 0.0, 0.0)))
    cell_frame = per_cell(pd.DataFrame(rows))
    point, low, high, n = p2_roi_diff_interval(
        cell_frame, rng=np.random.default_rng(0), n_resamples=500
    )
    assert n == 3
    assert low < point < high
    assert point == pytest.approx((0.03 + 0.03 + 0.01) / 3)


def test_cluster_interval_rejects_a_single_bookmaker() -> None:
    rows = _cell_rows("E0", "Bet365", "1X2", (0.1, 0.2, 0.0, 0.1))
    rows.extend(_cell_rows("SP1", "Bet365", "1X2", (0.2, 0.3, 0.1, 0.2)))
    cell_frame = per_cell(pd.DataFrame(rows))
    margins = pd.DataFrame(
        [
            {"league": "E0", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.07},
            {"league": "SP1", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.08},
        ]
    )
    rho, low, high, n = p1_cluster_interval(
        cell_frame, margins, rng=np.random.default_rng(0), n_resamples=50
    )
    assert n == 2
    assert np.isnan(rho) and np.isnan(low) and np.isnan(high)
