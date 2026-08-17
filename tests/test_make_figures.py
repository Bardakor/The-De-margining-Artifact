"""Tests for the figure data-preparation functions in scripts/make_figures.py.

Figures themselves (matplotlib rendering) are exempt from unit tests, but the
data-prep that decides which numbers reach the page is not, so it is tested
here the same way the rest of the study is: on small, hand-checkable inputs.

`scripts/` is not a package (it mirrors `make_tables.py`, imported the same
way `study.py` runs standalone), so it is put on `sys.path` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from make_figures import (  # noqa: E402
    calibration_candidates,
    divergence_target,
    longshot_index,
    margin_spread_by_book,
    roi_spread_per_cell,
)

KEYS = ["league", "book", "market"]


def _cell_row(league: str, book: str, market: str, method: str, roi: float) -> dict[str, object]:
    return {"league": league, "book": book, "market": market, "method": method, "roi": roi}


def test_roi_spread_per_cell_is_max_minus_min_over_methods() -> None:
    cells = pd.DataFrame(
        [
            _cell_row("E0", "Bet365", "1X2", "proportional", 0.10),
            _cell_row("E0", "Bet365", "1X2", "power", 0.12),
            _cell_row("E0", "Bet365", "1X2", "shin", 0.05),
            _cell_row("E0", "Bet365", "1X2", "odds_ratio", 0.08),
        ]
    )
    out = roi_spread_per_cell(cells)
    assert len(out) == 1
    assert out.loc[0, "roi_spread"] == pytest.approx(0.12 - 0.05)


def test_roi_spread_per_cell_ignores_undefined_roi() -> None:
    """A transform with no bets has an undefined (NaN) ROI. Treating that as
    zero would report agreement where there is no measurement, so it must be
    excluded from the max/min rather than pulled in as a value."""
    cells = pd.DataFrame(
        [
            _cell_row("E0", "Bet365", "1X2", "proportional", 0.10),
            _cell_row("E0", "Bet365", "1X2", "power", 0.20),
            _cell_row("E0", "Bet365", "1X2", "shin", float("nan")),
            _cell_row("E0", "Bet365", "1X2", "odds_ratio", 0.15),
        ]
    )
    out = roi_spread_per_cell(cells)
    assert out.loc[0, "roi_spread"] == pytest.approx(0.20 - 0.10)


def test_margin_spread_by_book_aggregates_one_row_per_book() -> None:
    cells = pd.DataFrame(
        [
            *[
                _cell_row("E0", "Bet365", "1X2", m, v)
                for m, v in zip(
                    ["proportional", "power", "shin", "odds_ratio"],
                    [0.10, 0.12, 0.05, 0.08],
                    strict=True,
                )
            ],
            *[
                _cell_row("SP1", "Bet365", "1X2", m, v)
                for m, v in zip(
                    ["proportional", "power", "shin", "odds_ratio"],
                    [0.00, 0.02, -0.01, 0.01],
                    strict=True,
                )
            ],
            *[
                _cell_row("E0", "Pinnacle", "1X2", m, v)
                for m, v in zip(
                    ["proportional", "power", "shin", "odds_ratio"],
                    [0.01, 0.015, 0.005, 0.012],
                    strict=True,
                )
            ],
        ]
    )
    margins = pd.DataFrame(
        [
            {"league": "E0", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.08},
            {"league": "SP1", "book": "Bet365", "market": "1X2", "mean_book_sum": 1.04},
            {"league": "E0", "book": "Pinnacle", "market": "1X2", "mean_book_sum": 1.02},
        ]
    )
    by_book = margin_spread_by_book(cells, margins)
    assert sorted(by_book["book"]) == ["Bet365", "Pinnacle"]
    bet365 = by_book.set_index("book").loc["Bet365"]
    # Bet365's two cells have roi_spread 0.07 (E0) and 0.03 (SP1); margin 1.08 and 1.04.
    assert bet365["roi_spread"] == pytest.approx((0.07 + 0.03) / 2)
    assert bet365["mean_book_sum"] == pytest.approx((1.08 + 1.04) / 2)


def test_calibration_candidates_sorted_ascending_by_half_life() -> None:
    payload = {
        "candidates": [
            {"half_life_days": 700.0, "mean_rps_paired": 0.2061},
            {"half_life_days": 30.0, "mean_rps_paired": 0.2223},
            {"half_life_days": 400.0, "mean_rps_paired": 0.2049},
        ]
    }
    out = calibration_candidates(payload)
    assert out["half_life_days"].tolist() == [30.0, 400.0, 700.0]
    # iloc[0] is the fastest-decaying candidate, iloc[-1] the no-decay baseline.
    assert out["mean_rps_paired"].iloc[0] == pytest.approx(0.2223)
    assert out["mean_rps_paired"].iloc[-1] == pytest.approx(0.2061)


def test_divergence_target_picks_highest_margin_row() -> None:
    margins = pd.DataFrame(
        [
            {"league": "E0", "book": "Pinnacle", "market": "1X2", "mean_book_sum": 1.02},
            {"league": "F2", "book": "Stake", "market": "1X2", "mean_book_sum": 1.13},
            {"league": "D2", "book": "Coral", "market": "1X2", "mean_book_sum": 1.09},
        ]
    )
    assert divergence_target(margins) == ("F2", "Stake", "1X2")


def test_divergence_target_rejects_empty_frame() -> None:
    with pytest.raises(ValueError):
        divergence_target(pd.DataFrame(columns=["league", "book", "market", "mean_book_sum"]))


def test_longshot_index_is_the_smallest_implied_probability() -> None:
    implied = np.array([0.40, 0.36, 0.38])
    assert longshot_index(implied) == 1


def test_longshot_index_rejects_empty_array() -> None:
    with pytest.raises(ValueError):
        longshot_index(np.array([]))
