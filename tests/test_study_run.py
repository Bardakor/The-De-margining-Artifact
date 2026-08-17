"""Archive assembly and the per-cell four-transform comparison.

Study spec §4 and §6. No test here touches the network: acquisition is tested
through its result type, and the comparison through synthetic forecasts and
odds whose answer is known in advance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footy.market.overround import apply_overround
from footy.study.acquire import Acquisition
from footy.study.run import (
    MARKET_OUTCOMES,
    closing_books,
    evaluate_cell,
    load_archive,
)

KEYS = ["league", "season", "kickoff", "home", "away"]


def odds_frame(n: int = 60, book: str = "Bet365", book_sum: float = 1.07) -> pd.DataFrame:
    """Closing 1X2 odds for n matches, margined to a known book sum."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(n):
        fair = rng.dirichlet(np.array([6.0, 4.0, 3.0]))
        offered = apply_overround(fair, book_sum)
        for outcome, price in zip(MARKET_OUTCOMES["1X2"], offered, strict=True):
            rows.append(
                {
                    "league": "E0",
                    "season": "2324",
                    "kickoff": pd.Timestamp("2023-08-01") + pd.Timedelta(days=i),
                    "home": f"H{i}",
                    "away": f"A{i}",
                    "book": book,
                    "market": "1X2",
                    "period": "close",
                    "outcome": outcome,
                    "decimal": float(price),
                }
            )
    return pd.DataFrame(rows)


def forecast_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for i in range(n):
        p = rng.dirichlet(np.array([6.0, 4.0, 3.0]))
        outcome = int(rng.choice(3, p=p))
        rows.append(
            {
                "league": "E0",
                "season": "2324",
                "kickoff": pd.Timestamp("2023-08-01") + pd.Timedelta(days=i),
                "home": f"H{i}",
                "away": f"A{i}",
                "home_goals": 2 if outcome == 0 else (1 if outcome == 1 else 0),
                "away_goals": 0 if outcome == 0 else (1 if outcome == 1 else 2),
                "outcome": outcome,
                "p_home": p[0],
                "p_draw": p[1],
                "p_away": p[2],
                "p_over25": 0.52,
                "p_under25": 0.48,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Acquisition bookkeeping
# --------------------------------------------------------------------------


def test_acquisition_separates_present_from_absent() -> None:
    """A 404 means the league did not run that season, which is ordinary. It
    must not be conflated with a request that failed."""
    a = Acquisition(
        downloaded=(Path("a.csv"),),
        cached=(Path("b.csv"),),
        absent=(("9394", "G1"),),
        failed=(("2324", "E0", "HTTP 500"),),
    )
    assert set(a.available) == {Path("a.csv"), Path("b.csv")}
    assert "1 downloaded" in a.summary()
    assert "1 absent" in a.summary()
    assert "1 failed" in a.summary()


def test_available_deduplicates() -> None:
    shared = Path("x.csv")
    a = Acquisition(downloaded=(shared,), cached=(shared,), absent=(), failed=())
    assert a.available == (shared,)


# --------------------------------------------------------------------------
# Archive loading
# --------------------------------------------------------------------------


def test_loading_an_empty_directory_is_an_error(tmp_path: Path) -> None:
    """Silently returning an empty archive would produce a study with no data
    and no complaint."""
    with pytest.raises(ValueError, match="no readable season files"):
        load_archive(tmp_path)


def test_unreadable_files_are_recorded_not_skipped(tmp_path: Path) -> None:
    good = tmp_path / "2324"
    good.mkdir()
    (good / "E0.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,12/08/2023,Arsenal,Chelsea,2,1\n"
    )
    (good / "XX.csv").write_text("Div,Date,NotAColumn\nE0,12/08/2023,1\n")

    archive = load_archive(tmp_path)
    assert archive.files_read == 1
    assert [name for name, _ in archive.unreadable] == ["XX/2324"]


def test_league_filter_is_respected(tmp_path: Path) -> None:
    season = tmp_path / "2324"
    season.mkdir()
    for league in ("E0", "D1"):
        (season / f"{league}.csv").write_text(
            f"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n{league},12/08/2023,A,B,2,1\n"
        )
    archive = load_archive(tmp_path, leagues={"E0"})
    assert set(archive.matches["league"]) == {"E0"}


# --------------------------------------------------------------------------
# Closing-book pivot
# --------------------------------------------------------------------------


def test_pivot_produces_one_row_per_match_with_a_column_per_outcome() -> None:
    wide = closing_books(odds_frame(n=10), "Bet365", "1X2")
    assert len(wide) == 10
    assert set(MARKET_OUTCOMES["1X2"]) <= set(wide.columns)


def test_matches_with_an_incomplete_closing_book_are_dropped() -> None:
    """A transform applied to a partial book normalises over the wrong support
    and returns a benchmark that is wrong, not merely noisy."""
    odds = odds_frame(n=10)
    odds = odds[~((odds["home"] == "H3") & (odds["outcome"] == "D"))]
    wide = closing_books(odds, "Bet365", "1X2")
    assert "H3" not in set(wide["home"])
    assert len(wide) == 9


def test_a_book_absent_from_the_data_yields_nothing() -> None:
    assert closing_books(odds_frame(), "NotABook", "1X2").empty


def test_opening_odds_are_never_used_as_a_closing_book() -> None:
    """Benchmarking against opening odds would be a silent methodological
    error, so the pivot must return nothing rather than fall back to them."""
    odds = odds_frame()
    odds["period"] = "open"
    assert closing_books(odds, "Bet365", "1X2").empty


# --------------------------------------------------------------------------
# The per-cell comparison
# --------------------------------------------------------------------------


def test_every_transform_is_evaluated_on_the_same_matches() -> None:
    """The comparison is only attributable to the transform if the arms share
    their matches exactly."""
    result = evaluate_cell(forecast_frame(), odds_frame(), league="E0", book="Bet365", market="1X2")
    assert result is not None
    assert len(result.per_method) == 4
    assert set(result.per_method["method"]) == {"proportional", "power", "shin", "odds_ratio"}
    assert result.n_matches == 60


def test_the_model_score_does_not_depend_on_the_transform() -> None:
    """The model is held fixed across arms; only the benchmark varies. A model
    score that moved between arms would mean the arms are not comparable."""
    result = evaluate_cell(forecast_frame(), odds_frame(), league="E0", book="Bet365", market="1X2")
    assert result is not None
    assert np.isfinite(result.model_rps)


def test_transforms_produce_different_benchmarks_on_a_margined_book() -> None:
    """If they agreed there would be no study."""
    result = evaluate_cell(
        forecast_frame(), odds_frame(book_sum=1.09), league="E0", book="Bet365", market="1X2"
    )
    assert result is not None
    assert result.per_method["bench_rps"].nunique() == 4


def test_disagreement_is_larger_on_a_softer_book() -> None:
    """Prediction P1 at the cell level: the transforms coincide as B -> 1."""

    def spread(book_sum: float) -> float:
        cell = evaluate_cell(
            forecast_frame(),
            odds_frame(book_sum=book_sum),
            league="E0",
            book="Bet365",
            market="1X2",
        )
        assert cell is not None
        values = cell.per_method["bench_rps"].to_numpy(dtype=float)
        return float(values.max() - values.min())

    assert spread(1.09) > spread(1.02)


def test_a_cell_with_no_overlapping_matches_returns_none() -> None:
    forecasts = forecast_frame()
    forecasts["season"] = "1999"
    assert evaluate_cell(forecasts, odds_frame(), league="E0", book="Bet365", market="1X2") is None


def test_a_cell_with_no_closing_odds_returns_none() -> None:
    odds = odds_frame()
    odds["period"] = "open"
    assert evaluate_cell(forecast_frame(), odds, league="E0", book="Bet365", market="1X2") is None


def test_sign_disagreement_is_detected_when_it_exists() -> None:
    result = evaluate_cell(forecast_frame(), odds_frame(), league="E0", book="Bet365", market="1X2")
    assert result is not None
    rois = result.per_method["roi"].to_numpy(dtype=float)
    finite = rois[np.isfinite(rois)]
    expected = bool(finite.size and finite.max() > 0.0 and finite.min() < 0.0)
    assert result.roi_signs_disagree == expected


def test_the_cell_evaluation_is_deterministic() -> None:
    first = evaluate_cell(forecast_frame(), odds_frame(), league="E0", book="Bet365", market="1X2")
    second = evaluate_cell(forecast_frame(), odds_frame(), league="E0", book="Bet365", market="1X2")
    assert first is not None and second is not None
    assert first.per_method.equals(second.per_method)


def test_merge_survives_a_csv_round_trip(tmp_path: Path) -> None:
    """Regression, three times over. Forecasts reach evaluate_cell through a
    CSV round trip while odds do not, so the two sides arrive with whatever
    dtypes pandas inferred. Season came back as int64 once, kickoff as a string
    against datetime64 another time, and parse_dates silently did nothing once
    an explicit dtype map was passed alongside it. Each killed a full run."""
    forecasts = forecast_frame()
    path = tmp_path / "E0.csv"
    forecasts.to_csv(path, index=False)

    # Deliberately the worst case: no parse_dates, no dtype hints at all.
    reloaded = pd.read_csv(path)
    assert not pd.api.types.is_datetime64_any_dtype(reloaded["kickoff"])

    cell = evaluate_cell(reloaded, odds_frame(), league="E0", book="Bet365", market="1X2")
    assert cell is not None
    assert cell.n_matches == 60


def test_merge_is_unaffected_by_key_whitespace() -> None:
    forecasts = forecast_frame()
    forecasts["home"] = forecasts["home"] + " "
    odds = odds_frame()
    cell = evaluate_cell(forecasts, odds, league="E0", book="Bet365", market="1X2")
    assert cell is not None
    assert cell.n_matches == 60
