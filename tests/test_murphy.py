"""MODEL.md §11.1 — the identity is not exact for continuous forecasts."""

from typing import Any

import pytest

from footy.eval.murphy import murphy_decomposition


def test_reproduces_the_model_md_table(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §11.1.

    The outcome vector is derived rather than transcribed; see the fixture's
    _outcomes_provenance note.
    """
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    tol = spec["tolerance"]

    assert terms.brier == pytest.approx(spec["brier"], abs=tol)
    assert terms.uncertainty == pytest.approx(spec["uncertainty"], abs=tol)
    assert terms.within_bin_variance == pytest.approx(spec["wbv"], abs=tol)

    three_way = terms.reliability - terms.resolution + terms.uncertainty
    assert three_way == pytest.approx(spec["three_way_sum"], abs=tol)


def test_the_four_term_identity_is_exact(model_md: dict[str, Any]) -> None:
    """MODEL.md §11.1: BS = REL - RES + UNC + WBV, to 1e-12."""
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    assert (
        terms.reliability - terms.resolution + terms.uncertainty + terms.within_bin_variance
    ) == pytest.approx(terms.brier, abs=1e-12)


def test_the_residual_is_the_within_bin_variance(model_md: dict[str, Any]) -> None:
    """Guards against anyone quietly simplifying back to three terms."""
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    residual = terms.brier - (terms.reliability - terms.resolution + terms.uncertainty)
    assert residual == pytest.approx(terms.within_bin_variance, abs=1e-12)


def test_wbv_vanishes_when_every_bin_holds_one_distinct_forecast() -> None:
    """The discrete case Murphy was writing about: the three-way identity is exact."""
    forecasts = [0.05, 0.05, 0.45, 0.45, 0.75, 0.75, 0.95, 0.95]
    outcomes = [0, 0, 1, 0, 1, 1, 1, 1]
    terms = murphy_decomposition(forecasts, outcomes, n_bins=10)
    assert terms.within_bin_variance == pytest.approx(0.0, abs=1e-12)
    assert (terms.reliability - terms.resolution + terms.uncertainty) == pytest.approx(
        terms.brier, abs=1e-12
    )


def test_a_perfectly_calibrated_forecast_has_zero_reliability() -> None:
    forecasts = [0.5] * 100
    outcomes = [i % 2 for i in range(100)]
    terms = murphy_decomposition(forecasts, outcomes, n_bins=10)
    assert terms.reliability == pytest.approx(0.0, abs=1e-12)


def test_a_base_rate_forecast_has_zero_resolution() -> None:
    """Saying the base rate every time means saying nothing."""
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    terms = murphy_decomposition([0.3] * 10, outcomes, n_bins=10)
    assert terms.resolution == pytest.approx(0.0, abs=1e-12)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        murphy_decomposition([0.5, 0.5], [1], n_bins=10)


def test_rejects_non_binary_outcomes() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        murphy_decomposition([0.5, 0.5], [0, 2], n_bins=10)
