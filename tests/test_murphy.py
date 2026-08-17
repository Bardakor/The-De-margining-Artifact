"""MODEL.md §11.1 — the identity is not exact for continuous forecasts."""

from typing import Any

import numpy as np
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


# --------------------------------------------------------------------------
# The fifth term — an error found in MODEL.md §11.1 itself
# --------------------------------------------------------------------------


def test_the_five_term_identity_is_exact_on_continuous_forecasts() -> None:
    """MODEL.md §11.1's four-term identity is NOT exact for arbitrary
    forecasts. Expanding (p-o)^2 within a bin leaves a cross term equal to
    minus twice the within-bin forecast-outcome covariance."""
    rng = np.random.default_rng(0)
    for n in (200, 1000, 4000):
        p = rng.uniform(0.02, 0.98, n)
        o = (rng.uniform(size=n) < p).astype(int)
        terms = murphy_decomposition(p, o, n_bins=10)
        assert abs(terms.residual) < 1e-12, f"n={n}: residual {terms.residual:.3e}"


def test_the_four_term_form_is_measurably_wrong_on_continuous_forecasts() -> None:
    """The complement of the test above: if the four-term form were also exact
    here, the fifth term would be redundant and should be removed."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 4000)
    o = (rng.uniform(size=4000) < p).astype(int)
    terms = murphy_decomposition(p, o, n_bins=10)
    assert abs(terms.four_term - terms.brier) > 1e-4
    assert abs(terms.within_bin_covariance) > 1e-5


def test_the_model_md_fixture_cannot_distinguish_the_two_forms(
    model_md: dict[str, Any],
) -> None:
    """Why the omission went unnoticed. Every bin in the specification's own
    example holds either one point, or two points sharing an outcome, so the
    covariance is identically zero and both forms agree there."""
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    assert terms.within_bin_covariance == pytest.approx(0.0, abs=1e-15)
    assert terms.four_term == pytest.approx(terms.brier, abs=1e-12)
    assert terms.reconstructed == pytest.approx(terms.brier, abs=1e-12)


def test_covariance_vanishes_when_a_bin_holds_one_distinct_forecast() -> None:
    """The discrete case Murphy wrote about: no within-bin spread, so no
    within-bin covariance either."""
    p = np.array([0.15, 0.15, 0.45, 0.45, 0.75, 0.75, 0.95, 0.95])
    o = np.array([0, 1, 1, 0, 1, 1, 1, 0])
    terms = murphy_decomposition(p, o, n_bins=10)
    assert terms.within_bin_variance == pytest.approx(0.0, abs=1e-15)
    assert terms.within_bin_covariance == pytest.approx(0.0, abs=1e-15)
    assert terms.three_term == pytest.approx(terms.brier, abs=1e-12)


def test_covariance_is_positive_when_higher_forecasts_pick_the_winners() -> None:
    """Inside a single bin, order the outcomes by the forecast. The covariance
    must then be strictly positive, which is exactly the information the
    four-term form discards."""
    p = np.array([0.31, 0.33, 0.35, 0.37, 0.39])
    o = np.array([0, 0, 0, 1, 1])
    terms = murphy_decomposition(p, o, n_bins=10)
    assert terms.within_bin_covariance > 0.0
    assert abs(terms.residual) < 1e-12


def test_reconstruction_holds_across_bin_counts() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.98, 1500)
    o = (rng.uniform(size=1500) < p).astype(int)
    for n_bins in (1, 2, 5, 10, 20, 50):
        assert abs(murphy_decomposition(p, o, n_bins=n_bins).residual) < 1e-12
