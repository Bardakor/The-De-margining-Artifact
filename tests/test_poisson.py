"""MODEL.md §2 — goals as a Poisson process."""

import math

import numpy as np
import pytest

from footy.core.poisson import poisson_pmf


def test_matches_the_naive_formula_for_small_k() -> None:
    lam = 1.6
    k = np.arange(6)
    expected = np.array([math.exp(-lam) * lam**i / math.factorial(i) for i in range(6)])
    assert poisson_pmf(k, lam) == pytest.approx(expected, abs=1e-12)


def test_stays_finite_where_the_naive_form_overflows() -> None:
    """MODEL.md §2: lam**k / k! overflows to NaN for large k; the log form does not."""
    result = poisson_pmf(np.arange(200), 1.6)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_sums_to_one_over_a_wide_support() -> None:
    assert poisson_pmf(np.arange(300), 12.0).sum() == pytest.approx(1.0, abs=1e-12)


def test_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        poisson_pmf(np.arange(5), 0.0)
