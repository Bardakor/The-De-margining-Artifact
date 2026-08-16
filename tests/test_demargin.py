"""MODEL.md §9 — recovering true probabilities from offered odds."""

from typing import Any

import numpy as np
import pytest

from footy.market.demargin import shin


def test_reproduces_the_model_md_round_trip(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §9."""
    spec = model_md["shin_round_trip"]
    recovered, z = shin(spec["implied"])
    assert recovered == pytest.approx(spec["recovered"], abs=spec["tolerance"])
    assert z == pytest.approx(spec["z"], abs=spec["tolerance"])


def test_recovered_probabilities_sum_to_one() -> None:
    recovered, _ = shin([0.5, 0.35, 0.25])
    assert float(recovered.sum()) == pytest.approx(1.0, abs=1e-12)


def test_a_fair_book_is_returned_unchanged() -> None:
    """With no margin there is no insider mass to remove."""
    fair = np.array([0.5, 0.3, 0.2])
    recovered, z = shin(fair)
    assert recovered == pytest.approx(fair, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_shortens_the_longshot_more_than_proportional_normalisation() -> None:
    """Shin attributes margin to insider trading, which concentrates on longshots."""
    implied = np.array([0.5, 0.35, 0.25])
    recovered, _ = shin(implied)
    proportional = implied / implied.sum()
    longshot = int(np.argmin(implied))
    assert recovered[longshot] < proportional[longshot]


def test_rejects_a_book_summing_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        shin([0.3, 0.3, 0.3])
