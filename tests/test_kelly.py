"""MODEL.md §10 — staking."""

import pytest

from footy.market.kelly import expected_value, kelly_fraction


def test_matches_the_closed_form() -> None:
    p, d = 0.55, 2.10
    b, q = d - 1.0, 1.0 - p
    assert kelly_fraction(p, d, fraction=1.0) == pytest.approx((b * p - q) / b, abs=1e-12)


def test_kelly_and_expected_value_never_disagree() -> None:
    """MODEL.md §10: b*p - q == d*p - 1, so edge, EV and Kelly agree by identity."""
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        for d in (1.2, 1.8, 2.5, 4.0, 11.0):
            assert (kelly_fraction(p, d, fraction=1.0) > 0.0) == (expected_value(p, d) > 0.0)


def test_negative_edge_returns_zero_not_a_reverse_bet() -> None:
    """MODEL.md §10: a negative f* means do not bet, not bet the other side."""
    assert kelly_fraction(0.2, 2.0) == 0.0


def test_quarter_kelly_is_the_default() -> None:
    p, d = 0.55, 2.10
    assert kelly_fraction(p, d) == pytest.approx(
        kelly_fraction(p, d, fraction=1.0) / 4.0, abs=1e-12
    )


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probability"):
        kelly_fraction(1.5, 2.0)
    with pytest.raises(ValueError, match="greater than 1"):
        kelly_fraction(0.5, 1.0)
