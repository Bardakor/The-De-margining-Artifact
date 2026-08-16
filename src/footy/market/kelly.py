"""Bankroll fraction maximising long-run logarithmic growth.

MODEL.md §10:

    f* = (b*p - q) / b,   b = d - 1,  q = 1 - p

Note b*p - q == d*p - 1, so Kelly, edge and expected value can never disagree
about whether a bet is worth taking.

A negative f* means do not bet -- not bet the other side, which has its own
price and its own assessment.

The default is quarter Kelly. Full Kelly is intolerably volatile once the
probability estimate itself carries error.
"""

DEFAULT_FRACTION = 0.25


def expected_value(probability: float, decimal_odds: float) -> float:
    """Expected profit per unit staked: d*p - 1."""
    _validate(probability, decimal_odds)
    return decimal_odds * probability - 1.0


def kelly_fraction(
    probability: float, decimal_odds: float, fraction: float = DEFAULT_FRACTION
) -> float:
    """Fraction of bankroll to stake, floored at zero."""
    _validate(probability, decimal_odds)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Kelly fraction must lie in (0, 1], got {fraction}")
    b = decimal_odds - 1.0
    full = (b * probability - (1.0 - probability)) / b
    return max(0.0, full * fraction)


def _validate(probability: float, decimal_odds: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    if not decimal_odds > 1.0:
        raise ValueError(f"decimal odds must be greater than 1, got {decimal_odds}")
