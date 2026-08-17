"""Economic evaluation: staking, return on investment, closing-line value.

Study spec §6.2. Prediction P3 — that the model's measured edge changes *sign*
between de-margining transforms — is an economic statement, so it is measured
here rather than in the scoring rules.

The betting rule is the one the spec names: back an outcome when the model's
probability exceeds the de-margined market probability for that outcome, and
settle at the offered price. That rule is deliberately simple, because the
study varies the *benchmark*, not the strategy. Changing both at once would
make any difference in measured edge unattributable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from footy.market.kelly import DEFAULT_FRACTION, kelly_fraction

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int_]


@dataclass(frozen=True)
class Settlement:
    """Per-selection staking and profit, aligned with the input rows."""

    stake: FloatArray
    profit: FloatArray
    selected: BoolArray

    @property
    def n_bets(self) -> int:
        return int(self.selected.sum())

    @property
    def turnover(self) -> float:
        return float(self.stake.sum())

    @property
    def total_profit(self) -> float:
        return float(self.profit.sum())

    @property
    def roi(self) -> float:
        """Profit per unit staked. NaN when nothing was staked.

        NaN rather than zero: "no bets placed" and "bets placed that broke
        even" are different findings, and averaging them together across the
        study's cells would hide the first inside the second.
        """
        return self.total_profit / self.turnover if self.turnover > 0.0 else float("nan")


def value_selections(
    model: FloatArray,
    benchmark: FloatArray,
    *,
    minimum_edge: float = 0.0,
) -> BoolArray:
    """Which selections the model rates above the de-margined benchmark.

    ``minimum_edge`` is an absolute probability threshold, applied identically
    across transforms so it cannot advantage one of them.
    """
    m = np.asarray(model, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    if m.shape != b.shape:
        raise ValueError(f"shape mismatch: model {m.shape} vs benchmark {b.shape}")
    if minimum_edge < 0.0:
        raise ValueError(f"minimum_edge must be non-negative, got {minimum_edge}")
    return np.asarray(m - b > minimum_edge, dtype=bool)


def settle(
    model: FloatArray,
    benchmark: FloatArray,
    offered_odds: FloatArray,
    won: BoolArray,
    *,
    staking: str = "flat",
    fraction: float = DEFAULT_FRACTION,
    minimum_edge: float = 0.0,
) -> Settlement:
    """Stake the value selections and settle them at the offered price.

    Args:
        model: Model probability for each selection.
        benchmark: De-margined market probability for the same selection.
        offered_odds: The price actually available, before de-margining.
        won: Whether each selection was the realised outcome.
        staking: ``"flat"`` for one unit per bet, ``"kelly"`` for fractional
            Kelly against the offered price.
        fraction: Kelly fraction, ignored when staking is flat.
        minimum_edge: Absolute probability threshold for taking a bet.

    Returns:
        Stakes and profits aligned with the inputs; unselected rows are zero.
    """
    m = np.asarray(model, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    d = np.asarray(offered_odds, dtype=np.float64)
    w = np.asarray(won, dtype=bool)
    if not (m.shape == b.shape == d.shape == w.shape):
        raise ValueError("model, benchmark, offered_odds and won must share a shape")
    if np.any(d <= 1.0):
        raise ValueError("offered odds must exceed 1")
    if staking not in ("flat", "kelly"):
        raise ValueError(f"staking must be 'flat' or 'kelly', got {staking!r}")

    selected = value_selections(m, b, minimum_edge=minimum_edge)
    stake = np.zeros_like(m)
    if staking == "flat":
        stake[selected] = 1.0
    else:
        stake[selected] = [
            kelly_fraction(float(p), float(price), fraction=fraction)
            for p, price in zip(m[selected], d[selected], strict=True)
        ]

    # Win returns stake*(d-1); a loss returns -stake.
    profit = np.where(w, stake * (d - 1.0), -stake)
    profit[~selected] = 0.0
    return Settlement(stake=stake, profit=profit, selected=selected)


def closing_line_value(
    taken_odds: FloatArray,
    closing_odds: FloatArray,
    selected: BoolArray | None = None,
) -> FloatArray:
    """Log ratio of the price taken to the closing price, per selection.

    Positive means the price shortened after the bet was struck — the market
    moved toward the selection, which is the standard evidence that a bet was
    priced better than the market's final word.

    Reported in logs so it is symmetric: taking 2.10 on a 2.00 close and taking
    2.00 on a 2.10 close are equal and opposite, which is not true of a ratio.
    """
    taken = np.asarray(taken_odds, dtype=np.float64)
    closing = np.asarray(closing_odds, dtype=np.float64)
    if taken.shape != closing.shape:
        raise ValueError("taken and closing odds must share a shape")
    if np.any(taken <= 1.0) or np.any(closing <= 1.0):
        raise ValueError("odds must exceed 1")

    clv = np.log(taken / closing)
    if selected is not None:
        mask = np.asarray(selected, dtype=bool)
        if mask.shape != clv.shape:
            raise ValueError("selected must share the odds shape")
        clv = np.where(mask, clv, 0.0)
    return np.asarray(clv, dtype=np.float64)


def outcome_won(outcome: IntArray, n_outcomes: int) -> BoolArray:
    """Flatten per-match outcome indices into a per-selection win mask.

    The study evaluates every outcome of every market as a separate selection,
    so a 3-outcome market becomes three rows and exactly one of them won.
    """
    idx = np.asarray(outcome, dtype=np.int_)
    if np.any((idx < 0) | (idx >= n_outcomes)):
        raise ValueError(f"outcome index outside [0, {n_outcomes})")
    grid = np.zeros((idx.size, n_outcomes), dtype=bool)
    grid[np.arange(idx.size), idx] = True
    return grid.reshape(-1)
