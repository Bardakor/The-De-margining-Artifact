"""Exponential time decay on the likelihood.

MODEL.md §4. Dixon and Coles' second contribution, after tau, was recognising
that team strength is not static and that an unweighted likelihood therefore
mis-states it. Each match contributes with weight

    phi(dt) = exp(-xi * dt)

where dt is the age of the match in days at the moment of the fit. xi controls
how fast the past is forgotten.

xi is selected once on a calibration period and then frozen — see the study
spec §5.3. Re-tuning it against evaluation data would let the model absorb
information about the very matches used to judge it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

SECONDS_PER_DAY = 86_400.0


def half_life_to_xi(days: float) -> float:
    """Convert a half-life in days to the decay rate xi.

    A half-life is the interpretable quantity — "matches a year old count half
    as much" — while xi is what the likelihood uses.
    """
    if not days > 0.0:
        raise ValueError(f"half-life must be strictly positive, got {days}")
    return float(np.log(2.0) / days)


def xi_to_half_life(xi: float) -> float:
    """Inverse of :func:`half_life_to_xi`."""
    if not xi > 0.0:
        raise ValueError(f"xi must be strictly positive, got {xi}")
    return float(np.log(2.0) / xi)


def match_age_days(
    kickoffs: pd.Series[Any] | npt.NDArray[np.datetime64],
    as_of: pd.Timestamp,
) -> npt.NDArray[np.float64]:
    """Age of each match in days at ``as_of``.

    Raises:
        ValueError: If any kickoff is at or after ``as_of``. That would mean a
            match is being used to fit a model that predicts it, which is the
            single point where leakage can enter the walk-forward protocol.
    """
    stamps = pd.to_datetime(pd.Series(kickoffs).reset_index(drop=True))
    if stamps.isna().any():
        raise ValueError("kickoffs contain missing timestamps")
    deltas = (pd.Timestamp(as_of) - stamps).dt.total_seconds().to_numpy(dtype=np.float64)
    if np.any(deltas <= 0.0):
        n_bad = int((deltas <= 0.0).sum())
        raise ValueError(
            f"{n_bad} match(es) at or after the fit cutoff {as_of}; "
            "a fit may only use strictly earlier matches"
        )
    return np.asarray(deltas / SECONDS_PER_DAY, dtype=np.float64)


def time_weights(
    kickoffs: pd.Series[Any] | npt.NDArray[np.datetime64],
    as_of: pd.Timestamp,
    xi: float,
) -> npt.NDArray[np.float64]:
    """Likelihood weight for each match: ``exp(-xi * age_in_days)``.

    ``xi = 0`` gives an unweighted fit, which is the Maher (1982) model and a
    useful baseline rather than an error.
    """
    if xi < 0.0:
        raise ValueError(f"xi must be non-negative, got {xi}")
    ages = match_age_days(kickoffs, as_of)
    return np.asarray(np.exp(-xi * ages), dtype=np.float64)
