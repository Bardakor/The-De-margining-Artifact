"""Box-score league factors from players who change league (pre-registration A20 ii).

Per metric, on player x season x league cells with league minutes ``M > 0`` and
count ``Y``::

    log E[Y] = log(M / 90) + alpha_p + gamma_L + z'b,      gamma = 0 for the reference league

with ``z = (age, age^2)`` in A20. A league match's count is then divided by
``exp(gamma_L)``. Pure: no I/O, randomness only through the
``numpy.random.Generator`` argument. Input is a table of cells with
``player_id``, ``league``, ``minutes``, the count column named by ``metric`` and
the covariate columns.

Assumptions:

- ``alpha_p`` is removed by conditioning on the player's total (Hausman, Hall &
  Griliches 1984): given the total, a player's counts are multinomial with cell
  probabilities ``pi`` proportional to ``exp(log(M / 90) + gamma_L + z'b)``. A
  player with a single cell, or with a total of zero, has a constant
  likelihood: he is dropped and counted. A constant added to a covariate is
  absorbed by ``alpha_p``, so age may be centred.
- The estimate solves the score equations ``sum((Y - total * pi) * x) = 0``
  (``x`` a cell's design row) with their analytic Jacobian, by
  ``scipy.optimize.root``. The log-likelihood is concave, so the root is its
  maximum; the solver's convergence test does not compare likelihood values,
  which near the optimum differ by less than rounding error.
- Pooling (A20; docs/unmeasured/league-strength.md §2.2, §3.2): a linking player
  of a league has at least ``min_minutes`` league minutes, summed over seasons,
  in it and in another league. A league with at least ``min_players`` of them
  has its own ``gamma``; the others share one.
- ``gamma`` is identified only if players seen under two league parameters chain
  every parameter to the reference: ``fit`` raises otherwise, and a bootstrap
  replicate that breaks a chain is NaN. If every linking player of a league has
  all of his count on one side, its ``gamma`` diverges and the solver raises.
- Bootstrap: the players who inform the likelihood are resampled with
  replacement (a player drawn k times has weight k); each replicate starts from
  the point estimate. NaN replicates are left out of the standard error and of
  the percentile interval, and counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import root
from scipy.sparse.csgraph import connected_components

REFERENCE = "GB1"
POOLED = "REST"
MIN_LINKING_PLAYERS = 100
MIN_LINKING_MINUTES = 450.0


@dataclass(frozen=True)
class Cells:
    """Cells of the players who inform the likelihood, sorted by player."""

    names: tuple[str, ...]  # league parameters (reference excluded), then covariates
    design: NDArray[np.float64]  # (cells, parameters): league dummies, then covariates
    offset: NDArray[np.float64]  # log(minutes / 90)
    count: NDArray[np.float64]
    player: NDArray[np.int64]  # per cell: 0 .. n_players - 1, non-decreasing
    start: NDArray[np.int64]  # first cell of each player
    total: NDArray[np.float64]  # count per player
    seen: NDArray[np.bool_]  # (players, 1 + league parameters): has a cell under it
    n_single_cell: int  # players dropped: one cell
    n_zero_count: int  # players dropped: several cells, total of zero


def league_parameters(
    cells: pd.DataFrame,
    *,
    min_players: int = MIN_LINKING_PLAYERS,
    min_minutes: float = MIN_LINKING_MINUTES,
    pooled: str = POOLED,
) -> pd.DataFrame:
    """Per league: its linking players and the parameter it gets (own code or ``pooled``)."""
    total = cells.groupby(["player_id", "league"])["minutes"].sum()
    enough = total[total >= min_minutes].reset_index()
    n_leagues = enough.groupby("player_id")["league"].transform("size")
    linking = enough.loc[n_leagues >= 2, "league"].value_counts()
    out = pd.DataFrame({"league": np.sort(cells["league"].unique())})
    out["n_linking"] = out["league"].map(linking).fillna(0).astype(np.int64)
    out["parameter"] = out["league"].where(out["n_linking"] >= min_players, pooled)
    return out


def pre_move(cells: pd.DataFrame, order: str) -> NDArray[np.bool_]:
    """Cells directly followed, in the player's ``order``, by a cell in another league."""
    s = cells.sort_values(["player_id", order], kind="stable")
    following = s.groupby("player_id")["league"].shift(-1)
    flag = following.notna() & (following != s["league"])
    return flag.reindex(cells.index).to_numpy(dtype=np.bool_)


def prepare_cells(
    cells: pd.DataFrame,
    metric: str,
    parameter_of: Mapping[str, str],
    *,
    covariates: Sequence[str] = ("age", "age2"),
    reference: str = REFERENCE,
) -> Cells:
    by_player = cells.groupby("player_id")[metric]
    several = by_player.transform("size") > 1
    scored = by_player.transform("sum") > 0
    used = cells[several & scored].sort_values("player_id", kind="stable")
    leagues = used["league"].unique()
    free = sorted({parameter_of[x] for x in leagues} - {reference})
    index = {name: k for k, name in enumerate([reference, *free])}
    league = used["league"].map({x: index[parameter_of[x]] for x in leagues}).to_numpy(np.int64)
    player_id = used["player_id"].to_numpy()
    start = np.flatnonzero(np.r_[True, player_id[1:] != player_id[:-1]])
    player = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(used)]))
    count = used[metric].to_numpy(np.float64)
    seen = np.zeros((len(start), len(index)), dtype=np.bool_)
    seen[player, league] = True
    design = np.zeros((len(used), len(free) + len(covariates)))
    own = np.flatnonzero(league)
    design[own, league[own] - 1] = 1.0
    design[:, len(free) :] = used[list(covariates)].to_numpy(np.float64)
    return Cells(
        names=(*free, *covariates),
        design=design,
        offset=np.log(used["minutes"].to_numpy(np.float64) / 90),
        count=count,
        player=player,
        start=start,
        total=np.add.reduceat(count, start),
        seen=seen,
        n_single_cell=int((~several).sum()),
        n_zero_count=int(cells.loc[several & ~scored, "player_id"].nunique()),
    )


def fit(c: Cells) -> pd.Series:
    """Conditional maximum-likelihood estimate of every parameter in ``c.names``."""
    identified = _identified(c.seen)
    if not identified.all():
        loose = [name for name, ok in zip(c.names, identified, strict=False) if not ok]
        raise ValueError(f"no chain of players links {loose} to the reference league")
    return pd.Series(_solve(c, np.ones(len(c.start)), np.zeros(len(c.names))), index=c.names)


def bootstrap(
    c: Cells, estimate: pd.Series, rng: np.random.Generator, *, n_boot: int = 200
) -> NDArray[np.float64]:
    """Player-bootstrap replicates of ``fit``, shape (n_boot, parameters)."""
    n = len(c.start)
    begin = estimate.to_numpy(np.float64)
    reps = np.full((n_boot, len(c.names)), np.nan)
    for b in range(n_boot):
        weight = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        if _identified(c.seen[weight > 0]).all():
            reps[b] = _solve(c, weight, begin)
    return reps


def summarise(
    estimate: pd.Series, replicates: NDArray[np.float64], *, level: float = 0.9
) -> pd.DataFrame:
    """Per parameter: estimate, bootstrap SE, percentile interval, replicates used."""
    lo, hi = np.nanquantile(replicates, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return pd.DataFrame(
        {
            "estimate": estimate,
            "se": np.nanstd(replicates, axis=0, ddof=1),
            "lo": lo,
            "hi": hi,
            "n_boot": np.isfinite(replicates).sum(axis=0),
        }
    )


def _identified(seen: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Per league parameter: chained to the reference by players seen under two parameters."""
    _, component = connected_components(seen.T @ seen, directed=False)
    return np.asarray(component[1:] == component[0], dtype=np.bool_)


def _solve(
    c: Cells, weight: NDArray[np.float64], begin: NDArray[np.float64]
) -> NDArray[np.float64]:
    res = root(_score, begin, args=(c, weight), jac=_score_jacobian, method="hybr")
    if not res.success:
        raise RuntimeError(f"conditional Poisson fit did not converge: {res.message}")
    return np.asarray(res.x, dtype=np.float64)


def _shares(theta: NDArray[np.float64], c: Cells) -> NDArray[np.float64]:
    """Per cell: its share ``pi`` of the player's ``exp(eta)``."""
    eta = c.offset + c.design @ theta
    expo = np.exp(eta - np.maximum.reduceat(eta, c.start)[c.player])
    return expo / np.add.reduceat(expo, c.start)[c.player]


def _score(
    theta: NDArray[np.float64], c: Cells, weight: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gradient of the negative conditional log-likelihood per unit of count.

    ``weight`` is per player. The log-likelihood is ``sum(Y * log(pi))`` over cells.
    """
    total = weight * c.total
    resid = total[c.player] * _shares(theta, c) - weight[c.player] * c.count
    return resid @ c.design / total.sum()


def _score_jacobian(
    theta: NDArray[np.float64], c: Cells, weight: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Jacobian of ``_score``: per player, ``total`` times the ``pi``-covariance of ``x``."""
    share = _shares(theta, c)
    total = weight * c.total
    mean = np.add.reduceat(share[:, None] * c.design, c.start)
    second = (c.design * (total[c.player] * share)[:, None]).T @ c.design
    return (second - (mean * total[:, None]).T @ mean) / total.sum()
