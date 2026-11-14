"""M6 Recognition: what the Ballon d'Or vote rewards (design §5 M6, pre-registration H6).

Two models of each edition's vote over its candidate universe, both explained by the same
blocks of features (individual statistics, impact, team success, reputation):

- OLS of log(1 + points) with edition fixed effects, and its leave-one-edition-out R²;
- a Plackett-Luce model of the ranked order with utility = features . beta, fitted by
  maximum likelihood (analytic gradient), and its McFadden pseudo-R².

For each model: the exact Shapley decomposition of explained variance across the blocks
(every coalition), residuals per player-edition, and a player-cluster bootstrap of the
Shapley shares.

Input: one DataFrame, one row per (edition, candidate), with columns ``edition``,
``player``, ``points`` (0 for unranked), ``rank`` (NaN for unranked) and the feature columns
named in ``blocks`` (block name -> columns). Pure: no I/O; randomness only through the
Generator passed to :func:`bootstrap_shares`.

Assumptions (Claude's calls)

- Features arrive aligned. Each edition's statistics, RAPM and team success belong to the
  season that edition voted on, so the caller who assembles the universe applies the
  edition -> season mapping. Reputation is the previous edition's points and the log market
  value at the vote. Every feature value must be finite.
- R² means the within-edition R². The edition fixed effects are always in the model and are
  not a block, so R²(no block) = 0 and the Shapley shares sum to the full model's R².
- Cross-validated R² leaves one edition out. Training on the other editions cannot estimate
  the held-out edition's fixed effect, so y and X are demeaned within each edition and the
  held-out edition is scored on its demeaned values. Its fixed effect is the one number
  taken from its own data.
- Plackett-Luce likelihood. Unranked candidates rank below every ranked player and have no
  order among themselves. Each ranked player i contributes exp(u_i) / sum_{j in R_i}
  exp(u_j), where R_i holds everyone not ranked strictly above i. Unranked candidates are in
  every R_i and contribute no numerator. Summing the full Plackett-Luce probability over
  every order of the unranked gives exactly this (top-K likelihood). Tied ranks follow the
  Breslow convention: tied players share one risk set. There is no intercept and no edition
  effect, because utilities are defined only up to a constant within an edition.
- McFadden pseudo-R² = 1 - LL(beta_hat) / LL(0), where LL(0) is the uniform model over the
  same universes.
- The bootstrap resamples players (each player is a cluster of all his editions), not
  editions. Five editions are too few clusters for an edition bootstrap, and a player's
  residual persists from one edition to the next. A player drawn twice in an edition ties
  with himself.
- A positive residual means the vote rewarded the player beyond what all the blocks
  predict. OLS residual: log(1 + points) - fitted. Plackett-Luce residual: expected position
  - observed position, with E[position_i] = 1 + sum_{j != i} P(j above i) and
  P(j above i) = 1 / (1 + exp(u_i - u_j)) (the pairwise marginal of a Plackett-Luce
  ranking). The observed position is the mid-rank, and the unranked share the bottom
  group's mid-rank.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
Blocks = Mapping[str, Sequence[str]]


def plackett_luce_loglik(
    beta: FloatArray, x: FloatArray, rank: FloatArray, edition: IntArray
) -> tuple[float, FloatArray]:
    """Log-likelihood of the ranked orders and its gradient in beta.

    ``rank`` is +inf for unranked candidates, and only the order of ranks within an
    edition matters.
    """
    u = x @ beta
    loglik = 0.0
    grad = np.zeros_like(beta)
    for e in np.unique(edition):
        in_e = edition == e
        order = np.argsort(rank[in_e], kind="stable")
        ue, xe, re = u[in_e][order], x[in_e][order], rank[in_e][order]
        ranked = np.isfinite(re)
        top = ue.max()
        w = np.exp(ue - top)
        tail_w = np.cumsum(w[::-1])[::-1]  # sum of w over sorted positions >= p
        tail_wx = np.cumsum((w[:, None] * xe)[::-1], axis=0)[::-1]
        start = np.searchsorted(re, re[ranked], side="left")  # first index of i's risk set
        loglik += float(np.sum(ue[ranked] - top - np.log(tail_w[start])))
        grad += xe[ranked].sum(axis=0) - (tail_wx[start] / tail_w[start, None]).sum(axis=0)
    return loglik, grad


@dataclass(frozen=True)
class PlackettLuceFit:
    beta: FloatArray
    loglik: float
    loglik_null: float
    converged: bool

    @property
    def pseudo_r2(self) -> float:
        """McFadden: 1 - LL(beta_hat) / LL(0)."""
        return 1.0 - self.loglik / self.loglik_null


def fit_plackett_luce(x: FloatArray, rank: FloatArray, edition: IntArray) -> PlackettLuceFit:
    """Maximum likelihood by BFGS on columns scaled to unit SD, returned on the input scale."""
    sd = x.std(axis=0)
    scale = np.where(sd > 0.0, sd, 1.0)
    z = x / scale

    def negative(gamma: FloatArray) -> tuple[float, FloatArray]:
        loglik, grad = plackett_luce_loglik(gamma, z, rank, edition)
        return -loglik, -grad

    zero = np.zeros(x.shape[1])
    result = minimize(negative, zero, jac=True, method="BFGS")
    return PlackettLuceFit(
        beta=np.asarray(result.x, dtype=np.float64) / scale,
        loglik=-float(result.fun),
        loglik_null=plackett_luce_loglik(zero, z, rank, edition)[0],
        converged=bool(result.success),
    )


def shapley(value: Callable[[frozenset[str]], float], names: Sequence[str]) -> dict[str, float]:
    """Exact Shapley values of the game ``value`` over ``names``, using every coalition."""
    n = len(names)
    cache: dict[frozenset[str], float] = {}

    def v(coalition: frozenset[str]) -> float:
        if coalition not in cache:
            cache[coalition] = value(coalition)
        return cache[coalition]

    shares = {}
    for name in names:
        others = [m for m in names if m != name]
        shares[name] = sum(
            factorial(k)
            * factorial(n - k - 1)
            / factorial(n)
            * (v(frozenset(c) | {name}) - v(frozenset(c)))
            for k in range(n)
            for c in combinations(others, k)
        )
    return shares


@dataclass(frozen=True)
class _Data:
    y: FloatArray  # log(1 + points)
    x: FloatArray  # every block's columns, in block order
    rank: FloatArray  # +inf for unranked
    edition: IntArray
    columns: dict[str, list[int]]  # block -> column indices into x

    def cols(self, coalition: frozenset[str]) -> list[int]:
        return [i for block, idx in self.columns.items() if block in coalition for i in idx]


def _data(frame: pd.DataFrame, blocks: Blocks) -> _Data:
    names = [c for cols in blocks.values() for c in cols]
    x = frame[names].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("feature columns must be finite")
    columns, start = {}, 0
    for block, cols in blocks.items():
        columns[block] = list(range(start, start + len(cols)))
        start += len(cols)
    codes, _ = pd.factorize(frame["edition"])
    return _Data(
        y=np.log1p(frame["points"].to_numpy(dtype=np.float64)),
        x=x,
        rank=np.nan_to_num(frame["rank"].to_numpy(dtype=np.float64), nan=np.inf),
        edition=np.asarray(codes, dtype=np.intp),
        columns=columns,
    )


def _demean(a: FloatArray, edition: IntArray) -> FloatArray:
    sums = np.zeros((int(edition.max()) + 1, *a.shape[1:]))
    np.add.at(sums, edition, a)
    counts = np.bincount(edition).reshape(-1, *([1] * (a.ndim - 1)))
    demeaned: FloatArray = a - (sums / counts)[edition]
    return demeaned


def _ols(x: FloatArray, y: FloatArray) -> FloatArray:
    """Least squares on demeaned data; minimum-norm solution when columns are collinear."""
    beta: FloatArray = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta


def _r2(resid: FloatArray, y: FloatArray) -> float:
    return 1.0 - float(resid @ resid) / float(y @ y)


def _cv_r2(x: FloatArray, y: FloatArray, edition: IntArray) -> float:
    """Leave-one-edition-out R² on within-edition demeaned x and y."""
    sse = 0.0
    for e in np.unique(edition):
        test = edition == e
        resid = y[test] - x[test] @ _ols(x[~test], y[~test])
        sse += float(resid @ resid)
    return 1.0 - sse / float(y @ y)


def _shares(d: _Data) -> tuple[dict[str, float], dict[str, float], bool]:
    """Shapley shares of the OLS within-R² and of the PL pseudo-R²; True if every PL fit
    converged."""
    yd, xd = _demean(d.y, d.edition), _demean(d.x, d.edition)
    converged: list[bool] = []

    def ols_value(coalition: frozenset[str]) -> float:
        c = d.cols(coalition)
        return _r2(yd - xd[:, c] @ _ols(xd[:, c], yd), yd) if c else 0.0

    def pl_value(coalition: frozenset[str]) -> float:
        c = d.cols(coalition)
        if not c:
            return 0.0
        fit = fit_plackett_luce(d.x[:, c], d.rank, d.edition)
        converged.append(fit.converged)
        return fit.pseudo_r2

    names = list(d.columns)
    return shapley(ols_value, names), shapley(pl_value, names), all(converged)


@dataclass(frozen=True)
class Recognition:
    ols_r2: float  # within-edition R² of the full model
    ols_cv_r2: float  # leave-one-edition-out
    pl_pseudo_r2: float  # McFadden
    ols_shares: dict[str, float]  # Shapley, sums to ols_r2
    pl_shares: dict[str, float]  # Shapley, sums to pl_pseudo_r2
    ols_beta: dict[str, float]
    pl_beta: dict[str, float]
    pl_converged: bool  # every Plackett-Luce fit behind the shares and the full model
    residuals: pd.DataFrame


def fit_recognition(frame: pd.DataFrame, blocks: Blocks) -> Recognition:
    """Both models, their Shapley decompositions and per player-edition residuals."""
    d = _data(frame, blocks)
    ols_shares, pl_shares, converged = _shares(d)
    yd, xd = _demean(d.y, d.edition), _demean(d.x, d.edition)
    beta = _ols(xd, yd)
    resid = yd - xd @ beta
    pl = fit_plackett_luce(d.x, d.rank, d.edition)
    u = d.x @ pl.beta
    expected = np.empty_like(u)
    observed = np.empty_like(u)
    for e in np.unique(d.edition):
        in_e = d.edition == e
        ue = u[in_e]
        expected[in_e] = 0.5 + expit(ue[None, :] - ue[:, None]).sum(axis=1)
        observed[in_e] = rankdata(d.rank[in_e], method="average")
    residuals = pd.DataFrame(
        {
            "edition": frame["edition"].to_numpy(),
            "player": frame["player"].to_numpy(),
            "points": frame["points"].to_numpy(),
            "rank": frame["rank"].to_numpy(),
            "ols_fitted": d.y - resid,
            "ols_residual": resid,
            "pl_utility": u,
            "pl_expected_position": expected,
            "observed_position": observed,
            "pl_residual": expected - observed,
        }
    )
    names = [c for cols in blocks.values() for c in cols]
    return Recognition(
        ols_r2=_r2(resid, yd),
        ols_cv_r2=_cv_r2(xd, yd, d.edition),
        pl_pseudo_r2=pl.pseudo_r2,
        ols_shares=ols_shares,
        pl_shares=pl_shares,
        ols_beta=dict(zip(names, beta.tolist(), strict=True)),
        pl_beta=dict(zip(names, pl.beta.tolist(), strict=True)),
        pl_converged=converged and pl.converged,
        residuals=residuals,
    )


def bootstrap_shares(
    frame: pd.DataFrame, blocks: Blocks, rng: np.random.Generator, n_boot: int = 200
) -> pd.DataFrame:
    """Shapley shares on ``n_boot`` player-cluster resamples.

    One row per (replicate, model in {"ols", "pl"}), with one column per block and a
    ``converged`` flag for the Plackett-Luce fits.
    """
    clusters = list(frame.groupby("player", sort=True).indices.values())
    rows: list[dict[str, object]] = []
    for b in range(n_boot):
        draw = rng.integers(len(clusters), size=len(clusters))
        sample = frame.iloc[np.concatenate([clusters[i] for i in draw])]
        ols, pl, converged = _shares(_data(sample, blocks))
        rows.append({"replicate": b, "model": "ols", **ols, "converged": True})
        rows.append({"replicate": b, "model": "pl", **pl, "converged": converged})
    return pd.DataFrame(rows)
