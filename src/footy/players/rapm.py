"""Regularised adjusted plus-minus on stints (design §5, M3).

Model. Each stint gives two rows, one per attacking side. In the row where side
``s`` attacks side ``o``::

    y = response_s * 90 / minutes
    E[y] = sum(offence_p, p on s) - sum(defence_q, q on o)
           + home[group] * [s is home] + intercept[group] + beta * c_s

with weight = stint minutes. ``defence`` is therefore goals *prevented* per 90
(higher is better, like ``offence``) and ``total = offence + defence``. Home
advantage and intercept are per group (competition-season, pre-registration
amendment A7: empty stadiums in 2019/20-2020/21 cut home advantage). ``c_s``
is the per-stint covariate taken from the home side's view (e.g. home Elo minus
away Elo): ``+c`` in the home-attacking row, ``-c`` in the away-attacking row.
The response is any pair of per-stint columns (goals by default; zone-xG for
and against work the same way). The penalty ``lam * ||(offence, defence)||^2``
acts on player columns only, through augmented rows; home, intercepts and the
covariate are unpenalised. Every fit is one call to
``scipy.sparse.linalg.lsqr`` on the column-scaled (Jacobi-preconditioned) system.

Assumptions, stated so callers and tests can rely on them:

- Players below ``min_minutes`` over the whole data share one replacement
  player (id ``REPLACEMENT``) with one offence and one defence column; two such
  players on the pitch count twice. It is penalised like any player.
- A zero-minute stint carries zero weight (its goals are ignored).
- The ridge resolves the confounding between the sum of player effects and the
  intercepts (every row has ~11 attackers) by centring players near 0.
- CV, bootstrap and split-half all resample whole games, never single stints.
- Split halves are fitted at the same ``lam`` as the full data; the
  half-correlation is averaged over splits, then stepped up by Spearman-Brown.
  It measures stability, and teammate leakage is stable across halves;
  ``attribution_factor`` removes that part (see its docstring).
- Each bootstrap replicate is also split in two halves of its games (a game
  drawn twice lands twice in the same half), so M4 can re-estimate the
  reliability inside every replicate.
- Per-season estimates reuse the pooled fit as prior: ``season = pooled + delta``
  with penalty ``lam_season * ||delta||^2``; home, intercepts and covariate are
  re-estimated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import lsqr

REPLACEMENT = -1
COMPONENTS = ("offence", "defence", "total")
LSQR_TOL = 1e-10

FloatArray = NDArray[np.float64]
Split = tuple[FloatArray, FloatArray, FloatArray, FloatArray]  # est_a, est_b, minutes_a, _b


@dataclass(frozen=True)
class Design:
    """Weighted design: columns are offence (n_players), defence (n_players),
    one home column per group, one intercept per group, then the covariate."""

    x: sp.csr_matrix
    y: FloatArray
    w: FloatArray
    game: NDArray[np.int64]
    stint_ids: NDArray[np.int64]
    players: pd.Index
    groups: pd.Index
    has_covariate: bool

    @property
    def n_players(self) -> int:
        return len(self.players)

    def player_minutes(self, row_weight: FloatArray | None = None) -> FloatArray:
        w = self.w if row_weight is None else self.w * row_weight
        return np.asarray(self.x[:, : self.n_players].T @ w, dtype=np.float64)


@dataclass(frozen=True)
class RapmFit:
    players: pd.DataFrame
    home: pd.Series
    intercepts: pd.Series
    covariate: float | None
    lam: float


@dataclass(frozen=True)
class Halves:
    """Estimates from two disjoint halves of the games; one row per split,
    one column per player. ``a`` and ``b`` map component -> frame."""

    a: dict[str, pd.DataFrame]
    b: dict[str, pd.DataFrame]
    minutes_a: pd.DataFrame
    minutes_b: pd.DataFrame


@dataclass(frozen=True)
class RapmBootstrap:
    """``replicates[c]`` has one row per replicate; row ``b`` of ``halves`` is
    the split of replicate ``b``."""

    se: pd.DataFrame
    replicates: dict[str, pd.DataFrame]
    halves: Halves


def build_design(
    stints: pd.DataFrame,
    stint_players: pd.DataFrame,
    *,
    home_col: str = "home_goals",
    away_col: str = "away_goals",
    group: str | None = None,
    covariate: str | None = None,
    min_minutes: float = 0.0,
) -> Design:
    """``group`` and ``covariate`` name columns of ``stints``."""
    st = stints.reset_index(drop=True)
    n = len(st)
    sp_ = stint_players.merge(st[["stint_id", "minutes"]], on="stint_id")
    played = sp_.groupby("player_id")["minutes"].sum()
    low = played.index[played < min_minutes]
    pid = sp_["player_id"].where(~sp_["player_id"].isin(low), REPLACEMENT)
    codes, players = pd.factorize(pid, sort=True)
    n_players = len(players)

    row_of = pd.Series(np.arange(n), index=st["stint_id"].to_numpy())
    s = row_of.loc[sp_["stint_id"].to_numpy()].to_numpy()
    is_home = (sp_["side"] == "home").to_numpy()
    attack_row = np.where(is_home, s, n + s)
    defend_row = np.where(is_home, n + s, s)

    if group is None:
        g_codes, groups = np.zeros(n, dtype=np.int64), pd.Index(["all"])
    else:
        g_codes, groups = pd.factorize(st[group], sort=True)
    home0, icpt0 = 2 * n_players, 2 * n_players + len(groups)
    both = np.arange(2 * n)
    rows = [attack_row, defend_row, np.arange(n), both]
    cols = [codes, n_players + codes, home0 + g_codes, icpt0 + np.concatenate([g_codes, g_codes])]
    vals = [np.ones(len(s)), -np.ones(len(s)), np.ones(n), np.ones(2 * n)]
    n_cols = icpt0 + len(groups)
    if covariate is not None:
        c = st[covariate].to_numpy(dtype=np.float64)
        rows.append(both)
        cols.append(np.full(2 * n, n_cols))
        vals.append(np.concatenate([c, -c]))
        n_cols += 1
    x = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * n, n_cols),
    ).tocsr()

    minutes = st["minutes"].to_numpy(dtype=np.float64)
    response = np.concatenate([st[home_col], st[away_col]]).astype(np.float64)
    w = np.concatenate([minutes, minutes])
    y = np.divide(response * 90.0, w, out=np.zeros_like(w), where=w > 0)
    game_codes, _ = pd.factorize(st["game_id"])
    return Design(
        x=x,
        y=y,
        w=w,
        game=np.concatenate([game_codes, game_codes]).astype(np.int64),
        stint_ids=st["stint_id"].to_numpy(dtype=np.int64),
        players=pd.Index(players),
        groups=pd.Index(groups),
        has_covariate=covariate is not None,
    )


def fit_rapm(d: Design, lam: float) -> RapmFit:
    return _to_fit(d, _solve(d, lam), lam)


def cv_lambda(
    d: Design, lams: Sequence[float], rng: np.random.Generator, *, n_folds: int = 5
) -> tuple[float, pd.Series]:
    """K-fold CV with folds of whole games; returns the best lam and the curve
    (minutes-weighted mean squared error of the per-90 response)."""
    fold = (rng.permutation(_n_games(d)) % n_folds)[d.game]
    err = np.zeros(len(lams))
    for k in range(n_folds):
        test = fold == k
        for i, lam in enumerate(lams):
            r = d.y[test] - d.x[test] @ _solve(d, lam, (~test).astype(np.float64))
            err[i] += float(d.w[test] @ r**2)
    curve = pd.Series(err / d.w.sum(), index=pd.Index(lams, name="lam"), name="cv_mse")
    return float(lams[int(np.argmin(err))]), curve


def bootstrap_rapm(
    d: Design, lam: float, rng: np.random.Generator, *, n_boot: int = 200
) -> RapmBootstrap:
    """Match-level bootstrap: games drawn with replacement; each replicate is
    fitted whole and on two halves of its games (three fits per replicate)."""
    n_games = _n_games(d)
    reps = np.empty((n_boot, d.n_players, 3))
    splits = []
    for b in range(n_boot):
        counts = np.bincount(rng.integers(n_games, size=n_games), minlength=n_games)
        reps[b] = _components(_solve(d, lam, counts[d.game].astype(np.float64)), d.n_players)
        splits.append(_split(d, lam, rng, counts.astype(np.float64)))
    replicates = {
        name: pd.DataFrame(reps[:, :, k], columns=d.players) for k, name in enumerate(COMPONENTS)
    }
    se = pd.DataFrame({name: df.std(ddof=1) for name, df in replicates.items()})
    return RapmBootstrap(se=se, replicates=replicates, halves=_stack(d, splits))


def split_halves(d: Design, lam: float, rng: np.random.Generator, *, n_repeats: int = 20) -> Halves:
    """Fits on random halves of the games, ``n_repeats`` independent splits."""
    ones = np.ones(_n_games(d))
    return _stack(d, [_split(d, lam, rng, ones) for _ in range(n_repeats)])


def split_half_reliability(
    halves: Halves, *, min_minutes: float, groups: pd.Series | None = None
) -> pd.DataFrame:
    """Spearman-Brown reliability of offence, defence and total, per group.

    In each split only players with at least ``min_minutes`` in both halves
    enter (never the replacement player). ``groups`` maps player id to a
    label; without it there is one label "all".
    """
    labels = _labels(halves.minutes_a.columns, groups)
    corrs = []
    for r in range(len(halves.minutes_a)):
        enough = (halves.minutes_a.iloc[r] >= min_minutes) & (
            halves.minutes_b.iloc[r] >= min_minutes
        )
        a = np.column_stack([halves.a[c].iloc[r] for c in COMPONENTS])
        b = np.column_stack([halves.b[c].iloc[r] for c in COMPONENTS])
        corrs.append(_group_corr(a, b, labels.where(enough.to_numpy())))
    out = _mean(corrs)
    r_ = out[list(COMPONENTS)]
    out[list(COMPONENTS)] = 2 * r_ / (1 + r_)
    return out


def attribution_factor(
    d: Design,
    lam: float,
    rng: np.random.Generator,
    *,
    min_minutes: float,
    n_sims: int = 5,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Share of an estimate's stable variance that is the player's own effect.

    Plants i.i.d. standard-normal offence and defence effects on the observed
    design, fits it without noise at ``lam`` and returns the mean over
    ``n_sims`` of the squared correlation between estimate and planted effect,
    per group (players with at least ``min_minutes``) and component. The ridge
    lets teammates and opponents leak into each other's estimates; both halves
    of a split share the schedule, so split-half reliability counts the leak as
    signal. ``split_half * attribution`` is the squared correlation of RAPM
    with the player's own effect, the reliability M4 needs. Assumes effects
    independent of who plays with whom.
    """
    labels = _labels(d.players, groups).where(d.player_minutes() >= min_minutes)
    p = d.n_players
    corrs = []
    for _ in range(n_sims):
        beta = rng.standard_normal(2 * p)
        y = np.asarray(d.x[:, : 2 * p] @ beta, dtype=np.float64)
        est = _components(_solve(replace(d, y=y), lam), p)
        corr = _group_corr(est, _components(beta, p), labels)
        corr[list(COMPONENTS)] **= 2
        corrs.append(corr)
    return _mean(corrs)


def fit_rapm_by_season(
    d: Design, pooled: RapmFit, season: pd.Series, lam_season: float
) -> pd.DataFrame:
    """Per-season effects shrunk toward the pooled ones.

    ``season`` labels each stint (indexed by stint_id); ``pooled`` is
    ``fit_rapm`` on the same design. Returns one row per (player_id, season).
    """
    n, p = len(d.stint_ids), d.n_players
    s_codes, s_labels = pd.factorize(season.reindex(d.stint_ids).to_numpy(), sort=True)
    n_s = len(s_labels)
    off, dfn = d.x[:, :p].tocoo(), d.x[:, p : 2 * p].tocoo()
    rows = np.concatenate([off.row, dfn.row])
    keys, inv = np.unique(
        np.concatenate([off.col, dfn.col]) * n_s + s_codes[rows % n], return_inverse=True
    )
    k = len(keys)
    cols = np.concatenate([inv[: off.nnz], k + inv[off.nnz :]])
    block = sp.coo_matrix(
        (np.concatenate([off.data, dfn.data]), (rows, cols)), shape=(2 * n, 2 * k)
    ).tocsr()
    base = pooled.players.reindex(d.players)
    prior = np.concatenate(
        [base["offence"].to_numpy()[keys // n_s], base["defence"].to_numpy()[keys // n_s]]
    )
    index = pd.MultiIndex.from_arrays(
        [d.players[keys // n_s], s_labels[keys % n_s]], names=["player_id", "season"]
    )
    ds = replace(
        d, x=sp.hstack([block, d.x[:, 2 * p :]]).tocsr(), y=d.y - block @ prior, players=index
    )
    delta = _solve(ds, lam_season)
    est = prior + delta[: 2 * k]
    out = pd.DataFrame(
        {"minutes": ds.player_minutes(), "offence": est[:k], "defence": est[k:]}, index=index
    )
    out["total"] = out["offence"] + out["defence"]
    return out


def _split(d: Design, lam: float, rng: np.random.Generator, counts: FloatArray) -> Split:
    """Halve the games at random; ``counts`` weighs each game (bootstrap)."""
    in_a = rng.permutation(len(counts)) < len(counts) // 2
    wa, wb = (counts * in_a)[d.game], (counts * ~in_a)[d.game]
    p = d.n_players
    est_a, est_b = (_components(_solve(d, lam, w), p) for w in (wa, wb))
    return est_a, est_b, d.player_minutes(wa), d.player_minutes(wb)


def _stack(d: Design, splits: list[Split]) -> Halves:
    def frame(i: int, k: int | None) -> pd.DataFrame:
        rows = [s[i] if k is None else s[i][:, k] for s in splits]
        return pd.DataFrame(np.array(rows), columns=d.players)

    return Halves(
        a={c: frame(0, k) for k, c in enumerate(COMPONENTS)},
        b={c: frame(1, k) for k, c in enumerate(COMPONENTS)},
        minutes_a=frame(2, None),
        minutes_b=frame(3, None),
    )


def _labels(players: pd.Index, groups: pd.Series | None) -> pd.Series:
    """Group label per player; NaN for the replacement player."""
    labels = (
        pd.Series("all", index=players, dtype=object)
        if groups is None
        else groups.reindex(players).astype(object)
    )
    return labels.where(players != REPLACEMENT)


def _group_corr(a: FloatArray, b: FloatArray, labels: pd.Series) -> pd.DataFrame:
    """Per-label correlation of each component of a and b; NaN labels skipped."""
    rows = {}
    for g in sorted(labels.dropna().unique()):
        ok = (labels == g).to_numpy()
        rows[g] = [np.corrcoef(a[ok, k], b[ok, k])[0, 1] for k in range(3)] + [ok.sum()]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=[*COMPONENTS, "n_players"])
    out.index.name = "group"
    return out


def _mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames).groupby(level=0).mean()


def _n_games(d: Design) -> int:
    return int(d.game.max()) + 1


def _solve(d: Design, lam: float, row_weight: FloatArray | None = None) -> FloatArray:
    """Ridge coefficients; ``row_weight`` multiplies the minutes weights
    (0 drops a row, 2 counts it twice)."""
    w = d.w if row_weight is None else d.w * row_weight
    keep = w > 0
    sw = np.sqrt(w[keep])
    a = sp.diags(sw) @ d.x[keep]
    n_pen = 2 * d.n_players
    pen = sp.csr_matrix(
        (np.full(n_pen, np.sqrt(lam)), (np.arange(n_pen), np.arange(n_pen))),
        shape=(n_pen, d.x.shape[1]),
    )
    a_aug = sp.vstack([a, pen]).tocsr()
    norms = np.sqrt(np.asarray(a_aug.multiply(a_aug).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    rhs = np.concatenate([sw * d.y[keep], np.zeros(n_pen)])
    z = lsqr(a_aug @ sp.diags(1.0 / norms), rhs, atol=LSQR_TOL, btol=LSQR_TOL)[0]
    return np.asarray(z / norms, dtype=np.float64)


def _components(coef: FloatArray, n_players: int) -> FloatArray:
    off, dfn = coef[:n_players], coef[n_players : 2 * n_players]
    return np.column_stack([off, dfn, off + dfn])


def _to_fit(d: Design, coef: FloatArray, lam: float) -> RapmFit:
    p, g = d.n_players, len(d.groups)
    players = pd.DataFrame(_components(coef, p), index=d.players, columns=COMPONENTS)
    players.insert(0, "minutes", d.player_minutes())
    return RapmFit(
        players=players,
        home=pd.Series(coef[2 * p : 2 * p + g], index=d.groups),
        intercepts=pd.Series(coef[2 * p + g : 2 * p + 2 * g], index=d.groups),
        covariate=float(coef[-1]) if d.has_covariate else None,
        lam=lam,
    )
