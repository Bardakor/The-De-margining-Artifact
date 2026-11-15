"""Dependence structure of player metrics and double counting (design §5, M2).

Input is a player-season frame: one column per metric (per-90 rates) plus
covariate columns (team Elo, league-season label, club id). Every function is
pure; randomness only through an explicit ``numpy.random.Generator``.

Assumptions, stated so callers and the paper can rely on them:

- Complete cases. A row with a missing or infinite value in a column used is an
  error (``ValueError``); the caller drops or imputes first.
- Correlation, PCA, factor analysis and composites work on standardised metrics
  (the correlation matrix), so the units of per-90 rates do not matter.
- Partial correlation = correlation of OLS residuals on an intercept, the
  numeric covariates (linearly) and one dummy per level of each categorical
  covariate (fixed effects). The design is dense: fine for ~10^4 rows and
  ~10^3 levels. The Spearman variant is the Spearman correlation of these
  linear residuals, not a rank-based partial correlation.
- Clustering distance 1 - |rho| treats a strong negative correlation as
  redundancy, like a positive one.
- Parallel analysis permutes each column independently (keeps the skewed
  marginals of per-90 rates) and retains the leading eigenvalues that exceed
  the null quantile, stopping at the first that does not.
- Factor analysis is maximum likelihood on the correlation matrix, minimising
  Joreskog's (1967) likelihood concentrated in the uniquenesses, with R's
  ``factanal`` start, bounds [0.005, 1] and gradient; varimax with Kaiser
  normalisation (R ``stats::varimax``); each factor's sign is set so its
  loadings sum to a positive number and factors are ordered by variance
  explained. Chi-square uses Bartlett's correction.
- Effective numbers use the eigenvalues of the correlation matrix, rounding
  negatives clipped to zero. Both count the noise floor, so k clean factors
  give more than k unless the metrics are exact copies.
- Distance correlation is the V-statistic of Szekely, Rizzo & Bakirov (2007):
  always >= 0 and biased upward by O(n^-1/2) under independence, so "zero"
  is judged by the permutation test. Columns of a block are standardised
  first. Memory is O(n^2): subsample beyond a few thousand rows.
- Team ICC(1): groups are club x league-season cells; league-season is a fixed
  stratum, so the ICC is the share of within-league-season variance that lies
  between clubs. The ANOVA estimate is not truncated at 0. It is an upper bound
  on the team's causal share: clubs that recruit similar players raise it too.
  The bootstrap resamples clubs with replacement (each drawn copy a distinct
  club, all its seasons kept) and reports the standard interval (A17).
- A composite is C = sum_k w_k z_k with z_k the z-score of metric k (sample mean,
  sd with ddof=1). Its diagnostics need the Pearson correlation of the same
  metrics: the variance algebra is linear.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, squareform
from scipy.special import ndtri
from scipy.stats import chi2

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PSI_BOUNDS = (0.005, 1.0)


def _values(frame: pd.DataFrame) -> FloatArray:
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("missing or infinite values: drop incomplete rows first")
    return values


def _corr(x: FloatArray) -> FloatArray:
    return np.asarray(np.corrcoef(x, rowvar=False), dtype=np.float64)


# --- 1. Correlation ---------------------------------------------------------


def correlation(
    metrics: pd.DataFrame, method: Literal["pearson", "spearman"] = "pearson"
) -> pd.DataFrame:
    _values(metrics)
    return metrics.corr(method=method)


def residualise(
    panel: pd.DataFrame,
    metrics: Sequence[str],
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
) -> pd.DataFrame:
    """OLS residuals of each metric on the covariates, categoricals as fixed effects."""
    y = _values(panel[list(metrics)])
    parts = [np.ones((len(panel), 1)), _values(panel[list(numeric)])]
    if categorical:
        groups = panel[list(categorical)]
        if groups.isna().to_numpy().any():
            raise ValueError("missing categorical covariate: drop incomplete rows first")
        parts.append(pd.get_dummies(groups.astype(str), dtype=np.float64).to_numpy())
    design = np.column_stack(parts)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return pd.DataFrame(y - design @ beta, index=panel.index, columns=list(metrics))


def partial_correlation(
    panel: pd.DataFrame,
    metrics: Sequence[str],
    numeric: Sequence[str] = (),
    categorical: Sequence[str] = (),
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:
    return correlation(residualise(panel, metrics, numeric, categorical), method)


# --- 2. Hierarchical clustering ---------------------------------------------


def cluster_metrics(corr: pd.DataFrame) -> tuple[FloatArray, list[str]]:
    """Average-linkage tree on 1 - |rho|, and the leaf order of the metrics for plotting."""
    distance = np.clip(1.0 - np.abs(corr.to_numpy(dtype=np.float64)), 0.0, None)
    np.fill_diagonal(distance, 0.0)
    tree = np.asarray(
        linkage(squareform(distance, checks=False), method="average"), dtype=np.float64
    )
    return tree, [str(corr.columns[i]) for i in leaves_list(tree)]


# --- 3. PCA and parallel analysis -------------------------------------------


@dataclass(frozen=True)
class PCA:
    eigenvalues: FloatArray  # descending; they sum to the number of metrics
    components: pd.DataFrame  # metrics x PC1..PCp, unit-norm, each summing >= 0


def pca(metrics: pd.DataFrame) -> PCA:
    eigenvalues, vectors = np.linalg.eigh(_corr(_values(metrics)))
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    vectors = vectors * np.where(vectors.sum(axis=0) < 0, -1.0, 1.0)
    names = [f"PC{i + 1}" for i in range(len(eigenvalues))]
    return PCA(eigenvalues, pd.DataFrame(vectors, index=metrics.columns, columns=names))


@dataclass(frozen=True)
class ParallelAnalysis:
    observed: FloatArray  # eigenvalues of the data, descending
    threshold: FloatArray  # null quantile of each ranked eigenvalue
    n_components: int


def parallel_analysis(
    metrics: pd.DataFrame,
    rng: np.random.Generator,
    n_repeats: int = 200,
    quantile: float = 0.95,
) -> ParallelAnalysis:
    """Horn (1965): keep components whose eigenvalue beats column-permuted data."""
    x = _values(metrics)
    observed = np.linalg.eigvalsh(_corr(x))[::-1]
    null = np.array(
        [np.linalg.eigvalsh(_corr(rng.permuted(x, axis=0)))[::-1] for _ in range(n_repeats)]
    )
    threshold = np.asarray(np.quantile(null, quantile, axis=0), dtype=np.float64)
    n_components = int(np.cumprod(observed > threshold).sum())
    return ParallelAnalysis(observed, threshold, n_components)


# --- 4. Maximum-likelihood factor analysis ----------------------------------


@dataclass(frozen=True)
class FactorAnalysis:
    loadings: pd.DataFrame  # metrics x F1..Fk, varimax-rotated
    uniquenesses: pd.Series
    discrepancy: float  # ML discrepancy F = log|Sigma| + tr(R Sigma^-1) - log|R| - p
    chi2: float  # Bartlett-corrected (n - 1 - (2p + 5)/6 - 2k/3) * F
    dof: int  # ((p - k)^2 - (p + k)) / 2
    p_value: float
    rmsea: float
    converged: bool


def _fa_given_psi(psi: FloatArray, r: FloatArray, k: int) -> tuple[float, FloatArray]:
    """Discrepancy and optimal unrotated loadings for fixed uniquenesses psi."""
    scale = 1.0 / np.sqrt(psi)
    theta, omega = np.linalg.eigh(r * np.outer(scale, scale))  # ascending
    rest = theta[: len(psi) - k]
    top, vectors = theta[::-1][:k], omega[:, ::-1][:, :k]
    loadings = np.sqrt(psi)[:, None] * vectors * np.sqrt(np.maximum(top - 1.0, 0.0))
    return float(np.sum(rest - np.log(rest) - 1.0)), loadings


def varimax(loadings: FloatArray, tol: float = 1e-5, max_iter: int = 1000) -> FloatArray:
    """Kaiser's varimax with row normalisation, as R's ``stats::varimax``."""
    norm = np.sqrt(np.sum(loadings**2, axis=1, keepdims=True))
    x = loadings / norm
    p, k = x.shape
    rotation = np.eye(k)
    criterion = 0.0
    for _ in range(max_iter):
        z = x @ rotation
        u, s, vt = np.linalg.svd(x.T @ (z**3 - z * np.sum(z**2, axis=0) / p))
        rotation = u @ vt
        previous, criterion = criterion, float(np.sum(s))
        if criterion < previous * (1.0 + tol):
            break
    return np.asarray((x @ rotation) * norm, dtype=np.float64)


def factor_analysis(metrics: pd.DataFrame, n_factors: int) -> FactorAnalysis:
    x = _values(metrics)
    n, p = x.shape
    k = n_factors
    dof = ((p - k) ** 2 - (p + k)) // 2
    if dof <= 0:
        raise ValueError(f"{k} factors on {p} metrics leave {dof} degrees of freedom")
    r = _corr(x)

    def objective(psi: FloatArray) -> tuple[float, FloatArray]:
        f, lam = _fa_given_psi(psi, r, k)
        return f, (np.sum(lam**2, axis=1) + psi - 1.0) / psi**2  # diag(Sigma - R) / psi^2

    start = (1.0 - 0.5 * k / p) / np.diag(np.linalg.inv(r))
    fit = minimize(
        objective,
        np.clip(start, *PSI_BOUNDS),
        jac=True,
        method="L-BFGS-B",
        bounds=[PSI_BOUNDS] * p,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    psi = np.asarray(fit.x, dtype=np.float64)
    discrepancy, lam = _fa_given_psi(psi, r, k)
    if k > 1:
        lam = varimax(lam)
    lam = lam * np.where(lam.sum(axis=0) < 0, -1.0, 1.0)
    lam = lam[:, np.argsort(-np.sum(lam**2, axis=0), kind="stable")]
    stat = (n - 1 - (2 * p + 5) / 6 - 2 * k / 3) * discrepancy
    return FactorAnalysis(
        loadings=pd.DataFrame(lam, index=metrics.columns, columns=[f"F{i + 1}" for i in range(k)]),
        uniquenesses=pd.Series(psi, index=metrics.columns, name="uniqueness"),
        discrepancy=discrepancy,
        chi2=stat,
        dof=dof,
        p_value=float(chi2.sf(stat, dof)),
        rmsea=float(np.sqrt(max(stat - dof, 0.0) / (dof * (n - 1)))),
        converged=bool(fit.success),
    )


# --- 5. Effective number of independent metrics -----------------------------


def participation_ratio(eigenvalues: FloatArray) -> float:
    """(sum lambda)^2 / sum lambda^2: p for independent metrics, 1 for one repeated metric."""
    lam = np.clip(eigenvalues, 0.0, None)
    return float(lam.sum() ** 2 / np.sum(lam**2))


def entropy_effective_number(eigenvalues: FloatArray) -> float:
    """exp(Shannon entropy of lambda / sum lambda)."""
    lam = np.clip(eigenvalues, 0.0, None)
    share = lam[lam > 0] / lam.sum()
    return float(np.exp(-np.sum(share * np.log(share))))


# --- 6. Distance correlation ------------------------------------------------


def _centred_distances(x: FloatArray) -> FloatArray:
    x = x.reshape(len(x), -1)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    d = np.asarray(cdist(x, x), dtype=np.float64)
    return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()


def _dcor(a: FloatArray, b: FloatArray) -> float:
    dcov2 = max(float(np.mean(a * b)), 0.0)
    return float(np.sqrt(dcov2 / np.sqrt(np.mean(a * a) * np.mean(b * b))))


def distance_correlation(x: FloatArray, y: FloatArray) -> float:
    """dCor of two samples (n,) or blocks (n, p), (n, q); 0 iff independent in population."""
    return _dcor(_centred_distances(x), _centred_distances(y))


def distance_correlation_test(
    x: FloatArray, y: FloatArray, rng: np.random.Generator, n_permutations: int = 999
) -> tuple[float, float]:
    """dCor and its permutation p-value (1 + #{T* >= T}) / (1 + R), permuting y's rows."""
    a, b = _centred_distances(x), _centred_distances(y)
    observed = np.mean(a * b)
    null = np.array(
        [
            np.mean(a * b[np.ix_(perm, perm)])
            for perm in (rng.permutation(len(b)) for _ in range(n_permutations))
        ]
    )
    return _dcor(a, b), float((1 + np.sum(null >= observed)) / (1 + n_permutations))


# --- 7. Team intraclass correlation -----------------------------------------


def _icc1(y: FloatArray, cell_key: IntArray, stratum_key: IntArray) -> FloatArray:
    """One-way ANOVA ICC(1) of each column of y, unequal cells, cells nested in fixed strata.

    E[MSB] = sigma_w^2 + n0 sigma_b^2 with n0 = (N - sum_g n_g^2 / N_s(g)) / (G - S),
    E[MSW] = sigma_w^2, so ICC = (MSB - MSW) / (MSB + (n0 - 1) MSW).
    """
    cell = np.unique(cell_key, return_inverse=True)[1]
    stratum = np.unique(stratum_key, return_inverse=True)[1]
    n, n_cells, n_strata = len(y), int(cell.max()) + 1, int(stratum.max()) + 1
    cell_stratum = np.zeros(n_cells, dtype=np.int64)
    cell_stratum[cell] = stratum
    size = np.bincount(cell).astype(np.float64)
    stratum_size = np.bincount(stratum).astype(np.float64)
    df_between, df_within = n_cells - n_strata, n - n_cells
    n0 = (n - np.sum(size**2 / stratum_size[cell_stratum])) / df_between
    out = np.empty(y.shape[1])
    for j, col in enumerate(y.T):
        cell_mean = np.bincount(cell, weights=col) / size
        stratum_mean = np.bincount(stratum, weights=col) / stratum_size
        msw = np.sum((col - cell_mean[cell]) ** 2) / df_within
        msb = np.sum(size * (cell_mean - stratum_mean[cell_stratum]) ** 2) / df_between
        out[j] = (msb - msw) / (msb + (n0 - 1.0) * msw)
    return out


def team_icc(
    panel: pd.DataFrame,
    metrics: Sequence[str],
    club: str,
    stratum: str | None,
    rng: np.random.Generator,
    n_boot: int = 1000,
    level: float = 0.95,
) -> pd.DataFrame:
    """ICC(1) of each metric between clubs within stratum, with a club bootstrap interval.

    Returns one row per metric: ``icc``, ``lo``, ``hi``. ``stratum=None`` is the plain
    one-way ICC with clubs as groups. The interval is the standard bootstrap interval
    icc +- z sd, sd the SD of the club-bootstrap estimates and z the normal quantile of
    ``level`` (amendment A17). The club bootstrap sits below the estimate (a club drawn
    twice adds identical cells, which shrinks the between-club mean square) while the
    estimate itself is about unbiased, so percentile intervals are shifted low and basic
    intervals high; the standard interval uses only the bootstrap's spread.
    """
    y = _values(panel[list(metrics)])
    clubs = pd.factorize(panel[club])[0].astype(np.int64)
    strata = (
        pd.factorize(panel[stratum])[0].astype(np.int64)
        if stratum is not None
        else np.zeros(len(panel), dtype=np.int64)
    )
    n_clubs, n_strata = int(clubs.max()) + 1, int(strata.max()) + 1
    estimate = _icc1(y, clubs * n_strata + strata, strata)
    order = np.argsort(clubs, kind="stable")
    rows = np.split(order, np.flatnonzero(np.diff(clubs[order])) + 1)
    boot = np.empty((n_boot, len(metrics)))
    for b in range(n_boot):
        drawn = [rows[c] for c in rng.integers(n_clubs, size=n_clubs)]
        idx = np.concatenate(drawn)
        copy = np.repeat(np.arange(n_clubs), [len(d) for d in drawn])
        boot[b] = _icc1(y[idx], copy * n_strata + strata[idx], strata[idx])
    half = ndtri((1 + level) / 2) * boot.std(axis=0, ddof=1)
    return pd.DataFrame(
        {"icc": estimate, "lo": estimate - half, "hi": estimate + half}, index=list(metrics)
    )


# --- 8. Double counting in weighted composites ------------------------------


def composite(metrics: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """V1-style composite C = sum_k w_k z_k over the metrics named in ``weights``."""
    x = _values(metrics[list(weights.index)])
    z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    return pd.Series(z @ weights.to_numpy(dtype=np.float64), index=metrics.index, name="C")


@dataclass(frozen=True)
class CompositeDiagnostics:
    inflation: float  # Var(C) / sum w^2; 1 when the metrics are uncorrelated
    shares: pd.DataFrame  # per metric: claimed w_k^2 / sum w^2, effective w_k Cov(z_k, C)/Var(C)
    pc1_share: float  # share of Var(C) along the first principal axis of the metrics


def composite_diagnostics(corr: pd.DataFrame, weights: pd.Series) -> CompositeDiagnostics:
    """Double-counting diagnostics of C = w'z, from the Pearson correlation R of z.

    Var(C) = w'Rw and Cov(z_k, C) = (Rw)_k, so the effective shares e_k = w_k (Rw)_k / w'Rw
    sum to 1 (they can be negative); with R = I, Var(C) = w'w and e_k equals the claimed
    share c_k = w_k^2 / w'w. Writing w'Rw = sum_j lambda_j (v_j'w)^2 over the eigenpairs
    of R, the PC1 share is lambda_1 (v_1'w)^2 / w'Rw.

    Duplicated metric. Let z_1 enter twice, as copies a and b (correlation 1) with
    weights a and b, beside independent metrics with S = sum_{k>=2} w_k^2. Then

        Var(C) = (a + b)^2 + S,     w'w = a^2 + b^2 + S,
        inflation = ((a + b)^2 + S) / (a^2 + b^2 + S) = 1 + 2ab / (a^2 + b^2 + S),
        e_a = a (a + b) / ((a + b)^2 + S),     c_a = a^2 / (a^2 + b^2 + S).

    Splitting z_1's weight w_1 in two (a = b = w_1 / 2) leaves C unchanged, so
    e_a + e_b = w_1^2 / (w_1^2 + S), z_1's real share, while the claimed share falls to
    (w_1^2 / 2) / (w_1^2 / 2 + S) and inflation is (w_1^2 + S) / (w_1^2 / 2 + S) > 1.
    Counting it twice at full weight (a = b = w_1) gives inflation
    (4 w_1^2 + S) / (2 w_1^2 + S) and e_a + e_b = 4 w_1^2 / (4 w_1^2 + S) against a
    claimed 2 w_1^2 / (2 w_1^2 + S). In both cases R's leading eigenpair is
    lambda = 2 along (copy a + copy b) / sqrt 2, so pc1_share = (a + b)^2 / ((a + b)^2 + S).
    """
    r = corr.loc[weights.index, weights.index].to_numpy(dtype=np.float64)
    w = weights.to_numpy(dtype=np.float64)
    rw = r @ w
    var_c, sum_w2 = float(w @ rw), float(w @ w)
    eigenvalues, vectors = np.linalg.eigh(r)
    shares = pd.DataFrame(
        {"claimed": w**2 / sum_w2, "effective": w * rw / var_c}, index=weights.index
    )
    return CompositeDiagnostics(
        inflation=var_c / sum_w2,
        shares=shares,
        pc1_share=float(eigenvalues[-1] * (vectors[:, -1] @ w) ** 2 / var_c),
    )
