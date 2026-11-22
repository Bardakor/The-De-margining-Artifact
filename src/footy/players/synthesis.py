"""Synthesis: one mixture over definitions, samples and the unmeasured (design §5, M8).

A draw is indexed by (s, b, u): ``s`` a specification of M5's full factorial,
``b`` one of the B sampling replicates of the inputs (player-season metrics
resampled by match, RAPM bootstrap replicate b), ``u`` one of U draws of the
unmeasured component. Specifications and replicates are finite populations,
equally weighted (or the specifications by ``spec_weights``, below); ``u`` is Gaussian
and integrated by Monte Carlo. A draw's score
for player i is ``y = x[s, b, i] + tau[s, b, i] * z[s, b, k, i]`` with ``x`` M5's score,
ranked within the segment (outfield, goalkeepers) as M5 does: 1 = best, ties share
the best rank, a player without a qualifying season (NaN score) ranks last.

Scale of the unmeasured component (Claude's derivation). In a position group g,
write true impact ``T = V + W`` with ``V`` the part the individual statistics
can see and ``W`` uncorrelated with it. M4's disattenuated share is
``sigma_g = 1 - R2_true = Var(W) / Var(T)``, so ``Var(W) = sigma_g / (1 - sigma_g) * Var(V)``.
Taking a box score as ``V`` up to scale gives
``tau^2 = sigma / (1 - sigma) * Var_g(x[s, b])``; a specification that is only
partly box uses ``omega * sigma_g`` in place of ``sigma_g``.

- ``sigma_g`` is capped at ``SHARE_CAP`` = 0.95 for this purpose only (the ratio
  diverges at 1). At the cap the unseen SD is 4.4 visible SDs and two players one
  visible SD apart keep their order with probability 0.57, so ranks inside the
  group are already near random and a higher cap changes little.
- ``omega[s, segment]`` (A13 (ii)) is the share of the specification's absolute
  weights on the box metrics (goals, non-penalty goals, assists), taken exactly
  from M5's weights on the original data (``stats_weight_share``) and held fixed
  across replicates. On-pitch goal difference counts with RAPM: as an on-pitch
  measure it already contains the unseen part. So ``box`` has ``omega = 1``,
  ``impact`` ``omega = 0`` (``tau = 0``: RAPM's noise is in b), and ``box_gd`` and
  ``hybrid`` lie in between per their weights (equal_z: 3/4 and 1/2).
- ``Var_g(x)``: per (s, b), the robust variance ``(1.4826 MAD)^2`` of the finite
  scores of the pool players of group g in the segment (equal to the variance
  under normality). Not the sample variance: a leader's lead would inflate it, so
  ``tau`` would grow with the lead it is testing and no lead, however large,
  could be decided (lead / tau is capped near ``sqrt(n_g / (2 ratio))``). A group
  with fewer than two finite scores takes the MAD of the segment's pooled
  deviations from group medians. If the pool is selected on the visible score,
  its spread understates the population's, so ``tau`` is conservative (too small).
- ``z``: standard normal, independent across players, draws and (s, b), and of
  the visible score.
- M4's share is held at its point estimate; its own uncertainty is not propagated.
- Optional on-pitch leakage term (post hoc robustness, ``onpitch_ratio``). RAPM's
  teammate leakage is stable across game resamples, so b does not contain it. Within
  group g write the stable part of a RAPM score ``S = own + leak``, own independent of
  leak, own's share of ``Var(S)`` the attribution factor ``A_g``. Ranking by own given
  S: ``E[own | S] = A_g S`` (rank-preserving) and ``Var(own | S) = A_g (1 - A_g) Var(S)``;
  in score units (divide by ``A_g``) the equivalent Gaussian draw has variance
  ``(1 - A_g) / A_g Var(S)``, with ``Var(S) = rel_g Var_g(x)`` (``rel_g`` RAPM's split-half
  reliability in g). It is added to the on-pitch share ``1 - omega`` of each
  specification: ``tau^2 = Var_g(x) [omega sigma_g / (1 - omega sigma_g) + (1 - omega)
  lambda_g]`` with ``lambda_g = (1 - A_g) / A_g rel_g`` the caller's ``onpitch_ratio``.
  On-pitch goal difference counts with the on-pitch part, as for omega; it is noisier
  than RAPM, so the term is conservative for box_gd and hybrid. Without it (default)
  impact specifications get ``tau = 0``.

- Optional specification weights (post hoc, A25 (iii); ``spec_weights``). The mixture
  over s is then the weighted one, ``w[s, segment]`` normalised to sum to 1 within a
  segment: every draw of specification s counts ``w_s`` in the rank histogram (P(#1),
  P(top-k), median, interval), in the pairwise probabilities and in the views (a
  family's mean rank is weighted within the family), and the decomposition takes its
  moments over s with these weights, ``Var_s(E[r|s]) = sum_s w_s (E[r|s] - E[r])^2``
  and the two ``E_s`` terms as weighted means, so the three terms still sum to the
  total. A specification of weight k is k equally weighted copies of itself.
  ``distinct_ranking_weights`` gives the weights under which each distinct ranking of
  the pool counts once. The median specification and its rank sets belong to the
  factorial, not to the mixture, and do not use the weights. Without weights (default)
  every specification counts the same and the arithmetic is unchanged.

Outputs.

- Per player: P(#1), P(top-3), P(top-10), median rank and 90% interval
  (inverted-CDF quantiles of the mixture's rank distribution).
- Pairwise: ``P(i above j) = P(r_i < r_j) + P(r_i = r_j) / 2`` averaged over (s, b),
  with u integrated exactly, ``Phi((x_i - x_j) / sqrt(tau_i^2 + tau_j^2))``
  (Rao-Blackwellised: the expectation of the Monte Carlo indicator, without its
  noise). Ties count half so the matrix is complementary.
- Nested decomposition of each player's rank:
  ``Var(r) = Var_s(E[r|s]) + E_s[Var_b(E[r|s,b])] + E_{s,b}[Var_u(r|s,b)]``
  (population variances over the equally weighted s and b). ``E_u[r_i | s, b] =
  1 + sum_j P(y_j > y_i)`` is computed exactly from the same Gaussian
  probabilities (ties count 0, as in min-ranking), ``E_u[r^2 | s, b]`` is the
  Monte Carlo mean over the U draws, ``Var_u = E_u[r^2] - E_u[r]^2`` (unbiased for
  any U >= 1) and the total is ``E[r^2] - E[r]^2`` from the same ingredients, so
  the three terms sum to it exactly. The specification and sampling terms carry
  no Monte Carlo noise (a plain Monte Carlo mean over U draws would leak about
  ``Var_u / U`` into the sampling term); the total differs from the plain variance
  of the draws only through the exact mean.
- Rank confidence sets at one fixed specification (``rank_confidence_sets``).
- Stats and impact views: each player's mean rank over the specifications of the
  box family (the stats view: individual statistics only) and of the impact
  family, over b and u. ``box_gd`` and ``hybrid`` mix the two kinds of measure, so
  they enter every other output but neither view.

The median specification (``median_specification``) is the specification whose
outfield ranking is nearest, in Spearman's footrule (L1 on ranks), to the
coordinatewise median of the outfield rankings across specifications, lowest
spec_id on ties. The coordinatewise median is the L1 centre of the rankings, so
this is the actual specification closest to the multiverse's centre; being one
specification, it gives a well-defined top 10 (H4) and a fixed estimator for the
rank confidence sets. Computed on the point estimate (the original data).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr
from scipy.stats import rankdata

from footy.players.multiverse import BOX, V1_LIKE_WEIGHTS, Multiverse, run_multiverse

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SEGMENTS = ("outfield", "goalkeeper")
STATS_FAMILY = "box"
IMPACT_FAMILY = "impact"
SHARE_CAP = 0.95
MAD_TO_SD = 1.4826
N_UNMEASURED = 4
LEVEL = 0.9
DECIDED = 0.9
_CHUNK = 2_000_000  # array elements per block of specifications


@dataclass(frozen=True)
class Synthesis:
    """``players``: one row per pool player (module doc). ``pairwise[segment]``:
    P(row player above column player), diagonal NaN. ``rank_sets``: confidence sets
    at ``fixed_spec``, one row per pool player."""

    players: pd.DataFrame
    pairwise: dict[str, pd.DataFrame]
    rank_sets: pd.DataFrame
    fixed_spec: int
    n_replicates: int
    n_unmeasured: int


def multiverse_runs(
    inputs: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
    stabilisation: Mapping[str, float],
    pool: Sequence[int],
    rng: np.random.Generator,
    *,
    expert_weights: Mapping[str, float] = V1_LIKE_WEIGHTS,
) -> Iterator[Multiverse]:
    """M5's result for each (player_season, impact_season) input, lazily.

    Every run gets a generator from the same seed (one draw from ``rng``), so the
    ridge CV folds are identical across inputs with the same players and the
    sampling layer holds only data noise. Pass the original data first::

        runs = multiverse_runs([original, *replicates], k, pool, rng)
        point = next(runs)
        omega = stats_weight_share(point.weights)
        result = synthesise(point.ranks, (r.ranks for r in runs), share, groups, rng,
                            omega=omega)
    """
    seed = int(rng.integers(2**63))
    for player_season, impact_season in inputs:
        yield run_multiverse(
            player_season,
            impact_season,
            stabilisation,
            pool,
            np.random.default_rng(seed),
            expert_weights=expert_weights,
        )


def stats_weight_share(weights: pd.DataFrame) -> pd.Series:
    """``omega`` per (spec_id, segment): the box metrics' share of the absolute weights.

    ``weights`` is ``Multiverse.weights`` of the original data (absolute weights sum
    to 1, so this is the sum of absolute weights on ``BOX``, A13 (ii)); on_pitch_gd
    counts with RAPM, not as a box metric. Written as a ratio so that box gives
    exactly 1 and impact exactly 0.
    """
    size = weights["weight"].abs()
    keys = [weights["spec_id"], weights["segment"].astype(str)]
    box = size.where(weights["metric"].isin(BOX), 0.0).groupby(keys).sum()
    return (box / size.groupby(keys).sum()).rename("omega")


def synthesise(
    point: pd.DataFrame,
    replicates: Iterable[pd.DataFrame],
    share: Mapping[str, float],
    groups: pd.Series,
    rng: np.random.Generator,
    *,
    omega: pd.Series,
    n_unmeasured: int = N_UNMEASURED,
    fixed_spec: int | None = None,
    onpitch_ratio: Mapping[str, float] | None = None,
    spec_weights: pd.Series | None = None,
) -> Synthesis:
    """The (s, b, u) mixture (module doc).

    ``point`` and each of ``replicates`` are M5 ranks frames (spec_id, family,
    player_id, segment, score) over the same specifications and pool; ``point``
    comes from the original data. ``share`` maps position group to M4's clipped
    unmeasured share; ``groups`` maps player id to position group; ``omega`` maps
    (spec_id, segment) to the box share of the weights (``stats_weight_share`` of
    M5's weights on the original data). ``fixed_spec`` defaults to the median
    specification. ``onpitch_ratio`` maps position group to ``lambda_g`` of the
    on-pitch leakage term (module doc); None leaves it out. ``spec_weights`` maps
    (spec_id, segment) to a non-negative weight, normalised within segment here (module
    doc); None weights every specification equally.
    """
    if n_unmeasured < 1:
        raise ValueError("n_unmeasured must be at least 1")
    specs = point.drop_duplicates("spec_id").sort_values("spec_id")
    spec_ids = specs["spec_id"].to_numpy(np.int64)
    family = specs["family"].astype(str).to_numpy()
    first = point.drop_duplicates("player_id")
    players = first["player_id"].to_numpy(np.int64)
    segment = first["segment"].astype(str).to_numpy()
    group = groups.reindex(players)
    if group.isna().any():
        raise ValueError(f"no position group for {players[group.isna().to_numpy()].tolist()}")
    missing = sorted(set(group) - set(share))
    if missing:
        raise ValueError(f"no unmeasured share for groups {missing}")
    sigma = np.array([share[g] for g in group], dtype=np.float64)
    if not ((sigma >= 0) & (sigma <= 1)).all():
        raise ValueError("unmeasured shares must lie in [0, 1]")
    om = omega.unstack().reindex(index=spec_ids, columns=segment).to_numpy(np.float64)
    if not ((om >= 0) & (om <= 1)).all():
        raise ValueError("omega must give every (spec_id, segment) a share in [0, 1]")
    eff = om * np.minimum(sigma, SHARE_CAP)[None, :]
    ratio = eff / (1.0 - eff)
    if onpitch_ratio is not None:
        missing = sorted(set(group) - set(onpitch_ratio))
        if missing:
            raise ValueError(f"no on-pitch ratio for groups {missing}")
        lam = np.array([onpitch_ratio[g] for g in group], dtype=np.float64)
        if (lam < 0).any():
            raise ValueError("on-pitch ratios must be non-negative")
        ratio = ratio + (1.0 - om) * lam[None, :]

    x_point = _score_table(point, spec_ids, players)
    if fixed_spec is None:
        fixed_spec = median_specification(point[point["segment"] == "outfield"])
    fixed = int(np.flatnonzero(spec_ids == fixed_spec)[0])
    parts = {s: np.flatnonzero(segment == s) for s in SEGMENTS if (segment == s).any()}
    weight: dict[str, FloatArray | None] = dict.fromkeys(parts)
    if spec_weights is not None:
        table = spec_weights.unstack().reindex(index=spec_ids)
        for s in parts:
            w = table[s].to_numpy(np.float64)
            if not (np.isfinite(w).all() and (w >= 0).all() and w.sum() > 0):
                raise ValueError(f"specification weights of the {s} segment are not usable")
            weight[s] = w / w.sum()
    acc = {
        s: _Accumulator(spec_ids.size, idx.size, pd.factorize(group.iloc[idx])[0], weight[s])
        for s, idx in parts.items()
    }
    boot_fixed = []
    for frame in replicates:
        x = _score_table(frame, spec_ids, players)
        for s, idx in parts.items():
            acc[s].add(x[:, idx], ratio[:, idx], rng, n_unmeasured)
        boot_fixed.append(x[fixed])
    n_rep = len(boot_fixed)
    if n_rep < 2:
        raise ValueError("need at least two replicates")

    stats_specs, impact_specs = family == STATS_FAMILY, family == IMPACT_FAMILY
    rows, pairwise, sets = [], {}, []
    boot = np.array(boot_fixed)
    for s, idx in parts.items():
        a = acc[s]
        summary = a.summary(n_rep)
        mean_rank = a.m1 / n_rep
        summary.insert(0, "segment", s)
        summary.insert(0, "player_id", players[idx])
        summary["stats_view"] = _family_mean(mean_rank, stats_specs, a.w)
        summary["impact_view"] = _family_mean(mean_rank, impact_specs, a.w)
        rows.append(summary)
        pairwise[s] = pd.DataFrame(
            a.pairwise(n_rep), index=pd.Index(players[idx], name="player_id"), columns=players[idx]
        )
        rs = rank_confidence_sets(x_point[fixed, idx], boot[:, idx])
        rs.insert(0, "segment", s)
        rs.insert(0, "player_id", players[idx])
        sets.append(rs)
    return Synthesis(
        players=pd.concat(rows, ignore_index=True),
        pairwise=pairwise,
        rank_sets=pd.concat(sets, ignore_index=True),
        fixed_spec=int(fixed_spec),
        n_replicates=n_rep,
        n_unmeasured=n_unmeasured,
    )


def distinct_ranking_weights(point_ranks: pd.DataFrame) -> pd.Series:
    """Weights per (spec_id, segment) under which each distinct ranking counts once.

    ``point_ranks`` is an M5 ranks frame of the original data (spec_id, player_id,
    segment, rank). Within a segment, the specifications that rank the pool
    identically share one unit of weight (A25 (iii)): a specification gets 1 / (number
    of specifications with its ranking), so a segment's weights sum to its number of
    distinct rankings.
    """
    parts = []
    for segment, part in point_ranks.groupby("segment", observed=True):
        table = part.pivot(index="spec_id", columns="player_id", values="rank")
        _, inverse, counts = np.unique(
            table.to_numpy(), axis=0, return_inverse=True, return_counts=True
        )
        index = pd.MultiIndex.from_product(
            [table.index, [str(segment)]], names=["spec_id", "segment"]
        )
        parts.append(pd.Series(1.0 / counts[inverse.ravel()], index=index))
    return pd.concat(parts).rename("weight")


def median_specification(ranks: pd.DataFrame) -> int:
    """spec_id nearest in footrule to the coordinatewise median ranking (module doc)."""
    table = ranks.pivot(index="spec_id", columns="player_id", values="rank").sort_index()
    footrule = (table - table.median(axis=0)).abs().sum(axis=1)
    return int(footrule.idxmin())


def rank_confidence_sets(
    point: FloatArray, boot: FloatArray, *, level: float = LEVEL
) -> pd.DataFrame:
    """Rank confidence sets from the bootstrap of pairwise score differences.

    Mogstad, Romano, Shaikh and Wilhelm (2024), single-step, two-sided. ``point``
    (n,) are the scores on the original data, ``boot`` (B, n) on the replicates;
    higher is better. With ``d_jk = x_j - x_k``, ``se_jk`` its bootstrap SD and
    ``t*_jk = |d*_jk - d_jk| / se_jk``, the critical value ``c`` is the ``level``
    quantile over replicates of ``max_k t*_jk`` (marginal set of player j) or of
    ``max_{j,k} t*_jk`` (simultaneous sets). Player k is above j when
    ``d_jk + c se_jk < 0`` and below when ``d_jk - c se_jk > 0``; the set is
    ``[1 + #above, n - #below]``. Players with a NaN score in the point estimate or
    any replicate get no set, and count as possibly above or below every other.
    ``naive_lo``/``naive_hi`` is the percentile interval of the bootstrap ranks.
    """
    n = point.size
    ok = np.isfinite(point) & np.isfinite(boot).all(axis=0)
    x, xb = point[ok], boot[:, ok]
    d = x[:, None] - x[None, :]
    db = xb[:, :, None] - xb[:, None, :]
    se = db.std(axis=0, ddof=1)
    dev = np.abs(db - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, dev / se, np.where(dev > 0, np.inf, 0.0))
    t_j = t.max(axis=2)  # the diagonal is 0
    out = pd.DataFrame({"point_rank": _rank_min(np.where(np.isfinite(point), point, -np.inf))})
    ranks = _rank_min(np.where(np.isfinite(boot), boot, -np.inf))
    for name, q in (("naive_lo", (1 - level) / 2), ("naive_hi", (1 + level) / 2)):
        out[name] = np.quantile(ranks, q, axis=0, method="inverted_cdf").astype(np.int64)
    critical = {
        "marginal": np.quantile(t_j, level, axis=0, method="inverted_cdf")[:, None],
        "simultaneous": np.quantile(t_j.max(axis=1), level, method="inverted_cdf"),
    }
    for name, c in critical.items():
        with np.errstate(invalid="ignore"):
            above = (d + c * se < 0).sum(axis=1)
            below = (d - c * se > 0).sum(axis=1)
        lo, hi = np.full(n, np.nan), np.full(n, np.nan)
        lo[ok], hi[ok] = 1 + above, n - below
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out


def kendalls_w(rankings: pd.DataFrame) -> float:
    """Kendall's W with the tie correction; one column per view, one row per player.

    Lower is better in every column (each column is turned into mid-ranks, so any
    scale with that orientation works); rows with a missing value are dropped.
    ``W = 12 S / (m^2 (n^3 - n) - m sum_j T_j)``, ``T_j = sum (t^3 - t)`` over tie
    groups of view j. Under independent random views E[W] = 1/m, not 0; the mean
    pairwise Spearman correlation ``(m W - 1) / (m - 1)`` has expectation 0.
    """
    r = rankings.dropna().to_numpy(np.float64)
    n, m = r.shape
    totals = rankdata(r, axis=0).sum(axis=1)
    s = float(((totals - totals.mean()) ** 2).sum())
    ties = sum(
        float((c**3 - c).sum()) for c in (np.unique(col, return_counts=True)[1] for col in r.T)
    )
    return 12.0 * s / (m**2 * (n**3 - n) - m * ties)


def view_agreement(
    players: pd.DataFrame, recognition: pd.Series, *, segment: str = "outfield"
) -> float:
    """Kendall's W of the stats view, the impact view and recognition.

    ``players`` is ``Synthesis.players``; ``recognition`` maps player id to Ballon
    d'Or points aggregated by the caller (higher is better). Players missing from
    ``recognition`` are dropped; give unranked pool players 0 to keep them.
    """
    view = players[players["segment"] == segment].set_index("player_id")
    table = pd.DataFrame(
        {
            "stats": view["stats_view"],
            "impact": view["impact_view"],
            "recognition": -recognition.reindex(view.index),
        }
    )
    return kendalls_w(table)


def verdicts(result: Synthesis, indices: pd.DataFrame) -> pd.DataFrame:
    """H1, H2, H4 and H5 with the numbers behind them (outfield).

    ``indices`` is ``multiverse.variance_indices`` of the point-estimate ranks.
    H1: share of pool pairs with P >= 0.9 or <= 0.1 is below 0.5. H2: the highest
    P(#1) is below 0.5. H4: over the players ranked 1-10 (ties included) by the
    median specification on the original data, the mean of specification /
    sampling variance exceeds 1 (players with both components 0 are skipped).
    H5: aggregation has the largest total-order index for the player with the
    highest P(#1).
    """
    view = result.players[result.players["segment"] == "outfield"].set_index("player_id")
    p = result.pairwise["outfield"].to_numpy()
    p = p[np.triu_indices(len(p), 1)]
    decided = float(((p >= DECIDED) | (1.0 - p >= DECIDED)).mean())
    leader = int(view["p_first"].idxmax())
    p_max = float(view["p_first"].max())
    sets = result.rank_sets[result.rank_sets["segment"] == "outfield"].set_index("player_id")
    top10 = sets.index[sets["point_rank"] <= 10]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (view.loc[top10, "var_specification"] / view.loc[top10, "var_sampling"]).to_numpy()
    ratio = ratio[~np.isnan(ratio)]
    mean_ratio = float(ratio.mean()) if ratio.size else np.nan
    total = indices[indices["player_id"] == leader].set_index("dimension")["total_order"]
    largest = str(total.idxmax()) if total.notna().any() else "undefined"
    rows = [
        ("H1", decided, "< 0.5", decided < 0.5, f"{p.size} outfield pairs"),
        ("H2", p_max, "< 0.5", p_max < 0.5, f"highest P(#1): player {leader}"),
        (
            "H4",
            mean_ratio,
            "> 1",
            mean_ratio > 1,
            f"{ratio.size} players ranked <= 10 by spec {result.fixed_spec}",
        ),
        (
            "H5",
            float(total.get("aggregation", np.nan)),
            "largest total-order index",
            largest == "aggregation",
            f"player {leader}: largest is {largest} ({total.max():.3f})",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["hypothesis", "statistic", "threshold", "supported", "detail"]
    )


class _Accumulator:
    """Running sums over (b, u) for one segment."""

    def __init__(self, n_specs: int, n: int, codes: IntArray, w: FloatArray | None) -> None:
        self.codes = codes
        self.w = w  # specification weights summing to 1; None: equal
        # player x rank - 1: draws, or their weights
        self.hist: IntArray | FloatArray = (
            np.zeros((n, n), dtype=np.int64) if w is None else np.zeros((n, n))
        )
        self.m1 = np.zeros((n_specs, n))  # sum_b E_u[r | s, b]
        self.m2 = np.zeros((n_specs, n))  # sum_b E_u[r | s, b]^2
        self.vu = np.zeros((n_specs, n))  # sum_b Var_u(r | s, b)
        self.r1, self.r2 = np.zeros(n), np.zeros(n)  # sum_{s,b} (w_s) E_u[r], E_u[r^2]
        self.iu, self.ju = np.triu_indices(n, 1)
        self.inc_i, self.inc_j = np.eye(n)[self.iu], np.eye(n)[self.ju]  # pair -> player
        self.pairs = np.zeros(self.iu.size)  # sum_{s,b} (w_s) P(i above j | s, b), i < j

    def add(self, x: FloatArray, ratio: FloatArray, rng: np.random.Generator, n_u: int) -> None:
        n_specs, n = x.shape
        tau2 = ratio * _visible_variance(x, self.codes)
        tau = np.sqrt(tau2)
        x = np.where(np.isnan(x), -np.inf, x)
        offset = n * np.arange(n)
        step = max(1, _CHUNK // (n * max(n_u, (n - 1) // 2, 1)))
        for lo in range(0, n_specs, step):
            sl = slice(lo, lo + step)
            # tau = 0 gives d / 0 = +-inf (Phi = 1 or 0); a tie gives NaN
            with np.errstate(invalid="ignore", divide="ignore"):
                d = x[sl][:, self.iu] - x[sl][:, self.ju]
                p = ndtr(d / np.sqrt(tau2[sl][:, self.iu] + tau2[sl][:, self.ju]))
            tie = np.isnan(p)
            above, below = np.where(tie, 0.0, p), np.where(tie, 0.0, 1.0 - p)
            mean = 1.0 + below @ self.inc_i + above @ self.inc_j
            z = rng.standard_normal((mean.shape[0], n_u, n))
            r = _rank_min(x[sl, None, :] + tau[sl, None, :] * z)
            cell = (r - 1 + offset).ravel()
            square = (r.astype(np.float64) ** 2).mean(axis=1)
            self.m1[sl] += mean
            self.m2[sl] += mean**2
            self.vu[sl] += square - mean**2
            if self.w is None:
                self.pairs += np.where(tie, 0.5, p).sum(axis=0)
                self.hist += np.bincount(cell, minlength=n * n).reshape(n, n)
                self.r1 += mean.sum(axis=0)
                self.r2 += square.sum(axis=0)
            else:
                w = self.w[sl]
                self.pairs += w @ np.where(tie, 0.5, p)
                draws = np.broadcast_to(w[:, None, None], r.shape).ravel()
                counted = np.bincount(cell, weights=draws, minlength=n * n).reshape(n, n)
                self.hist = self.hist + counted
                self.r1 += w @ mean
                self.r2 += w @ square

    def summary(self, n_rep: int) -> pd.DataFrame:
        n = self.hist.shape[0]
        cdf = np.cumsum(self.hist, axis=1) / self.hist.sum(axis=1, keepdims=True)
        n_draws = self.m1.shape[0] * n_rep if self.w is None else n_rep
        total = self.r2 / n_draws - (self.r1 / n_draws) ** 2
        e_s = self.m1 / n_rep
        if self.w is None:
            parts = (
                e_s.var(axis=0),
                (self.m2 / n_rep - e_s**2).mean(axis=0),
                (self.vu / n_rep).mean(axis=0),
            )
        else:
            parts = (
                self.w @ (e_s - self.w @ e_s) ** 2,
                self.w @ (self.m2 / n_rep - e_s**2),
                self.w @ (self.vu / n_rep),
            )
        out = pd.DataFrame(
            {
                "p_first": cdf[:, 0],
                "p_top3": cdf[:, min(3, n) - 1],
                "p_top10": cdf[:, min(10, n) - 1],
                "median_rank": _inverse_cdf(cdf, 0.5),
                "rank_lo": _inverse_cdf(cdf, (1 - LEVEL) / 2),
                "rank_hi": _inverse_cdf(cdf, (1 + LEVEL) / 2),
                "var_total": total,
                "var_specification": parts[0],
                "var_sampling": parts[1],
                "var_unmeasured": parts[2],
            }
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            for part in ("specification", "sampling", "unmeasured"):
                out[f"share_{part}"] = out[f"var_{part}"] / total
        return out

    def pairwise(self, n_rep: int) -> FloatArray:
        n = self.hist.shape[0]
        n_specs = self.m1.shape[0] if self.w is None else 1
        p = np.full((n, n), np.nan)
        p[self.iu, self.ju] = self.pairs / (n_specs * n_rep)
        p[self.ju, self.iu] = 1.0 - p[self.iu, self.ju]
        return p


def _family_mean(
    mean_rank: FloatArray, specs: NDArray[np.bool_], w: FloatArray | None
) -> FloatArray | float:
    """A view: each player's mean rank over the family's specifications (weighted by w)."""
    if not specs.any():
        return np.nan
    if w is None:
        return np.asarray(mean_rank[specs].mean(axis=0), dtype=np.float64)
    return np.asarray(w[specs] @ mean_rank[specs] / w[specs].sum(), dtype=np.float64)


def _visible_variance(x: FloatArray, codes: IntArray) -> FloatArray:
    """(specs, players): robust variance of the finite scores of each player's group;
    the pooled one (deviations from group medians) where the group has fewer than two."""
    mad = np.full(x.shape, np.nan)
    dev = np.full(x.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        for g in np.unique(codes):
            cols = codes == g
            a = np.abs(x[:, cols] - np.nanmedian(x[:, cols], axis=1, keepdims=True))
            enough = np.isfinite(x[:, cols]).sum(axis=1, keepdims=True) >= 2
            mad[:, cols] = np.where(enough, np.nanmedian(a, axis=1, keepdims=True), np.nan)
            dev[:, cols] = np.where(enough, a, np.nan)
        mad = np.where(np.isnan(mad), np.nanmedian(dev, axis=1, keepdims=True), mad)
    return np.asarray(np.nan_to_num((MAD_TO_SD * mad) ** 2), dtype=np.float64)


def _score_table(frame: pd.DataFrame, spec_ids: IntArray, players: IntArray) -> FloatArray:
    si = pd.Index(spec_ids).get_indexer(pd.Index(frame["spec_id"]))
    pi = pd.Index(players).get_indexer(pd.Index(frame["player_id"]))
    if len(frame) != spec_ids.size * players.size or (si < 0).any() or (pi < 0).any():
        raise ValueError("a replicate's specifications or pool differ from the point estimate's")
    x = np.full((spec_ids.size, players.size), np.nan)
    x[si, pi] = frame["score"].to_numpy(np.float64)
    return x


def _rank_min(y: FloatArray) -> IntArray:
    """Ranks along the last axis: 1 = highest, ties share the best rank, -inf last."""
    key = -y
    order = np.argsort(key, axis=-1)
    ordered = np.take_along_axis(key, order, axis=-1)
    n = y.shape[-1]
    start = np.ones(ordered.shape, dtype=bool)
    start[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    first = np.maximum.accumulate(np.where(start, np.arange(n), 0), axis=-1)
    ranks = np.empty(order.shape, dtype=np.int64)
    np.put_along_axis(ranks, order, first + 1, axis=-1)
    return ranks


def _inverse_cdf(cdf: FloatArray, q: float) -> IntArray:
    """Smallest rank whose cumulative probability reaches ``q``, per row."""
    return np.asarray(np.argmax(cdf >= q - 1e-12, axis=1) + 1, dtype=np.int64)
