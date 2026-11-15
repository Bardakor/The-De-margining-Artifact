"""Recovery study on the real schedule (design §5, M9; pre-registration A9).

Effects are planted on a schedule that is given, not generated: the real
stints, who was on the pitch in each, and therefore every player's real minutes
per game. Only goals and box scores are drawn. Model, stated so tests and the
paper can rely on it:

- Player effects: ``offence ~ N(0, offence_sd[g]^2)`` and
  ``defence ~ N(0, defence_sd[g]^2)``, independent, ``g`` the player's
  position group; or, if ``simulate_on_schedule`` is given ``effects``, those
  values exactly (no draw; ``offence_sd`` and ``defence_sd`` still set
  ``sd_t`` below). ``component`` names the effect the box score can see:
  ``t = offence``, ``defence`` or ``offence + defence`` (``total``, the
  default), with planted SD ``sd_t[g]``.
- Box-score latent: ``z = sqrt(m[g]) * t / sd_t[g] + sqrt(1 - m[g]) * e``,
  ``e ~ N(0, 1)``, ``m = measurable_share``. In the population of a group a
  share ``m[g]`` of the variance of ``t`` is explained by ``z``;
  ``truth.seen = sqrt(m) * sd_t * z`` is that measurable part, as
  ``synthetic.py``'s ``seen``. (In ``synthetic.py`` ``z`` is a summand of
  offence; both give the same joint law of ``t`` and ``z``. This form also
  covers ``total``.) If ``sd_t[g] = 0`` then ``z = sqrt(1 - m) * e``.
- Shot rate per 90 ``max(SHOT_RATE_FLOOR, shot_rate_mean[g] + shot_rate_sd[g] * z)``;
  per player-match, shots are Poisson with that rate times
  ``exp(shock) * minutes / 90`` and goals Binomial(shots, ``conversion[g]``).
  Box score and stint goals are independent given the effects, apart from
  the match shock.
- Stint goals, as ``synthetic.py``: a side's rate per 90 is
  ``base_rate[k] +- home_advantage[k] / 2 + sum(offence on the side) -
  sum(defence of the opponents) + shock``, floored at ``RATE_FLOOR``, Poisson
  with that rate times minutes / 90. ``k`` is the stint's label in the
  ``group`` column (competition-season); RAPM's intercept is
  ``base_rate - home_advantage / 2``. ``shock ~ N(0, match_shock_sd^2)`` per
  team-match, as ``synthetic.py``'s ``match_shock_sd`` (none by default).
- No covariate effect is planted. If ``covariate`` is given, M3 still fits it
  so it runs as on the real data.

Assumptions of the M3/M4 runs: one ``min_minutes`` both pools players into
RAPM's replacement column and keeps them out of M4 (they are the same players);
M4 sees per-90 shots and goals only; the attribution factor is passed in
(computed once for the real estimate: it depends on the schedule and ``lam``
only, so it is the same in every simulation).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from footy.players.construct_gap import (
    H3_HIGH,
    H3_LOW,
    Estimator,
    Splits,
    position_contrast,
    split_stats,
)
from footy.players.rapm import build_design, fit_rapm, split_halves
from footy.players.synthetic import RATE_FLOOR, SHOT_RATE_FLOOR

STATS = ("shots", "goals")

FloatArray = NDArray[np.float64]
PerGroup = float | Mapping[Any, float]
Component = Literal["offence", "defence", "total"]


@dataclass(frozen=True)
class SimParams:
    """Generator parameters (module docstring). ``offence_sd`` to
    ``conversion`` are a float or a mapping position group -> float;
    ``base_rate`` and ``home_advantage`` a float or a mapping from the labels
    of the stints' ``group`` column."""

    offence_sd: PerGroup = 0.05
    defence_sd: PerGroup = 0.05
    measurable_share: PerGroup = 0.5
    component: Component = "total"
    shot_rate_mean: PerGroup = 1.5
    shot_rate_sd: PerGroup = 0.8
    conversion: PerGroup = 0.11
    base_rate: PerGroup = 1.35
    home_advantage: PerGroup = 0.3
    match_shock_sd: float = 0.0


@dataclass(frozen=True)
class Simulated:
    """``stints`` is the input with new home_goals / away_goals; ``stint_players``
    is the input; ``player_match`` has game_id, player_id, minutes, shots, goals
    (one row per player-match with minutes > 0); ``truth`` has player_id,
    position_group, offence, defence, total, seen, shot_rate."""

    stints: pd.DataFrame
    stint_players: pd.DataFrame
    player_match: pd.DataFrame
    truth: pd.DataFrame


@dataclass(frozen=True)
class ContrastBootstrap:
    """``sims``: one row per simulation, M4's output (``diff``,
    ``diff_clipped`` and its own 90% player-bootstrap interval) plus
    ``error = diff - planted`` and ``error_clipped``. ``summary``: rows raw and
    clipped; the real estimate, the planted contrast, the error's mean and SD,
    the parametric interval ``ci_low`` / ``ci_high``, and two coverages over
    the simulations (``position_contrast`` docstring for the M4 interval)."""

    sims: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class RankRecovery:
    """``sims``: one row per simulation (planted top-two gap, whether RAPM total
    and the box composite put the planted #1 first, their mean absolute rank
    errors). ``by_gap``: the same averaged in equal-count bins of the gap."""

    sims: pd.DataFrame
    by_gap: pd.DataFrame


def simulate_on_schedule(
    stints: pd.DataFrame,
    stint_players: pd.DataFrame,
    positions: pd.Series,
    rng: np.random.Generator,
    params: SimParams,
    *,
    group: str | None = None,
    effects: pd.DataFrame | None = None,
) -> Simulated:
    """Plant effects and draw goals and box scores on a given schedule.

    ``stints`` and ``stint_players`` are in the M3 schema (``rapm.build_design``);
    ``positions`` maps player id -> position group and must cover every player
    when a position parameter is a mapping. ``group`` names the stints column
    that keys ``base_rate`` and ``home_advantage``. ``effects`` (player_id,
    offence, defence), covering every player, replaces the drawn effects;
    everything else is drawn as without it.
    """
    st = stints.reset_index(drop=True)
    n = len(st)
    row = pd.Index(st["stint_id"]).get_indexer(stint_players["stint_id"])
    home = (stint_players["side"] == "home").to_numpy()
    codes, players = pd.factorize(stint_players["player_id"], sort=True)
    pos = positions.reindex(players).to_numpy()
    n_p = len(players)

    if effects is None:
        off = _per(params.offence_sd, pos) * rng.standard_normal(n_p)
        dfn = _per(params.defence_sd, pos) * rng.standard_normal(n_p)
    else:
        given = effects.set_index("player_id").reindex(players)[["offence", "defence"]]
        if given.isna().to_numpy().any():
            raise ValueError("effects must cover every player of stint_players")
        off, dfn = (given[c].to_numpy(dtype=np.float64) for c in ("offence", "defence"))
    t = {"offence": off, "defence": dfn, "total": off + dfn}[params.component]
    sd_t, m = _component_sd(params, pos), _per(params.measurable_share, pos)
    std = np.divide(t, sd_t, out=np.zeros(n_p), where=sd_t > 0)
    z = np.sqrt(m) * std + np.sqrt(1 - m) * rng.standard_normal(n_p)
    shot_rate = np.maximum(
        SHOT_RATE_FLOOR, _per(params.shot_rate_mean, pos) + _per(params.shot_rate_sd, pos) * z
    )

    game, games = pd.factorize(st["game_id"])
    shock = np.zeros((len(games), 2))
    if params.match_shock_sd > 0:
        shock = rng.normal(0.0, params.match_shock_sd, size=shock.shape)
    labels = st[group].to_numpy() if group is not None else np.full(n, "all", dtype=object)
    base, adv = _per(params.base_rate, labels), _per(params.home_advantage, labels)
    minutes = st["minutes"].to_numpy(dtype=np.float64)

    def on_side(values: FloatArray, mask: NDArray[np.bool_]) -> FloatArray:
        return np.bincount(row[mask], weights=values[codes[mask]], minlength=n)

    goals = []
    for k, (mask, sign) in enumerate(((home, 1.0), (~home, -1.0))):
        rate = base + sign * adv / 2 + on_side(off, mask) - on_side(dfn, ~mask) + shock[game, k]
        goals.append(rng.poisson(np.maximum(RATE_FLOOR, rate) * minutes / 90))

    key = game[row] * n_p + codes
    pm_key, inv = np.unique(key, return_inverse=True)
    pm_minutes = np.bincount(inv, weights=minutes[row])
    pm_away = np.zeros(len(pm_key), dtype=np.int64)
    pm_away[inv] = ~home
    pm_game, pm_player = np.divmod(pm_key, n_p)
    expected = shot_rate[pm_player] * np.exp(shock[pm_game, pm_away]) * pm_minutes / 90
    shots = rng.poisson(expected)
    scored = rng.binomial(shots, _per(params.conversion, pos)[pm_player])
    pm = pd.DataFrame(
        {
            "game_id": games[pm_game],
            "player_id": players[pm_player],
            "minutes": pm_minutes,
            "shots": shots,
            "goals": scored,
        }
    )
    truth = pd.DataFrame(
        {
            "player_id": players,
            "position_group": pos,
            "offence": off,
            "defence": dfn,
            "total": off + dfn,
            "seen": np.sqrt(m) * sd_t * z,
            "shot_rate": shot_rate,
        }
    )
    return Simulated(
        stints=stints.assign(home_goals=goals[0], away_goals=goals[1]),
        stint_players=stint_players,
        player_match=pm[pm["minutes"] > 0].reset_index(drop=True),
        truth=truth,
    )


def parametric_bootstrap_contrast(
    stints: pd.DataFrame,
    stint_players: pd.DataFrame,
    positions: pd.Series,
    real: pd.Series,
    attribution: Mapping[str, float],
    params: SimParams,
    rng: np.random.Generator,
    *,
    lam: float,
    min_minutes: float,
    n_sims: int = 200,
    n_repeats: int = 5,
    group: str | None = None,
    covariate: str | None = None,
    estimator: Estimator = "cross",
    high: Sequence[str] = H3_HIGH,
    low: Sequence[str] = H3_LOW,
    n_boot: int = 200,
    level: float = 0.9,
) -> ContrastBootstrap:
    """A9's decision interval for H3.

    ``params`` holds the effect SDs and measurable shares estimated on the real
    data, ``real`` is ``construct_gap.position_contrast`` on the real data and
    ``attribution`` the factors it used. Each of ``n_sims`` simulations draws
    goals and box scores on the real schedule and runs M3 (``fit_rapm`` at
    ``lam``, ``split_halves`` with ``n_repeats`` splits) and M4
    (``position_contrast`` on ``params.component``) as on the real data.

    The planted contrast is fixed: per pool, ``1 - sum(m * sd_t^2) / sum(sd_t^2)``
    over the players entering M4 (players with at least ``min_minutes``).

    Interval: basic (pivotal) bootstrap, ``[T - q_hi, T - q_lo]`` with ``T``
    the real estimate and ``q`` the quantiles of ``estimate - planted``. M4 is
    biased on finite data (ridge leakage, the reliability corrections,
    clipping); the basic interval treats the error as a pivot and so removes
    the bias, where a percentile interval of the simulated estimates would add
    it a second time (it centres on ``planted + bias`` with ``T`` already
    biased). A studentized interval would need an inner bootstrap per
    simulation. Fewer splits than the real estimate make the estimates
    noisier and the interval conservative.

    Coverages: ``coverage_m4`` is the share of simulations whose own M4
    interval (player bootstrap, ``n_boot`` draws) covers the planted contrast;
    ``coverage_loo`` is the same for the parametric interval built around each
    simulation's estimate from the other simulations' errors (close to
    ``level`` by construction; its real test is across independent leagues).
    """
    entering = _played(stints, stint_players).loc[lambda s: s >= min_minutes].index
    pos = positions.reindex(entering).to_numpy()
    planted = _planted_share(params, pos, high) - _planted_share(params, pos, low)

    rows = []
    for _ in range(n_sims):
        sim = simulate_on_schedule(stints, stint_players, positions, rng, params, group=group)
        d = build_design(
            sim.stints, stint_players, group=group, covariate=covariate, min_minutes=min_minutes
        )
        fit = fit_rapm(d, lam)
        halves = split_halves(d, lam, rng, n_repeats=n_repeats)
        c = params.component
        splits = Splits(halves.a[c], halves.b[c], split_stats(sim.player_match, halves, STATS))
        contrast = position_contrast(
            fit.players[c], _per90(sim.player_match), splits, attribution, positions, rng,
            high=high, low=low, estimator=estimator, n_boot=n_boot, level=level,
        )  # fmt: skip
        rows.append(contrast)
    sims = pd.DataFrame(rows).reset_index(drop=True)
    sims["error"] = sims["diff"] - planted
    sims["error_clipped"] = sims["diff_clipped"] - planted

    summary = {}
    for kind, sfx in (("raw", ""), ("clipped", "_clipped")):
        err, t = sims[f"error{sfx}"].to_numpy(), float(real[f"diff{sfx}"])
        q_lo, q_hi = _quantiles(err, level)
        loo = [_quantiles(np.delete(err, s), level) for s in range(len(err))]
        covered = [lo <= e <= hi for (lo, hi), e in zip(loo, err, strict=True)]
        m4 = (sims[f"ci_low{sfx}"] <= planted) & (planted <= sims[f"ci_high{sfx}"])
        summary[kind] = {
            "estimate": t,
            "planted": planted,
            "mean_error": float(err.mean()),
            "sd_error": float(err.std(ddof=1)),
            "ci_low": t - q_hi,
            "ci_high": t - q_lo,
            "coverage_m4": float(m4.mean()),
            "coverage_loo": float(np.mean(covered)),
        }
    return ContrastBootstrap(sims=sims, summary=pd.DataFrame.from_dict(summary, orient="index"))


def rank_recovery(
    stints: pd.DataFrame,
    stint_players: pd.DataFrame,
    positions: pd.Series,
    pool: Sequence[int],
    params: SimParams,
    rng: np.random.Generator,
    *,
    lam: float,
    min_minutes: float = 0.0,
    n_sims: int = 200,
    n_bins: int = 4,
    group: str | None = None,
    covariate: str | None = None,
) -> RankRecovery:
    """How often the planted #1 of ``pool`` is ranked first, by RAPM total and
    by a box-score composite, against the planted gap between the top two.

    Every simulation plants fresh effects for everyone (``params``), pool
    included, so the gap varies from one simulation to the next. Ranks are
    within the pool, by planted ``total`` (ties broken at random, so with all
    effects equal the planted #1 is a uniform draw). Box composite: the mean of
    the per-90 shots and goals, each z-scored within position group over the
    players with at least ``min_minutes`` (the multiverse's equal-z, within-
    group specification). Bins have equal counts of simulations.
    """
    ids = pd.Index(pool)
    played = _played(stints, stint_players)
    eligible = played.index[played >= min_minutes]
    if not ids.isin(eligible).all():
        raise ValueError("every pool player needs at least min_minutes on the schedule")
    rows = []
    for _ in range(n_sims):
        sim = simulate_on_schedule(stints, stint_players, positions, rng, params, group=group)
        d = build_design(
            sim.stints, stint_players, group=group, covariate=covariate, min_minutes=min_minutes
        )
        rapm = fit_rapm(d, lam).players.loc[ids, "total"].to_numpy()
        stats = _per90(sim.player_match).loc[eligible]
        g = positions.reindex(eligible)
        z = (stats - stats.groupby(g).transform("mean")) / stats.groupby(g).transform("std")
        box = z.mean(axis=1).loc[ids].to_numpy()
        planted = sim.truth.set_index("player_id").loc[ids, "total"].to_numpy()
        true_rank = _ranks(planted, rng)
        first = int(np.argmin(true_rank))
        top_two = np.sort(planted)[::-1][:2]
        row = {"gap": float(top_two[0] - top_two[1])}
        for name, est in (("rapm", rapm), ("box", box)):
            rank = _ranks(est, rng)
            row[f"top_{name}"] = float(rank[first] == 1)
            row[f"mae_{name}"] = float(np.abs(rank - true_rank).mean())
        rows.append(row)
    sims = pd.DataFrame(rows)
    bins = pd.qcut(sims["gap"].rank(method="first"), n_bins, labels=False)
    by_gap = sims.groupby(bins).agg(
        n=("gap", "size"),
        gap_low=("gap", "min"),
        gap_high=("gap", "max"),
        gap_mean=("gap", "mean"),
        p_top_rapm=("top_rapm", "mean"),
        p_top_box=("top_box", "mean"),
        mae_rapm=("mae_rapm", "mean"),
        mae_box=("mae_box", "mean"),
    )
    by_gap.index.name = "gap_bin"
    return RankRecovery(sims=sims, by_gap=by_gap)


def _per(value: PerGroup, labels: NDArray[Any]) -> FloatArray:
    """A float, or a mapping looked up for every label (KeyError if missing)."""
    if isinstance(value, Mapping):
        return np.array([value[g] for g in labels], dtype=np.float64)
    return np.full(len(labels), float(value))


def _component_sd(params: SimParams, pos: NDArray[Any]) -> FloatArray:
    sd_o, sd_d = _per(params.offence_sd, pos), _per(params.defence_sd, pos)
    return {"offence": sd_o, "defence": sd_d, "total": np.hypot(sd_o, sd_d)}[params.component]


def _planted_share(params: SimParams, pos: NDArray[Any], pool: Sequence[str]) -> float:
    """Population unmeasured share of a pool: 1 - sum(m * sd_t^2) / sum(sd_t^2)."""
    p = pos[pd.Series(pos).isin(list(pool)).to_numpy()]
    v = _component_sd(params, p) ** 2
    return float(1 - (_per(params.measurable_share, p) * v).sum() / v.sum())


def _played(stints: pd.DataFrame, stint_players: pd.DataFrame) -> pd.Series:
    """Total minutes per player over the schedule (as ``rapm.build_design``)."""
    sp_ = stint_players.merge(stints[["stint_id", "minutes"]], on="stint_id")
    return sp_.groupby("player_id")["minutes"].sum()


def _per90(pm: pd.DataFrame) -> pd.DataFrame:
    g = pm.groupby("player_id")[["minutes", *STATS]].sum()
    return g[list(STATS)].div(g["minutes"], axis=0) * 90


def _ranks(x: FloatArray, rng: np.random.Generator) -> NDArray[np.int64]:
    """Rank 1 = largest; ties broken at random."""
    perm = rng.permutation(len(x))
    order = perm[np.argsort(-x[perm], kind="stable")]
    rank = np.empty(len(x), dtype=np.int64)
    rank[order] = np.arange(1, len(x) + 1)
    return rank


def _quantiles(x: FloatArray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
