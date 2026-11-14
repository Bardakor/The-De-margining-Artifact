from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from footy.players.construct_gap import (
    Estimator,
    Splits,
    position_contrast,
    split_stats,
    unmeasured_share,
)
from footy.players.rapm import (
    Design,
    Halves,
    RapmFit,
    attribution_factor,
    bootstrap_rapm,
    build_design,
    fit_rapm,
    split_halves,
)
from footy.players.synthetic import League, simulate_league

LAM = 1e4  # what cv_lambda picks on these leagues (tests/players/test_rapm.py)
POOLS = {"CB": "CB+DM", "DM": "CB+DM", "CF": "CF+W", "W": "CF+W"}
STATS = ["shots", "goals"]


@dataclass(frozen=True)
class Inputs:
    m: float
    league: League
    design: Design
    fit: RapmFit
    halves: Halves
    stats: pd.DataFrame
    split_stats: list[tuple[pd.DataFrame, pd.DataFrame]]
    positions: pd.Series
    # attribution factors per estimator, index "all" or the two pools
    attribution: dict[Estimator, pd.DataFrame]
    attribution_pools: dict[Estimator, pd.DataFrame]

    def splits(self, comp: str) -> Splits:
        return Splits(self.halves.a[comp], self.halves.b[comp], self.split_stats)


def _per90(pm: pd.DataFrame) -> pd.DataFrame:
    g = pm.groupby("player_id")[["minutes", *STATS]].sum()
    return g[STATS].div(g["minutes"], axis=0) * 90


def _inputs(m: float, seed: int, match_shock_sd: float = 0.0) -> Inputs:
    """30 clubs x 25 players x 6 seasons (~35k stints, 750 players)."""
    rng = np.random.default_rng(seed)
    lg = simulate_league(
        rng, n_teams=30, squad_size=25, n_seasons=6, offence_sd=0.1, defence_sd=0.1,
        measurable_share=m, match_shock_sd=match_shock_sd,
    )  # fmt: skip
    d = build_design(lg.stints, lg.stint_players)
    fit = fit_rapm(d, LAM)
    positions = pd.Series(rng.choice(list(POOLS), size=len(fit.players)), index=fit.players.index)
    halves = split_halves(d, LAM, np.random.default_rng(2), n_repeats=5)

    def attribution(groups: pd.Series | None) -> dict[Estimator, pd.DataFrame]:
        return {
            est: attribution_factor(
                d,
                LAM,
                np.random.default_rng(3),
                min_minutes=1000,
                n_sims=3,
                groups=groups,
                halves=est == "cross",
            )  # fmt: skip
            for est in ("cross", "within")
        }

    pm = lg.player_match
    return Inputs(
        m, lg, d, fit, halves, _per90(pm), split_stats(pm, halves, STATS), positions,
        attribution(None), attribution(positions.map(POOLS)),
    )  # fmt: skip


@pytest.fixture(scope="module", params=[0.3, 0.7])
def inputs(request: pytest.FixtureRequest) -> Inputs:
    return _inputs(float(request.param), seed=0)


def _at(df: pd.DataFrame, group: str, comp: str) -> float:
    return float(df[comp].to_dict()[group])


def _share(
    x: Inputs,
    comp: str,
    estimator: Estimator = "cross",
    attribution: float | None = None,
    n_boot: int = 100,
    reps: bool = False,
) -> pd.Series:
    replicates = None
    if reps:
        boot = bootstrap_rapm(x.design, LAM, np.random.default_rng(7), n_boot=n_boot)
        rep_stats = split_stats(x.league.player_match, boot.halves, STATS)
        replicates = (
            boot.replicates[comp],
            Splits(boot.halves.a[comp], boot.halves.b[comp], rep_stats),
        )
    att = _at(x.attribution[estimator], "all", comp) if attribution is None else attribution
    out = unmeasured_share(
        x.fit.players[comp], x.stats, x.splits(comp), {"all": att},
        pd.Series("all", index=x.fit.players.index), np.random.default_rng(4),
        estimator=estimator, n_boot=n_boot, replicates=replicates,
    )  # fmt: skip
    return pd.Series(out.loc["all"])


def test_offence_share_recovers_one_minus_m(inputs: Inputs) -> None:
    """Planted: a share m of offence variance drives the shot rate. Tolerance
    0.15: over 12-16 leagues like this one the within estimator's error was
    +0.01..+0.075 with SD 0.055; the cross estimator's is smaller."""
    target = 1 - inputs.m
    res = _share(inputs, "offence")
    assert abs(res["share"] - target) < 0.15
    assert res["r2_true_clipped"] == pytest.approx(min(max(res["r2_true"], 0), 1))
    # Without disattenuation the share is biased upward, far beyond tolerance.
    assert res["share_naive"] > target + 0.15
    # Split-half alone (attribution 1) over-states RAPM's reliability through
    # teammate leakage, so it under-corrects.
    split_half_only = _share(inputs, "offence", attribution=1.0)
    assert res["share"] < split_half_only["share"] < res["share_naive"]


def test_estimators_agree_without_match_shocks(inputs: Inputs) -> None:
    within, cross = (
        _share(inputs, "offence", est, n_boot=10)["share"] for est in ("within", "cross")
    )
    assert abs(within - cross) < 0.06


def test_match_shocks_bias_within_low_and_cross_recovers() -> None:
    """One shock per team-match (SD 1 goal per 90, exp(shock) on shot rates)
    moves both a player's shots and his stints' goals. Means over three
    leagues, planted share 0.5: within sits ~0.13 low, cross within ~0.01."""
    shares: dict[Estimator, list[float]] = {"within": [], "cross": []}
    for seed in (0, 1, 2):
        x = _inputs(0.5, seed, match_shock_sd=1.0)
        for est in shares:
            shares[est].append(float(_share(x, "offence", est, n_boot=10)["share"]))
    within, cross = np.mean(shares["within"]), np.mean(shares["cross"])
    assert within < 0.5 - 0.08
    assert abs(cross - 0.5) < 0.08
    assert all(w < c for w, c in zip(shares["within"], shares["cross"], strict=True))


def test_both_reliabilities_vary_across_bootstrap_draws(inputs: Inputs) -> None:
    """A8: the joint bootstrap re-estimates rel_RAPM and rel_pred in every draw."""
    res = _share(inputs, "offence")
    for rel in ("rel_rapm", "rel_pred"):
        assert res[f"{rel}_ci_high"] - res[f"{rel}_ci_low"] > 0.01
        assert res[f"{rel}_ci_low"] < res[rel] < res[f"{rel}_ci_high"]
    assert res["ci_low"] < res["share"] < res["ci_high"]


def test_defence_share_is_one(inputs: Inputs) -> None:
    """No statistic in the generator sees defence."""
    res = _share(inputs, "defence")
    assert res["share_clipped"] > 0.9
    assert res["ci_low_clipped"] > 0.8


def test_contrast_recovers_a_planted_difference(inputs: Inputs) -> None:
    """CB+DM players carry their defence RAPM (share ~1), CF+W their offence
    RAPM (share ~1 - m): the contrast should be ~m and its interval exclude 0."""
    cfw = inputs.positions.index[inputs.positions.map(POOLS) == "CF+W"]
    cbdm = inputs.positions.index.difference(cfw)
    p, h = inputs.fit.players, inputs.halves
    mixed = pd.concat([p.loc[cfw, "offence"], p.loc[cbdm, "defence"]])
    half_a, half_b = (
        pd.concat([side["offence"][cfw], side["defence"][cbdm]], axis=1) for side in (h.a, h.b)
    )
    pools = inputs.attribution_pools["cross"]
    attribution = {"CB+DM": _at(pools, "CB+DM", "defence"), "CF+W": _at(pools, "CF+W", "offence")}
    out = position_contrast(
        mixed, inputs.stats, Splits(half_a, half_b, inputs.split_stats), attribution,
        inputs.positions, np.random.default_rng(5), n_boot=100,
    )  # fmt: skip
    # Pools of ~375 players: SD of each share ~0.08, of the difference ~0.11.
    assert abs(out["diff_clipped"] - inputs.m) < 0.25
    assert out["ci_low_clipped"] > 0


def test_contrast_interval_covers_zero_when_pools_are_alike(inputs: Inputs) -> None:
    pools = inputs.attribution_pools["cross"]
    out = position_contrast(
        inputs.fit.players["offence"], inputs.stats, inputs.splits("offence"),
        {g: _at(pools, g, "offence") for g in ("CB+DM", "CF+W")}, inputs.positions,
        np.random.default_rng(6), n_boot=100,
    )  # fmt: skip
    assert out["ci_low"] < 0 < out["ci_high"]


def test_rapm_replicates_widen_the_interval(inputs: Inputs) -> None:
    if inputs.m != 0.7:
        pytest.skip("one league is enough")
    plain = _share(inputs, "offence", n_boot=30)
    joint = _share(inputs, "offence", n_boot=30, reps=True)
    assert joint["share"] == pytest.approx(plain["share"])
    assert joint["ci_low"] < joint["share"] < joint["ci_high"]
    assert joint["ci_high"] - joint["ci_low"] > plain["ci_high"] - plain["ci_low"]
    assert joint["rel_rapm_ci_high"] - joint["rel_rapm_ci_low"] > 0.03
    one_replicate = inputs.fit.players[["offence"]].T  # 1 row; the splits have 5
    with pytest.raises(ValueError, match="replicates"):
        unmeasured_share(
            inputs.fit.players["offence"], inputs.stats, inputs.splits("offence"), {"all": 0.9},
            pd.Series("all", index=inputs.fit.players.index), np.random.default_rng(4),
            n_boot=2, replicates=(one_replicate, inputs.splits("offence")),
        )  # fmt: skip
