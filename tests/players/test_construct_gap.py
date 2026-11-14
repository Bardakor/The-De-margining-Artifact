from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from footy.players.construct_gap import position_contrast, unmeasured_share
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


@dataclass(frozen=True)
class Inputs:
    m: float
    league: League
    design: Design
    fit: RapmFit
    halves: Halves
    stats: pd.DataFrame
    stats_a: pd.DataFrame
    stats_b: pd.DataFrame
    positions: pd.Series
    attribution: pd.DataFrame  # index "all"
    attribution_pools: pd.DataFrame  # index "CB+DM", "CF+W"


def _per90(pm: pd.DataFrame) -> pd.DataFrame:
    g = pm.groupby("player_id")[["minutes", "shots", "goals"]].sum()
    return g[["shots", "goals"]].div(g["minutes"], axis=0) * 90


@pytest.fixture(scope="module", params=[0.3, 0.7])
def inputs(request: pytest.FixtureRequest) -> Inputs:
    """30 clubs x 25 players x 6 seasons (~35k stints, 750 players)."""
    m = float(request.param)
    rng = np.random.default_rng(0)
    lg = simulate_league(
        rng,
        n_teams=30,
        squad_size=25,
        n_seasons=6,
        offence_sd=0.1,
        defence_sd=0.1,
        measurable_share=m,
    )
    d = build_design(lg.stints, lg.stint_players)
    fit = fit_rapm(d, LAM)
    pm = lg.player_match
    half = rng.random(int(pm["game_id"].max()) + 1) < 0.5
    in_a = half[pm["game_id"].to_numpy()]
    positions = pd.Series(rng.choice(list(POOLS), size=len(fit.players)), index=fit.players.index)

    def attribution(groups: pd.Series | None) -> pd.DataFrame:
        return attribution_factor(
            d, LAM, np.random.default_rng(3), min_minutes=1000, n_sims=3, groups=groups
        )

    return Inputs(
        m, lg, d, fit, split_halves(d, LAM, np.random.default_rng(2), n_repeats=5),
        _per90(pm), _per90(pm[in_a]), _per90(pm[~in_a]), positions,
        attribution(None), attribution(positions.map(POOLS)),
    )  # fmt: skip


def _at(df: pd.DataFrame, group: str, comp: str) -> float:
    return float(df[comp].to_dict()[group])


def _share(
    x: Inputs, comp: str, attribution: float, n_boot: int = 100, reps: bool = False
) -> pd.Series:
    replicates = None
    if reps:
        boot = bootstrap_rapm(x.design, LAM, np.random.default_rng(7), n_boot=n_boot)
        replicates = (boot.replicates[comp], boot.halves.a[comp], boot.halves.b[comp])
    out = unmeasured_share(
        x.fit.players[comp], (x.halves.a[comp], x.halves.b[comp]), {"all": attribution},
        x.stats, x.stats_a, x.stats_b, pd.Series("all", index=x.fit.players.index),
        np.random.default_rng(4), n_boot=n_boot, replicates=replicates,
    )  # fmt: skip
    return pd.Series(out.loc["all"])


def test_offence_share_recovers_one_minus_m(inputs: Inputs) -> None:
    """Planted: a share m of offence variance drives the shot rate. Tolerance
    0.15: over 16 leagues like this one the error was +0.03 +- 0.06 (m = 0.3)
    and +0.075 +- 0.06 (m = 0.7)."""
    target = 1 - inputs.m
    res = _share(inputs, "offence", _at(inputs.attribution, "all", "offence"))
    assert abs(res["share"] - target) < 0.15
    assert res["r2_true_clipped"] == pytest.approx(min(max(res["r2_true"], 0), 1))
    # Without disattenuation the share is biased upward, far beyond tolerance.
    assert res["share_naive"] > target + 0.15
    # Split-half alone (attribution 1) over-states RAPM's reliability through
    # teammate leakage, so it under-corrects.
    split_half_only = _share(inputs, "offence", 1.0)
    assert res["share"] < split_half_only["share"] < res["share_naive"]


def test_both_reliabilities_vary_across_bootstrap_draws(inputs: Inputs) -> None:
    """A8: the joint bootstrap re-estimates rel_RAPM and rel_pred in every draw."""
    res = _share(inputs, "offence", _at(inputs.attribution, "all", "offence"))
    for rel in ("rel_rapm", "rel_pred"):
        assert res[f"{rel}_ci_high"] - res[f"{rel}_ci_low"] > 0.01
        assert res[f"{rel}_ci_low"] < res[rel] < res[f"{rel}_ci_high"]
    assert res["ci_low"] < res["share"] < res["ci_high"]


def test_defence_share_is_one(inputs: Inputs) -> None:
    """No statistic in the generator sees defence."""
    res = _share(inputs, "defence", _at(inputs.attribution, "all", "defence"))
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
    attribution = {
        "CB+DM": _at(inputs.attribution_pools, "CB+DM", "defence"),
        "CF+W": _at(inputs.attribution_pools, "CF+W", "offence"),
    }
    out = position_contrast(
        mixed, (half_a, half_b), attribution, inputs.stats, inputs.stats_a,
        inputs.stats_b, inputs.positions, np.random.default_rng(5), n_boot=100,
    )  # fmt: skip
    # Pools of ~375 players: SD of each share ~0.08, of the difference ~0.11,
    # and the CF+W share runs ~0.03-0.08 high (see the offence test).
    assert abs(out["diff_clipped"] - inputs.m) < 0.25
    assert out["ci_low_clipped"] > 0


def test_contrast_interval_covers_zero_when_pools_are_alike(inputs: Inputs) -> None:
    attribution = {g: _at(inputs.attribution_pools, g, "offence") for g in ("CB+DM", "CF+W")}
    out = position_contrast(
        inputs.fit.players["offence"], (inputs.halves.a["offence"], inputs.halves.b["offence"]),
        attribution, inputs.stats, inputs.stats_a, inputs.stats_b, inputs.positions,
        np.random.default_rng(6), n_boot=100,
    )  # fmt: skip
    assert out["ci_low"] < 0 < out["ci_high"]


def test_rapm_replicates_widen_the_interval(inputs: Inputs) -> None:
    if inputs.m != 0.7:
        pytest.skip("one league is enough")
    attribution = _at(inputs.attribution, "all", "offence")
    plain = _share(inputs, "offence", attribution, n_boot=30)
    joint = _share(inputs, "offence", attribution, n_boot=30, reps=True)
    assert joint["share"] == pytest.approx(plain["share"])
    assert joint["ci_low"] < joint["share"] < joint["ci_high"]
    assert joint["ci_high"] - joint["ci_low"] > plain["ci_high"] - plain["ci_low"]
    assert joint["rel_rapm_ci_high"] - joint["rel_rapm_ci_low"] > 0.03
    boot = bootstrap_rapm(inputs.design, LAM, np.random.default_rng(7), n_boot=2)
    with pytest.raises(ValueError, match="replicates"):
        unmeasured_share(
            inputs.fit.players["offence"], (inputs.halves.a["offence"], inputs.halves.b["offence"]),
            {"all": attribution}, inputs.stats, inputs.stats_a, inputs.stats_b,
            pd.Series("all", index=inputs.fit.players.index), np.random.default_rng(4), n_boot=3,
            replicates=(boot.replicates["offence"], boot.halves.a["offence"],
                        boot.halves.b["offence"]),
        )  # fmt: skip
