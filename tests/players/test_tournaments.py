import numpy as np
import pandas as pd
import pytest

from footy.players import tournaments as tr

FEATURES = ("f1", "f2", "f3")
TRUE_R2 = 0.5


def _league(
    rng: np.random.Generator, n_players: int, n_matches: int, shock: float = 0.0
) -> pd.DataFrame:
    """Planted truth. True per-90 feature rates F ~ N(0, I); the true target rate is
    T = 0.5 F1 + 0.5 F2 + u with var(u) = 0.5, so var(T) = 1 and the box score sees
    R² = 0.5 of it. Each 90-minute match adds N(0, 2²) to every feature rate and
    N(0, 8) to the target rate, so with 8 matches the unit means have reliability 2/3
    (features, and the F1 + F2 composite) and 1/2 (target). ``shock`` adds a
    match-level term shared by f1, f2 and the target (a dominant team-match)."""
    f = rng.normal(size=(n_players, 3))
    t = f @ np.array([0.5, 0.5, 0.0]) + rng.normal(scale=np.sqrt(0.5), size=n_players)
    player = np.repeat(np.arange(n_players), n_matches)
    delta = shock * rng.normal(size=len(player))
    x = f[player] + rng.normal(scale=2.0, size=(len(player), 3))
    x[:, :2] += delta[:, None]
    y = t[player] + rng.normal(scale=np.sqrt(8.0), size=len(player)) + delta
    pm = pd.DataFrame(x, columns=list(FEATURES))
    return pm.assign(player_id=player, season="S", minutes=90.0, target=y)


def _run(pm: pd.DataFrame, seed: int, **kwargs: int) -> pd.Series:
    out = tr.visibility(pm, ["target"], FEATURES, np.random.default_rng(seed), **kwargs)
    return out.iloc[0]


def test_planted_visibility_is_recovered_by_both_estimators() -> None:
    pm = _league(np.random.default_rng(1), n_players=3000, n_matches=8)
    got = _run(pm, 2, n_cv=5, n_splits=100, n_boot=1)
    assert got["rel_target"] == pytest.approx(0.5, abs=0.04)
    assert got["rel_box"] == pytest.approx(2 / 3, abs=0.04)
    assert got["r2_cv"] == pytest.approx(TRUE_R2 * 0.5 * 2 / 3, abs=0.03)
    assert got["r2_disatt"] == pytest.approx(TRUE_R2, abs=0.08)
    assert got["r2_cross"] == pytest.approx(TRUE_R2, abs=0.08)


def test_shared_match_noise_inflates_r2_disatt_but_not_r2_cross() -> None:
    pm = _league(np.random.default_rng(3), n_players=3000, n_matches=8, shock=2.0)
    got = _run(pm, 4, n_cv=5, n_splits=100, n_boot=1)
    assert got["r2_cross"] == pytest.approx(TRUE_R2, abs=0.1)
    assert got["r2_disatt"] > TRUE_R2 + 0.5  # about 2 in population


def test_player_bootstrap_interval_covers_the_truth() -> None:
    pm = _league(np.random.default_rng(5), n_players=800, n_matches=8)
    # Each player has two units (two "competitions"): the bootstrap must resample players.
    pm = pd.concat([pm, pm.assign(season="T")], ignore_index=True)
    got = _run(pm, 6, n_cv=2, n_splits=20, n_boot=60)
    assert (got["n_units"], got["n_players"]) == (1600, 800)
    for name in ("r2_disatt", "r2_cross"):
        assert got[f"{name}_lo"] < TRUE_R2 < got[f"{name}_hi"]
        assert got[f"{name}_lo"] < got[name] < got[f"{name}_hi"]
    assert got["boot_nan"] == 0.0


def test_unit_group_is_the_minutes_modal_group() -> None:
    pm = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1, 2, 2],
            "season": ["A", "A", "A", "B", "A", "A"],
            "position_group": ["W", "CF", "CF", "FB", "CM", "DM"],
            "minutes": [90.0, 45.0, 30.0, 90.0, 60.0, 60.0],
        }
    )
    got = tr.unit_groups(pm)
    assert got["position_group"].to_dict() == {(1, "A"): "W", (1, "B"): "FB", (2, "A"): "CM"}
    assert got["minutes"].to_dict() == {(1, "A"): 165.0, (1, "B"): 90.0, (2, "A"): 120.0}
