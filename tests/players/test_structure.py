import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster
from scipy.stats import pearsonr

from footy.players.structure import (
    cluster_metrics,
    composite,
    composite_diagnostics,
    correlation,
    distance_correlation,
    distance_correlation_test,
    entropy_effective_number,
    factor_analysis,
    parallel_analysis,
    partial_correlation,
    participation_ratio,
    pca,
    team_icc,
)

# Planted two-factor structure: metrics m0-m3 on F1, m4-m7 on F2, two cross-loadings.
LOADINGS = np.array(
    [[0.80, 0], [0.75, 0.20], [0.70, 0], [0.60, 0], [0, 0.80], [0, 0.70], [0.20, 0.65], [0, 0.60]]
)
UNIQUENESS = 1 - np.sum(LOADINGS**2, axis=1)
NAMES = [f"m{i}" for i in range(8)]
CONGRUENCE_MIN = 0.95  # Lorenzo-Seva & ten Berge (2006): >= 0.95, factors "equal"


def _two_factor(seed: int, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 8)) * np.sqrt(UNIQUENESS)
    return pd.DataFrame(rng.standard_normal((n, 2)) @ LOADINGS.T + noise, columns=NAMES)


def _congruence(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def test_parallel_analysis_retains_the_two_planted_factors() -> None:
    result = parallel_analysis(_two_factor(1), np.random.default_rng(2), n_repeats=100)
    assert result.n_components == 2
    assert result.observed[1] > result.threshold[1] > result.observed[2]


def test_factor_analysis_recovers_planted_loadings_and_fit() -> None:
    data = _two_factor(3)
    fa = factor_analysis(data, 2)
    est = fa.loadings.to_numpy()
    best = max(
        (abs(_congruence(est[:, 0], LOADINGS[:, i])), abs(_congruence(est[:, 1], LOADINGS[:, j])))
        for i, j in [(0, 1), (1, 0)]
    )
    assert fa.converged
    assert min(best) > CONGRUENCE_MIN
    np.testing.assert_allclose(fa.uniquenesses.to_numpy(), UNIQUENESS, atol=0.05)
    np.testing.assert_allclose(np.sum(est**2, axis=1) + fa.uniquenesses, 1.0, atol=1e-5)
    assert fa.dof == 13 and fa.rmsea < 0.05 and fa.p_value > 0.01
    one = factor_analysis(data, 1)
    assert one.p_value < 1e-6 and one.rmsea > 0.1


def test_effective_number_matches_population_eigenvalues() -> None:
    population = np.linalg.eigvalsh(LOADINGS @ LOADINGS.T + np.diag(UNIQUENESS))
    sample = pca(_two_factor(4)).eigenvalues
    assert participation_ratio(sample) == pytest.approx(participation_ratio(population), abs=0.1)
    assert entropy_effective_number(sample) == pytest.approx(
        entropy_effective_number(population), abs=0.1
    )


def test_effective_number_counts_exact_copies_once() -> None:
    # Three independent metrics, each present twice: eigenvalues (2, 2, 2, 0, 0, 0).
    block = np.kron(np.eye(3), np.ones((2, 2)))
    eigenvalues = np.linalg.eigvalsh(block)
    assert participation_ratio(eigenvalues) == pytest.approx(3.0)
    assert entropy_effective_number(eigenvalues) == pytest.approx(3.0)
    assert participation_ratio(np.ones(5)) == pytest.approx(5.0)


def test_clustering_splits_the_two_blocks() -> None:
    tree, order = cluster_metrics(correlation(_two_factor(5), "spearman"))
    labels = fcluster(tree, 2, criterion="maxclust")
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]
    assert {order.index(m) < 4 for m in NAMES[:4]} in ({True}, {False})
    assert sorted(order) == NAMES


def test_partial_correlation_removes_covariate_induced_correlation() -> None:
    rng = np.random.default_rng(6)
    n = 4000
    league_season = rng.integers(0, 20, n)
    elo = rng.normal(1500, 100, n) + 20 * league_season
    shared = rng.normal(0, 1, 20)[league_season] + (elo - 1700) / 100
    e = rng.multivariate_normal(np.zeros(3), np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 1]]), n)
    panel = pd.DataFrame(
        {
            "a": shared + e[:, 0],
            "b": 0.5 * shared + e[:, 1],
            "c": -shared + e[:, 2],
            "elo": elo,
            "league_season": [f"L{g}" for g in league_season],
        }
    )
    raw = correlation(panel[["a", "b", "c"]])
    part = partial_correlation(panel, ["a", "b", "c"], ["elo"], ["league_season"])
    assert raw["c"]["a"] < -0.4
    assert part.loc["a", "c"] == pytest.approx(0.0, abs=0.05)
    assert part.loc["b", "c"] == pytest.approx(0.0, abs=0.05)
    assert part.loc["a", "b"] == pytest.approx(0.3, abs=0.05)


def test_distance_correlation_sees_a_square_pearson_misses() -> None:
    rng = np.random.default_rng(7)
    half = rng.uniform(-1, 1, 150)
    x = np.concatenate([half, -half])  # symmetric sample: Pearson(x, x^2) is exactly 0
    dcor, p = distance_correlation_test(x, x**2, rng, n_permutations=499)
    assert abs(pearsonr(x, x**2).statistic) < 1e-12
    assert dcor > 0.4 and p < 0.01


def test_distance_correlation_near_zero_when_independent() -> None:
    rng = np.random.default_rng(8)
    assert distance_correlation(rng.standard_normal(2000), rng.standard_normal(2000)) < 0.06
    _, p = distance_correlation_test(
        rng.standard_normal(300), rng.standard_normal(300), rng, n_permutations=499
    )
    assert p > 0.05
    block = rng.standard_normal((200, 2))
    assert distance_correlation(block, block) == pytest.approx(1.0)


def test_team_icc_recovers_planted_share_within_league_season() -> None:
    rng = np.random.default_rng(9)
    clubs, seasons, icc = 30, 12, 0.25
    frames = []
    for s in range(seasons):
        offset = rng.normal(0, 3)
        for c in range(clubs):
            size = int(rng.integers(4, 30))
            team = offset + rng.normal(0, np.sqrt(icc))
            frames.append(
                pd.DataFrame(
                    {
                        "club_id": c,
                        "league_season": f"S{s}",
                        "team_metric": team + rng.normal(0, np.sqrt(1 - icc), size),
                        "pure_player": offset + rng.normal(0, 1, size),
                    }
                )
            )
    panel = pd.concat(frames, ignore_index=True)
    metrics = ["team_metric", "pure_player"]
    out = team_icc(panel, metrics, "club_id", "league_season", rng, n_boot=300)
    assert out.loc["team_metric", "icc"] == pytest.approx(icc, abs=0.05)
    assert out["lo"]["team_metric"] < icc < out["hi"]["team_metric"]
    assert out.loc["pure_player", "icc"] == pytest.approx(0.0, abs=0.03)
    # Club-season groups without the league-season stratum: league offsets pose as team effect.
    panel["club_season"] = panel["club_id"].astype(str) + panel["league_season"]
    naive = team_icc(panel, metrics, "club_season", None, rng, n_boot=10)
    assert naive["icc"]["pure_player"] > 0.5


def test_independent_metrics_are_not_inflated() -> None:
    rng = np.random.default_rng(10)
    data = pd.DataFrame(rng.standard_normal((5000, 4)), columns=list("abcd"))
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("abcd"))
    diag = composite_diagnostics(correlation(data), weights)
    assert diag.inflation == pytest.approx(1.0, abs=0.05)
    np.testing.assert_allclose(diag.shares["effective"], diag.shares["claimed"], atol=0.02)
    # The diagnostic is the variance algebra of the composite actually built.
    c = composite(data, weights)
    assert c.var() / np.sum(weights**2) == pytest.approx(diag.inflation, rel=1e-9)


@pytest.mark.parametrize("split", [0.5, 1.0])  # weight split in two; full weight twice
def test_duplicated_metric_matches_derivation(split: float) -> None:
    w1, others = 0.4, np.array([0.3, 0.3])
    a = b = split * w1
    s = float(np.sum(others**2))
    corr = pd.DataFrame(
        np.eye(4), index=["z1a", "z1b", "z2", "z3"], columns=["z1a", "z1b", "z2", "z3"]
    )
    corr.loc["z1a", "z1b"] = corr.loc["z1b", "z1a"] = 1.0
    weights = pd.Series([a, b, *others], index=corr.index)
    diag = composite_diagnostics(corr, weights)
    var_c, sum_w2 = (a + b) ** 2 + s, a**2 + b**2 + s
    assert diag.inflation == pytest.approx(var_c / sum_w2)
    assert diag.shares.loc["z1a", "effective"] == pytest.approx(a * (a + b) / var_c)
    assert diag.shares.loc["z1a", "claimed"] == pytest.approx(a**2 / sum_w2)
    assert diag.shares["effective"].sum() == pytest.approx(1.0)
    assert diag.pc1_share == pytest.approx((a + b) ** 2 / var_c)
    if split == 0.5:  # composite unchanged: z1 keeps its real share, its claimed share drops
        assert diag.shares.loc[["z1a", "z1b"], "effective"].sum() == pytest.approx(
            w1**2 / (w1**2 + s)
        )
        assert diag.shares.loc[["z1a", "z1b"], "claimed"].sum() < w1**2 / (w1**2 + s)
