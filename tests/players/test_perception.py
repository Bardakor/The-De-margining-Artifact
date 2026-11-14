"""M6 recognition on synthetic votes with a planted truth.

Votes are Plackett-Luce draws (Gumbel-max: sort utility + Gumbel noise). The top
``n_ranked`` are ranked, everyone else is unranked, and points are the Borda count
n_ranked + 1 - rank.
"""

from itertools import permutations

import numpy as np
import pandas as pd

from footy.players.perception import (
    FloatArray,
    IntArray,
    bootstrap_shares,
    fit_plackett_luce,
    fit_recognition,
    plackett_luce_loglik,
)


def _simulate(
    rng: np.random.Generator,
    betas: dict[str, list[float]],
    n_editions: int,
    n_candidates: int,
    n_ranked: int,
    bonus: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One row per (edition, candidate). ``bonus`` is utility no block sees, given to p0 in
    edition 0, whose features are then set to the population mean (0)."""
    blocks = {name: [f"{name}_{k}" for k in range(len(b))] for name, b in betas.items()}
    beta = np.concatenate([np.asarray(b, dtype=np.float64) for b in betas.values()])
    columns = [c for cols in blocks.values() for c in cols]
    frames = []
    for e in range(n_editions):
        x = rng.standard_normal((n_candidates, beta.size))
        if e == 0 and bonus:
            x[0] = 0.0
        u = x @ beta
        if e == 0:
            u[0] += bonus
        order = np.argsort(-(u + rng.gumbel(size=n_candidates)))
        rank = np.full(n_candidates, np.nan)
        rank[order[:n_ranked]] = np.arange(1, n_ranked + 1)
        frame = pd.DataFrame(x, columns=columns)
        frame.insert(0, "edition", 2021 + e)
        frame.insert(1, "player", [f"p{i}" for i in range(n_candidates)])
        frame.insert(2, "points", np.where(np.isnan(rank), 0.0, n_ranked + 1 - rank))
        frame.insert(3, "rank", rank)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), blocks


def test_unranked_likelihood_is_the_sum_over_their_orders() -> None:
    """Unranked rank below every ranked player, in no order: the likelihood equals the full
    Plackett-Luce probability summed over every complete order that agrees with the
    ranking."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((7, 2))
    beta = np.array([0.7, -1.2])
    rank = np.array([2.0, np.inf, 1.0, np.inf, np.inf, 1.0, np.inf])
    edition: IntArray = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.intp)
    w = np.exp(x @ beta)

    def full_order_probability(order: tuple[int, ...]) -> float:
        p = 1.0
        for k, i in enumerate(order):
            p *= w[i] / w[list(order[k:])].sum()
        return p

    expected = 1.0
    for members in ([0, 1, 2, 3, 4], [5, 6]):
        expected *= sum(
            full_order_probability(order)
            for order in permutations(members)
            if all(rank[a] <= rank[b] for a, b in zip(order, order[1:], strict=False))
        )
    loglik, _ = plackett_luce_loglik(beta, x, rank, edition)
    assert np.isclose(np.exp(loglik), expected, rtol=1e-12)


def test_analytic_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((60, 4))
    rank = np.full(60, np.inf)
    rank[:15] = rng.integers(1, 12, size=15)  # ties included
    rank[30:40] = np.arange(1, 11)
    edition: IntArray = np.repeat(np.arange(2), 30).astype(np.intp)
    beta = rng.standard_normal(4)
    _, grad = plackett_luce_loglik(beta, x, rank, edition)
    h = 1e-6
    numeric: FloatArray = np.array(
        [
            (
                plackett_luce_loglik(beta + h * step, x, rank, edition)[0]
                - plackett_luce_loglik(beta - h * step, x, rank, edition)[0]
            )
            / (2 * h)
            for step in np.eye(4)
        ]
    )
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-7)


def test_plackett_luce_recovers_planted_beta() -> None:
    """30 editions x 100 candidates, top 30 ranked; tolerance 0.12 on every coefficient."""
    planted = [1.0, -0.5, 0.8]
    frame, blocks = _simulate(np.random.default_rng(3), {"f": planted}, 30, 100, 30)
    edition: IntArray = np.asarray(pd.factorize(frame["edition"])[0], dtype=np.intp)
    fit = fit_plackett_luce(
        frame[blocks["f"]].to_numpy(dtype=np.float64),
        np.nan_to_num(frame["rank"].to_numpy(dtype=np.float64), nan=np.inf),
        edition,
    )
    assert fit.converged
    np.testing.assert_allclose(fit.beta, planted, atol=0.12)


def test_shapley_sums_to_r2_follows_planted_strength_and_ignores_noise() -> None:
    betas = {"stats": [0.15, 0.15], "impact": [0.5], "team": [1.2, 0.8], "noise": [0.0, 0.0]}
    frame, blocks = _simulate(np.random.default_rng(4), betas, 10, 120, 30)
    fit = fit_recognition(frame, blocks)
    assert fit.pl_converged
    assert np.isclose(sum(fit.ols_shares.values()), fit.ols_r2, rtol=0, atol=1e-12)
    assert np.isclose(sum(fit.pl_shares.values()), fit.pl_pseudo_r2, rtol=0, atol=1e-12)
    for shares in (fit.ols_shares, fit.pl_shares):
        assert shares["team"] > shares["impact"] > shares["stats"]
        assert abs(shares["noise"]) < 0.005
    assert fit.ols_r2 - 0.02 < fit.ols_cv_r2 < fit.ols_r2


def test_exactly_collinear_duplicate_blocks_split_credit_equally() -> None:
    frame, blocks = _simulate(
        np.random.default_rng(5), {"stats": [0.3], "team": [1.0, 0.7]}, 8, 100, 30
    )
    frame["copy_0"] = 2.0 * frame["team_0"] + frame["team_1"]
    frame["copy_1"] = frame["team_0"] - frame["team_1"]
    fit = fit_recognition(frame, {**blocks, "team_copy": ["copy_0", "copy_1"]})
    assert abs(fit.ols_shares["team"] - fit.ols_shares["team_copy"]) < 1e-10
    assert abs(fit.pl_shares["team"] - fit.pl_shares["team_copy"]) < 1e-6


def test_residuals_flag_the_player_the_vote_overrates() -> None:
    """p0 in edition 2021 gets utility no block sees; he has the largest residual there."""
    frame, blocks = _simulate(np.random.default_rng(6), {"team": [1.0, 0.6]}, 6, 80, 30, bonus=5.0)
    residuals = fit_recognition(frame, blocks).residuals
    first = residuals[residuals["edition"] == 2021]
    assert first.loc[first["ols_residual"].idxmax(), "player"] == "p0"
    assert first.loc[first["pl_residual"].idxmax(), "player"] == "p0"


def test_bootstrap_is_reproducible_and_keeps_the_planted_order() -> None:
    frame, blocks = _simulate(
        np.random.default_rng(7), {"stats": [0.2], "team": [1.2, 0.8]}, 5, 80, 30
    )
    first = bootstrap_shares(frame, blocks, np.random.default_rng(8), n_boot=10)
    again = bootstrap_shares(frame, blocks, np.random.default_rng(8), n_boot=10)
    pd.testing.assert_frame_equal(first, again)
    assert len(first) == 20
    assert first["converged"].all()
    assert (first["team"] > first["stats"]).all()
