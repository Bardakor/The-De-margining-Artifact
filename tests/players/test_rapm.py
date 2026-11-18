from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from footy.players.rapm import (
    REPLACEMENT,
    Design,
    RapmFit,
    attribution_factor,
    bootstrap_rapm,
    build_design,
    cv_lambda,
    fit_rapm,
    fit_rapm_by_season,
    split_half_reliability,
    split_halves,
)
from footy.players.synthetic import AGE_PEAK, League, simulate_league

LAMS = [1e3, 3e3, 1e4, 3e4, 1e5]
SD = 0.1  # planted offence / defence sd, goals per 90


@dataclass(frozen=True)
class Fitted:
    league: League
    stints: pd.DataFrame
    design: Design
    lam: float
    curve: pd.Series


def _with_season(lg: League) -> pd.DataFrame:
    return lg.stints.merge(lg.games[["game_id", "season"]], on="game_id")


@pytest.fixture(scope="module")
def six_seasons() -> Fitted:
    """20 clubs x 20 players x 6 seasons, transfers on: ~14k stints, 400 players."""
    lg = simulate_league(
        np.random.default_rng(0),
        n_teams=20,
        squad_size=20,
        n_seasons=6,
        offence_sd=SD,
        defence_sd=SD,
    )
    st = _with_season(lg)
    d = build_design(st, lg.stint_players, group="season")
    lam, curve = cv_lambda(d, LAMS, np.random.default_rng(1))
    return Fitted(lg, st, d, lam, curve)


def test_recovers_planted_effects_and_home_advantage(six_seasons: Fitted) -> None:
    # A player has ~11,300 minutes, so even an effect estimated alone would have
    # noise variance 121.5 / 11,300 = 0.0108 (Poisson, 1.35 goals/90) against a
    # planted variance of 0.01: reliability <= 0.48 and correlation <= 0.69.
    # Teammate confounding costs more; 0.5 leaves room for that, not for a bug.
    fit = fit_rapm(six_seasons.design, six_seasons.lam)
    joined = fit.players.join(six_seasons.league.truth.set_index("player_id"), rsuffix="_true")
    for comp in ("offence", "defence"):
        assert np.corrcoef(joined[comp], joined[f"{comp}_true"])[0, 1] > 0.5
    # One home advantage per season (A7). The schedule is balanced, so each one
    # should track that season's raw home-minus-away rate per 90 (which has SD
    # ~0.096 around the planted 0.3 across seasons and seeds; 2020/21 here is a
    # 3.8-SD draw at 0.66), and their mean the planted 0.3 (SE ~0.04).
    st = six_seasons.stints.groupby("season")[["home_goals", "away_goals", "minutes"]].sum()
    raw = (st["home_goals"] - st["away_goals"]) * 90 / st["minutes"]
    assert (fit.home - raw).abs().max() < 0.05
    assert abs(fit.home.mean() - 0.3) < 0.1


def test_home_advantage_is_estimated_per_group() -> None:
    """Two leagues, planted home advantage 0.6 and 0.0, fitted together."""
    parts = []
    for k, home in enumerate((0.6, 0.0)):
        lg = simulate_league(
            np.random.default_rng(20 + k), n_teams=20, n_seasons=2, home_advantage=home
        )
        off = 100_000 * k
        st = lg.stints.assign(
            stint_id=lg.stints["stint_id"] + off, game_id=lg.stints["game_id"] + off, league=k
        )
        sp_ = lg.stint_players.assign(
            stint_id=lg.stint_players["stint_id"] + off,
            player_id=lg.stint_players["player_id"] + off,
        )
        parts.append((st, sp_))
    st, sp_ = (pd.concat([p[i] for p in parts], ignore_index=True) for i in (0, 1))
    fit = fit_rapm(build_design(st, sp_, group="league"), 1e4)
    # SE ~ sqrt(2 * 121.5 / (760 games * 90)) = 0.06 per league.
    assert abs(fit.home[0] - 0.6) < 0.2
    assert abs(fit.home[1] - 0.0) < 0.2


def test_cv_picks_an_interior_lambda(six_seasons: Fitted) -> None:
    curve = six_seasons.curve
    assert LAMS[0] < six_seasons.lam < LAMS[-1]
    assert curve.min() < min(curve.iloc[0], curve.iloc[-1])


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-3 players a side drawn from a pool of 8: unequal sides make it full rank."""
    rng = np.random.default_rng(11)
    stints, players = [], []
    for s in range(1, 301):
        ids = rng.permutation(8) + 1
        n_home, n_away = rng.integers(1, 4, size=2)
        players += [(s, int(p), "home") for p in ids[:n_home]]
        players += [(s, int(p), "away") for p in ids[n_home : n_home + n_away]]
        stints.append(
            (s, (s + 1) // 2, int(rng.integers(5, 60)), int(rng.poisson(1.0)),
             int(rng.poisson(0.8)), float(rng.normal()), "ab"[s % 3 == 0])
        )  # fmt: skip
    cols = ["stint_id", "game_id", "minutes", "home_goals", "away_goals", "c", "g"]
    return pd.DataFrame(stints, columns=cols), pd.DataFrame(
        players, columns=["stint_id", "player_id", "side"]
    )


def _dense_least_squares(st: pd.DataFrame, sp_: pd.DataFrame) -> NDArray[np.float64]:
    """Weighted least squares written out row by row, independent of rapm.py.
    Columns: offence 1..8, defence 1..8, home a, home b, intercept a, intercept b, c."""
    cols = ("minutes", "c", "home_goals", "away_goals")
    num = {k: st[k].to_numpy(dtype=np.float64) for k in cols}
    x, y, w = [], [], []
    for i, s in enumerate(st["stint_id"].to_numpy()):
        on = sp_[sp_["stint_id"] == s]
        minutes, c = num["minutes"][i], num["c"][i]
        goals = {"home": num["home_goals"][i], "away": num["away_goals"][i]}
        for side, other, sign in (("home", "away", 1), ("away", "home", -1)):
            v = np.zeros(21)
            g = int(st["g"].iat[i] == "b")
            for p in on.loc[on["side"] == side, "player_id"].to_numpy():
                v[p - 1] += 1
            for p in on.loc[on["side"] == other, "player_id"].to_numpy():
                v[8 + p - 1] -= 1
            v[16 + g] = sign == 1
            v[18 + g] = 1
            v[20] = sign * c
            x.append(v)
            y.append(goals[side] * 90 / minutes)
            w.append(minutes)
    xa, sw = np.array(x), np.sqrt(np.array(w))
    assert np.linalg.matrix_rank(xa) == 21
    solution = np.linalg.lstsq(xa * sw[:, None], np.array(y) * sw, rcond=None)[0]
    return np.asarray(solution, dtype=np.float64)


def test_lambda_limits_least_squares_and_zero() -> None:
    st, sp_ = _toy()
    d = build_design(st, sp_, group="g", covariate="c")
    ls = _dense_least_squares(st, sp_)
    near = fit_rapm(d, 1e-9)
    got = np.concatenate(
        [
            near.players["offence"].to_numpy(),
            near.players["defence"].to_numpy(),
            near.home.to_numpy(),
            near.intercepts.to_numpy(),
            [near.covariate or 0.0],
        ]
    )
    np.testing.assert_allclose(got, ls, atol=1e-6)
    far = fit_rapm(d, 1e12)
    assert far.players[["offence", "defence"]].abs().to_numpy().max() < 1e-6


def test_bootstrap_standard_errors_shrink_with_more_games() -> None:
    # Near-unpenalised (lam = 30 minutes): a ridge SE is sqrt(n) * sigma / (n + lam)
    # per player, which only falls with the minutes n once n is well above lam.
    se = []
    for n_seasons in (1, 2, 4):
        lg = simulate_league(
            np.random.default_rng(5), n_seasons=n_seasons, offence_sd=SD, defence_sd=SD
        )
        d = build_design(lg.stints, lg.stint_players)
        boot = bootstrap_rapm(d, 30.0, np.random.default_rng(6), n_boot=20)
        assert boot.replicates["offence"].shape == (20, d.n_players)
        assert boot.halves.a["offence"].shape == (20, d.n_players)
        se.append(boot.se.median())
    assert (se[1] < se[0]).all() and (se[2] < se[1]).all()
    assert (se[2] < 0.6 * se[0]).all()  # 4x the games: sqrt(1/4) = 0.5 unpenalised


def test_reliability_grows_with_seasons(six_seasons: Fitted) -> None:
    lg2 = simulate_league(
        np.random.default_rng(0),
        n_teams=20,
        squad_size=20,
        n_seasons=2,
        offence_sd=SD,
        defence_sd=SD,
    )
    d2 = build_design(_with_season(lg2), lg2.stint_players, group="season")
    d6 = six_seasons.design
    comps = ["offence", "defence"]

    def rel(d: Design, lam: float) -> pd.Series:
        halves = split_halves(d, lam, np.random.default_rng(2), n_repeats=3)
        return pd.Series(split_half_reliability(halves, min_minutes=500).loc["all", comps])

    # Near-unpenalised, split-half reliability grows with the data.
    assert (rel(d6, 1e3) > rel(d2, 1e3) + 0.05).all()

    # At the CV lam, teammate leakage (stable across halves) props up the short
    # league's split-half; corrected by the attribution factor it grows again.
    def own(d: Design) -> pd.Series:
        att = attribution_factor(d, 1e4, np.random.default_rng(3), min_minutes=500, n_sims=2)
        return rel(d, 1e4) * att.loc["all", comps]

    assert (own(d6) > own(d2) + 0.1).all()


def test_split_half_times_attribution_is_the_squared_correlation_with_truth(
    six_seasons: Fitted,
) -> None:
    d, lam = six_seasons.design, six_seasons.lam
    fit = fit_rapm(d, lam)
    truth = six_seasons.league.truth.set_index("player_id").loc[fit.players.index]
    halves = split_halves(d, lam, np.random.default_rng(3), n_repeats=5)
    rel = split_half_reliability(halves, min_minutes=1000)
    att = attribution_factor(d, lam, np.random.default_rng(4), min_minutes=1000, n_sims=3)
    for comp in ("offence", "defence"):
        validity = np.corrcoef(fit.players[comp], truth[comp])[0, 1] ** 2
        split_half, att_c = rel[comp].to_dict()["all"], att[comp].to_dict()["all"]
        assert att_c < 0.95  # teammates leak into each other's estimates
        # Over 16 larger leagues split-half x attribution ran +0.02 to +0.035 above
        # the squared correlation with truth, whose own SD is ~0.025.
        assert abs(split_half * att_c - validity) < min(0.1, abs(split_half - validity))


def test_season_estimates_equal_pooled_when_season_penalty_is_infinite(
    six_seasons: Fitted,
) -> None:
    d = six_seasons.design
    pooled = fit_rapm(d, six_seasons.lam)
    season = six_seasons.stints.set_index("stint_id")["season"]
    by_season = fit_rapm_by_season(d, pooled, season, 1e12)
    played = six_seasons.league.player_match.groupby(["player_id", "season"]).size()
    assert set(by_season.index) == set(played.index)
    base = pooled.players.loc[by_season.index.get_level_values("player_id")]
    for comp in ("offence", "defence", "total"):
        np.testing.assert_allclose(by_season[comp].to_numpy(), base[comp].to_numpy(), atol=1e-6)
    loose = fit_rapm_by_season(d, pooled, season, 1e3)
    assert (loose["offence"].to_numpy() - base["offence"].to_numpy()).std() > 0.01


def test_low_minute_players_share_one_replacement_column() -> None:
    lg = simulate_league(np.random.default_rng(8), n_seasons=1)
    minutes = lg.player_match.groupby("player_id")["minutes"].sum()
    cut = float(minutes.quantile(0.3))
    d = build_design(lg.stints, lg.stint_players, min_minutes=cut)
    fit = fit_rapm(d, 1e4)
    kept = minutes.index[minutes >= cut]
    assert set(fit.players.index) == set(kept) | {REPLACEMENT}
    assert fit.players.loc[REPLACEMENT, "minutes"] == minutes[minutes < cut].sum()


def test_generic_response_columns_and_covariate_sign() -> None:
    """Adding gamma * c per 90 to home goals and removing it from away goals
    must move the covariate coefficient by exactly gamma and nothing else."""
    lg = simulate_league(np.random.default_rng(9))
    st = lg.stints.assign(c=np.random.default_rng(10).normal(size=len(lg.stints)))
    shift = 0.25 * st["c"] * st["minutes"] / 90
    st = st.assign(home_x=st["home_goals"] + shift, away_x=st["away_goals"] - shift)
    base = fit_rapm(build_design(st, lg.stint_players, covariate="c"), 1e4)
    moved = fit_rapm(
        build_design(st, lg.stint_players, home_col="home_x", away_col="away_x", covariate="c"),
        1e4,
    )
    assert base.covariate is not None and moved.covariate is not None
    assert moved.covariate - base.covariate == pytest.approx(0.25, abs=1e-6)
    pd.testing.assert_frame_equal(moved.players, base.players, atol=1e-6)


def test_bootstrap_counts_reproduce_each_replicate() -> None:
    lg = simulate_league(np.random.default_rng(12), offence_sd=SD, defence_sd=SD)
    d = build_design(lg.stints, lg.stint_players)
    boot = bootstrap_rapm(d, 1e4, np.random.default_rng(13), n_boot=3)
    n_games = len(d.game_ids)
    assert boot.counts.shape == (3, n_games)
    assert (boot.counts.sum(axis=1) == n_games).all()
    # The halves of a replicate split its own draw.
    np.testing.assert_array_equal(
        boot.halves.games_a.to_numpy() + boot.halves.games_b.to_numpy(), boot.counts
    )
    for b in range(3):
        refit = fit_rapm(replace(d, w=d.w * boot.counts[b][d.game]), 1e4)
        for comp in ("offence", "defence", "total"):
            np.testing.assert_allclose(
                refit.players[comp].to_numpy(), boot.replicates[comp].iloc[b].to_numpy(), atol=1e-9
            )


def test_season_game_weights_act_as_duplicated_games(six_seasons: Fitted) -> None:
    d, st, lg = six_seasons.design, six_seasons.stints, six_seasons.league
    pooled = fit_rapm(d, six_seasons.lam)
    season = st.set_index("stint_id")["season"]
    plain = fit_rapm_by_season(d, pooled, season, 1e4)
    ones = fit_rapm_by_season(d, pooled, season, 1e4, weights=np.ones(len(d.game_ids)))
    pd.testing.assert_frame_equal(ones, plain)

    weights = np.random.default_rng(14).integers(0, 3, size=len(d.game_ids))
    weighted = fit_rapm_by_season(d, pooled, season, 1e4, weights=weights)
    # The same data written out: weight-0 games dropped, weight-2 games twice.
    w_of = pd.Series(weights, index=d.game_ids)
    copies, players = [], []
    for k in (1, 2):
        keep = st[st["game_id"].map(w_of) >= k]
        off = 10**9 * (k - 1)
        copies.append(keep.assign(stint_id=keep["stint_id"] + off, game_id=keep["game_id"] + off))
        sp_ = lg.stint_players[lg.stint_players["stint_id"].isin(keep["stint_id"])]
        players.append(sp_.assign(stint_id=sp_["stint_id"] + off))
    st2, sp2 = pd.concat(copies, ignore_index=True), pd.concat(players, ignore_index=True)
    d2 = build_design(st2, sp2, group="season")
    assert d2.players.equals(d.players)
    dup = fit_rapm_by_season(d2, pooled, st2.set_index("stint_id")["season"], 1e4)
    assert set(dup.index) <= set(weighted.index)
    pd.testing.assert_frame_equal(weighted.loc[dup.index], dup, atol=1e-6)
    # A player-season seen only in weight-0 games keeps the pooled estimate.
    unseen = weighted.drop(dup.index)
    assert (unseen["minutes"] == 0).all()
    base = pooled.players.loc[unseen.index.get_level_values("player_id")]
    np.testing.assert_allclose(unseen["offence"].to_numpy(), base["offence"].to_numpy())


# --- league means (pre-registration A20 i, vii) ---------------------------------------

MEANS = (0.0, -0.02, -0.04, -0.06, -0.03)  # planted; the last league is weakly linked


@dataclass(frozen=True)
class LeagueFitted:
    league: League
    plain: Design  # no league columns
    design: Design  # league columns, the first league is the reference
    lam: float
    fit_plain: RapmFit
    fit: RapmFit


def _shares(lg: League) -> pd.DataFrame:
    """League shares from the truth table, without the first league (the reference)."""
    shares = lg.truth.set_index("player_id").filter(like="share_")
    return shares.rename(columns=lambda c: c.removeprefix("share_")).iloc[:, 1:]


def _league_designs(lg: League) -> tuple[Design, Design]:
    """The same stints without and with league columns, group = competition-season."""
    st = lg.stints.merge(lg.games[["game_id", "season", "competition_id"]], on="game_id")
    st["group"] = st["competition_id"] + " " + st["season"]
    return (
        build_design(st, lg.stint_players, group="group"),
        build_design(st, lg.stint_players, group="group", league_shares=_shares(lg)),
    )


@pytest.fixture(scope="module")
def five_leagues() -> LeagueFitted:
    """Four leagues linked by the cup and by 3% of players changing league each season,
    and a fifth with no cup games and one swap a season; each 10 clubs x 18 players x
    6 seasons, ~22k stints in all. lam is chosen without the league columns, as on the
    real data (A20 i)."""
    lg = simulate_league(
        np.random.default_rng(30),
        n_teams=10,
        squad_size=18,
        n_seasons=6,
        offence_sd=SD,
        defence_sd=SD,
        league_means=MEANS,
        cross_transfer_rate=0.03,
        weak_link_swaps=1,
    )
    plain, design = _league_designs(lg)
    lam, _ = cv_lambda(plain, LAMS, np.random.default_rng(31))
    return LeagueFitted(lg, plain, design, lam, fit_rapm(plain, lam), fit_rapm(design, lam))


def test_league_means_recover_the_planted_levels(five_leagues: LeagueFitted) -> None:
    linked = np.concatenate([[0.0], five_leagues.fit.leagues.to_numpy()[:3]])
    slope = float(np.polyfit(MEANS[:4], linked, 1)[0])
    # Each mean has a sampling SD of ~0.007 (the cup gives 96 games per pair of
    # leagues), so the slope over these four levels has SD 0.13: over 16 other seeds
    # it averaged 0.97. Here it is 1.14, and the largest error of a mean is 0.014.
    assert abs(slope - 1) < 0.3
    assert np.abs(linked - MEANS[:4]).max() < 0.025


def test_league_columns_improve_the_ranking_across_leagues(five_leagues: LeagueFitted) -> None:
    truth = five_leagues.league.truth.set_index("player_id")
    total = (truth["offence"] + truth["defence"])[truth["share_SYN5"] == 0]
    without, with_ = (
        float(f.players["total"].reindex(total.index).corr(total, method="spearman"))
        for f in (five_leagues.fit_plain, five_leagues.fit)
    )
    # Spearman with the true total, players of the four linked leagues pooled: 0.519
    # without the league columns, 0.542 with (16 other seeds: 0.547 and 0.559, higher
    # with the columns in every one). Over all five leagues it falls from 0.519 to
    # 0.410: the weakly linked league's own mean is noise (next test), which is why
    # A20 pools such leagues.
    assert with_ > without + 0.005


def test_weakly_linked_league_has_a_wide_bootstrap_spread(five_leagues: LeagueFitted) -> None:
    d, lam = five_leagues.design, five_leagues.lam
    boot = bootstrap_rapm(d, lam, np.random.default_rng(32), n_boot=12)
    sd = boot.leagues.std(ddof=1)
    # 10 movers and no cup game: bootstrap SD 0.093 (estimate +0.044, planted -0.03)
    # against at most 0.008 for a linked league (16 other seeds: ratio 6 to 21).
    assert sd["SYN5"] > 4 * sd.drop("SYN5").max()
    # A replicate is the fit on its own draw of games, league terms included.
    refit = fit_rapm(replace(d, w=d.w * boot.counts[0][d.game]), lam)
    np.testing.assert_allclose(refit.leagues.to_numpy(), boot.leagues.iloc[0].to_numpy(), atol=1e-9)
    for comp in ("offence", "defence", "total"):
        np.testing.assert_allclose(
            refit.players[comp].to_numpy(), boot.replicates[comp].iloc[0].to_numpy(), atol=1e-9
        )


def test_league_columns_change_nothing_when_all_planted_means_are_zero() -> None:
    lg = simulate_league(
        np.random.default_rng(33),
        n_teams=10,
        squad_size=18,
        n_seasons=6,
        offence_sd=SD,
        defence_sd=SD,
        league_means=(0.0, 0.0, 0.0, 0.0),
        cross_transfer_rate=0.03,
    )
    plain, design = _league_designs(lg)
    lam, _ = cv_lambda(plain, LAMS, np.random.default_rng(34))
    without, with_ = fit_rapm(plain, lam), fit_rapm(design, lam)
    # The three free means have SD ~0.010 around 0 (10 other seeds); here the largest
    # is 0.014 in absolute value and the two rankings correlate at 0.997 (Spearman
    # with the true total: 0.515 without the columns, 0.513 with).
    assert with_.leagues.abs().max() < 0.03
    rho = without.players["total"].corr(with_.players["total"], method="spearman")
    assert rho > 0.99


def test_league_mean_takes_a_planted_shift_and_reaches_each_player_by_his_share() -> None:
    """Adding g * z per 90 to home goals and removing it from away goals, z = the
    SYN2 shares on the pitch (home minus away), must move the SYN2 mean by exactly
    g, each player's offence and defence by g times his share, and nothing else."""
    lg = simulate_league(
        np.random.default_rng(35),
        n_seasons=3,
        league_means=(0.0, 0.0, 0.0),
        cross_transfer_rate=0.1,
    )
    shares = _shares(lg)
    on = lg.stint_players.join(shares, on="player_id")
    signed = on["SYN2"] * np.where(on["side"] == "home", 1.0, -1.0)
    z = signed.groupby(on["stint_id"]).sum().reindex(lg.stints["stint_id"]).to_numpy()
    shift = 0.25 * z * lg.stints["minutes"] / 90
    st = lg.stints.assign(
        home_x=lg.stints["home_goals"] + shift, away_x=lg.stints["away_goals"] - shift
    )
    minutes = lg.player_match.groupby("player_id")["minutes"].sum()
    cut = float(minutes.quantile(0.2))  # the lowest fifth share the replacement column
    base = fit_rapm(build_design(st, lg.stint_players, league_shares=shares, min_minutes=cut), 1e4)
    moved = fit_rapm(
        build_design(
            st,
            lg.stint_players,
            home_col="home_x",
            away_col="away_x",
            league_shares=shares,
            min_minutes=cut,
        ),
        1e4,
    )
    np.testing.assert_allclose((moved.leagues - base.leagues).to_numpy(), [0.25, 0.0], atol=1e-6)
    share = shares["SYN2"].reindex(base.players.index, fill_value=0.0)  # replacement: none
    assert REPLACEMENT in share.index and (share > 0).sum() > 100
    for comp, times in (("offence", 1), ("defence", 1), ("total", 2)):
        np.testing.assert_allclose(
            (moved.players[comp] - base.players[comp]).to_numpy(),
            0.25 * times * share.to_numpy(),
            atol=1e-6,
        )
    with pytest.raises(ValueError, match="without league shares"):
        build_design(st, lg.stint_players, league_shares=shares.iloc[1:])


def test_season_estimates_keep_the_pooled_league_means(five_leagues: LeagueFitted) -> None:
    d, pooled, lg = five_leagues.design, five_leagues.fit, five_leagues.league
    season = _with_season(lg).set_index("stint_id")["season"]
    by_season = fit_rapm_by_season(d, pooled, season, 1e4)
    # The same fit written without league columns: their pooled part goes into the
    # response, the prior is the pooled deviation, the league term is added back.
    means = pooled.leagues.to_numpy()
    term = pd.Series(d.shares @ means, index=d.players)
    plain = replace(five_leagues.plain, y=d.y - d.x[:, -len(means) :] @ means)
    prior = replace(pooled, players=pooled.players[["offence", "defence"]].sub(term, axis=0))
    expected = fit_rapm_by_season(
        plain, replace(prior, leagues=pd.Series(dtype=float)), season, 1e4
    )
    add = term.reindex(by_season.index.get_level_values("player_id")).to_numpy()
    assert np.abs(add).max() > 0.05 and by_season.index.equals(expected.index)
    for comp, times in (("offence", 1), ("defence", 1), ("total", 2)):
        np.testing.assert_allclose(
            by_season[comp].to_numpy(), expected[comp].to_numpy() + times * add, atol=1e-9
        )
    # With an infinite season penalty every player-season is the pooled estimate.
    fixed = fit_rapm_by_season(d, pooled, season, 1e12)
    base = pooled.players.loc[fixed.index.get_level_values("player_id")]
    np.testing.assert_allclose(fixed["total"].to_numpy(), base["total"].to_numpy(), atol=1e-6)


def test_ageing_movers_make_a_league_linked_by_movers_look_stronger() -> None:
    """Direction only (A20 vii). Two leagues linked by ten swaps a season and no cup;
    players lose 0.02 per year above the peak age and the older ones move down. The
    model's player effects are constant, so a mover down is credited in the weaker
    league with what he was in the stronger one. Without sampling noise: the weak
    league's mean fitted to the expected goals with ageing, against the same fit when
    every player has his minutes-weighted average effect in every season."""
    slope = 0.02
    lg = simulate_league(
        np.random.default_rng(36),
        n_teams=10,
        squad_size=18,
        n_seasons=6,
        offence_sd=SD,
        defence_sd=SD,
        league_means=(0.0, -0.05),
        weak_link_swaps=10,
        age_slope=slope,
    )
    st = _with_season(lg)
    d = build_design(st, lg.stint_players, league_shares=_shares(lg))
    p = d.n_players
    truth = lg.truth.set_index("player_id").loc[d.players]
    year = np.tile(st["season"].str[:4].astype(int).to_numpy() - 2020, 2)
    years = np.arange(6)
    fade = slope * np.maximum(0, truth["age"].to_numpy()[:, None] + years - AGE_PEAK)
    base = np.concatenate([truth["offence"].to_numpy(), truth["defence"].to_numpy()])
    ageing, minutes = np.zeros(len(d.y)), np.zeros((p, len(years)))
    for s in years:
        rows = np.flatnonzero(year == s)
        ageing[rows] = d.x[rows, : 2 * p] @ (base - np.tile(fade[:, s], 2))
        minutes[:, s] = d.x[rows, :p].T @ d.w[rows]
    average = (fade * minutes).sum(axis=1) / minutes.sum(axis=1)
    constant = d.x[:, : 2 * p] @ (base - np.tile(average, 2))
    with_ageing, without = (
        float(fit_rapm(replace(d, y=np.asarray(y)), 1e4).leagues["SYN2"])
        for y in (ageing, constant)
    )
    # -0.058 with ageing against -0.076 without: the gap is understated (8 other
    # seeds: by 0.016 to 0.028 at a gap of about 0.06). With a cup carrying the link
    # the same comparison moves a mean by less than 0.003, the other way.
    assert with_ageing > without + 0.005


def test_reference_league_is_only_an_origin_when_the_sides_are_equal() -> None:
    """With a player short on one side the league columns sum to the difference in
    players, so the mean fixed at 0 is a constraint: unless that difference is the
    covariate, the free means take part of the extra player's effect. Without noise
    and without player effects: planted means and 0.9 goals per 90 per extra player
    (the real data's value), one side a player short in every tenth stint."""
    lg = simulate_league(
        np.random.default_rng(37),
        n_seasons=3,
        league_means=(0.0, 0.0, 0.0),
        cross_transfer_rate=0.1,
    )
    shares, on = _shares(lg), lg.stint_players
    first = ~on.duplicated(["stint_id", "side"])
    short = np.where(on["side"] == "home", on["stint_id"] % 20 == 0, on["stint_id"] % 20 == 10)
    on = on[~(first & short)]
    sign = np.where(on["side"] == "home", 1.0, -1.0)

    def home_minus_away(values: NDArray[np.float64]) -> NDArray[np.float64]:
        sums = pd.Series(values * sign).groupby(on["stint_id"].to_numpy()).sum()
        return np.asarray(sums.reindex(lg.stints["stint_id"]), dtype=np.float64)

    men = home_minus_away(np.ones(len(on)))
    planted = {"SYN2": -0.03, "SYN3": -0.06}
    rate = 0.9 * men + sum(
        mean * home_minus_away(shares[league].reindex(on["player_id"]).to_numpy())
        for league, mean in planted.items()
    )
    goals = rate * lg.stints["minutes"] / 90
    st = lg.stints.assign(men=men, home_x=goals, away_x=-goals)
    with_men, without = (
        fit_rapm(
            build_design(
                st, on, home_col="home_x", away_col="away_x", covariate=c, league_shares=shares
            ),
            1e4,
        )
        for c in ("men", None)
    )
    np.testing.assert_allclose(with_men.leagues.to_numpy(), list(planted.values()), atol=1e-6)
    assert with_men.covariate == pytest.approx(0.9, abs=1e-6)
    # Without the covariate the free means come out at -0.018 and -0.046: too high by
    # 0.012 and 0.014 here (10% of the minutes a player short), by 0.06 on the real data.
    assert (without.leagues.to_numpy() > np.array(list(planted.values())) + 0.005).all()
