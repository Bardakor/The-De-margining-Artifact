"""League-strength-adjusted analysis on the real data (pre-registration A20, post hoc).

Stages:

- rapm (A20 i): the primary RAPM (run_impact.py: same stints, competition-season groups,
  900-minute columns, lambda and lambda_season read from results/players/rapm/) with one
  unpenalised mean per league, the Premier League fixed at 0. A player's league shares are
  his league-scope minutes of the panel over the window; a stint player without any is in
  the pseudo-league OTHER, which has its own mean. A league has its own mean with at least
  100 stint games against clubs of other covered leagues or at least 150 players with 450
  minutes in it and in another league; the other leagues share the mean REST. Pooled fit,
  per-season fits (pooled means kept), and the primary's 200 bootstrap draws of games
  (results/players/rapm/bootstrap_pooled.npz), each refitted pooled and per season.
- rapm_men (A22, the fit the league-adjusted ranking uses): the same with one more
  unpenalised column, the players on the pitch, home side minus away side. Where the
  sides are unequal (5% of the minutes) the league columns no longer sum to 0, so fixing
  one league at 0 is a
  constraint and not a normalisation: every other league's mean then also carries the
  effect of an extra player. This column takes that effect.
- factors (A20 ii, A21): box-score league factors from players who change league, with
  and without the team-context control (scripts/players/_league_factors_stage.py);
  results/players/league/.
- validate (A23 i, descriptive): the league means of rapm_men and the factors of the
  ``context`` model against club results, which use no player and no mover, and against
  UEFA's association coefficients (``stage_validate``); levels.csv (one row per league),
  summary.json and levels_bootstrap.npz (the club-results levels on the primary's game
  resamples, for A24) in results/players/league/validation/.

Outputs in results/players/rapm_league/ (rapm) and results/players/rapm_league_men/
(rapm_men): league_shares.csv, league_means.csv, rapm_pooled.csv, rapm_season.csv,
pool_rapm.csv, summary.json ({"value", "description"}), and bootstrap_pooled.npz and
bootstrap_season.npz in the format of run_impact.py's (offence and defence include the
league term), the first with two more arrays: leagues and league_means[B, leagues].
Stage timings in results/players/rapm_league/timings.json.

The only resampling reuses the primary's game counts. validate alone draws random numbers,
for the games the primary does not have, from task "validate" (run_impact.rng).

Usage:
    uv run python scripts/players/run_league.py                  # every stage
    uv run python scripts/players/run_league.py --stage rapm     # one stage
"""

from __future__ import annotations

import argparse
import functools
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _league_factors_stage import LEAGUE_DIR, stage_factors
from run_impact import (
    LEVEL,
    MIN_MINUTES,
    RAPM_DIR,
    data,
    lam,
    lam_season,
    read_json,
    record,
    rng,
    season_of_stints,
    to_csv,
    write_json,
)

from footy.players.elo import attack_defence, least_squares_ratings
from footy.players.paths import PAPER, PROCESSED, RESULTS
from footy.players.rapm import COMPONENTS, REPLACEMENT, build_design, fit_rapm, fit_rapm_by_season

REFERENCE, REST, OTHER = "GB1", "REST", "OTHER"
MIN_GAMES = 100  # stint games against clubs of other covered leagues
MIN_LINKING = 150  # players with LINK_MINUTES in the league and in another one
LINK_MINUTES = 450
CLUBS = {  # Transfermarkt club ids
    "Bayern Munich": 27,
    "Manchester City": 281,
    "Liverpool": 31,
    "Arsenal": 11,
    "Real Madrid": 418,
    "Barcelona": 131,
    "Paris Saint-Germain": 583,
    "Inter": 46,
    "Porto": 720,
    "Benfica": 294,
    "Sporting CP": 336,
}
RAPM_LEAGUE_DIR = RESULTS / "rapm_league"
RAPM_MEN_DIR = RESULTS / "rapm_league_men"
VALIDATION_DIR = LEAGUE_DIR / "validation"
ON_PITCH = 22  # an average team of league L is 22 mu_L goals per 90 from the reference's


@functools.cache
def panel() -> pd.DataFrame:
    columns = ["player_id", "competition_id", "scope", "club_id", "minutes"]
    return pd.read_csv(PROCESSED / "panel_player_match.csv.gz", usecols=columns)


def league_minutes() -> pd.DataFrame:
    """League-scope minutes over the window, player x league."""
    league = panel()[panel()["scope"] == "league"]
    return league.groupby(["player_id", "competition_id"])["minutes"].sum().unstack(fill_value=0)


def club_seasons(games: pd.DataFrame) -> pd.DataFrame:
    """club, season, league: the covered league a club plays league games in that season."""
    domestic = games[games["competition_type"] == "domestic_league"]
    out = pd.concat(
        [
            domestic[[f"{side}_club_id", "season", "competition_id"]].set_axis(
                ["club", "season", "league"], axis=1
            )
            for side in ("home", "away")
        ]
    ).drop_duplicates()
    if out.drop_duplicates(["club", "league"])["club"].duplicated().any():
        raise ValueError("a club in two covered leagues")
    return out


def league_links(minutes: pd.DataFrame) -> pd.DataFrame:
    """The two counts of the A20 rule per league and whether it has its own mean."""
    games = pd.read_csv(
        PROCESSED / "tm_games.csv.gz",
        usecols=[
            "game_id",
            "competition_id",
            "competition_type",
            "season",
            "home_club_id",
            "away_club_id",
        ],
    )
    league_of = club_seasons(games).drop_duplicates("club").set_index("club")["league"]
    played = games[games["game_id"].isin(data().stints["game_id"].unique())]
    home, away = (played[f"{side}_club_id"].map(league_of) for side in ("home", "away"))
    cross = home.notna() & away.notna() & (home != away)
    enough = minutes >= LINK_MINUTES
    out = pd.DataFrame(
        {
            "games_other_leagues": pd.concat([home[cross], away[cross]]).value_counts(),
            "linking_players": enough[enough.sum(axis=1) >= 2].sum(),
        }
    )
    out = out.reindex(minutes.columns).fillna(0).astype(int).rename_axis("league")
    out["own_mean"] = (out["games_other_leagues"] >= MIN_GAMES) | (
        out["linking_players"] >= MIN_LINKING
    )
    return out


def club_means(totals: pd.DataFrame) -> pd.DataFrame:
    """Mean of each column of ``totals`` (index player_id) over a club's players, weighted
    by their panel minutes for the club (every competition), players with own columns."""
    rows = panel()[panel()["club_id"].isin(CLUBS.values())]
    minutes = rows.groupby(["club_id", "player_id"])["minutes"].sum().reset_index()
    minutes = minutes[minutes["player_id"].isin(totals.index)]
    weighted = totals.reindex(minutes["player_id"]).mul(minutes["minutes"].to_numpy(), axis=0)
    sums = weighted.groupby(minutes["club_id"].to_numpy()).sum()
    out = sums.div(minutes.groupby("club_id")["minutes"].sum(), axis=0)
    return out.loc[list(CLUBS.values())].set_axis(list(CLUBS), axis=0).rename_axis("club")


def run(folder: Path, covariate: str | None) -> None:
    dd, lam_, ls = data(), lam(), lam_season()
    lo, hi = (1 - LEVEL) / 2, (1 + LEVEL) / 2

    # league shares and the A20 rule
    minutes = league_minutes()
    links = league_links(minutes)
    ids = pd.Index(np.sort(dd.stint_players["player_id"].unique()), name="player_id")
    minutes = minutes.reindex(ids, fill_value=0)
    minutes[OTHER] = (minutes.sum(axis=1) == 0).astype(int)
    shares = minutes.div(minutes.sum(axis=1), axis=0)
    column = pd.Series(np.where(links["own_mean"], links.index, REST), index=links.index)
    column[OTHER] = OTHER
    by_column = shares.T.groupby(column).sum().T
    long = shares.rename_axis(columns="league").stack().rename("share").reset_index()
    to_csv(long[long["share"] > 0], folder / "league_shares.csv")
    own = links.index[links["own_mean"]]
    print(f"leagues with their own mean ({len(own)}): {', '.join(own)}")
    print(f"sharing {REST}: {', '.join(links.index[~links['own_mean']])}")
    print(links.sort_values("games_other_leagues", ascending=False).to_string())
    n_other = int((shares[OTHER] > 0).sum())
    print(f"stint players {len(ids):,}; without league minutes ({OTHER}) {n_other:,}", flush=True)

    # design, pooled and per-season fits
    stints = dd.stints.assign(men=dd.stints["home_n"] - dd.stints["away_n"])
    d = build_design(
        stints,
        dd.stint_players,
        group="group",
        covariate=covariate,
        league_shares=by_column.drop(columns=REFERENCE),
        min_minutes=MIN_MINUTES,
    )
    primary = np.load(RAPM_DIR / "bootstrap_pooled.npz")
    same = np.array_equal(primary["game_ids"], d.game_ids.to_numpy())
    if not same or not np.array_equal(primary["players"], d.players.to_numpy()):
        raise ValueError("games or players differ from the primary bootstrap")
    counts = primary["counts"]
    n_boot = len(counts)
    season = season_of_stints()
    fit = fit_rapm(d, lam_)
    by_season = fit_rapm_by_season(d, fit, season, ls)
    out = by_season.reset_index()
    out.insert(2, "position_group", dd.positions.reindex(out["player_id"]).to_numpy())
    out["position_group"] = out["position_group"].fillna("replacement")
    to_csv(out, folder / "rapm_season.csv")

    # the primary's bootstrap draws, refitted pooled and per season
    pooled = {c: np.empty((n_boot, d.n_players)) for c in COMPONENTS[:2]}
    means = np.empty((n_boot, len(d.leagues)))
    men = []
    seasonal = {
        c: np.empty((n_boot, len(by_season)), np.float32) for c in ("minutes", *COMPONENTS[:2])
    }
    t0 = time.perf_counter()
    for b in range(n_boot):
        fit_b = fit_rapm(replace(d, w=d.w * counts[b][d.game]), lam_)
        for c in pooled:
            pooled[c][b] = fit_b.players[c].to_numpy()
        means[b] = fit_b.leagues.to_numpy()
        men.append(fit_b.covariate)
        est = fit_rapm_by_season(d, fit_b, season, ls, weights=counts[b])
        if not est.index.equals(by_season.index):
            raise ValueError("per-season keys differ between replicates")
        for c in seasonal:
            seasonal[c][b] = est[c].to_numpy(np.float32)
        if (b + 1) % 20 == 0:
            print(f"  replicate {b + 1}/{n_boot} ({time.perf_counter() - t0:.0f} s)", flush=True)
    np.savez_compressed(
        folder / "bootstrap_pooled.npz",
        game_ids=primary["game_ids"],
        counts=counts,
        players=d.players.to_numpy(np.int64),
        offence=pooled["offence"].astype(np.float32),
        defence=pooled["defence"].astype(np.float32),
        lam=lam_,
        leagues=d.leagues.to_numpy().astype(str),
        league_means=means,
    )
    np.savez_compressed(
        folder / "bootstrap_season.npz",
        player_id=by_season.index.get_level_values("player_id").to_numpy(np.int64),
        season=by_season.index.get_level_values("season").to_numpy().astype(str),
        lam_season=ls,
        **seasonal,
    )

    # league means
    reps = pd.DataFrame(means, columns=d.leagues)
    by_mean = pd.DataFrame(
        {
            "mu": fit.leagues,
            "se": reps.std(ddof=1),
            "ci_low": reps.quantile(lo),
            "ci_high": reps.quantile(hi),
        }
    )
    by_mean.loc[REFERENCE] = 0.0
    table = links.reindex(column.index).assign(column=column).join(by_mean, on="column")
    table["own_mean"] = table["column"] == table.index
    table = table.sort_values("mu", ascending=False).rename_axis("league").reset_index()
    to_csv(table, folder / "league_means.csv")
    print(table.round(4).to_string(index=False), flush=True)

    # players
    term = d.shares @ fit.leagues.to_numpy()
    players = fit.players.copy()
    players.insert(1, "position_group", dd.positions.reindex(players.index).fillna("replacement"))
    total = pooled["offence"] + pooled["defence"]
    for c, r in (*pooled.items(), ("total", total)):
        players[f"se_{c}"] = r.std(axis=0, ddof=1)
    players["delta_offence"] = players["offence"] - term
    players["delta_defence"] = players["defence"] - term
    players["league_term"] = term
    to_csv(players.rename_axis("player_id").reset_index(), folder / "rapm_pooled.csv")

    # pool players, the primary beside
    before = pd.read_csv(RAPM_DIR / "pool_rapm.csv")
    pool = before[["player", "tm_player_id", "position_group", "is_goalkeeper"]].copy()
    at = d.players.get_indexer(pool["tm_player_id"])
    if (at < 0).any():
        raise ValueError("a pool player without own columns")
    pool["league_term"] = term[at]
    pool["total"] = players["total"].to_numpy()[at]
    pool["se_total"] = players["se_total"].to_numpy()[at]
    pool["total_ci_low"], pool["total_ci_high"] = np.quantile(total[:, at], [lo, hi], axis=0)
    pool["rank_outfield"] = (
        pool["total"].where(~pool["is_goalkeeper"]).rank(ascending=False, method="min")
    )
    shown = ["total", "se_total", "total_ci_low", "total_ci_high", "rank_outfield"]
    pool[[f"{c}_primary" for c in shown]] = before[shown]
    pool = pool.sort_values(["is_goalkeeper", "rank_outfield"])
    to_csv(pool, folder / "pool_rapm.csv")
    pd.set_option("display.width", 200)
    print(pool.head(20).round(3).to_string(index=False))

    # headline numbers
    both = players.drop(index=REPLACEMENT)[["total"]].join(
        pd.read_csv(RAPM_DIR / "fit_pooled.csv").set_index("player_id")["total"].rename("primary")
    )
    clubs = club_means(both)
    print(clubs.round(4).to_string())
    outfield = pool[~pool["is_goalkeeper"]]
    entries: dict[str, tuple[Any, str]] = {
        "lambda": (lam_, "Ridge penalty: the primary's (results/players/rapm/cv.json)"),
        "n_boot": (n_boot, "Bootstrap replicates: the primary's draws of games"),
        "n_leagues_own": (len(own), "Leagues with their own mean, the reference included"),
        "leagues_rest": (list(links.index[~links["own_mean"]]), f"Leagues sharing {REST}"),
        "n_players_other": (n_other, f"Stint players without league minutes ({OTHER})"),
        "spearman_total_all": (
            both["total"].corr(both["primary"], method="spearman"),
            "Spearman between primary and league-adjusted RAPM total, own-column players",
        ),
        "spearman_total_pool": (
            pool["total"].corr(pool["total_primary"], method="spearman"),
            "Same, pool players",
        ),
        "pool_top20": (
            outfield.nsmallest(20, "rank_outfield")["player"].tolist(),
            "Pool outfield players, top 20 by league-adjusted RAPM total",
        ),
        "pool_top20_primary": (
            outfield.nsmallest(20, "rank_outfield_primary")["player"].tolist(),
            "Same by primary RAPM total",
        ),
    }
    for league, row in by_mean.iterrows():
        entries[f"mu_{league}"] = (row["mu"], f"League mean of {league} (goals per 90, GB1 = 0)")
        entries[f"mu_se_{league}"] = (row["se"], f"Its bootstrap SE ({n_boot} game resamples)")
    for club, row in clubs.iterrows():
        what = f"minutes-weighted mean RAPM total of {club}'s players"
        entries[f"club_total_primary_{club}"] = (row["primary"], f"Primary: {what}")
        entries[f"club_total_league_{club}"] = (row["total"], f"League-adjusted: {what}")
    if fit.covariate is not None:
        entries["men"] = (fit.covariate, "Goals per 90 per extra player on the pitch")
        entries["men_se"] = (float(np.std(men, ddof=1)), "Its bootstrap SE")
    record(folder, entries)
    for key in ("spearman_total_all", "spearman_total_pool"):
        print(f"{key}: {entries[key][0]:.4f}")


def stage_rapm() -> None:
    run(RAPM_LEAGUE_DIR, None)


def stage_rapm_men() -> None:
    run(RAPM_MEN_DIR, "men")


def stage_validate() -> None:
    """A23 (i): league levels from club results and from UEFA, beside 22 mu and the factors.

    Games: the club games of tm_games.csv.gz (the window) between two clubs of covered
    leagues, a club's league being the one it plays league games in (it has one over the
    window; ``club_seasons``). One rating per club over the window
    (``footy.players.elo``); a league's value is the mean over its club-seasons, so a club
    counts once per season it plays in the league.

    - Club-rating level: least squares of goal difference on home rating - away rating + a
      home term; the league's mean rating minus the reference's. Goals per game, the scale
      of 22 mu (goals per 90).
    - Goal multiplier from club results: Poisson model of goals scored, attack - defence
      + home. Against club j every attack scores exp(-defence_j) times its rate against a
      defence of 0, so against the clubs of league L it scores c_L = mean of exp(-defence_j)
      over L's club-seasons. Multiplier of L = c_reference / c_L: what a goal against L's
      defences is worth in goals against the reference's. The movers' exp(-gamma_L) is that
      worth for a player who changes league, and so also changes team-mates.
    - Head-to-head: the games between a club of the league and one of the reference; goal
      difference per game from the league's side, SE = SD / sqrt(games). Not adjusted for
      which clubs meet or where.
    - UEFA: mean of the association's season coefficients in
      paper-unmeasured/data/uefa_coefficients.csv over the seasons it has a value for.

    Choices A23 does not fix (Claude's calls):

    - A league is rated on club results with at least ``MIN_GAMES`` games against clubs of
      other covered leagues, the game count of the A20 rule on these games (A20's other
      count is of players). The others stay in the fit, keep their head-to-head and are
      left out of every comparison.
    - SEs: both fits on the game resamples of the rapm_men bootstrap (the primary's), so a
      replicate has its league means and its club levels from the same draw of games and
      the SE of club level - 22 mu is the SD of that difference over replicates. The two
      share the games between leagues: as if independent, that SE would be about 1.6 times
      larger. The games of the fit without stints, absent from those draws, are resampled
      among themselves. Against the movers' factors (a player bootstrap) the SE of the
      difference is taken as if independent.
    - A comparison takes the leagues with both values: own mean or own factor (a league
      sharing REST is left out), rated, UEFA member. The reference is one of them: fixing
      it at 0 or 1 only sets the origin.
    - Sensitivity: the same level from the games in which both clubs play league games in
      a covered league in the game's own season (no promoted or relegated club's cup game,
      no European game of a league before its first season in the data).

    levels_bootstrap.npz (for A24): leagues, rated, club_level[leagues] and
    club_level_boot[B, leagues] (league minus the reference, goals per game; NaN where not
    rated), and game_ids and counts[B, games] copied from
    results/players/rapm_league_men/bootstrap_pooled.npz. Row b of club_level_boot is the
    fit with the games weighted by row b of counts (the games without stints by their own
    draw), so it is the replicate of row b of that file's league_means; a reader checks it
    by comparing the two files' game_ids and counts.
    """
    columns = ["game_id", "competition_id", "competition_type", "season"]
    columns += ["home_club_id", "away_club_id", "home_goals", "away_goals"]
    games = pd.read_csv(PROCESSED / "tm_games.csv.gz", usecols=columns)
    seasons = club_seasons(games)
    share = pd.crosstab(seasons["league"], seasons["club"], normalize="index")
    leagues, clubs = share.index, share.columns
    games = games[
        (games["competition_type"] != "national_team_competition")
        & games["home_club_id"].isin(clubs)
        & games["away_club_id"].isin(clubs)
    ].reset_index(drop=True)
    league_of = seasons.drop_duplicates("club").set_index("club")["league"]
    home_league, away_league = (
        games[f"{side}_club_id"].map(league_of) for side in ("home", "away")
    )
    cross = home_league != away_league
    in_league = pd.MultiIndex.from_frame(seasons[["club", "season"]])
    same_season = np.logical_and(
        *(
            pd.MultiIndex.from_frame(games[[f"{side}_club_id", "season"]]).isin(in_league)
            for side in ("home", "away")
        )
    )

    def links(used: pd.Series) -> pd.Series:
        """Games against clubs of other covered leagues, per league."""
        sides = pd.concat([home_league[used], away_league[used]])
        return sides.value_counts().reindex(leagues, fill_value=0)

    # the two fits, on the games and on the game resamples of the rapm_men bootstrap
    home, away = (clubs.get_indexer(games[f"{side}_club_id"]) for side in ("home", "away"))
    home_goals, away_goals = games[["home_goals", "away_goals"]].to_numpy(np.float64).T
    of_league, ref, n, k = share.to_numpy(), leagues.get_loc(REFERENCE), len(games), len(leagues)

    def estimate(weights: np.ndarray) -> np.ndarray:
        """Per league the club-rating level, then the goal multiplier; then the home term."""
        rating, home_term = least_squares_ratings(
            home, away, home_goals - away_goals, weights, len(clubs)
        )
        defence = attack_defence(home, away, home_goals, away_goals, weights, len(clubs))[1]
        level, conceded = of_league @ rating, of_league @ np.exp(-defence)
        return np.r_[level - level[ref], conceded[ref] / conceded, home_term]

    point = estimate(np.ones(n))
    boot = np.load(RAPM_MEN_DIR / "bootstrap_pooled.npz")
    counts = boot["counts"]
    drawn = pd.Index(boot["game_ids"]).get_indexer(games["game_id"])
    rest = np.flatnonzero(drawn < 0)  # games without stints
    g = rng("validate")
    n_boot = len(counts)
    reps = np.empty((n_boot, len(point)))
    t0 = time.perf_counter()
    for b in range(n_boot):
        weights = counts[b][drawn].astype(np.float64)
        weights[rest] = np.bincount(g.integers(len(rest), size=len(rest)), minlength=len(rest))
        reps[b] = estimate(weights)
        if (b + 1) % 50 == 0:
            print(f"  replicate {b + 1}/{n_boot} ({time.perf_counter() - t0:.0f} s)", flush=True)
    se = reps.std(axis=0, ddof=1)
    mu_reps = pd.DataFrame(ON_PITCH * boot["league_means"], columns=boot["leagues"].astype(str))
    paired_se = (pd.DataFrame(reps[:, :k], columns=leagues) - mu_reps).std(ddof=1)

    # head-to-head with the reference, from the league's side
    versus = cross & ((home_league == REFERENCE) | (away_league == REFERENCE))
    hosts = away_league[versus] == REFERENCE
    margin = (games["home_goals"] - games["away_goals"])[versus]
    h2h = (
        margin.where(hosts, -margin)
        .groupby(home_league[versus].where(hosts, away_league[versus]))
        .agg(["size", "mean", "sem"])
    )

    mu = pd.read_csv(RAPM_MEN_DIR / "league_means.csv").set_index("league")
    factors = pd.read_csv(LEAGUE_DIR / "factors.csv")
    factors = factors[factors["model"] == "context"].set_index("league")
    uefa = pd.read_csv(PAPER / "data" / "uefa_coefficients.csv").set_index("league")
    n_links = links(cross)
    rated = n_links >= MIN_GAMES
    table = pd.DataFrame(
        {
            "own_mean": mu["own_mean"],
            "mu22": ON_PITCH * mu["mu"],
            "mu22_se": ON_PITCH * mu["se"],
            "games_other_leagues": n_links,
            "rated": rated,
            "club_level": pd.Series(point[:k], leagues).where(rated),
            "club_level_se": pd.Series(se[:k], leagues).where(rated),
            "club_level_minus_mu22_se": paired_se.where(rated),
            "h2h_games": h2h["size"].reindex(leagues, fill_value=0),
            "h2h_gd": h2h["mean"],
            "h2h_gd_se": h2h["sem"],
            "goal_multiplier_clubs": pd.Series(point[k:-1], leagues).where(rated),
            "goal_multiplier_clubs_se": pd.Series(se[k:-1], leagues).where(rated),
        },
        index=leagues,
    )
    metrics = ("goals", "np_goals")
    for metric in metrics:
        f = factors[factors["metric"] == metric]
        table[f"{metric}_multiplier_movers"] = f["multiplier"]
        table[f"{metric}_multiplier_movers_se"] = f["multiplier"] * f["se"]  # delta method
    table["own_factor"] = factors.groupby(level=0)["status"].first() != "pooled"
    table["uefa_mean"] = uefa.filter(like="/").mean(axis=1)
    table = table.sort_values("mu22", ascending=False)
    to_csv(table.reset_index(), VALIDATION_DIR / "levels.csv")
    np.savez_compressed(
        VALIDATION_DIR / "levels_bootstrap.npz",
        leagues=leagues.to_numpy().astype(str),
        rated=rated.to_numpy(),
        club_level=np.where(rated, point[:k], np.nan),
        club_level_boot=np.where(rated, reps[:, :k], np.nan),
        game_ids=boot["game_ids"],
        counts=counts,
    )
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(), flush=True)

    # agreement, each pair over the leagues with both values
    mu22, club = table["mu22"].where(table["own_mean"]), table["club_level"]
    uefa_mean = table["uefa_mean"]
    pairs = {  # a, b, SE of a - b and how it is obtained, what they are
        "mu_club": (
            club,
            mu22,
            table["club_level_minus_mu22_se"],
            "over the shared game resamples",
            "the club-rating level and 22 mu",
        ),
        "mu_uefa": (mu22, uefa_mean, None, "", "22 mu and the UEFA coefficient"),
        "club_uefa": (club, uefa_mean, None, "", "the club-rating level and the UEFA coefficient"),
    } | {
        f"multiplier_{metric}": (
            table["goal_multiplier_clubs"],
            table[f"{metric}_multiplier_movers"].where(table["own_factor"]),
            np.hypot(table["goal_multiplier_clubs_se"], table[f"{metric}_multiplier_movers_se"]),
            "as if the two were independent",
            f"the goal multiplier from club results and the movers' factor for {metric}",
        )
        for metric in metrics
    }
    entries: dict[str, tuple[Any, str]] = {
        "n_games": (n, "Club games between two clubs of covered leagues"),
        "n_games_other_leagues": (int(cross.sum()), "Of which between two leagues"),
        "n_clubs": (len(clubs), "Clubs rated"),
        "n_club_seasons": (len(seasons), "Club-seasons the league values are means over"),
        "n_boot": (n_boot, "Game resamples behind the SEs: the primary's draws"),
        "n_games_resampled_apart": (len(rest), "Games without stints, resampled among themselves"),
        "home_term": (point[-1], "Home term of the least-squares rating, goals per game"),
        "home_term_se": (se[-1], "Its bootstrap SE"),
        "leagues_rated": (
            list(leagues[rated]),
            f"Leagues rated on club results: at least {MIN_GAMES} games against clubs of other "
            "covered leagues",
        ),
        "leagues_not_rated": (
            n_links[~rated].sort_values(ascending=False).to_dict(),
            "Leagues not rated, with those games",
        ),
    }
    for key, (a, b, gap_se, how, what) in pairs.items():
        both = a.notna() & b.notna()
        entries[f"n_{key}"] = (int(both.sum()), f"Leagues with {what}")
        for method in ("pearson", "spearman"):
            entries[f"{method}_{key}"] = (
                a.corr(b, method=method),
                f"{method.capitalize()} correlation between {what}, {both.sum()} leagues",
            )
        if gap_se is None:
            continue
        gap = pd.DataFrame({"difference": a - b, "se": gap_se})[both]
        gap["z"] = gap["difference"] / gap["se"]  # the reference: 0 / 0, dropped below
        gap = gap.dropna().sort_values("z", key=np.abs, ascending=False)
        entries[f"mean_difference_{key}"] = (
            gap["difference"].mean(),
            f"Mean of {what.replace(' and ', ' minus ', 1)}, the reference left out",
        )
        entries[f"n_2se_{key}"] = (
            int((gap["z"].abs() >= 2).sum()),
            f"Leagues of the {len(gap)} where they differ by 2 SEs or more "
            f"(SE of the difference {how})",
        )
        entries[f"disagreements_{key}"] = (
            gap.head(5).reset_index().to_dict("records"),
            f"The five largest differences in SEs, {what.replace(' and ', ' minus ', 1)}",
        )
    entries["slope_mu_club"] = (
        (club * mu22).sum() / (mu22[club.notna()] ** 2).sum(),
        "Least-squares slope through the reference of the club-rating level on 22 mu",
    )

    # sensitivity: both clubs in a covered league in the game's own season
    stricter = pd.Series(estimate(same_season.astype(np.float64))[:k], leagues)
    kept = rated & (links(cross & same_season) >= MIN_GAMES)
    shift = (stricter - club)[kept]
    entries |= {
        "same_season_games": (
            int(same_season.sum()),
            "Sensitivity: games with both clubs in a covered league in the game's own season",
        ),
        "same_season_leagues": (
            int(kept.sum()),
            f"Rated leagues with {MIN_GAMES} games against other leagues among them too",
        ),
        "same_season_max_shift": (
            shift.abs().max(),
            f"Largest change of their club-rating level ({shift.abs().idxmax()})",
        ),
        "same_season_pearson": (
            stricter[kept].corr(club[kept]),
            "Pearson correlation between the two levels over them",
        ),
    }
    record(VALIDATION_DIR, entries)
    for key, (value, description) in entries.items():
        if not isinstance(value, (list, dict)):
            print(f"{key}: {value:.6g}  ({description})")


STAGES: dict[str, Callable[[], None]] = {
    "rapm": stage_rapm,
    "rapm_men": stage_rapm_men,
    "factors": stage_factors,
    "validate": stage_validate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--stage", choices=list(STAGES))
    which.add_argument("--from", dest="start", choices=list(STAGES))
    args = parser.parse_args()
    names = list(STAGES)
    if args.stage:
        names = [args.stage]
    elif args.start:
        names = names[names.index(args.start) :]
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        STAGES[name]()
        seconds = time.perf_counter() - t0
        timings = read_json(RAPM_LEAGUE_DIR / "timings.json")
        timings[name] = {"seconds": seconds, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(RAPM_LEAGUE_DIR / "timings.json", timings)
        print(f"=== {name}: {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
