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

Outputs in results/players/rapm_league/ (rapm) and results/players/rapm_league_men/
(rapm_men): league_shares.csv, league_means.csv, rapm_pooled.csv, rapm_season.csv,
pool_rapm.csv, summary.json ({"value", "description"}), and bootstrap_pooled.npz and
bootstrap_season.npz in the format of run_impact.py's (offence and defence include the
league term), the first with two more arrays: leagues and league_means[B, leagues].
Stage timings in results/players/rapm_league/timings.json.

No random numbers are drawn: the only resampling reuses the primary's game counts.

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
from _league_factors_stage import stage_factors
from run_impact import (
    LEVEL,
    MIN_MINUTES,
    RAPM_DIR,
    data,
    lam,
    lam_season,
    read_json,
    record,
    season_of_stints,
    to_csv,
    write_json,
)

from footy.players.paths import PROCESSED, RESULTS
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


@functools.cache
def panel() -> pd.DataFrame:
    columns = ["player_id", "competition_id", "scope", "club_id", "minutes"]
    return pd.read_csv(PROCESSED / "panel_player_match.csv.gz", usecols=columns)


def league_minutes() -> pd.DataFrame:
    """League-scope minutes over the window, player x league."""
    league = panel()[panel()["scope"] == "league"]
    return league.groupby(["player_id", "competition_id"])["minutes"].sum().unstack(fill_value=0)


def league_links(minutes: pd.DataFrame) -> pd.DataFrame:
    """The two counts of the A20 rule per league and whether it has its own mean."""
    games = pd.read_csv(
        PROCESSED / "tm_games.csv.gz",
        usecols=["game_id", "competition_id", "competition_type", "home_club_id", "away_club_id"],
    )
    domestic = games[games["competition_type"] == "domestic_league"]
    club_league = pd.concat(
        [
            domestic[[f"{side}_club_id", "competition_id"]].set_axis(["club", "league"], axis=1)
            for side in ("home", "away")
        ]
    ).drop_duplicates()
    if club_league["club"].duplicated().any():
        raise ValueError("a club in two covered leagues")
    league_of = club_league.set_index("club")["league"]
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


STAGES: dict[str, Callable[[], None]] = {
    "rapm": stage_rapm,
    "rapm_men": stage_rapm_men,
    "factors": stage_factors,
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
