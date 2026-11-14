"""Download the Transfermarkt snapshot, build the processed tables, print the checks.

Usage:
    uv run python scripts/players/acquire_transfermarkt.py

Design §4.1/§4.3. The download is skipped for files already cached. The summary
prints row counts, the stint reconciliation checks, the validation of stint-derived
appearances against Transfermarkt's own rows, and the Elo fit. The per competition x
season coverage goes to results/players/coverage/tm_coverage.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.players import stints, transfermarkt
from footy.players.paths import RESULTS

KEY = ["game_id", "player_id"]
POOL_COMPETITIONS = (
    "GB1",
    "ES1",
    "L1",
    "IT1",
    "FR1",
    "CL",
    "EL",
    "FIWC",
    "EURO",
    "COPA",
    "AFCN",
    "SA1",
    "MLS1",
    "KLUB",
)


def _in(rows: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        pd.MultiIndex.from_frame(rows[KEY]).isin(pd.MultiIndex.from_frame(other[KEY]))
    )


def main() -> None:
    print("download:", transfermarkt.download())
    t = transfermarkt.build()
    for name in (
        "tm_games",
        "tm_appearances",
        "tm_players",
        "tm_valuations",
        "tm_stints",
        "tm_stint_players",
        "team_elo",
    ):
        print(f"{name:18s} {len(t[name]):>9,d} rows")
    print("tm_appearances by source:", t["tm_appearances"]["source"].value_counts().to_dict())

    games, sel, st = t["tm_games"], t["stint_selection"], t["tm_stints"]
    complete = sel["complete"]
    print(
        f"\nlineups: {len(sel):,d} window games, {complete.sum():,d} with 11 starters a side "
        f"({complete.mean():.1%}), {(sel['home_starters'] + sel['away_starters'] == 0).sum():,d}"
        f" with no lineup; {(complete & ~sel['has_stints']).sum():,d} complete games dropped"
        " for a change at an unknown minute"
    )

    rec = stints.reconcile(st, games).merge(sel[["game_id", "unknown_minute_goals"]])
    bad = rec[~rec["match"]]
    diff = (bad["home_goals_stints"] + bad["away_goals_stints"]) - (
        bad["home_goals"] + bad["away_goals"]
    )
    category = np.select(
        [bad["unknown_minute_goals"] > 0, diff < 0, diff > 0],
        ["goal at unknown minute", "fewer goal events than score", "more goal events than score"],
        "same total, wrong side",
    )
    print(
        f"\n[1] goals reconcile in {rec['match'].mean():.2%} of {len(rec):,d} stint games "
        f"({rec['match'].sum():,d} / {complete.sum():,d} = "
        f"{rec['match'].sum() / complete.sum():.2%} of all complete-lineup games)"
    )
    print(pd.Series(category).value_counts().to_string())

    derived = t["tm_appearances"][t["tm_appearances"]["source"] == "stints"]
    tm_rows = t["tm_rows"]
    both = derived.merge(tm_rows, on=KEY, suffixes=("", "_tm"))
    gap = (both["minutes"] - both["minutes_tm"]).abs()
    print(f"\n[2] stint-derived vs Transfermarkt rows, {len(both):,d} shared (game, player):")
    print(f"    minutes within +-2: {(gap <= 2).mean():.2%} (within +-5: {(gap <= 5).mean():.2%})")
    for col in ("goals", "assists", "yellow", "red"):
        print(f"    {col:8s} equal: {(both[col] == both[f'{col}_tm']).mean():.3%}")
    tm_stint_games = tm_rows[tm_rows["game_id"].isin(st["game_id"])]
    only_tm = tm_stint_games[~_in(tm_stint_games, derived)]
    print(
        f"    Transfermarkt rows in stint games with no derived row: {len(only_tm):,d} "
        f"(their minutes: {only_tm['minutes'].value_counts().head(3).to_dict()})"
    )
    filled = derived[(derived["minutes"] >= 1) & ~_in(derived, tm_rows)]
    print(
        f"    derived player-games (minutes >= 1) with no Transfermarkt row: {len(filled):,d} "
        "by competition:"
    )
    print(filled["competition_id"].value_counts().to_string())

    outside = ~(st["home_n"].between(7, 11) & st["away_n"].between(7, 11))
    print(
        f"\n[3] stints with a side outside 7..11: {outside.sum():,d} of {len(st):,d} "
        f"({outside.mean():.3%}) in {st.loc[outside, 'game_id'].nunique():,d} games; "
        f"max side {max(st['home_n'].max(), st['away_n'].max())}, "
        f"min side {min(st['home_n'].min(), st['away_n'].min())}"
    )

    print("\n[4] Elo (squared error of expected home score):")
    print(t["elo_report"].to_string(index=False))

    coverage = t["coverage"]
    path = RESULTS / "coverage" / "tm_coverage.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"\ncoverage ({len(coverage):,d} competition-seasons) -> {path}; pool competitions:")
    pool = coverage[coverage["competition_id"].isin(POOL_COMPETITIONS)]
    print(
        pool.groupby("competition_id")
        .agg(
            seasons=("season", ",".join),
            games=("games", "sum"),
            with_stints=("games_with_stints", "sum"),
            player_games=("player_games", "sum"),
            only_from_stints=("player_games_only_from_stints", "sum"),
            first=("first_date", "min"),
            last=("last_date", "max"),
        )
        .reindex(list(POOL_COMPETITIONS))
        .to_string()
    )


if __name__ == "__main__":
    main()
