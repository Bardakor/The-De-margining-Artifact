"""Download the Transfermarkt snapshot, build the processed tables, print the checks.

Usage:
    uv run python scripts/players/acquire_transfermarkt.py

Design §4.1/§4.3. The download is skipped for files already cached. The summary
prints row counts, the stint reconciliation checks, the Elo fit and the coverage
of the pool-relevant competitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.players import stints, transfermarkt

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

    mins = t["tm_appearances"][t["tm_appearances"]["game_id"].isin(st["game_id"])].merge(
        stints.player_minutes(st, t["tm_stint_players"]),
        on=["game_id", "player_id"],
        how="left",
        suffixes=("", "_stints"),
    )
    gap = (mins["minutes_stints"].fillna(0) - mins["minutes"]).abs()
    print(
        f"\n[2] minutes within +-2 of appearances for {(gap <= 2).mean():.2%} of "
        f"{len(mins):,d} (game, player) rows; within +-5: {(gap <= 5).mean():.2%}; "
        f"player absent from stints: {mins['minutes_stints'].isna().sum():,d}"
    )

    outside = ~(st["home_n"].between(7, 11) & st["away_n"].between(7, 11))
    print(
        f"\n[3] stints with a side outside 7..11: {outside.sum():,d} of {len(st):,d} "
        f"({outside.mean():.3%}) in {st.loc[outside, 'game_id'].nunique():,d} games; "
        f"max side {max(st['home_n'].max(), st['away_n'].max())}, "
        f"min side {min(st['home_n'].min(), st['away_n'].min())}"
    )

    print("\n[4] Elo (squared error of expected home score):")
    print(t["elo_report"].to_string(index=False))

    print("\ncoverage of pool-relevant competitions (window):")
    has_apps = set(t["tm_appearances"]["game_id"])
    rows = []
    for cid in POOL_COMPETITIONS:
        g = games[games["competition_id"] == cid]
        s = sel[sel["game_id"].isin(g["game_id"])]
        rows.append(
            {
                "competition": cid,
                "games": len(g),
                "first": g["date"].min(),
                "last": g["date"].max(),
                "seasons": ",".join(sorted(g["season"].unique())),
                "complete_lineups": int(s["complete"].sum()),
                "with_stints": int(s["has_stints"].sum()),
                "with_appearances": int(g["game_id"].isin(has_apps).sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
