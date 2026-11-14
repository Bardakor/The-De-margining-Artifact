"""Download StatsBomb open data, build sb_matches / sb_shots / sb_player_match, fit xT.

Prints the checks of the StatsBomb build: coverage per competition-season, goals
against the final score, minutes per player and per team-match, and the xT grid.

Usage:
    uv run python scripts/players/acquire_statsbomb.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from footy.players import statsbomb as sb
from footy.players import xt
from footy.players.paths import PROCESSED, RESULTS


def coverage(tables: sb.Tables) -> pd.DataFrame:
    competitions = sb.read_json(sb.RAW_DIR / "competitions.json.gz")
    rows = []
    for name, season, cid, sid in sb.resolve_seasons(competitions):
        ids = [m["match_id"] for m in sb.read_json(sb.RAW_DIR / f"matches/{cid}/{sid}.json.gz")]
        got = tables.matches[
            (tables.matches["competition"] == name) & (tables.matches["season"] == season)
        ]
        rows.append(
            {
                "competition": name,
                "season": season,
                "in_matches_file": len(ids),
                "sb_matches_rows": len(got),
                "event_files": sum((sb.RAW_DIR / f"events/{i}.json.gz").exists() for i in ids),
                "lineup_files": sum((sb.RAW_DIR / f"lineups/{i}.json.gz").exists() for i in ids),
                "with_player_rows": tables.player_match.loc[
                    tables.player_match["sb_match_id"].isin(ids), "sb_match_id"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows)


def goal_check(tables: sb.Tables) -> pd.DataFrame:
    shots = tables.shots[tables.shots["outcome"] == "Goal"]
    goals = shots.groupby(["sb_match_id", "team"]).size().rename("shot_goals")
    og = tables.own_goals.set_index(["sb_match_id", "team"])["own_goals"]
    m = tables.matches
    out = m[["sb_match_id", "competition", "home_team", "away_team", "home_score", "away_score"]]
    for side in ("home", "away"):
        key = pd.MultiIndex.from_arrays([m["sb_match_id"], m[f"{side}_team"]])
        out = out.assign(
            **{
                f"{side}_from_events": goals.reindex(key).fillna(0).to_numpy()
                + og.reindex(key).fillna(0).to_numpy()
            }
        )
    out = out.assign(
        ok=(out["home_from_events"] == out["home_score"])
        & (out["away_from_events"] == out["away_score"])
    )
    return out


def telescoping_check(tables: sb.Tables) -> tuple[int, int, float, float]:
    """A real possession made only of chained successful moves: sum of xt_added vs ends."""
    for mid in tables.matches["sb_match_id"]:
        flat = sb.flatten_events(mid, sb.read_json(sb.RAW_DIR / f"events/{mid}.json.gz"))
        is_move = (flat["type"] == "Carry") | (
            (flat["type"] == "Pass") & ~flat["kind"].isin(sb.SET_PIECES)
        )
        for _, group in flat[is_move].groupby("possession"):
            ok = (group["type"] == "Carry") | group["outcome"].isna()
            chained = (
                xt.cell(group["x"].iloc[1:], group["y"].iloc[1:])
                == xt.cell(group["end_x"].iloc[:-1], group["end_y"].iloc[:-1])
            ).all()
            if len(group) >= 8 and ok.all() and chained:
                added = xt.xt_added(
                    tables.grid, group["x"], group["y"], group["end_x"], group["end_y"]
                ).sum()
                end = xt.value(tables.grid, group["end_x"].iloc[-1], group["end_y"].iloc[-1])
                start = xt.value(tables.grid, group["x"].iloc[0], group["y"].iloc[0])
                return int(mid), int(group["possession"].iloc[0]), float(added), float(end - start)
    raise RuntimeError("no chained possession found")


def main() -> None:
    n = sb.acquire()
    print(f"acquire: {n} matches listed, all event and lineup files cached")
    tables = sb.build()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tables.matches.to_csv(PROCESSED / "sb_matches.csv.gz", index=False)
    tables.shots.to_csv(PROCESSED / "sb_shots.csv.gz", index=False)
    tables.player_match.to_csv(PROCESSED / "sb_player_match.csv.gz", index=False)
    out = RESULTS / "xt"
    out.mkdir(parents=True, exist_ok=True)
    sb.grid_frame(tables.grid).to_csv(out / "grid.csv", index=False)
    open_play = tables.shots[tables.shots["shot_type"] == "Open Play"]
    fit = {
        "iterations": tables.grid.iterations,
        "max_change_last_sweep": tables.grid.max_change,
        "open_play_shots": len(open_play),
        "open_play_goals": int((open_play["outcome"] == "Goal").sum()),
        "move_attempts": len(tables.attempts),
        "successful_moves": int(tables.attempts["success"].sum()),
    }
    (out / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")

    pd.set_option("display.width", 160)
    print("\n[1] coverage per competition-season")
    cov = coverage(tables)
    print(cov.to_string(index=False))
    print(f"total matches: {cov['sb_matches_rows'].sum()}")

    print("\n[2] goals from sb_shots + own goals vs final score")
    goals = goal_check(tables)
    print(f"matches agreeing: {goals['ok'].sum()} / {len(goals)} = {goals['ok'].mean():.4f}")
    print(goals.loc[~goals["ok"]].to_string(index=False))

    print("\n[3] minutes")
    pm = tables.player_match
    print(f"rows {len(pm)}; minutes min {pm['minutes'].min():.2f} max {pm['minutes'].max():.2f}")
    over, zero = (pm["minutes"] > 130).sum(), (pm["minutes"] == 0).sum()
    print(f"rows with minutes > 130: {over}; == 0: {zero}")
    team = pm.groupby(["sb_match_id", "team"])["minutes"].sum().reset_index()
    team["ratio"] = team["minutes"] / (11 * team["sb_match_id"].map(tables.match_minutes))
    q = team["ratio"].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])
    print("per team-match sum / (11 x match length), quantiles:")
    print(q.round(4).to_string())
    print(f"team-matches with ratio < 0.97: {(team['ratio'] < 0.97).sum()} of {len(team)}")
    length = tables.match_minutes
    lo, med, hi = length.min(), length.median(), length.max()
    print(f"match length (min): min {lo:.1f} median {med:.1f} max {hi:.1f}")
    print(f"players on the pitch with no event: {tables.on_pitch_without_events}")
    print(f"event players with no minutes: {tables.event_players_without_minutes}")

    print("\n[4] xT grid")
    print(json.dumps(fit, indent=2))
    values = tables.grid.values
    by_third = [values[0:5].mean(), values[5:11].mean(), values[11:16].mean()]
    print(f"mean xT for x < 37.5, 37.5-82.5, >= 82.5: {np.round(by_third, 5).tolist()}")
    print(f"max cell {values.max():.4f} at {np.unravel_index(values.argmax(), values.shape)}")
    print(f"column maxima along the pitch: {np.round(values.max(axis=1), 4).tolist()}")
    mid, possession, added, diff = telescoping_check(tables)
    print(
        f"telescoping (match {mid}, possession {possession}): sum xt_added {added:.6f}, "
        f"xT(end) - xT(start) {diff:.6f}, |diff| {abs(added - diff):.2e}"
    )

    print("\nrows: ", {"sb_matches": len(tables.matches), "sb_shots": len(tables.shots),
                       "sb_player_match": len(pm)})  # fmt: skip


if __name__ == "__main__":
    main()
