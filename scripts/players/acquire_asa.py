"""Download the MLS seasons of pool players from the American Soccer Analysis API, build
asa_mls_pool, print the checks (pre-registration A20 vi).

Usage:
    uv run python scripts/players/acquire_asa.py

Seven requests, each cached under data/players/raw/asa/ and skipped once on disk.
asa_mls_pool has one row per pool player x MLS regular season before 2025 with no MLS
game in panel_player_match. The 2025 season is downloaded only to compare ASA with the
panel game by game. Fields, assumptions and licence (CC BY-SA 4.0): footy.players.asa.
"""

from __future__ import annotations

import pandas as pd

from footy.players import asa
from footy.players.paths import PAPER, PROCESSED

PANEL_FROM = 2025  # first MLS season in the panel: earlier ones are filled, this one compared
SEASONS = "2020,2021,2022,2023,2024,2025"  # the MLS seasons of the study window up to it
STAGE = "Regular Season"
FIELDS = (
    "minutes=minutes_played; goals=goals; penalty_goals=goals[shot_pattern=Penalty]; "
    "np_goals=goals-penalty_goals; assists=primary_assists; "
    "minutes_scaled=sum over games of minutes_played*90/expanded_minutes"
)
COLUMNS = [
    "tm_player_id", "player", "asa_player_id", "season", "club", "minutes", "goals", "np_goals",
    "penalty_goals", "assists", "minutes_scaled", "asa_season_name", "asa_stage_name",
    "asa_fields",
]  # fmt: skip
COUNTS = ["minutes", "goals", "penalty_goals", "assists"]


def panel_check(by_game: pd.DataFrame, mls: pd.DataFrame) -> pd.DataFrame:
    """The panel's MLS 2025 rows of the matched players, each with the ASA game of that
    player that kicked off within a day of the panel's date."""
    panel = mls[mls["season"] == asa.study_season(PANEL_FROM)]
    panel = panel.assign(date=pd.to_datetime(panel["date"]).astype("datetime64[s]"))
    panel = panel.rename(columns={"player_id": "tm_player_id"})
    games = by_game.assign(date=by_game["kickoff"].dt.tz_localize(None).astype("datetime64[s]"))
    names = by_game[["tm_player_id", "player"]].drop_duplicates()
    return pd.merge_asof(
        panel.merge(names, on="tm_player_id").sort_values("date"),
        games.drop(columns="player").sort_values("date"),
        on="date",
        by="tm_player_id",
        direction="nearest",
        tolerance=pd.Timedelta(days=1),
        suffixes=("_tm", "_asa"),
    )


def main() -> None:
    pool = pd.read_csv(PAPER / "data" / "pool.csv").merge(
        pd.read_csv(PROCESSED / "tm_players.csv.gz")[["player_id", "date_of_birth"]],
        left_on="tm_player_id",
        right_on="player_id",
        validate="1:1",
    )
    root = asa.RAW_DIR
    players = asa.ensure(root, "players", "mls/players", {})
    teams = asa.ensure(root, "teams", "mls/teams", {})
    pairs = asa.match_players(pool, pd.DataFrame(players))
    both = pairs["same_name"] & pairs["same_birth_date"]
    matched = pairs.loc[both, ["tm_player_id", "player", "asa_player_id"]]
    if matched["tm_player_id"].duplicated().any():
        raise ValueError("a pool player matches two ASA players")

    query = {
        "player_id": ",".join(sorted(matched["asa_player_id"])),
        "season_name": SEASONS,
        "stage_name": STAGE,
        "split_by_seasons": "true",
        "split_by_teams": "true",
    }
    penalty = {"shot_pattern": "Penalty"}
    split_by_games = {"split_by_games": "true"}
    rows = asa.ensure(root, "xgoals_seasons", "mls/players/xgoals", query)
    penalty_rows = asa.ensure(root, "xgoals_seasons_penalty", "mls/players/xgoals", query | penalty)
    game_rows = asa.ensure(root, "xgoals_games", "mls/players/xgoals", query | split_by_games)
    game_penalty_rows = asa.ensure(
        root, "xgoals_games_penalty", "mls/players/xgoals", query | split_by_games | penalty
    )
    game_ids = ",".join(sorted({row["game_id"] for row in game_rows}))
    games = asa.ensure(root, "games", "mls/games", {"game_id": game_ids})

    seasons = asa.player_seasons(matched, rows, penalty_rows, teams)
    by_game = asa.player_games(matched, game_rows, game_penalty_rows, games)
    keys = ["tm_player_id", "asa_season_name"]
    # the season rows must be the sums of the game rows before the two are combined
    pd.testing.assert_frame_equal(
        seasons.set_index(keys)[COUNTS], by_game.groupby(keys)[COUNTS].sum()
    )
    per_season = by_game.groupby(keys, as_index=False).agg(
        games=("asa_game_id", "size"), minutes_scaled=("minutes_scaled", "sum")
    )
    seasons = seasons.merge(per_season, on=keys, validate="1:1")
    seasons["minutes_scaled"] = seasons["minutes_scaled"].round().astype(int)

    panel = pd.read_csv(
        PROCESSED / "panel_player_match.csv.gz",
        usecols=["player_id", "date", "season", "competition_id", *COUNTS],
    )
    mls = panel[panel["competition_id"] == "MLS1"]
    in_panel = pd.MultiIndex.from_frame(seasons[["tm_player_id", "season"]]).isin(
        pd.MultiIndex.from_frame(mls[["player_id", "season"]])
    )
    fill = seasons[(seasons["asa_season_name"] < PANEL_FROM) & ~in_panel]
    fill.assign(asa_stage_name=STAGE, asa_fields=FIELDS)[COLUMNS].to_csv(
        PROCESSED / "asa_mls_pool.csv.gz", index=False
    )

    pd.set_option("display.width", 200)
    print(f"[1] pool players matched to an ASA player (same name and birth date): {both.sum()}")
    asa_seasons = pd.DataFrame(players).set_index("player_id")["season_name"]
    listing = pairs.assign(asa_seasons=pairs["asa_player_id"].map(asa_seasons))
    print(listing[both].drop(columns=["same_name", "same_birth_date"]).to_string(index=False))
    print("same name, other birth date (not matched):")
    print(listing[pairs["same_name"] & ~both].to_string(index=False))
    print(f"same birth date, other name (not matched): {(~pairs['same_name']).sum()} pairs")

    print(f"\n[2] ASA regular-season rows of the matched players, seasons {SEASONS}")
    print(seasons.assign(written=seasons.index.isin(fill.index)).to_string(index=False))
    print(f"asa_mls_pool: {len(fill)} rows written")

    print("\n[3] panel MLS 2025 rows of the matched players against the ASA game of that day")
    check = panel_check(by_game, mls)
    pairs_of = [f"{c}_{side}" for c in COUNTS for side in ("tm", "asa")]
    found = check[check["asa_game_id"].notna()]
    summary = found.groupby("player")[[*pairs_of, "minutes_scaled"]].sum().round().astype(int)
    summary.insert(0, "panel_rows", check.groupby("player").size())
    summary.insert(1, "with_asa_game", found.groupby("player").size())
    print(summary.T.to_string())
    differs = found[
        (found["goals_tm"] != found["goals_asa"])
        | (found["penalty_goals_tm"] != found["penalty_goals_asa"])
        | (found["assists_tm"] != found["assists_asa"])
    ]
    print("games where goals, penalty goals or assists differ:")
    print(differs[["player", "date", *pairs_of[2:]]].to_string(index=False))
    partial = found[found["minutes_tm"] < 90]
    print(f"games with fewer than 90 panel minutes: {len(partial)}; minutes there: panel "
          f"{partial['minutes_tm'].sum()}, ASA {partial['minutes_asa'].sum()}, "
          f"scaled {partial['minutes_scaled'].sum():.0f}")  # fmt: skip


if __name__ == "__main__":
    main()
