"""Link StatsBomb to Transfermarkt, build StatsBomb stints, the player panels and the pool.

Reads the processed tables (tm_*, team_elo, sb_matches), raw Transfermarkt game_lineups,
game_events and players (candidate sets, names, citizenship, own goals) and raw
StatsBomb events and lineups (timeline, nicknames). Reads no ESPN file.

Writes data/players/processed/{sb_stints, sb_stint_players, sb_appearances,
player_positions, panel_player_match, panel_player_season}.csv.gz,
results/players/linking/*.csv and paper-unmeasured/data/pool.csv, then prints the checks.

Usage:
    uv run python scripts/players/build_panel.py
"""

from __future__ import annotations

import gzip
import json

import pandas as pd

from footy.players import linking, panel, stints
from footy.players.paths import PAPER, PROCESSED, RAW, RESULTS
from footy.players.window import WINDOW_START

OUT = RESULTS / "linking"
TM_RAW = RAW / "transfermarkt"
SB_RAW = RAW / "statsbomb"

# Every pool decision not taken by the rule in panel.match_pool, with its evidence.
MANUAL_POOL: dict[str, tuple[int, str]] = {
    "Alisson Becker": (
        105470,
        "V1-only name, so no club to confirm; four TM players are named 'Alisson'; chosen the "
        "goalkeeper born 1992-10-02 (Liverpool, Brazil; 24,605 window minutes), the others "
        "are a Sao Paulo midfielder, a Shakhtar winger and a 26-minute full-back",
    ),
    "Moisés Caicedo": (
        687626,
        "V1-only name; the only TM player named 'Moisés Caicedo' (Brighton, Chelsea, "
        "Ecuador); the other candidates are single-name 'Moisés' (CSKA, Fortaleza/Santos)",
    ),
    "Álex Grimaldo": (
        193082,
        "TM lists him as 'Alejandro Grimaldo' (Álex is the short form), below the name "
        "threshold; Ballon d'Or 2024 club Bayer Leverkusen, which he played for (Benfica "
        "before), Spain",
    ),
}


def read_sb(kind: str, match_id: int) -> panel.Json:
    return json.loads(gzip.decompress((SB_RAW / kind / f"{match_id}.json.gz").read_bytes()))


def load() -> dict[str, pd.DataFrame]:
    tables = {
        name: pd.read_csv(PROCESSED / f"{name}.csv.gz", low_memory=False)
        for name in (
            "tm_games", "tm_appearances", "tm_stints", "tm_stint_players", "team_elo",
            "sb_matches",
        )
    }  # fmt: skip
    tables["tm_games"]["date"] = pd.to_datetime(tables["tm_games"]["date"])
    ids = set(tables["tm_games"]["game_id"])
    lineups = pd.read_csv(
        TM_RAW / "game_lineups.csv.gz",
        usecols=["game_id", "player_id", "club_id", "player_name", "type"],
    )
    events = pd.read_csv(
        TM_RAW / "game_events.csv.gz",
        usecols=[
            "game_id", "minute", "type", "club_id", "player_id", "description",
            "player_in_id", "player_assist_id",
        ],
    )  # fmt: skip
    tables["lineups"] = lineups[lineups["game_id"].isin(ids)]
    tables["events"] = events[events["game_id"].isin(ids)]
    tables["profiles"] = pd.read_csv(
        TM_RAW / "players.csv.gz",
        usecols=["player_id", "name", "country_of_citizenship", "last_season", "sub_position"],
    )
    return tables


def event_players(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """game_id, club_id, player_id of everyone a Transfermarkt event names, on his side
    (an own goal's scorer belongs to the club not credited)."""
    ev = events.merge(games[["game_id", "home_club_id", "away_club_id"]], on="game_id")
    own = (ev["type"] == "Goals") & ev["description"].fillna("").str.contains("Own-goal")
    other = ev["home_club_id"].where(ev["club_id"] == ev["away_club_id"], ev["away_club_id"])
    frames = [
        pd.DataFrame({"game_id": ev["game_id"], "club_id": ev["club_id"].where(~own, other),
                      "player_id": ev["player_id"]}),
        pd.DataFrame({"game_id": ev["game_id"], "club_id": ev["club_id"],
                      "player_id": ev["player_in_id"]}),
        pd.DataFrame({"game_id": ev["game_id"], "club_id": ev["club_id"],
                      "player_id": ev["player_assist_id"]}),
    ]  # fmt: skip
    out = pd.concat(frames).dropna()
    return out.astype(int).drop_duplicates()


def main() -> None:
    t = load()
    games, profiles = t["tm_games"], t["profiles"]
    names = profiles.set_index("player_id")["name"]
    OUT.mkdir(parents=True, exist_ok=True)

    # --- games
    sbm = t["sb_matches"]
    game_links = linking.link_sb_games(sbm, games)
    early = pd.to_datetime(sbm["date"]) < pd.Timestamp(WINDOW_START)
    reason = pd.Series("", index=sbm.index).where(game_links["status"] == "linked", "no_tm_game")
    game_links["reason"] = reason.where(~early, "before_window")
    game_links = sbm[["sb_match_id", "competition", "season", "date", "home_team",
                      "away_team"]].merge(game_links, on="sb_match_id")  # fmt: skip
    game_links.to_csv(OUT / "game_links.csv", index=False)
    teams = linking.team_clubs(sbm, game_links, games)
    linked_ids = sorted(teams["sb_match_id"].unique())

    # --- StatsBomb matches and players
    parsed = [panel.parse_sb_match(int(m), read_sb("events", int(m))) for m in linked_ids]
    squad = pd.DataFrame(
        [
            (m, team["team_name"], p["player_id"], p["player_name"], p["player_nickname"])
            for m in linked_ids
            for team in read_sb("lineups", int(m))
            for p in team["lineup"]
        ],
        columns=["sb_match_id", "team", "sb_player_id", "player_name", "player_nickname"],
    )
    on_pitch = pd.concat(
        [
            pd.concat([m.starters[["team", "sb_player_id"]],
                       m.timeline.loc[m.timeline["kind"] == "sub", ["team", "sb_player_in_id"]]
                       .rename(columns={"sb_player_in_id": "sb_player_id"})])
            .assign(sb_match_id=m.sb_match_id)
            for m in parsed
        ]
    ).astype({"sb_player_id": int})  # fmt: skip
    sb_players = on_pitch.merge(squad, on=["sb_match_id", "team", "sb_player_id"], how="left")

    # --- Transfermarkt candidates
    ev_players = event_players(t["events"], games)
    lineups = t["lineups"].rename(columns={"player_name": "name"})
    with_lineups = set(lineups["game_id"])
    ev_named = ev_players.assign(name=ev_players["player_id"].map(names))
    game_ids = set(teams["game_id"])
    tm_game = pd.concat(
        [
            lineups[lineups["game_id"].isin(game_ids)],
            ev_named[ev_named["game_id"].isin(game_ids - with_lineups)],
        ]
    )[["game_id", "club_id", "player_id", "name"]].dropna()
    tm_team = pd.concat([lineups, ev_named])[["club_id", "player_id", "name"]].dropna()
    tm_team = tm_team[tm_team["club_id"].isin(teams["club_id"])].drop_duplicates(
        ["club_id", "player_id"]
    )
    recent = profiles[profiles["last_season"] >= 2019].dropna(subset=["country_of_citizenship"])
    tm_country = pd.DataFrame(
        {
            "country": recent["country_of_citizenship"].map(linking.normalise_team),
            "player_id": recent["player_id"],
            "name": recent["name"],
        }
    )
    mapping, issues = linking.link_sb_players(sb_players, teams, tm_game, tm_team, tm_country)
    mapping = mapping.merge(
        squad.drop_duplicates("sb_player_id")[["sb_player_id", "player_name", "player_nickname"]],
        on="sb_player_id",
        how="left",
    ).assign(tm_name=lambda f: f["tm_player_id"].map(names))
    mapping.to_csv(OUT / "player_links.csv", index=False)
    issues.merge(mapping[["sb_player_id", "player_name", "tm_name"]], on="sb_player_id").to_csv(
        OUT / "player_link_issues.csv", index=False
    )
    to_tm = mapping.dropna(subset=["tm_player_id"]).set_index("sb_player_id")["tm_player_id"]

    # --- StatsBomb stints where Transfermarkt has none
    no_tm = set(games["game_id"]) - set(t["tm_stints"]["game_id"])
    sb_games = teams[teams["game_id"].isin(no_tm)]
    sb_parsed = [m for m in parsed if m.sb_match_id in set(sb_games["sb_match_id"])]
    sb_stints, sb_stint_players, sb_apps = panel.sb_tables(sb_parsed, sb_games, games, to_tm)
    for name, table in (
        ("sb_stints", sb_stints), ("sb_stint_players", sb_stint_players),
        ("sb_appearances", sb_apps),
    ):  # fmt: skip
        table.to_csv(PROCESSED / f"{name}.csv.gz", index=False, date_format="%Y-%m-%d")

    # --- panels
    all_stints = pd.concat([t["tm_stints"], sb_stints], ignore_index=True)
    all_players = pd.concat([t["tm_stint_players"], sb_stint_players], ignore_index=True)
    apps = t["tm_appearances"]
    apps = pd.concat([apps[~apps["game_id"].isin(sb_stints["game_id"])], sb_apps],
                     ignore_index=True)  # fmt: skip
    apps["date"] = pd.to_datetime(apps["date"])
    groups = panel.position_groups(apps, profiles)
    groups.to_csv(PROCESSED / "player_positions.csv.gz", index=False)
    sb_cols = panel.sb_player_columns(parsed, teams, to_tm, all_stints, all_players, games)
    pm = panel.player_match(apps, games, all_stints, all_players, t["team_elo"], groups, sb_cols)
    ps = panel.player_season(pm)
    pm.to_csv(PROCESSED / "panel_player_match.csv.gz", index=False, date_format="%Y-%m-%d")
    ps.to_csv(PROCESSED / "panel_player_season.csv.gz", index=False)

    # --- pool
    v1 = pd.read_csv(PAPER / "data" / "v1_ranking.csv")
    bdo = pd.read_csv(PAPER / "data" / "ballon_dor.csv")
    club_name = pd.concat(
        [games[["home_club_id", "home_club_name"]].set_axis(["club_id", "club_name"], axis=1),
         games[["away_club_id", "away_club_name"]].set_axis(["club_id", "club_name"], axis=1)]
    ).dropna().drop_duplicates("club_id").set_index("club_id")["club_name"]  # fmt: skip
    club_minutes = pm.groupby(["player_id", "club_id"], as_index=False).agg(
        minutes=("minutes", "sum")
    )
    club_minutes["club_name"] = club_minutes["club_id"].map(club_name)
    window_players = profiles[profiles["player_id"].isin(pm["player_id"])]
    pool, candidates = panel.match_pool(
        panel.pool_names(v1, bdo), window_players, club_minutes, groups, MANUAL_POOL
    )
    pool.drop(columns="note").to_csv(PAPER / "data" / "pool.csv", index=False)
    pool.to_csv(OUT / "pool_matching.csv", index=False)
    candidates.to_csv(OUT / "pool_candidates.csv", index=False)

    report(game_links, mapping, issues, sb_players, teams, to_tm, t, games, sb_stints,
           parsed, sb_parsed, sb_games, apps, all_stints, pm, ps, pool)  # fmt: skip


def report(
    game_links: pd.DataFrame,
    mapping: pd.DataFrame,
    issues: pd.DataFrame,
    sb_players: pd.DataFrame,
    teams: pd.DataFrame,
    to_tm: pd.Series,
    t: dict[str, pd.DataFrame],
    games: pd.DataFrame,
    sb_stints: pd.DataFrame,
    parsed: list[panel.SbMatch],
    sb_parsed: list[panel.SbMatch],
    sb_games: pd.DataFrame,
    apps: pd.DataFrame,
    all_stints: pd.DataFrame,
    pm: pd.DataFrame,
    ps: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    pd.set_option("display.width", 200)
    print("== game links per StatsBomb competition-season")
    rates = game_links.groupby(["competition", "season"]).agg(
        matches=("sb_match_id", "size"),
        linked=("status", lambda s: int((s == "linked").sum())),
        before_window=("reason", lambda s: int((s == "before_window").sum())),
        no_tm_game=("reason", lambda s: int((s == "no_tm_game").sum())),
    )
    rates["rate_in_window"] = rates["linked"] / (rates["matches"] - rates["before_window"])
    print(rates.to_string())
    rates.to_csv(OUT / "game_link_rates.csv")
    linked = game_links[game_links["status"] == "linked"]
    print("ambiguous game links:", int((game_links["status"] == "ambiguous").sum()),
          "| duplicate game ids:", int(linked["game_id"].duplicated().sum()))  # fmt: skip

    print("== player links")
    rows = sb_players.merge(teams, on=["sb_match_id", "team"])
    rows["tm"] = rows["sb_player_id"].map(to_tm)
    per_match = rows.groupby("sb_match_id")["tm"].apply(lambda s: s.notna().mean())
    print(f"players on the pitch: {len(rows)}, linked {rows['tm'].notna().mean():.4f}; "
          f"per match: mean {per_match.mean():.4f}, min {per_match.min():.3f}, "
          f"matches at 100% {(per_match == 1).mean():.3f}")  # fmt: skip
    print("methods:", mapping["method"].value_counts(dropna=False).to_dict())
    print("issues:", issues["kind"].value_counts().to_dict())
    dup_game = rows.dropna(subset=["tm"]).duplicated(["game_id", "tm"]).sum()
    dup_tm = mapping.dropna(subset=["tm_player_id"])["tm_player_id"].duplicated().sum()
    print(f"duplicate mappings: within game {dup_game}, TM id used by two SB ids {dup_tm}, "
          f"SB id with two TM ids {mapping['sb_player_id'].duplicated().sum()}")  # fmt: skip
    per_match.rename("player_link_rate").to_csv(OUT / "player_link_rate_per_match.csv")
    link_validation(parsed, teams, to_tm, t["events"])

    print("== StatsBomb stints (games without Transfermarkt stints)")
    rec = stints.reconcile(sb_stints, games)
    print(f"games {sb_stints['game_id'].nunique()}, stints {len(sb_stints)}, appearances "
          f"{int((apps['source'] == 'statsbomb').sum())}; goals = final score in "
          f"{rec['match'].mean():.4f} ({int(rec['match'].sum())}/{len(rec)})")  # fmt: skip
    pseudo = int((apps["player_id"] < 0).sum())
    print(
        f"pseudo-id (unlinked) appearances: {pseudo}; side counts outside 11 in stints: "
        f"{int(((sb_stints['home_n'] != 11) | (sb_stints['away_n'] != 11)).sum())}"
    )
    by_comp = rec.merge(games[["game_id", "competition_id", "season"]], on="game_id")
    print(by_comp.groupby(["competition_id", "season"])["match"].agg(["size", "mean"]))

    print("== panel")
    print(f"panel_player_match rows {len(pm)}, players {pm['player_id'].nunique()}, games "
          f"{pm['game_id'].nunique()}; panel_player_season rows {len(ps)}")  # fmt: skip
    print("scope:", pm["scope"].value_counts().to_dict())
    print("sb_linked rows:", int(pm["sb_linked"].sum()))
    minutes_check(pm, apps, all_stints)
    goals_check(pm, apps, t, games, sb_parsed, sb_games)
    seasons_check(games)

    print("== pool")
    print(f"names {len(pool)}, mapped {int(pool['tm_player_id'].notna().sum())}, goalkeepers "
          f"{int(pool['is_goalkeeper'].sum())}, TM id used twice "
          f"{int(pool['tm_player_id'].dropna().duplicated().sum())}")  # fmt: skip
    print(pool["match_method"].value_counts().to_dict())
    print(pool["confidence"].value_counts().to_dict())
    print(pool[pool["match_method"] != "name+club"].to_string())
    holes(pool, ps)


def link_validation(
    parsed: list[panel.SbMatch], teams: pd.DataFrame, to_tm: pd.Series, events: pd.DataFrame
) -> None:
    """Independent check of the player links: substitutions (off, on) and goal scorers of
    the linked StatsBomb matches found among Transfermarkt's own events for that game."""
    line = pd.concat([m.timeline.assign(sb_match_id=m.sb_match_id) for m in parsed])
    line = line.merge(teams, on=["sb_match_id", "team"])
    line["off"] = line["sb_player_id"].map(to_tm)
    line["on"] = line["sb_player_in_id"].map(to_tm)
    tm_subs = events[events["type"] == "Substitutions"]
    tm_goals = events[(events["type"] == "Goals")
                      & ~events["description"].fillna("").str.contains("Own-goal")]  # fmt: skip
    for label, sb, tm, cols in (
        ("substitutions (off, on)", line[line["kind"] == "sub"], tm_subs, ["off", "on"]),
        ("goal scorers", line[(line["kind"] == "goal") & ~line["own_goal"]], tm_goals, ["off"]),
    ):
        sb = sb[sb["game_id"].isin(tm["game_id"])]
        tm_cols = ["player_id", "player_in_id"][: len(cols)]
        known = set(
            tm[["game_id", *tm_cols]].dropna().astype(int).itertuples(index=False, name=None)
        )
        found = [
            tuple(int(v) if pd.notna(v) else -1 for v in r)
            for r in sb[["game_id", *cols]].itertuples(index=False, name=None)
        ]
        print(
            f"{label} also in Transfermarkt events: "
            f"{sum(f in known for f in found) / max(len(found), 1):.4f} of {len(found)}"
        )
    # fmt: skip


def minutes_check(pm: pd.DataFrame, apps: pd.DataFrame, all_stints: pd.DataFrame) -> None:
    length = all_stints.groupby("game_id")["end_min"].max()
    off = apps.assign(off=((apps["red"] > 0) | (apps["yellow"] >= 2)).astype(int))
    sent = off.groupby(["game_id", "club_id"])["off"].sum()
    side = pm[pm["game_id"].isin(length.index)].groupby(["game_id", "club_id"])["minutes"].sum()
    frame = side.to_frame().join(sent.rename("sent_off")).fillna({"sent_off": 0}).reset_index()
    frame["length"] = frame["game_id"].map(length)
    full = 11 * frame["length"]
    low = full - frame["sent_off"] * frame["length"] - 11
    frame["ok"] = (frame["minutes"] >= low) & (frame["minutes"] <= full + 11)
    clean = frame["sent_off"] == 0
    frame["ok"] = frame["ok"] & (~clean | ((frame["minutes"] - full).abs() <= 11))
    per_game = frame.groupby("game_id")["ok"].all()
    print(f"minutes per (game, club) within tolerance: {frame['ok'].mean():.4f} of sides, "
          f"{per_game.mean():.4f} of {len(per_game)} stint games")  # fmt: skip
    frame.to_csv(OUT / "minutes_check.csv", index=False)


def goals_check(
    pm: pd.DataFrame,
    apps: pd.DataFrame,
    t: dict[str, pd.DataFrame],
    games: pd.DataFrame,
    sb_parsed: list[panel.SbMatch],
    sb_games: pd.DataFrame,
) -> None:
    ev = t["events"]
    own = ev[(ev["type"] == "Goals") & ev["description"].fillna("").str.contains("Own-goal")]
    own_tm = own.groupby(["game_id", "club_id"]).size()
    sb_line = pd.concat([m.timeline.assign(sb_match_id=m.sb_match_id) for m in sb_parsed])
    sb_own = sb_line[sb_line["own_goal"]].merge(sb_games, on=["sb_match_id", "team"])
    own_sb = sb_own.groupby(["game_id", "club_id"]).size()
    sb_ids = set(sb_games["game_id"])
    owns = pd.concat([own_tm[~own_tm.index.get_level_values(0).isin(sb_ids)], own_sb])
    scored = pm.groupby(["game_id", "club_id"])["goals"].sum()
    with_events = set(ev["game_id"]) | sb_ids
    g = games[games["game_id"].isin(with_events) & games["game_id"].isin(pm["game_id"])]
    rows = []
    for side in ("home", "away"):
        key = pd.MultiIndex.from_arrays([g["game_id"], g[f"{side}_club_id"]])
        total = scored.reindex(key).fillna(0).to_numpy() + owns.reindex(key).fillna(0).to_numpy()
        rows.append(total == g[f"{side}_goals"].to_numpy())
    ok = pd.Series(rows[0] & rows[1], index=g["game_id"])
    sb_ok = ok[ok.index.isin(sb_ids)]
    print(f"goals (player non-own + own) = final score: {ok.mean():.4f} of {len(ok)} games "
          f"with events; StatsBomb-stint games {sb_ok.mean():.4f} of {len(sb_ok)}")  # fmt: skip
    zero = apps[(apps["minutes"] == 0) & (apps["goals"] > 0)]["game_id"]
    print(f"  failing games with a goal by a 0-minute player (dropped by minutes >= 1): "
          f"{int((~ok & ok.index.isin(zero)).sum())} of {int((~ok).sum())}")  # fmt: skip
    ok.rename("goals_match").to_csv(OUT / "goals_check.csv")


def seasons_check(games: pd.DataFrame) -> None:
    study = panel.study_seasons(games)
    changed = games.assign(study_season=study)[study != games["season"]]
    table = changed.groupby(["competition_id", "season", "study_season"]).size()
    print("season labels changed by the first-match rule (TM label -> study label):")
    print(table.to_string())
    table.rename("games").to_csv(OUT / "season_relabels.csv")


def holes(pool: pd.DataFrame, ps: pd.DataFrame) -> None:
    seasons = [f"{y}/{(y + 1) % 100:02d}" for y in range(2019, 2026)]
    allm = (
        ps[ps["scope"] == "all"]
        .pivot_table(index="player_id", columns="season", values="minutes", aggfunc="sum")
        .reindex(columns=seasons)
        .fillna(0)
        .astype(int)
    )
    table = (
        pool.dropna(subset=["tm_player_id"])
        .set_index("player")["tm_player_id"]
        .map(lambda pid: allm.loc[pid] if pid in allm.index else pd.Series(0, index=seasons))
    )
    frame = pd.DataFrame(list(table), index=table.index)
    frame.to_csv(OUT / "pool_season_minutes.csv")
    print("pool players' minutes per season (all scopes):")
    print(frame.to_string())


if __name__ == "__main__":
    main()
