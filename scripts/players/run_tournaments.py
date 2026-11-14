"""M7(a): how much event-visible on-ball value the box score sees (exploratory).

Reads sb_player_match and sb_matches, keeps outfield player-competitions with at least
MIN_MINUTES minutes, and runs ``footy.players.tournaments.visibility`` per position
group (and pooled over outfield groups) in two scopes: every StatsBomb competition, and
national-team tournaments only (the club seasons are one team's matches, so their
qualifying players are almost all from that one team). Writes
results/players/tournaments/visibility.csv and samples.json and prints a summary.

Usage:
    uv run python scripts/players/run_tournaments.py
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from footy.players import statsbomb as sb
from footy.players import tournaments as tr
from footy.players.paths import PROCESSED, RESULTS

SEED = 20261114
MIN_MINUTES = 270.0
MIN_UNITS = 30
N_BOOT = 1000
TOURNAMENTS = ("FIFA World Cup", "UEFA Euro", "Copa America", "African Cup of Nations")
GROUPS = ("ALL", "CB", "FB", "DM", "CM", "AM", "W", "CF")


def load() -> pd.DataFrame:
    pm = pd.read_csv(PROCESSED / "sb_player_match.csv.gz")
    matches = pd.read_csv(PROCESSED / "sb_matches.csv.gz").set_index("sb_match_id")
    pm["competition"] = pm["sb_match_id"].map(matches["competition"])
    pm["season"] = pm["competition"] + " " + pm["sb_match_id"].map(matches["season"])
    pm["position_group"] = pm["position"].map(sb.POSITION_GROUPS)
    missing = pm["position_group"].isna()
    if missing.any():
        raise ValueError(f"unmapped positions: {set(pm.loc[missing, 'position'])}")
    pm["tackles_interceptions"] = pm["tackles"] + pm["interceptions"]
    return pm


def main() -> None:
    pm = load()
    units = tr.unit_groups(pm).rename(columns={"position_group": "group"})
    eligible = units[(units["minutes"] >= MIN_MINUTES) & (units["group"] != "GK")]
    pm = pm.join(eligible[["group"]], on=["player_id", "season"], how="inner")
    rng = np.random.default_rng(SEED)
    scopes = {"all": pm, "tournaments": pm[pm["competition"].isin(TOURNAMENTS)]}

    frames = []
    samples: dict[str, Any] = {}
    for scope, data in scopes.items():
        samples[scope] = {"matches": int(data["sb_match_id"].nunique()), "groups": {}}
        for group in GROUPS:
            sub = data if group == "ALL" else data[data["group"] == group]
            n_units = len(sub.groupby(["player_id", "season"]))
            samples[scope]["groups"][group] = {
                "units": n_units,
                "players": int(sub["player_id"].nunique()),
                "player_matches": len(sub),
                "minutes_median_per_unit": float(sub.groupby(["player_id", "season"])["minutes"]
                                                 .sum().median()),
            }  # fmt: skip
            if n_units < MIN_UNITS:
                continue
            out = tr.visibility(sub, tr.TARGETS, tr.BOX_FEATURES, rng, n_boot=N_BOOT)
            frames.append(out.reset_index().assign(scope=scope, group=group))
            print(f"done {scope} {group} ({n_units} units)", flush=True)

    club = pm[~pm["competition"].isin(TOURNAMENTS)]
    club_units = club.groupby(["season", "player_id"])["team"].agg(lambda t: t.mode()[0])
    by_team = club_units.reset_index().groupby(["season", "team"]).size()
    by_team.index = by_team.index.map(" | ".join)
    samples["club_units_by_team"] = by_team.astype(int).to_dict()
    samples["settings"] = {
        "seed": SEED, "min_minutes": MIN_MINUTES, "min_units": MIN_UNITS, "n_boot": N_BOOT,
        "level": 0.95, "box_features": list(tr.BOX_FEATURES), "targets": list(tr.TARGETS),
        "exploratory": True,
    }  # fmt: skip

    table = pd.concat(frames, ignore_index=True)
    table = table[["scope", "group", *table.columns.drop(["scope", "group"])]]
    out_dir = RESULTS / "tournaments"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "visibility.csv", index=False, float_format="%.4f")
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    for scope in scopes:
        print(f"\n[{scope}] samples:", {g: v["units"] for g, v in samples[scope]["groups"].items()})
        sub = table[table["scope"] == scope]
        for name in ("r2_cross", "r2_disatt"):
            cell = (
                sub[name].map("{:.2f}".format)
                + " ["
                + sub[f"{name}_lo"].map("{:.2f}".format)
                + ", "
                + sub[f"{name}_hi"].map("{:.2f}".format)
                + "]"
            )
            pivot = sub.assign(cell=cell).pivot(index="target", columns="group", values="cell")
            print(f"\n{name} (95% player-bootstrap CI), {scope}:")
            print(pivot.reindex(index=list(tr.TARGETS), columns=[g for g in GROUPS if g in pivot]))
    print(f"\nwrote {out_dir / 'visibility.csv'} ({len(table)} rows) and samples.json")


if __name__ == "__main__":
    main()
