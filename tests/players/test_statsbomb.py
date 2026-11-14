import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from footy.players import statsbomb as sb
from footy.players import xt

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Any:
    return json.loads((FIXTURES / f"sb_{name}.json").read_text())


@pytest.fixture(scope="module")
def flat() -> pd.DataFrame:
    return sb.flatten_events(1, _load("events"))


@pytest.fixture(scope="module")
def table(flat: pd.DataFrame) -> pd.DataFrame:
    minutes = sb.event_minutes(1, _load("events"))
    return sb.player_match_table(minutes, sb.player_counts(flat)).set_index("player_id")


def test_progressive_rule_on_toy_coordinates() -> None:
    # (start, end): 60 -> 30 from goal; exactly 25% closer; 24% closer; starts at x = 47.9;
    # starts at x = 48 exactly; backwards.
    starts = np.array([[60, 40], [80, 40], [80, 40], [47.9, 40], [48, 40], [100, 40]])
    ends = np.array([[90, 40], [90, 40], [89.6, 40], [110, 40], [110, 40], [90, 40]])
    got = sb.is_progressive(starts[:, 0], starts[:, 1], ends[:, 0], ends[:, 1])
    assert got.tolist() == [True, True, False, False, True, False]


def test_penalty_box_bounds() -> None:
    assert sb.in_box([102, 101.9, 110, 110, 110], [40, 40, 18, 62, 62.1]).tolist() == [
        True,
        False,
        True,
        True,
        False,
    ]


def test_minutes_from_starting_xi_substitution_and_card_events() -> None:
    events = _load("events")
    assert sb.period_lengths(events) == {1: 47 * 60.0, 2: 48 * 60.0}
    minutes = sb.event_minutes(1, events).set_index("player_id")["minutes"]
    expected = {
        1: 95.0,  # full match, stoppage included
        2: 46.5,  # substituted at 46:30, in first-half stoppage
        6: 48.5,  # his replacement: 0.5 + 48
        7: 47.0,  # off at half time: whole first half
        3: 48.0,  # on at half time
        4: 62.0,  # second yellow at 15:00 of the second half: 47 + 15
        11: 95.0,  # two minutes off for treatment are not subtracted
        12: 95.0,
    }
    assert minutes.to_dict() == pytest.approx(expected)
    assert 13 not in minutes.index  # sent off from the bench, never played


def test_shootout_dropped_and_goals_match_score(flat: pd.DataFrame) -> None:
    assert flat["period"].max() == 2
    shots = sb.shots_table(flat)
    assert len(shots) == 1
    assert shots.loc[0, "shot_type"] == "Open Play"
    match = sb.matches_table("Cup", "2022", _load("matches")).iloc[0]
    assert bool(match["knockout"])
    og = sb.own_goals(flat).set_index("team")["own_goals"]
    home_goals = (shots["outcome"] == "Goal").sum() + og.get("Home FC", 0)
    assert (home_goals, og.get("Away FC", 0)) == (match["home_score"], match["away_score"])


def _row(table: pd.DataFrame, player: int, *columns: str) -> list[float]:
    values: list[float] = (
        table[list(columns)].to_numpy(dtype=float)[table.index.get_loc(player)].tolist()
    )
    return values


def test_player_counts(table: pd.DataFrame) -> None:
    assert _row(table, 1, "passes", "passes_completed", "progressive_passes") == [1, 1, 1]
    assert _row(table, 1, "shots", "touches_in_box", "sb_xg") == [1, 1, pytest.approx(0.3)]
    prog = ("progressive_carries", "progressive_passes", "passes_into_box", "key_passes")
    assert _row(table, 3, *prog) == [1, 1, 1, 1]
    assert _row(table, 3, "touches_in_box", "sb_xa") == [0, pytest.approx(0.3)]
    # throw-in completed but never progressive; second pass incomplete
    assert _row(table, 2, "passes", "passes_completed", "progressive_passes") == [2, 1, 0]
    assert _row(table, 4, "passes_completed", "progressive_passes") == [1, 0]  # from x = 40
    assert _row(table, 6, "dribbles_completed") == [1]
    assert _row(table, 11, "pressures", "blocks", "shots") == [1, 1, 0]  # shoot-out dropped
    assert _row(table, 12, "tackles", "interceptions", "clearances") == [1, 1, 1]


def test_xt_added_credits_successful_open_play_moves(
    flat: pd.DataFrame, table: pd.DataFrame
) -> None:
    attempts = sb.moves(flat)
    assert len(attempts) == 5  # throw-in excluded
    assert attempts["success"].tolist() == [True, False, True, True, True]  # P2 incomplete
    grid = xt.XTGrid(np.arange(xt.N_X * xt.N_Y, dtype=float).reshape(xt.N_X, xt.N_Y), 1, 0.0)
    out = sb.add_xt(table.reset_index(), attempts, grid).set_index("player_id")["xt_added"]

    def v(x: float, y: float) -> float:
        return float(xt.value(grid, x, y))

    assert out[3] == pytest.approx(v(110, 40) - v(90, 40))  # carry then pass telescopes
    assert out[1] == pytest.approx(v(90, 40) - v(60, 40))
    assert out[2] == 0.0  # throw-in and incomplete pass get nothing
    assert out[11] == 0.0


def test_ensure_never_refetches(tmp_path: Path) -> None:
    cached = tmp_path / "events" / "1.json.gz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(gzip.compress(b"[]"))
    manifest: dict[str, Any] = {}
    assert sb.ensure(tmp_path, "events/1.json", manifest, delay=0.0) == cached
    assert sb.read_json(cached) == []
    assert manifest == {}


def test_resolve_seasons_rejects_missing() -> None:
    comps = [
        {"competition_name": n, "season_name": s, "competition_gender": "male",
         "competition_id": i, "season_id": i}
        for i, (n, s) in enumerate(sb.SEASONS)
    ]  # fmt: skip
    assert [r[2] for r in sb.resolve_seasons(comps)] == list(range(len(sb.SEASONS)))
    with pytest.raises(ValueError, match="0 entries"):
        sb.resolve_seasons(comps[1:])
