from datetime import date

from footy.players.window import in_window, season_label


def test_season_label_turns_over_on_first_of_july() -> None:
    assert season_label(date(2021, 6, 30)) == "2020/21"
    assert season_label(date(2021, 7, 1)) == "2021/22"
    assert season_label(date(2099, 8, 1)) == "2099/00"


def test_window_bounds_are_inclusive() -> None:
    assert in_window(date(2020, 1, 1))
    assert in_window(date(2026, 8, 31))
    assert not in_window(date(2019, 12, 31))
    assert not in_window(date(2026, 9, 1))
