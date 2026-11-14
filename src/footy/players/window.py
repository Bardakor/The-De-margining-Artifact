"""Study window and season labels (design §3)."""

from datetime import date

WINDOW_START = date(2020, 1, 1)
WINDOW_END = date(2026, 8, 31)


def in_window(day: date) -> bool:
    return WINDOW_START <= day <= WINDOW_END


def season_label(day: date) -> str:
    """Club season running 1 July to 30 June, e.g. 2021-09-01 -> '2021/22'."""
    start = day.year if day.month >= 7 else day.year - 1
    return f"{start}/{(start + 1) % 100:02d}"
