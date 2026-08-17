"""The football-data.co.uk column registry.

Spec §3.1. Ingest discovers headers against this registry rather than assuming
a fixed schema, so the registry's internal consistency is load-bearing: a name
that maps to the wrong bookmaker, market or period would mis-attribute prices
and corrupt every downstream book sum without raising anything.
"""

import dataclasses

import pytest

from footy.data.columns import (
    ASIAN_COLUMNS,
    BOOKMAKERS,
    COUNT_COLUMNS,
    KNOWN_COLUMNS,
    ODDS_MAP,
    PINNACLE_ALIASES,
    RESULT_COLUMNS,
    OddsColumn,
)


def test_every_bookmaker_has_a_full_1x2_and_ou_set() -> None:
    for stem in BOOKMAKERS:
        for outcome in ("H", "D", "A"):
            assert f"{stem}{outcome}" in ODDS_MAP
            assert f"{stem}C{outcome}" in ODDS_MAP
        for flag in (">2.5", "<2.5"):
            assert f"{stem}{flag}" in ODDS_MAP
            assert f"{stem}C{flag}" in ODDS_MAP


def test_closing_columns_insert_c_before_the_outcome() -> None:
    """MODEL-adjacent convention: B365H is opening, B365CH is closing."""
    assert ODDS_MAP["B365H"] == OddsColumn("Bet365", "1X2", "H", "open")
    assert ODDS_MAP["B365CH"] == OddsColumn("Bet365", "1X2", "H", "close")
    assert ODDS_MAP["B365>2.5"] == OddsColumn("Bet365", "OU25", "O", "open")
    assert ODDS_MAP["B365C<2.5"] == OddsColumn("Bet365", "OU25", "U", "close")


def test_opening_and_closing_are_never_conflated() -> None:
    """Benchmarking against opening odds instead of closing would be a silent
    methodological error, so no name may carry both periods."""
    for name, spec in ODDS_MAP.items():
        expected = "close" if _looks_closing(name, spec.book) else "open"
        assert spec.period == expected, f"{name} claims period {spec.period}"


def _looks_closing(name: str, book: str) -> bool:
    if name.startswith(("PC", "PSC")):
        return True
    stems = [s for s, b in BOOKMAKERS.items() if b == book]
    return bool(stems) and name.startswith(f"{stems[0]}C")


def test_bfd_resolves_to_betfair_draw_not_the_betfred_stem() -> None:
    """`BF` + `D` collides with the `BFD` stem. football-data.co.uk publishes
    BFH/BFD/BFA as Betfair, and Betfred as BFDH/BFDD/BFDA, so Betfair must win
    the bare name. Pinned because the collision is invisible on inspection."""
    assert ODDS_MAP["BFD"] == OddsColumn("Betfair", "1X2", "D", "open")
    assert ODDS_MAP["BFDH"] == OddsColumn("Betfred", "1X2", "H", "open")
    assert ODDS_MAP["BFDD"] == OddsColumn("Betfred", "1X2", "D", "open")


def test_bfd_is_the_only_stem_collision() -> None:
    """If a new bookmaker introduces another collision it must be adjudicated
    deliberately, not discovered later as mis-attributed prices."""
    collisions = {
        f"{stem}{outcome}"
        for stem in BOOKMAKERS
        for outcome in ("H", "D", "A")
        if f"{stem}{outcome}" in BOOKMAKERS
    }
    assert collisions == {"BFD"}


def test_pinnacle_aliases_and_canonicals_agree_on_meaning() -> None:
    """The alias must describe the same price as its canonical, or preferring
    one over the other would change what is measured."""
    for preferred, aliases in PINNACLE_ALIASES:
        assert len(preferred) == len(aliases)
        for canonical, alias in zip(preferred, aliases, strict=True):
            assert ODDS_MAP[canonical] == ODDS_MAP[alias], f"{canonical} vs {alias}"


def test_pinnacle_alias_groups_are_registered() -> None:
    for preferred, aliases in PINNACLE_ALIASES:
        for name in (*preferred, *aliases):
            assert name in ODDS_MAP, name


def test_every_odds_name_is_a_known_column() -> None:
    assert set(ODDS_MAP) <= KNOWN_COLUMNS


def test_result_and_odds_namespaces_do_not_overlap() -> None:
    """A name in both would be skipped as a result column and never priced."""
    assert RESULT_COLUMNS.isdisjoint(set(ODDS_MAP))


def test_asian_and_count_columns_are_known_but_not_priced() -> None:
    """Asian handicap columns are recognised so they do not trip the loud
    failure, but they carry a line rather than a simplex and are not ingested."""
    assert ASIAN_COLUMNS <= KNOWN_COLUMNS
    assert COUNT_COLUMNS <= KNOWN_COLUMNS
    assert ASIAN_COLUMNS.isdisjoint(set(ODDS_MAP))
    assert COUNT_COLUMNS.isdisjoint(set(ODDS_MAP))


def test_markets_and_outcomes_are_from_the_declared_domain() -> None:
    for name, spec in ODDS_MAP.items():
        assert spec.market in ("1X2", "OU25"), name
        assert spec.period in ("open", "close"), name
        expected = ("H", "D", "A") if spec.market == "1X2" else ("O", "U")
        assert spec.outcome in expected, name


def test_odds_column_is_immutable() -> None:
    """Specs are shared across every row of the odds table; mutation would be
    action at a distance."""
    spec = ODDS_MAP["B365H"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.book = "tampered"  # type: ignore[misc]
