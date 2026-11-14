import pandas as pd
import pytest

from footy.players import linking

sim = linking.name_similarity
THRESHOLD = linking.PLAYER_THRESHOLD


def test_normalise_name_folds_accents_letters_apostrophes_and_hyphens() -> None:
    assert linking.normalise_name("Vinícius Júnior") == "vinicius junior"
    assert linking.normalise_name("N'Golo Kanté") == "ngolo kante"
    assert linking.normalise_name("Heung-Min Son") == "heung min son"
    assert linking.normalise_name("Martin Ødegaard") == "martin odegaard"
    assert linking.normalise_name("Łukasz Skorupski") == "lukasz skorupski"
    assert linking.normalise_name("1. Fußballclub") == "1 fussballclub"
    assert linking.normalise_name("İlkay Gündoğan") == "ilkay gundogan"


def test_accents_and_case_do_not_matter() -> None:
    assert linking.name_similarity("Vinícius Júnior", "Vinicius Junior") == 1.0
    assert linking.name_similarity("KYLIAN MBAPPÉ", "Kylian Mbappe") == 1.0


def test_word_order_does_not_matter() -> None:
    assert linking.name_similarity("Son Heung-min", "Heung-min Son") == 1.0
    assert linking.name_similarity("Kim Min-jae", "Min-jae Kim") == 1.0


def test_short_name_inside_full_legal_name() -> None:
    full = "Vinícius José Paixão de Oliveira Júnior"
    assert sim(full, "Vinicius Junior") == pytest.approx(0.95)
    assert sim("Rodrigo Hernández Cascante", "Rodri") >= THRESHOLD
    assert sim("Rodri", "Rodrigo Hernández") >= THRESHOLD
    assert sim("Rodri", "Rodrigo Hernández") < sim("Rodri", "Rodri")


def test_nickname_beats_full_name() -> None:
    names = ("Lionel Andrés Messi Cuccittini", "Lionel Messi")
    assert linking.best_name_similarity(names, "Lionel Messi") == 1.0
    assert linking.best_name_similarity(("Pedro González López", "Pedri"), "Pedri") == 1.0
    assert linking.best_name_similarity(("Pedro González López", None), "Pedri") < 0.8


def test_particles_and_transliteration() -> None:
    assert sim("Gabriel Fernando de Jesus", "Gabriel Jesus") == pytest.approx(0.95)
    assert sim("Hassan Al Heidos", "Hasan Al-Haydos") >= THRESHOLD
    assert sim("Abdelkarim Hassan Al Haj Fadlalla", "Hasan Al-Haydos") < THRESHOLD


def test_team_normalisation() -> None:
    assert linking.normalise_team("1. Fußballclub Heidenheim 1846") == "heidenheim"
    assert linking.team_similarity("FC Heidenheim", "1. Fußballclub Heidenheim 1846") == 1.0
    assert linking.team_similarity("Côte d'Ivoire", "Ivory Coast") == 1.0
    assert linking.team_similarity("Turkey", "Turkiye") == 1.0
    assert linking.team_similarity("Lyon", "Olympique Lyon") == 1.0
    assert linking.team_similarity("Real Madrid", "Real Sociedad") == 0.5


def _tm_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "date": pd.to_datetime(
                ["2022-12-02", "2022-12-01", "2022-12-10", "2022-12-10", "2022-12-20"]
            ),
            "home_club_name": ["Canada", "Croatia", "Spain", "Spain", "Wales"],
            "away_club_name": ["Morocco", "Belgium", "Portugal", "Portugal", "Iran"],
        }
    )


def test_link_sb_games_date_tolerance_orientation_ambiguity() -> None:
    sb = pd.DataFrame(
        {
            "sb_match_id": [10, 11, 12, 13],
            "date": ["2022-12-01", "2022-12-01", "2022-12-10", "2022-12-17"],
            "home_team": ["Canada", "Belgium", "Spain", "Wales"],
            "away_team": ["Morocco", "Croatia", "Portugal", "Iran"],
        }
    )
    out = linking.link_sb_games(sb, _tm_games()).set_index("sb_match_id")
    assert out.loc[10, "game_id"] == 1 and out.loc[10, "status"] == "linked"  # +1 day
    assert out.loc[11, "game_id"] == 2 and bool(out.loc[11, "swapped"])
    assert out.loc[12, "status"] == "ambiguous" and pd.isna(out.loc[12, "game_id"])
    assert out.loc[13, "status"] == "no_candidate"  # three days off


def _link(
    sb: pd.DataFrame,
    tm_game: pd.DataFrame,
    tm_team: pd.DataFrame | None = None,
    tm_country: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = pd.DataFrame({"sb_match_id": [1], "team": ["France"], "game_id": [7], "club_id": [3]})
    empty_team = pd.DataFrame(columns=["club_id", "player_id", "name"])
    empty_country = pd.DataFrame(columns=["country", "player_id", "name"])
    return linking.link_sb_players(
        sb,
        teams,
        tm_game,
        empty_team if tm_team is None else tm_team,
        empty_country if tm_country is None else tm_country,
    )


def _sb(names: list[tuple[int, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sb_match_id": 1,
            "sb_player_id": [i for i, _, _ in names],
            "player_name": [n for _, n, _ in names],
            "player_nickname": [k for _, _, k in names],
            "team": "France",
        }
    )


def test_two_players_with_the_same_surname_in_one_lineup() -> None:
    sb = _sb(
        [
            (1, "Theo Bernard François Hernández", None),
            (2, "Lucas François Bernard Hernández Pi", None),
            (3, "Kylian Mbappé Lottin", "Kylian Mbappé"),
        ]
    )
    tm = pd.DataFrame(
        {
            "game_id": 7,
            "club_id": 3,
            "player_id": [100, 200, 300],
            "name": ["Lucas Hernández", "Kylian Mbappé", "Theo Hernández"],
        }
    )
    mapping, issues = _link(sb, tm)
    got = mapping.set_index("sb_player_id")["tm_player_id"].to_dict()
    assert got == {1: 300, 2: 100, 3: 200}
    assert (mapping["method"] == "game").all() and issues.empty


def test_one_to_one_assignment_and_pool_tiers() -> None:
    sb = _sb(
        [
            (1, "Lautaro Javier Martínez", "Lautaro Martínez"),
            (2, "Lisandro Martínez", None),
            (3, "Damián Emiliano Martínez Romero", "Emiliano Martínez"),
            (4, "Nobody Known", None),
        ]
    )
    # Only one Martínez in the game's events: the other two must not both claim him.
    tm_game = pd.DataFrame(
        {"game_id": [7], "club_id": [3], "player_id": [10], "name": ["Lautaro Martínez"]}
    )
    tm_team = pd.DataFrame(
        {
            "club_id": [3, 3],
            "player_id": [10, 20],
            "name": ["Lautaro Martínez", "Lisandro Martínez"],
        }
    )
    tm_country = pd.DataFrame(
        {
            "country": [linking.normalise_team("France")] * 2,
            "player_id": [30, 31],
            "name": ["Emiliano Martínez", "Emiliano Insúa"],
        }
    )
    mapping, issues = _link(sb, tm_game, tm_team, tm_country)
    got = mapping.set_index("sb_player_id")
    assert got.loc[1, "tm_player_id"] == 10 and got.loc[1, "method"] == "game"
    assert got.loc[2, "tm_player_id"] == 20 and got.loc[2, "method"] == "team_pool"
    assert got.loc[3, "tm_player_id"] == 30 and got.loc[3, "method"] == "country"
    assert pd.isna(got.loc[4, "tm_player_id"])
    assert issues.set_index("sb_player_id").loc[4, "kind"] == "unlinked"
    assert not mapping["tm_player_id"].dropna().duplicated().any()
