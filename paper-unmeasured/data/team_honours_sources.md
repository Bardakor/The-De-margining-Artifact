# Team honours 2021-2026 (M6 team-success block): rules, sources and checks

Curated 2026-11-16 for `team_honours.csv`, the input of the post-hoc robustness stage
`recognition_curated` of `scripts/players/run_ranking.py`. The primary M6 analysis (A14)
computes the four team-success indicators from tm_games; its own checks found errors that
bias the team block down. This file lists every (edition, player, indicator) of
`results/players/recognition/universe.csv` whose value should differ under the rules below.
Post hoc and exploratory: none of this was pre-registered.

## Rules (fixed before any curated fit)

- **Row format.** One row per (edition, player_id, indicator) that changes; columns edition,
  player_id, name, indicator (league_title | ucl_winner | ucl_finalist | intl_winner),
  computed (0/1, as in universe.csv), curated (0/1), reason, source_primary, source_check.
  A player with two changed indicators in one edition has two rows. Several URLs in one cell
  are separated by " ; " (and " | " between two tournaments).
- **League title** = the official national champion of the season the edition votes on
  (A14: 2022-2026 vote on 2021/22-2025/26, 2021 on 2020/21), play-offs and championship
  rounds included, for a club the player played league minutes for that season, read from
  the panel (panel_player_match, scope league, minutes > 0). A player with league minutes for
  two clubs counts if either is the champion. A season still unfinished at the vote has no
  champion (unchanged from the primary).
- **UCL winner / losing finalist** = the season's final, 2025/26 included. Membership as in
  the primary: at least one UCL minute for the club that season in the panel.
- **Senior international tournament winner** = World Cup, continental championships (Euro,
  Copa América, AFCON, Asian Cup, Gold Cup) and the UEFA and CONCACAF Nations League
  finals; not one-off matches (Finalissima). A player counts only if he is in the winning
  team's final squad list (cited). Each tournament takes `footy.players.window.season_label`
  of its first match date (the study's rule for tournaments), then the edition by A14's
  mapping. For the Nations Leagues the tournament is the finals, so the date is the finals'
  first match (see "Readings not taken").
- **Verification.** Every changed fact (a champion or final result that differs from or is
  missing in the computed data) is checked against two independent sources, Wikipedia plus
  one other page actually fetched. Memory is not a source. Where a fact could not be
  verified, the computed value is kept and the gap is stated below.

## Method

- Wikipedia: raw wikitext (`index.php?title=...&action=raw`) fetched 2026-11-16, read from
  the infobox (`winners`, `champion`, `team1score`, `details`) and, for squads, from the
  `{{nat fs g player}}` entries of the winning team's section (prose on withdrawals read too).
  Squad entries were matched to pool.csv by accent-normalised name; for every winner, the
  pool players of that nationality not found in the squad were checked by searching the
  section text (none was a parsing miss).
- Second sources: fetched with the web-fetch tool (each URL is in the CSV row). Search-engine
  summaries were used only to find URLs, never as a source. BBC Sport could not be fetched
  by the tool and is not used.

## Result: 50 changed indicators, all 0 -> 1

| Indicator | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Total |
|---|---|---|---|---|---|---|---|
| league_title | | | 1 | | 2 | | 3 |
| ucl_winner | | | | | | 10 | 10 |
| ucl_finalist | | | | | | 6 | 6 |
| intl_winner | 2 | 5 | 4 | 1 | 9 | 10 | 31 |

No computed 1 had to become 0.

## League titles

Every league-season holding a universe player was checked against the Wikipedia season page
(`winners` field). Computed = official in all of them except:

| League-season | Computed leader | Official champion (Wikipedia) | Universe players at either club | Change |
|---|---|---|---|---|
| BE1 2021/22 | Union Saint-Gilloise | Club Brugge (2021-22 Belgian First Division A) | none (Caicedo: Beerschot) | none |
| BE1 2022/23 | KRC Genk | Royal Antwerp (2022-23 Belgian Pro League) | Willian Pacho (Antwerp, 2,899 league min) | Pacho 2023 league_title 0 -> 1 |
| BE1 2023/24 | Union Saint-Gilloise | Club Brugge | none in BE1 | none |
| BE1 2024/25 | KRC Genk | Union SG | none in BE1 | none |
| UKR1 2021/22 | Shakhtar Donetsk | title not awarded | none (Dovbyk: Dnipro-1) | none |
| MLS1 2024/25 (the 2025 MLS season) | Philadelphia Union | Inter Miami, MLS Cup 2025 (3-1 v Vancouver, 6 Dec 2025) | Messi, Suárez (Inter Miami); Son (LAFC) | Messi, Suárez 2025 league_title 0 -> 1 |

Second sources for the two changes: Antwerp, ESPN (5 Jun 2023: title clinched 4 Jun 2023 on
the last play-off day, 47 points vs 46); Inter Miami, mlssoccer.com match report (6 Dec 2025).

Checked and equal (one source, Wikipedia season page): Premier League, La Liga, Serie A,
Bundesliga, Ligue 1 2020/21-2025/26; Eredivisie 2020/21-2025/26; Primeira Liga
2020/21-2025/26; Süper Lig 2020/21-2025/26; Russian Premier League 2020/21, 2021/22;
Ukrainian Premier League 2020/21, 2022/23; Saudi Pro League 2024/25, 2025/26 (Al-Ittihad,
Al-Nassr); Brasileirão 2025 = BRA1 2024/25 (Flamengo). Denmark and Greece (also wrong in
the computed table) hold no universe player. MLS1 and BRA1 2025/26 (the 2026 seasons) were
unfinished at the vote: no title, as in the primary.

Note on the mapping: in the study's season labels MLS1 2024/25 and BRA1 2024/25 are the
2025 calendar seasons, which end after the 2025 ceremony. The rule keeps A14's mapping and
only corrects the champion.

## UCL finals

Wikipedia "List of European Cup and UEFA Champions League finals" agrees with the five
computed finals (2020/21 Chelsea-Man City, 2021/22 Real Madrid-Liverpool, 2022/23 Man
City-Inter, 2023/24 Real Madrid-Dortmund, 2024/25 PSG-Inter). The 2025/26 final, absent from
tm_games (its CL games stop at the semi-finals, 6 May 2026): Paris Saint-Germain 1-1
Arsenal after extra time, PSG 4-3 on penalties, 30 May 2026, Puskás Aréna, Budapest
(Wikipedia "2026 UEFA Champions League final"; theScore). Universe players with 2025/26 UCL
minutes: PSG, Hakimi, Vitinha, Dembélé, Kvaratskhelia, Nuno Mendes, Marquinhos, Doué, João
Neves, Fabián Ruiz, Pacho (ucl_winner); Arsenal, Saka, Rice, Ødegaard, Saliba, Gyökeres,
Gabriel Magalhães (ucl_finalist).

## International tournaments

| Tournament | First match | Season -> edition | Winner (result) | In data? | Universe pool players in the final squad | Change |
|---|---|---|---|---|---|---|
| Euro 2020 | 2021-06-11 | 2020/21 -> 2021 | Italy (1-1, 3-2 pens v England) | yes | Chiellini, Jorginho, Barella, Bonucci | none |
| Copa América 2021 | 2021-06-13 | 2020/21 -> 2021 | Argentina (1-0 v Brazil) | no | Messi, Lautaro Martínez | 0 -> 1 |
| CONCACAF NL Finals 2021 | 2021-06-03 | 2020/21 -> 2021 | United States | no | none | none |
| Gold Cup 2021 | 2021-07-10 | 2021/22 -> 2022 | United States (1-0 a.e.t. v Mexico) | no | none | none |
| UEFA NL Finals 2021 | 2021-10-06 | 2021/22 -> 2022 | France (2-1 v Spain) | no | Mbappé, Benzema, Griezmann, Upamecano | 0 -> 1 |
| AFCON 2021 | 2022-01-09 | 2021/22 -> 2022 | Senegal (0-0, 4-2 pens v Egypt) | no | Mané | 0 -> 1 |
| World Cup 2022 | 2022-11-20 | 2022/23 -> 2023 | Argentina (3-3, 4-2 pens v France) | yes | Messi, Lautaro, Julián Álvarez, Mac Allister | none |
| UEFA NL Finals 2023 | 2023-06-14 | 2022/23 -> 2023 | Spain (0-0, 5-4 pens v Croatia) | no | Rodri, Carvajal, Olmo, Fabián Ruiz | 0 -> 1 |
| CONCACAF NL Finals 2023 | 2023-06-15 | 2022/23 -> 2023 | United States | no | none | none |
| Gold Cup 2023 | 2023-06-24 | 2022/23 -> 2023 | Mexico (1-0 v Panama) | no | none (Quiñones not in squad) | none |
| Asian Cup 2023 | 2024-01-12 | 2023/24 -> 2024 | Qatar (3-1 v Jordan) | yes | none | none |
| AFCON 2023 | 2024-01-13 | 2023/24 -> 2024 | Ivory Coast (2-1 v Nigeria) | no | Haller | 0 -> 1 |
| CONCACAF NL Finals 2024 | 2024-03-21 | 2023/24 -> 2024 | United States | no | none | none |
| Euro 2024 | 2024-06-14 | 2023/24 -> 2024 | Spain (2-1 v England) | yes | Rodri, Lamine Yamal, Pedri, Nico Williams, Carvajal, Olmo, Fabián Ruiz, Ferran Torres, Grimaldo, Cucurella | none |
| Copa América 2024 | 2024-06-20 | 2023/24 -> 2024 | Argentina (1-0 a.e.t. v Colombia) | yes | Messi, Lautaro, Julián Álvarez, Mac Allister | none |
| CONCACAF NL Finals 2025 | 2025-03-20 | 2024/25 -> 2025 | Mexico (2-1 v Panama) | no | Quiñones | 0 -> 1 |
| UEFA NL Finals 2025 | 2025-06-04 | 2024/25 -> 2025 | Portugal (2-2, 5-3 pens v Spain) | no | Ronaldo, Bruno Fernandes, Bernardo Silva, Vitinha, Nuno Mendes, Rúben Dias, João Neves, Rafael Leão | 0 -> 1 |
| Gold Cup 2025 | 2025-06-14 | 2024/25 -> 2025 | Mexico (2-1 v United States) | no | Quiñones (unused substitute in the final, US Soccer) | same row as above |
| AFCON 2025 | 2025-12-21 | 2025/26 -> 2026 | Morocco, awarded 3-0 (see below) | yes | Hakimi | none |
| World Cup 2026 | 2026-06-11 | 2025/26 -> 2026 | Spain (1-0 a.e.t. v Argentina, 19 Jul 2026) | group stage only, no final | Rodri, Lamine Yamal, Pedri, Cubarsí, Nico Williams, Olmo, Grimaldo, Fabián Ruiz, Ferran Torres, Cucurella | 0 -> 1 |

Goalkeepers (Donnarumma, Emiliano Martínez, Maignan, Unai Simón, Bounou) are in some of
these squads but not in the universe (A14). Julián Álvarez (Copa América 2021) has no panel
minutes in 2020/21 and is not in the 2021 universe. Not in the winning squad although of that
nationality: Mac Allister (Copa 2021); Kanté, Dembélé and others (UEFA NL 2021); Nico
Williams (withdrew injured 5 Jun 2023), Pedri and others (UEFA NL 2023); Cancelo
(UEFA NL 2025); Carvajal (omitted, World Cup 2026).

For the six tournaments in the data, no universe pool player of the winning squad had 0
minutes (so the squad rule changes nothing there), and no computed 1 is outside a squad.
No CONCACAF Nations League Finals were played in 2025/26 (concacaf.com, 6 Jan 2026: the
2026/27 edition's Finals are in March 2027). Whether any UEFA Nations League finals fell in
2025/26 was not checked.

Second sources (one per tournament, URLs in the CSV): copaamerica.com (Copa 2021); Cyprus
Mail + uefa.com roll of honour (UEFA NL 2021); Al Jazeera (AFCON 2021); Fox News + uefa.com
(UEFA NL 2023); Sky Sports (AFCON 2023); concacaf.com (CONCACAF NL 2025); KSL/AP + uefa.com
(UEFA NL 2025); ussoccer.com (Gold Cup 2025); ESPN (World Cup 2026). Squad membership has one
source each, the Wikipedia squad list (which cites the federations' announcements).

## AFCON 2025: the 0-3 in the data is the official awarded score, not an error

- On the pitch: Senegal 1-0 Morocco after extra time, 18 Jan 2026, Rabat (Pape Gueye),
  after Senegal's players left the pitch for about 14 minutes in protest at a late penalty
  awarded to Morocco.
- 17 Mar 2026: the CAF Appeal Board ruled that Senegal forfeited the final (Articles 82 and
  84 of the AFCON regulations), recorded it as a 3-0 Morocco win and declared Morocco
  champions. Sources: Wikipedia "2025 Africa Cup of Nations final"; Al Jazeera, 19 Mar 2026.
- Senegal appealed to the Court of Arbitration for Sport (filed 25 Mar 2026); the hearing was
  on 8 Oct 2026 in Lausanne and no award had been published (Morocco World News, 8 Oct 2026;
  Wikipedia). Morocco is therefore the official winner as of 2026-11-16, and the computed
  indicator (Hakimi 2026 = 1, Mané 2026 = 0) is kept.
- If CAS reverses the decision: Mané 2026 intl_winner 0 -> 1 and Hakimi 2026 1 -> 0. Not run.

## Not verified, not curated, or readings not taken

- **No panel league minutes.** 15 universe rows have no league minutes in the panel (their
  minutes that season come from other competitions; their league is not among tm_games'
  domestic leagues that season): Palmer, Gvardiol, Gyökeres, Olise, Quiñones (2021);
  Gyökeres (2022); Suárez, Gyökeres (2023); Messi, Benzema, Kanté, Ronaldo, Suárez, Fabinho,
  Quiñones (2024). The rule reads the club from the panel, so these rows are left as
  computed (0) and whether their clubs won a league title was not checked.
- **Squad membership** rests on one source (Wikipedia squad lists); only Quiñones' Gold Cup
  2025 place was seen in a second source.
- **Nations League season.** Under the panel's own definition (an edition = one competition
  id and Transfermarkt season, dated by its first match, which for the Nations League would
  be the league phase), the UEFA NL titles of 2021 and 2023 could fall one edition earlier.
  The league-phase dates were not checked and this reading was not run.
- **Oceania.** The OFC Nations Cup is not in the rule's list; no pool player is from Oceania.
- **Unchanged leagues** were checked on one source only (Wikipedia), not two.
