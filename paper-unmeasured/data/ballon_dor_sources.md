# Ballon d'Or (men) 2021-2026: sources and checks

Curated 2026-11-14 for `ballon_dor.csv` (210 rows = 7 editions x 30 ranked players: 2019, 2021-2026; 2019 serves only as the "previous edition" for 2021).
Scoring: 2021-2023 five picks per voter (6-4-3-2-1 = 16 pts/voter). 2024-2026 ten picks per voter (15-12-10-8-7-5-4-3-2-1 = 67 pts/voter).

## Official sources: not reached
- ballondor.com: HTTP 403 to the fetch tool; not worked around.
- lequipe.fr / France Football: fetch blocked by the tool.
- Consequence: no edition was checked against the official page. Every edition was checked against 2-3 independent outlets instead (URLs in each CSV row). Where sources disagreed, no official tiebreaker was available (see Disagreements).

## Arithmetic check (independent of any outlet)
Sum of the 30 published totals = voters x points-per-voter exactly, so no points are missing or mistyped in any edition:
| Edition | Sum of points | Implied voters |
|---|---|---|
| 2021 | 2720 | 170 x 16 |
| 2022 | 1488 | 93 x 16 |
| 2023 | 1472 | 92 x 16 (UEFA/dailymercato: 92 journalists) |
| 2024 | 6633 | 99 x 67 |
| 2025 | 6700 | 100 x 67 (SI: 100 journalists) |
| 2026 | 6700 | 100 x 67 |

Total points are not published as a figure by any source; the sums above are ours.

## 2020
Cancelled (COVID-19). No edition, no rows.

## 2019 (added; only as the "previous edition" reputation feature for 2021)
- Rows: 30 (210 rows in the CSV with 2019). All 30 have points. Sum = 2816 = 176 x 16 (dailymercato also states 176 journalists, 2,816 points in total).
- Ceremony: 2 Dec 2019, Theatre du Chatelet, Paris.
- Voting period: calendar year 2019 (Wikipedia: "best footballers in the world for 2019"; nominees announced 21 Oct 2019). Not a club season.
- Scoring: 6-4-3-2-1 per ballot.
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2019_Ballon_d%27Or (raw wikitext) ; https://www.topendsports.com/sport/soccer/awards/ballondor-2019.htm ; https://www.dailymercato.com/news/ballon-d-or-2019-messi-n-a-gagne-qu-avec-7-points-d-avance-sur-van-dijk (top places, voter count and total only; full table not read).
- Ranks, ties and points identical between Wikipedia and topendsports. Ties: 20 (Aubameyang, Tadic), 24 (Koulibaly, ter Stegen), 26 (Benzema, Wijnaldum), 28 (Joao Felix, Marquinhos, van de Beek). No disagreement found. Official page not reachable, same caveat as above.
- Club: last club listed, i.e. at the ceremony (De Jong, Griezmann, Barcelona; Hazard, Real Madrid; De Ligt, Juventus; Joao Felix, Atletico Madrid). Wikipedia lists both clubs for these.

## Per edition
Number ranked = 30 in every edition. All 30 have published points in every edition (no empty `points`).

### 2021
- Ceremony: 29 Nov 2021, Theatre du Chatelet, Paris.
- Voting period: calendar year 2021 (Wikipedia: "best footballers in the world in 2021"; nominees announced 8 Oct 2021). Not a club season.
- Voters: 170 (implied by arithmetic only).
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2021_Ballon_d%27Or ; https://www.dailymercato.com/news/le-classement-complet-du-ballon-dor-2021 ; https://www.topendsports.com/sport/soccer/awards/ballondor-2021.htm ; https://www.thenationalnews.com/sport/football/2021/11/30/lionel-messi-wins-for-record-seventh-time-2021-ballon-dor-rankings-in-full (ranks only, no points).
- Points identical across Wikipedia (raw wikitext), dailymercato, topendsports.

### 2022
- Ceremony: 17 Oct 2022, Theatre du Chatelet, Paris.
- Voting period: 2021-22 club season (first season-based edition). Exact dates not stated in the sources read.
- Voters: 93 (implied by arithmetic only).
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2022_Ballon_d%27Or ; https://www.dailymercato.com/news/ballon-dor-2022-le-detail-du-classement-avec-les-points ; https://www.topendsports.com/sport/soccer/awards/ballondor-2022.htm ; https://www.sportsmole.co.uk/football/man-utd/ballon-dor/news/final-ballon-dor-rankings-in-full-as-karim-benzema-takes-top-prize_497228.html (ranks only).
- Points identical across Wikipedia, dailymercato, topendsports.

### 2023
- Ceremony: 30 Oct 2023, Theatre du Chatelet, Paris.
- Voting period: 1 Aug 2022 to 31 Jul 2023 (Wikipedia). Season-based.
- Voters: 92 journalists (dailymercato).
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2023_Ballon_d%27Or ; https://www.uefa.com/news/0278-15d78e1c84bc-6afe5b9cc349-1000--ballon-d-or-2023/ ; https://www.topendsports.com/sport/soccer/awards/ballondor-2023.htm ; https://www.dailymercato.com/news/ballon-dor-le-top-10-detaille (top 10 only).
- Ranks and points identical across Wikipedia, UEFA, topendsports (tie only at 28).

### 2024
- Ceremony: 28 Oct 2024, Theatre du Chatelet, Paris.
- Voting period: 1 Aug 2023 to 31 Jul 2024 (Wikipedia). Season-based.
- Voters: 99 (implied by arithmetic); TNT: one journalist per top-100 FIFA nation; Rodri's maximum possible was 1,485 = 99 x 15.
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2024_Ballon_d%27Or ; https://www.topendsports.com/sport/soccer/awards/ballondor-2024.htm ; https://www.tntsports.co.uk/football/ballon-dor-complete-standings-rodri-41-points-clear-vinicius-jr-full-list-votes-revealed_sto20052531/story.shtml
- Ranks and points identical across the three except the tie labels below.

### 2025
- Ceremony: 22 Sep 2025, Theatre du Chatelet, Paris.
- Voting period: 1 Aug 2024 to 13 Jul 2025 (Wikipedia; Club World Cup final). Season-based.
- Voters: 100 journalists (SI).
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2025_Ballon_d%27Or ; https://www.si.com/soccer/2025-ballon-dor-voting-breakdown-revealed (full 30) ; https://www.uefa.com/uefachampionsleague/news/029e-1eeae3a29e5c-64e217276491-1000--2025-ballon-d-or-voting-results-official-rankings (top 10 only).
- No ties published; equal points at 22/23 (21) and 24/25 (20) are ranked separately by both Wikipedia and SI.

### 2026 (EXISTS; announced and held before 2026-11-14)
- Ceremony: 26 Oct 2026, London Palladium (first edition outside France/Switzerland). Nominees announced 8 Sep 2026 (BeIN Sports / France Football, per search result; not re-verified on a page).
- Voting period: 3 Aug 2025 to 19 Jul 2026 (Wikipedia, citing ballondor.com regulations; 19 Jul 2026 is the World Cup final date). Season-based. This ends before the study window's end (2026-08-31), so it falls inside the window.
- Voters: 100 (implied by arithmetic; SI: journalists from FIFA member associations each rank 10).
- Source URLs: Wikipedia https://en.wikipedia.org/wiki/2026_Ballon_d%27Or ; https://www.si.com/soccer/2026-ballon-dor-voting-breakdown-revealed-kane-defeats-mbappe-slimmest-margins (full 30 as reported) ; https://sports.yahoo.com/articles/top-10-ballon-dor-rankings-235628080.html (top 10 only). A ballondor.com article exists (https://ballondor.com/news/posts/2026-mens-ballon-dor-the-full-ranking-and-the-28-points-that-kept-mbappe-from-the-trophy) but returned 403; not used.
- One Yahoo article gives the Kane margin as 18 points; its own table, SI and Wikipedia give 28 (1,055 vs 1,027). Treated as an outlet error.

## Disagreements (all about tie labels or club; no points disagreement anywhere)
Rule applied: rank is shared only where the sources agree the players are tied ("="); ties are never invented from equal points alone, because outlets rank some equal-points players separately (tiebreaks). The check script counts such pairs.

1. 2021, ranks 12/13, 17/18, 19/20. Wikipedia and topendsports show Lukaku/Chiellini tied 12th (26), Suarez/Kjaer tied 17th (8), Mount/Mahrez tied 19th (7). dailymercato and The National (AFP list) rank them separately (12 Lukaku, 13 Chiellini, 14 Bonucci; 17 Suarez, 18 Kjaer; 19 Mount, 20 Mahrez) while still marking ties at 21, 26, 29. CSV follows dailymercato/The National. Official page not reachable; confidence medium. Impact: rank only.
2. 2022, ranks 20-30. Wikipedia puts all 11 zero-point players at 20. dailymercato, topendsports and Sports Mole give Ronaldo 20, Kane 21, tie 22 (Alexander-Arnold, Foden, B. Silva), tie 25 (Kimmich, Maignan, Rudiger, Cancelo, Nkunku, Nunez). CSV follows the three-outlet version. Wikipedia and the others agree on ties at 14 (Fabinho/Leao, 2 pts) and 17 (Casemiro/Diaz/Vlahovic, 1 pt); Haller (13th, 2 pts) is ranked separately by all.
3. 2022, Leao points. topendsports shows "Not listed" for Leao; Wikipedia and dailymercato give 2. Used 2 (confirmed by the 1488 arithmetic).
4. 2024. topendsports additionally ties Dias/Saliba (23), Rice/Vitinha (26); Wikipedia and TNT rank them separately. CSV: separate; only 29 (Dovbyk/Hummels) is tied (all three agree).
5. 2026, rank 29. SI marks "T-29" (Mane, Saliba); Wikipedia lists both as 29. Agreement; recorded as tie. SI's page text does not number 30th place (consistent with the tie).
6. 2025 topendsports page is stale (shows 5-pick scoring, six names); not used.

## Club column
Rule: club as published, taking the most recent club listed when a source lists several (for example "Paris puis Miami"), i.e. the club at or near the ceremony. Sources disagree because Wikipedia tables list the club of the season for some players. Where this changes the value:
- 2022: Mane Bayern Munich (Wikipedia: Liverpool); Lewandowski Barcelona (Bayern); Haaland Manchester City (Dortmund); Haller Borussia Dortmund (Ajax); Casemiro Manchester United (Real Madrid); Nunez Liverpool (Benfica); Ruediger Real Madrid (Chelsea). Source: topendsports, dailymercato.
- 2021: Ronaldo Manchester United (Juventus before Aug 2021); Donnarumma PSG; Lukaku Chelsea; Messi PSG.
- 2023 (UEFA, last club listed): Bounou Al-Hilal (Wikipedia: Sevilla); Benzema Al-Ittihad (Real Madrid); Gundogan Barcelona; Bellingham Real Madrid; Kane, Kim Min-jae Bayern Munich; Onana Manchester United; Gvardiol Manchester City; Kolo Muani PSG; Messi Inter Miami.
- 2024: club as published by Wikipedia, topendsports and TNT, even where the player moved before the ceremony: Olmo RB Leipzig (joined Barcelona Aug 2024), Dovbyk Girona (joined Roma Jul 2024). Treat these two as published-at-nomination clubs; join on player name, not club.
- 2025 (SI, last club listed): Donnarumma Manchester City (Wikipedia: PSG); Gyokeres Arsenal (Sporting CP); Wirtz Liverpool (Bayer Leverkusen); McTominay Napoli; Kvaratskhelia PSG.
- 2026: Wikipedia table as is (Diaz Bayern Munich, Quinones Al-Qadsiah, Mane Al Nassr, Messi Inter Miami).
Club is a disambiguation hint only; use `player` (and nationality) for linking to Transfermarkt.

## Names
Spelled as the player's full common name with diacritics (e.g. "Vinícius Júnior", "Rodri", "Kylian Mbappé", "İlkay Gündoğan", "Rúben Dias", "Gabriel Magalhães"). Not matched against Transfermarkt in this task; the linker should normalise accents.

## Ties in the CSV (rank shared)
2021: 21 (Fernandes, Lautaro), 26 (Moreno, Barella, Dias), 29 (Modric, Azpilicueta). 2022: 14, 17, 22, 25. 2023: 28. 2024: 29. 2025: none. 2026: 29.
