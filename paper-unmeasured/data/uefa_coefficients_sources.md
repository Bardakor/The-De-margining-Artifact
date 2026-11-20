# UEFA association club coefficients (men) 2019/20-2025/26: sources and checks

Curated 2026-11-20 for `uefa_coefficients.csv` (23 rows = the 23 covered leagues whose association is a UEFA member; rows in the order of the 2025/26 ranking).
Columns: `league` (competition code of our data), `association` (UEFA's name), one column per season 2019/20-2025/26 (the association's season coefficient), `total_5y` (sum of 2021/22-2025/26, the association ranking at the end of 2025/26), `source_primary`, `source_check`.
Used by pre-registration A23 (i) (`scripts/players/run_league.py --stage validate`): the mean of an association's season values over the seasons it has one.

## What a season coefficient is
UEFA's page on the calculation (https://www.uefa.com/nationalassociations/uefarankings/country/, read 2026-11-20; it reproduces Annex D of the Champions League regulations): the points obtained by all the association's clubs in the three UEFA club competitions in a season, divided by the number of its clubs that took part. European games only, clubs only. The ranking is the sum over five seasons.

## Sources: two per value, both read with the fetch tool, nothing from memory
1. UEFA (primary): the coefficient data service on UEFA's own host, `comp.uefa.com/v2/coefficients` with `coefficientType=MEN_ASSOCIATION`, `coefficientRange=OVERALL`.
   - `seasonYear=2026`: seasons 2022-2026 (2021/22-2025/26) and the total. `lastUpdateDate` 2026-07-03T14:22:35+0000.
   - `seasonYear=2024`: seasons 2020-2024 (2019/20-2023/24) and the total. `lastUpdateDate` 2025-03-01T11:06:35+0000.
   - Full URLs in every CSV row.
2. kassiesa.net (check; not a UEFA site): "UEFA Country Ranking 2026" https://kassiesa.net/uefa/data/method5/crank2026.html (21/22-25/26) and "UEFA Country Ranking 2024" https://kassiesa.net/uefa/data/method5/crank2024.html (19/20-23/24). Whether the site computes the coefficients from results itself or takes them from UEFA was not checked on the pages read.

## UEFA's ranking page: table not reached
- https://www.uefa.com/nationalassociations/uefarankings/country/ (with and without `?year=2026`) gave the fetch tool only the article on the calculation, no table of values. Not worked around in a browser.
- That this page draws its table from the data service above was not checked. The service is on uefa.com and carries UEFA's `lastUpdateDate`; that is the ground for calling it UEFA's own.
- Each JSON response is about 121,000 characters; the fetch tool read the first 100,000. The 2026 response was read down to rank 44 and the 23 associations rank 1-28; all 23 were present in the part of the 2024 response read (ranks 1-26).

## Checks
- Agreement: 7 season values x 23 associations = 161 values, each identical in UEFA's service and on kassiesa.net. The seasons 2021/22-2023/24 are in both rankings, so they were read twice from each source: identical. The 23 totals of the 2025/26 ranking and the 23 totals of the 2023/24 ranking are identical between the two sources.
- Arithmetic (independent of any source agreeing with another): in each of the four tables read, the five season values of every association sum exactly to the published total (28 associations checked in each 2026 table, 23 and 26 in the 2024 tables).
- The comparison was run by a script on the four tables as returned by the fetch tool, not by eye.
- Values carry three decimals. UEFA's service drops trailing zeros (23.0); the CSV writes three decimals. Values such as 11.666 and 4.666 show that the third decimal is cut, not rounded (our reading of the numbers, not a statement read in a source).

## Russia (RU1): four seasons left empty
- UEFA statement of 2 May 2022 (https://www.uefa.com/uefaeuropaleague/news/0275-150cf4b3a571-e696f0fab20c-1000--suspension-of-russian-teams-and-clubs/): no Russian club takes part in the 2022/23 UEFA club competitions, and for 2022/23 Russia is assigned the lowest number of points it earned in the last five seasons, 4.333 for the men's association club coefficient.
- Both sources publish 4.333 for Russia in each of 2022/23, 2023/24, 2024/25 and 2025/26; kassiesa.net lists 0 teams for Russia in both rankings. No UEFA statement was read for the three seasons after 2022/23.
- These four values are assigned, not earned in matches, so the four cells are **empty** in the CSV. 2019/20-2021/22 (4.666, 4.333, 5.300) are as published; 2020/21 is an earned 4.333.
- `total_5y` for Russia is the published 22.632 = 5.300 + 4 x 4.333. It is the only row whose season cells do not sum to `total_5y`.

## League code -> association
Read from the club names of each competition's league games in `data/players/processed/tm_games.csv.gz`: GB1 England, ES1 Spain, IT1 Italy, L1 Germany, FR1 France, PO1 Portugal, NL1 Netherlands, BE1 Belgium, TR1 Türkiye, SC1 Scotland, GR1 Greece, DK1 Denmark, RU1 Russia, UKR1 Ukraine, A1 Austria (Altach, Wolfsberger AC), C1 Switzerland (Lugano, Winterthur), TS1 Czechia (Karviná, České Budějovice), KR1 Croatia (Dinamo Zagreb, Osijek), NO1 Norway, SE1 Sweden, PL1 Poland, RO1 Romania, SER1 Serbia.

## No row
ARG1, AUS1, BRA1, COL1, JAP1, MEX1, MLS1, RSK1 (South Korea: Pohang, Jeju), SA1: their countries are not among the 55 associations of the kassiesa.net tables. No UEFA coefficient exists for them.

## Not used
- The 2026/27 season, in progress on the curation date and after the study window.
- UEFA's club coefficients (per club) and the bonus points detail: only the association's season value is in the file.
