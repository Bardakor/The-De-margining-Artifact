# League strength — literature, data facts, and a minimal fix

Written 2026-11-18 for paper 2 (branch `research/unmeasured-remainder`). Read-only
on code and data; nothing here has been fitted. All counts come from
`data/players/processed/` (Transfermarkt snapshot 2026-09-05, window 2020-01-01 to
2026-07-06). League codes are Transfermarkt's: `L1` Germany, `KR1` **Croatia**,
`RSK1` South Korea, `A1` Austria, `C1` Switzerland, `TS1` Czechia, `SA1` Saudi
Arabia, `SC1` Scotland.

**Bottom line.** (a) RAPM: add one block of unpenalised league-mean columns so the
ridge shrinks each player toward his league's mean, not toward zero (Pantuso &
Hvattum's league term). (b) Box rates: a player-fixed-effects Poisson model with
league effects identified from movers (a regression NHLe), applied as a
multiplicative deflator. Do **not** use our in-house Elo for anything between
leagues: its league levels are an artefact (section 2.4).

---

## 1. Literature and practice

"Verified" means I opened the page or PDF named in the URL on 2026-11-17/18.
Where I could open only an abstract or a preprint, the row says so.

### 1.1 League factors inside plus-minus models (football)

| Source | Opened | What it does / finds |
|---|---|---|
| Sæbø & Hvattum (2015), NIK. `saebo2015transfer` | Full PDF, https://www.ntnu.no/ojs/index.php/nikt/article/view/5265/4741 | Ridge plus-minus on segments, a "twelfth man" home dummy, dismissal dummies, exponential time discount. 14 competitions incl. Champions League and Europa League, 20,217 matches, 120,834 segments, 15,884 players. **No league or competition term**: leagues are tied only by the European games and by players, as in our A12 fit. |
| Kharrat, McHale & López Peña (2020), EJOR. `kharrat2020plusminus` | arXiv version in full, https://ar5iv.labs.arxiv.org/html/1706.04943 | Ridge plus-minus (goals, xG, expected points), 11 European leagues, 20,868 games, 129,988 segments, 10,983 players; the data description mentions no European cup games. **One coefficient per league, estimated jointly with the players**; the stated identification is players who move between leagues (a player counts for a league after 6 games there in the season, or if most of his games in the previous 18 months were there). Normalised league strength: Premier League 0.88, Bundesliga 0.75, La Liga and Serie A 0.64, Eredivisie last at 0.11. |
| Pantuso & Hvattum (2021), TOP. `pantuso2021transfer` | arXiv version in full, https://ar5iv.labs.arxiv.org/html/1911.04689 ; metadata from Crossref | Player term = `β_p + age curve + (1/|B_p|) Σ_{b∈B_p} β^B_b`, where `B_p` is the set of leagues the player has played in: **the league term is the average of the coefficients of his leagues**. Ratings are shrunk toward similar team-mates, not toward zero. More than 84,000 matches, leagues of 25 countries plus international club and national-team tournaments. The paper says this generalises the "tournament factor" of Sæbø & Hvattum (2019). |
| Sæbø & Hvattum (2019), JSA. `saebo2019financial` | **Not opened** (publisher returns 403). Known only through Pantuso & Hvattum's description above. | Tournament factor: a player who debuts in a high-level league gets a higher rating than one who debuts in a low-level league. Do not cite for model details. |
| Hvattum (2020), Applied Sciences. `hvattum2020offdef`; Hvattum & Gelade (2021), IJCSS. `hvattum2021bottomup` | Abstracts only (MDPI 403; https://reference-global.com/article/10.2478/ijcss-2021-0002) | Offence/defence split of the same model family; plus-minus beats event-based ratings on reliability and validity. Whether and how the league term enters was **not verified**. |

Take-away: the two verified models that compare players across leagues both add an
explicit league term to the regression. A plain ridge with only European games as
links (Sæbø & Hvattum 2015, and our A12) has no such term.

### 1.2 League translation factors from players who change leagues

| Source | Opened | What it does / finds |
|---|---|---|
| Desjardins, "League Equivalencies" (hockey; PDF dated 3 Jan 2005, usually cited as 2004). `desjardins2005equivalencies` | Full PDF, http://hockeyanalytics.com/Research_files/League_Equivalencies.pdf | League quality = points per game in the NHL in year 2 divided by points per game in league X in year 1, over players with ≥ 40 games in each. AHL ≈ 0.45, Russian league 0.91, NCAA 0.33. Two stated biases: the factor depends on age (0.65–0.68 for 18–19-year-olds vs 0.45 for players in their 20s), and movers **lose power-play time** in the better league, so the factor understates the weaker league by an estimated 20–30%. |
| Bacon (2021), "NHL Equivalency and Prospect Projection Models". `bacon2021nhle` | Full page, https://hockeystats.com/methodology/nhle | Lists the weaknesses of the classic ratio (players develop between year 1 and year 2; per-game averages weight a 1-game player like an 82-game player; only leagues that feed the NHL directly). Network NHLe: same-season transitions only, totals not averages, indirect paths chained (KHL→AHL→NHL), each path weighted by `1/2^connections ×` its smallest transition count, ≥ 8 players per link. KHL 0.772, SHL 0.566, AHL 0.389. |
| "thecomeonman", Predicting transfer successes (blog, undated, football) | Full page, https://thecomeonman.github.io/PredictingTransferSuccesses | VAEP per minute after/before a move; a log-linear model with a "from" and a "to" factor per league. The author's own caveats: movers are not a random sample; individual moves scatter widely around the median; age did not help; team effects were weak. Not peer-reviewed: use as a practice example only. |
| Dinsdale & Gallagher (2022), "Transfer Portal", arXiv:2201.11533. `dinsdale2022transfer` | Abstract only, https://arxiv.org/abs/2201.11533 | A neural model predicts a player's output at a new club from player, team and league representations. States that post-move performance depends on style, team-mates' ability, league quality and role. |
| Baseball: Bill James's Major League Equivalencies; Davenport Translations | **Not opened** (Baseball Prospectus serves a bot wall). | Do not cite. |
| Smarterscout league adjustments | **Not opened**: `smarterscout.com/faq` now redirects to `vantagefc.com`, which has no method text. A search snippet says adjustments are calibrated per metric on players who moved between leagues. | Do not cite. |
| StatsBomb, Twenty3, American Soccer Analysis | No public method page on league translation found. | Nothing to cite. |

Econometric anchor for "group effects identified from movers": Abowd, Kramarz &
Margolis (1999), worker and firm fixed effects (metadata verified on Crossref).
Andrews et al. (2008) show that with few movers the estimated fixed effects are
noisy and their correlation is biased ("limited mobility bias"; abstract verified
on Crossref). Hausman, Hall & Griliches (1984) is the fixed-effects Poisson
reference (metadata verified).

### 1.3 Cross-league team ratings and their comparability

| Source | Opened | What it does / finds |
|---|---|---|
| Csató (2024), IJPAS. `csato2024elo` | arXiv version in full, https://ar5iv.labs.arxiv.org/html/2304.09078 ; metadata from Crossref | Compares the UEFA club coefficient with ClubElo (K = 20, home advantage and margin of victory) on 19 Champions League seasons. Elo wins on every measure (group matches: 73.3% vs 70.7% accuracy). States that in ClubElo a league's average rises **only** when its clubs beat expectations in European competition; England averaged 1,778 and Hungary 1,303 on 30 June 2021. |
| Shelopugin & Sirotkin (2023), arXiv:2310.11459. `shelopugin2023leagues` | Full text, https://ar5iv.labs.arxiv.org/html/2310.11459 | Modified Glicko-2 on about 366,000 matches (first and second divisions, cups, continental cups; third tier and below **excluded**). Ratings are **renormalised after every season** so that the mean cannot inflate; promoted and relegated clubs get a league-specific shift; newcomers from weaker leagues start lower. Warns that Europe and South America cannot be compared reliably (too few games between them). Log loss 0.5832 vs 0.5949 for plain Glicko-2. |
| UEFA association coefficient. `uefa2026coefficients` | Full page, https://www.uefa.com/nationalassociations/uefarankings/country/about/ | Season coefficient = points of the association's clubs in the three UEFA competitions (2 per win, 1 per draw, halved in qualifying, plus bonus points) divided by the number of its clubs; ranking = sum over five seasons. Uses European games only, and only UEFA members. |
| Opta Power Rankings (Segar, 31 Dec 2024; Furniss, 11 Jun 2025). `furniss2025leagues` | https://theanalyst.com/articles/opta-power-rankings/ ; https://theanalyst.com/articles/strongest-leagues-in-the-world-opta-power-rankings-june-2025 | Over 13,000 men's clubs on a 0–100 scale; a win over a stronger side earns more. League strength = mean rating of the league's clubs: Premier League 92.6, Serie A and La Liga 87.0, Bundesliga 86.3, Ligue 1 85.5, Portugal 79.8, Brazil 79.4, Eredivisie 78.8, Argentina 78.6, MLS 78.5; Saudi Pro League 29th. The pages I opened do not state the algorithm ("hierarchical Elo" appears only in a search snippet). Proprietary: an external benchmark, not an input. |
| Aldous (2017), Statistical Science. `aldous2017elo` | Abstract via Crossref, https://doi.org/10.1214/17-STS628 | Little foundational work exists on how accurate Elo-type ratings are; his simulations question the rule that about 30 games suffice. |
| FiveThirtyEight SPI | **Not opened**: the methodology pages now redirect to abcnews.com. | Do not cite. |
| ClubElo method page | **Not opened**: `clubelo.com/System` returns only rankings. Described through Csató (2024). | Cite Csató, not the site. |

Take-away: an Elo that includes domestic results can order clubs across European
leagues well (Csató), but only if league levels move through games between
leagues and nothing else. Systems built for this purpose exclude lower tiers and
renormalise (Shelopugin & Sirotkin). Ours does neither (section 2.4).

### 1.4 Opponent-strength adjustment of box-score rates

| Source | Opened | What it does / finds |
|---|---|---|
| McHale & Szczepański (2014), JRSS A. `mchale2014goalscoring` | Abstract and metadata via Crossref (full text paywalled) | Two mixed-effects models (shot generation, shot conversion) estimate scoring ability and beat the naive goals-per-minute rate at predicting next-season goals. Whether team and opponent effects enter was **not verified**. |
| Dixon & Coles (1997), JRSS C. `dixon1997modelling` | Metadata via Crossref | The standard team-level attack/defence Poisson model: a team's goal rate is its attack times the opponent's defence. This is the model behind any multiplicative opponent adjustment. |
| KenPom adjusted efficiency; Sports-Reference SRS; DVOA | **Not opened** (403). | Do not cite. |

I found no verified football paper that adjusts individual per-90 rates for
opponent strength at match level. The verified practice for individual rates is
league translation from movers (1.2).

---

## 2. Data facts from our tables

### 2.1 Coverage that constrains everything

- 32 domestic leagues, but only **14 cover the whole window** (BE1, DK1, ES1, FR1,
  GB1, GR1, IT1, L1, NL1, PO1, RU1, SC1, TR1, UKR1). The other 18 exist only for
  2024/25 and 2025/26.
- Calendar-year leagues carry the label of the previous European season: MLS
  "2024/25" is the 2025 season (22 Feb–19 Oct 2025).
- MLS "2024/25": only **357 of 509 games have line-ups** (Inter Miami 20 of 34).
  Messi has 18 appearances and 1,620 minutes there. This is a hole inside the
  season we do have.
- `AFAC` is the **AFC Asian Cup 2023**, not AFCON. AFCON 2025 is `AFCN`.

### 2.2 Movers (what a translation model would use)

Unit: a player-season-league with ≥ 900 league minutes. 29,513 such rows, 13,367
players.

- Followed by ≥ 900 minutes in a **different** league the next season: **1,883**
  origin player-seasons (1,538 players).
- Same, within two seasons: **3,294** origin player-seasons (1,959 players).
- Same season, two leagues, ≥ 900 in each: 120.
- Players with ≥ 450 league minutes (window total) in each of two or more
  leagues: **3,664** ("linking players"; the number a Poisson model with a
  minutes offset can use).
- At a threshold of 10 shared players, 30 leagues form one connected set; COL1
  and RSK1 fall out.

Per league ("out" = leaves this league, "in" = arrives; "≤2" = within two seasons;
out-flows of the 18 late leagues are truncated by the window):

| League | Player-seasons ≥900 | Out next | In next | Out ≤2 | In ≤2 | Linking players (≥450 in two leagues) |
|---|---:|---:|---:|---:|---:|---:|
| FR1 | 1,866 | 246 | 187 | 425 | 235 | 726 |
| GB1 | 2,195 | 160 | 270 | 306 | 318 | 631 |
| IT1 | 2,261 | 189 | 184 | 351 | 231 | 641 |
| ES1 | 2,289 | 145 | 158 | 271 | 193 | 529 |
| L1 | 1,866 | 145 | 144 | 280 | 167 | 519 |
| TR1 | 2,093 | 105 | 198 | 182 | 254 | 637 |
| PO1 | 1,863 | 168 | 80 | 307 | 103 | 550 |
| BE1 | 1,516 | 162 | 69 | 283 | 102 | 550 |
| NL1 | 1,642 | 150 | 73 | 269 | 105 | 462 |
| GR1 | 1,119 | 61 | 88 | 109 | 122 | 398 |
| DK1 | 837 | 76 | 27 | 134 | 36 | 280 |
| RU1 | 1,437 | 50 | 46 | 100 | 56 | 218 |
| SA1 | 542 | 18 | 83 | 18 | 128 | 197 |
| SC1 | 1,083 | 42 | 31 | 72 | 38 | 192 |
| MLS1 | 574 | 11 | 57 | 11 | 82 | 200 |
| PL1 | 565 | 19 | 31 | 19 | 54 | 171 |
| BRA1 | 507 | 14 | 36 | 14 | 54 | 188 |
| UKR1 | 1,278 | 29 | 6 | 50 | 11 | 115 |
| C1 | 359 | 19 | 18 | 19 | 31 | 127 |
| A1 | 312 | 16 | 12 | 16 | 23 | 85 |
| KR1 | 292 | 12 | 14 | 12 | 22 | 81 |
| SER1 | 455 | 12 | 12 | 12 | 18 | 89 |
| RO1 | 403 | 8 | 11 | 8 | 19 | 83 |
| SE1 | 200 | 5 | 8 | 5 | 17 | 94 |
| AUS1 | 315 | 5 | 9 | 5 | 13 | 54 |
| TS1 | 463 | 9 | 9 | 9 | 9 | 71 |
| MEX1 | 163 | 0 | 8 | 0 | 16 | 69 |
| ARG1 | 235 | 0 | 9 | 0 | 13 | 86 |
| JAP1 | 202 | 3 | 3 | 3 | 7 | 50 |
| NO1 | 224 | 3 | 1 | 3 | 3 | 71 |
| RSK1 | 162 | 1 | 2 | 1 | 2 | 19 |
| COL1 | 195 | 0 | 1 | 0 | 1 | 22 |

Largest flows, both directions summed, within two seasons: FR1–GB1 145, ES1–GB1
142, GB1–IT1 137, GB1–L1 127, FR1–IT1 122, ES1–FR1 85, FR1–L1 80, PO1–TR1 78,
ES1–IT1 77, IT1–TR1 75, FR1–TR1 66, IT1–L1 64, BE1–FR1 61, L1–NL1 55, FR1–PO1 52.

Big-5 matrix, directed, within two seasons (row = from, column = to):

| | GB1 | ES1 | IT1 | L1 | FR1 |
|---|---:|---:|---:|---:|---:|
| GB1 | – | 72 | 50 | 45 | 42 |
| ES1 | 70 | – | 37 | 10 | 33 |
| IT1 | 87 | 40 | – | 20 | 50 |
| L1 | 82 | 27 | 44 | – | 34 |
| FR1 | 103 | 52 | 72 | 46 | – |

Between the big 5 and the rest (within two seasons; from big 5 / to big 5): NL1
58/150, BE1 44/135, PO1 56/123, DK1 9/53, TR1 193/52, RU1 21/36, SC1 11/25, UKR1
1/20, GR1 65/18, **SA1 98/7, MLS1 38/2, BRA1 37/3**. The links to Saudi Arabia,
MLS and Brazil run one way.

Age at arrival (movers within two seasons): big 5 → other 28.8 years, other → big
5 25.0. By destination: SA1 28.8, MLS1 28.0, TR1 28.0, BRA1 28.6; GB1 25.0, L1
25.7, IT1 25.9. Direction of move and age are confounded.

### 2.3 Games between leagues

Club games in CL, EL, ECL (with qualifiers), Club World Cup and UEFA Super Cup
between clubs of two **different covered leagues**: **2,587** (2,431 with stints).
Another 1,318 have one side from a league we do not cover. Continental games hold
31,610 of 286,376 Transfermarkt stints (11%). National-team games: 373 (200 with
Transfermarkt stints), plus 166 StatsBomb stint games.

Games against another covered league, per league and season:

| League | 19/20 | 20/21 | 21/22 | 22/23 | 23/24 | 24/25 | 25/26 | Total | vs uncovered | of which Club World Cup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB1 | 22 | 77 | 78 | 68 | 75 | 83 | 104 | 507 | 41 | 12 |
| ES1 | 21 | 64 | 72 | 59 | 62 | 79 | 93 | 450 | 46 | 10 |
| IT1 | 14 | 59 | 61 | 75 | 68 | 83 | 76 | 436 | 40 | 6 |
| L1 | 24 | 53 | 59 | 69 | 64 | 74 | 84 | 427 | 37 | 9 |
| FR1 | 9 | 34 | 58 | 40 | 56 | 66 | 79 | 342 | 35 | 7 |
| NL1 | 4 | 33 | 59 | 46 | 40 | 63 | 49 | 294 | 50 | 0 |
| PO1 | 8 | 31 | 44 | 44 | 42 | 50 | 57 | 276 | 45 | 5 |
| BE1 | 4 | 32 | 30 | 44 | 50 | 51 | 40 | 251 | 56 | 0 |
| TR1 | 4 | 13 | 34 | 36 | 32 | 44 | 41 | 204 | 64 | 0 |
| TS1 | 0 | 25 | 32 | 26 | 30 | 43 | 40 | 196 | 72 | 0 |
| SC1 | 6 | 20 | 37 | 24 | 30 | 36 | 36 | 189 | 43 | 0 |
| A1 | 6 | 28 | 34 | 26 | 26 | 33 | 34 | 187 | 59 | 3 |
| GR1 | 4 | 27 | 18 | 6 | 49 | 27 | 45 | 176 | 81 | 0 |
| DK1 | 5 | 13 | 34 | 30 | 22 | 27 | 27 | 158 | 59 | 0 |
| NO1 | 0 | 12 | 20 | 24 | 28 | 35 | 38 | 157 | 66 | 0 |
| UKR1 | 6 | 28 | 20 | 26 | 20 | 24 | 22 | 146 | 54 | 0 |
| C1 | 3 | 12 | 16 | 18 | 30 | 35 | 31 | 145 | 75 | 0 |
| PL1 | 0 | 11 | 18 | 16 | 20 | 25 | 33 | 123 | 89 | 0 |
| SE1 | 2 | 6 | 16 | 24 | 10 | 31 | 23 | 112 | 76 | 0 |
| SER1 | 0 | 11 | 22 | 12 | 22 | 24 | 18 | 109 | 61 | 0 |
| KR1 | 0 | 24 | 16 | 14 | 16 | 9 | 17 | 96 | 63 | 0 |
| RO1 | 2 | 8 | 10 | 18 | 6 | 14 | 23 | 81 | 81 | 0 |
| RU1 | 0 | 28 | 26 | 0 | 0 | 0 | 0 | 54 | 4 | 0 |
| BRA1 | 0 | 1 | 1 | 1 | 1 | 0 | 14 | 18 | 7 | 18 |
| MEX1 | 0 | 3 | 0 | 0 | 1 | 0 | 7 | 11 | 2 | 11 |
| SA1 | 0 | 0 | 1 | 2 | 0 | 0 | 5 | 8 | 5 | 8 |
| MLS1 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 8 | 3 | 8 |
| ARG1 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 1 | 5 |
| JAP1 | 0 | 0 | 0 | 0 | 2 | 0 | 3 | 5 | 1 | 5 |
| RSK1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 3 | 2 | 3 |
| AUS1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| COL1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(A club's league is taken from the same season when we have it, otherwise from
any season, which is how pre-2024 games of the 18 late leagues get a league.)

Mean goal difference per game in those games, raw (not adjusted for who met
whom): GB1 +0.91, IT1 +0.42, L1 +0.41, ES1 +0.25, FR1 +0.18, PO1 +0.16, NL1 +0.09,
NO1 −0.08, GR1 −0.11, DK1 −0.16, BE1 −0.18, TR1 −0.21, SC1 −0.65, A1 −0.80.

Saudi Arabia, MLS, Brazil, Argentina, Mexico, Japan and South Korea meet other
covered leagues **only at the Club World Cup** (3 to 18 games each). Australia and
Colombia never do.

### 2.4 Mean pre-match Elo by league and season — and why it is not league strength

Mean of both sides' pre-match Elo over the league's own games:

| League | 19/20 | 20/21 | 21/22 | 22/23 | 23/24 | 24/25 | 25/26 | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GB1 | 1634 | 1650 | 1667 | 1688 | 1708 | 1736 | 1764 | 1697 |
| ES1 | 1633 | 1650 | 1670 | 1687 | 1699 | 1727 | 1739 | 1690 |
| L1 | 1643 | 1661 | 1671 | 1689 | 1698 | 1712 | 1724 | 1689 |
| **DK1** | 1626 | 1653 | 1663 | 1667 | 1685 | 1701 | 1723 | **1679** |
| IT1 | 1592 | 1614 | 1637 | 1649 | 1660 | 1668 | 1680 | 1646 |
| NL1 | 1572 | 1577 | 1600 | 1609 | 1615 | 1622 | 1624 | 1607 |
| RU1 | 1575 | 1583 | 1599 | 1606 | 1616 | 1613 | 1624 | 1605 |
| GR1 | 1618 | 1595 | 1592 | 1590 | 1602 | 1603 | 1623 | 1602 |
| **FR1** | 1546 | 1559 | 1571 | 1579 | 1616 | 1620 | 1632 | **1591** |
| PO1 | 1550 | 1554 | 1566 | 1579 | 1587 | 1588 | 1589 | 1575 |
| UKR1 | 1574 | 1578 | 1569 | 1563 | 1579 | 1566 | 1570 | 1571 |
| SC1 | 1574 | 1562 | 1562 | 1565 | 1554 | 1568 | 1577 | 1565 |
| TR1 | 1534 | 1528 | 1546 | 1571 | 1567 | 1563 | 1576 | 1555 |
| A1 | – | – | – | – | – | 1550 | 1553 | 1551 |
| KR1 | – | – | – | – | – | 1547 | 1552 | 1550 |
| TS1 | – | – | – | – | – | 1547 | 1552 | 1550 |
| BE1 | 1523 | 1512 | 1522 | 1526 | 1567 | 1578 | 1589 | 1545 |
| NO1 | – | – | – | – | – | 1537 | 1561 | 1543 |
| SE1 | – | – | – | – | – | 1534 | 1546 | 1537 |
| PL1 | – | – | – | – | – | 1529 | 1546 | 1537 |
| C1 | – | – | – | – | – | 1533 | 1537 | 1535 |
| RO1 | – | – | – | – | – | 1516 | 1524 | 1520 |
| SER1 | – | – | – | – | – | 1507 | 1516 | 1512 |
| BRA1 | – | – | – | – | – | 1506 | 1525 | 1512 |
| SA1 | – | – | – | – | – | 1500 | 1520 | 1510 |
| MEX1 | – | – | – | – | – | 1500 | 1502 | 1502 |
| ARG1 | – | – | – | – | – | 1500 | 1503 | 1502 |
| JAP1 | – | – | – | – | – | 1501 | – | 1501 |
| MLS1 | – | – | – | – | – | 1500 | 1500 | 1500 |
| COL1 | – | – | – | – | – | – | 1499 | 1499 |
| RSK1 | – | – | – | – | – | 1499 | 1499 | 1499 |
| AUS1 | – | – | – | – | – | 1500 | 1493 | 1497 |

These levels are mostly an artefact of how the Elo is built:

1. **Domestic cups feed points into some leagues and not others.** Lower-division
   clubs are in no covered league, enter at 1,500 and lose. Net Elo points gained
   inside the window, by source (recomputed from the pre-match ratings with K = 20,
   H = 60):

   | League | From domestic cups | From continental games | Cup points per current club |
   |---|---:|---:|---:|
   | ES1 | +2,613 | +354 | +131 |
   | DK1 | +1,939 | −571 | +162 |
   | L1 | +1,472 | +557 | +82 |
   | NL1 | +1,254 | −438 | +70 |
   | GB1 | +1,225 | +1,865 | +61 |
   | SC1 | +1,000 | −767 | +83 |
   | GR1 | +965 | −863 | +69 |
   | IT1 | +467 | +981 | +23 |
   | PO1 | +167 | +36 | +9 |
   | FR1, BE1, TR1 | 0 (no cup in the data) | +995, +292, −179 | 0 |

   All of the cup points come from games against clubs outside the covered
   leagues, which lost 12,476 points in total. For Spain, Denmark, Germany, the
   Netherlands, Scotland and Greece the cup inflow is larger than the
   continental flow. The paper's sentence that "cross-league European games tie
   the leagues together" is therefore incomplete.
2. **Result:** Denmark's league mean (1,679) is above Serie A's (1,646) and Ligue
   1's (1,591), although Danish clubs average −0.16 goals per game against other
   covered leagues and French clubs +0.18.
3. **The 18 late leagues enter at 1,500 in 2024/25 and stay there.** Brazil (1,512),
   Argentina (1,502) and MLS (1,500) sit at the bottom; Opta ranks them 9th, 11th
   and 12th in the world (section 1.3).
4. Most leagues covered throughout drift upward (2019/20 to 2025/26: GB1 +130,
   ES1 +106, DK1 +97, L1 +81), so the level also measures time in the data.

Within a league, Elo differences are unaffected by the first three points and
remain usable. Between leagues, `team_elo` and `opp_elo` are not. This also
touches the existing `elo_partialled` context level and M2, which use `team_elo`
pooled over leagues.

### 2.5 Pool exposure outside the big 5

Pool: 106 players. **31** have a season with ≥ 900 league minutes outside the big 5
(57 player-seasons): SA1 10 players / 18 seasons, PO1 8 / 15, TR1 5 / 6, MLS1 3 / 4,
NL1 3 / 6, BRA1 2 / 2, BE1 1, RU1 1 / 2, UKR1 1 / 3. 64 pool players have
qualifying seasons in two or more leagues, so most of the pool are movers
themselves. Every league that holds a pool player-season has at least 115
linking players.

---

## 3. Recommendation

### 3.0 What goes wrong in A12, in one equation

In the row where side `s` attacks side `o`,
`E[y] = Σ_{p∈s} off_p − Σ_{q∈o} def_q + home_g + icpt_g`.
Add a constant `c` to the offence **and** defence of every player of one league.
Every domestic row is unchanged (`+11c − 11c`). So the league's level is not
identified by its own games at all. Only rows with players of two leagues on the
pitch constrain it, and the ridge charges `λc²` for every one of the league's player columns
(two per player, hundreds to thousands per league). The ridge wins, every league is centred near
zero, and dominating any league looks the same. The competition-season intercept
does not help: it absorbs the league's scoring level, not its quality level.

### 3.1 (a) RAPM — league-mean columns (recommended)

**Model.** Write each player's effects as his league's mean plus a deviation, and
penalise only the deviation:

```
off_p = δoff_p + Σ_L s_pL · μ_L        def_p = δdef_p + Σ_L s_pL · μ_L

E[y_row] = Σ_{p∈att} δoff_p − Σ_{q∈def} δdef_q
           + Σ_L μ_L · (n_att,L − n_def,L)  +  home_g·[att is home] + icpt_g

penalty = λ · ( ‖δoff‖² + ‖δdef‖² )          (μ unpenalised)
```

- `s_pL` = share of player `p`'s domestic-league minutes in the window played in
  league `L` (fixed per player, sums to 1). Players with no minutes in a covered
  league get a pseudo-league `OTHER`.
- `n_att,L = Σ_{p∈att} s_pL`: how many of the attacking eleven "belong" to `L`.
- One league is the reference (`μ_GB1 = 0`), because the columns sum to
  `11 − 11 = 0` in every 11-v-11 row.
- Reported rating: `total_p = δoff_p + δdef_p + 2·Σ_L s_pL μ_L`.

This is Pantuso & Hvattum's league term with minutes weights instead of an
unweighted average, inside our existing ridge.

**Code change.** `z_L = n_home,L − n_away,L` is a per-stint number that enters as
`+z_L` in the home-attacking row and `−z_L` in the away-attacking row. That is
exactly the sign rule of the existing `covariate` in `rapm.build_design`. The
change is to accept a list of covariate columns instead of one, and to compute
the `z_L` columns in `run_impact.data()`. No new framework; λ is re-chosen by the
same CV; the game bootstrap gives a standard error for every `μ_L`. Per-season
fits keep the pooled `μ` fixed.

**What identifies `μ_L`.** Every row where `z_L ≠ 0`:
1. games between clubs of two leagues (2,431 stint games between covered leagues);
2. domestic games with a mover on the pitch: if a player has `s = (0.5, 0.5)` in
   leagues A and B, his rows in league A carry `+0.5(μ_B − μ_A)` and his rows in B
   carry the opposite, so how his teams do in each league, given everyone's `δ`,
   identifies the gap;
3. national-team games, where each side mixes leagues (366 stint games).

**Which leagues get their own `μ`** (rule fixed on link counts only, before any
fit): ≥ 100 games against other covered leagues, or ≥ 150 linking players. That
gives 24 leagues: the 20 with ≥ 100 games (GB1 … SER1 in the table of 2.3) plus
RU1, MLS1, SA1, BRA1 through movers. The other eight (KR1, RO1, ARG1, MEX1, AUS1,
JAP1, COL1, RSK1) share one `μ_REST`. No pool player has a qualifying season in
those eight.

**What it cannot fix.**
- One level per league for the whole window: a league that changes (Saudi Arabia
  after 2023) gets its average.
- The same shift for every player of a league, and the same for offence and
  defence.
- Saudi Arabia, MLS and Brazil are identified by movers and by 8 to 18 Club World
  Cup games only; expect wide intervals and report them.
- The within-league ridge shrinkage is unchanged.

**Known failure modes, with direction.**
- *Age.* RAPM has no age curve, and `δ_p` is constant over the window. Older
  players move down (28.8 years) and are worse than their window average when
  they arrive; younger players move up (25.0) and are better than theirs. Both
  make the weak league look **stronger** than it is, so gaps are understated.
- *Ridge leakage.* `μ` is identified largely by each league's European
  representatives. Their shrunk `δ` understates how dominant they are at home;
  the more dominant the representatives, the more of that leaks into a higher
  `μ` for their league. Again gaps are understated.
- *Limited mobility* (Andrews et al. 2008): with few movers the `μ` of a weakly
  linked league is noisy; the pooling rule and the bootstrap interval are the
  protection.
- Both directional biases are conservative: the correction is too small, not too
  large.

**Validation on a synthetic league.** `simulate_league` makes one league. Extend
it to several leagues and plant:
1. league means `μ_L` added to the offence and defence of every player of league
   `L`, e.g. `(0, −0.015, −0.04)` goals per 90 (average-team gaps of 0.33 and 0.88
   goals per 90, since a team's gap is `22·μ`);
2. a cup: each season the top four of every league (by last season's points, so
   that participants are selected on dominance as in reality) play each other;
3. a cross-league transfer rate (e.g. 3% of players per season), plus a variant
   where older players move down and their effects decline with a planted age
   slope;
4. one league with no cup games and under 20 movers.

Checks: slope of `μ̂_L` on planted `μ_L` near 1 and bootstrap coverage; Spearman
between estimated and true total over all leagues pooled, with and without the
league columns; `μ̂ ≈ 0` and unchanged ranks when all planted means are zero (no
harm); the unlinked league returns a wide interval; the age variant shows the
predicted understatement.

**Fallback, if the 24 free means prove too noisy:** the same columns with leagues
grouped into a few tiers fixed in advance from an external ranking (UEFA
association coefficients for Europe). I would not make this primary: it imports
an outside ordering and has no value for non-UEFA leagues.

### 3.2 (b) Box-score rates — league effects from movers (recommended)

**Model**, one per metric (goals, non-penalty goals, assists), on
player × season × league cells with league minutes `M > 0` (no minutes threshold;
the offset does the weighting):

```
Y_ptL ~ Poisson( (M_ptL / 90) · exp( α_p + γ_L + b1·age_pt + b2·age_pt² ) ),   γ_GB1 = 0
```

`α_p` is a player fixed effect, removed by the conditional likelihood (Hausman,
Hall & Griliches 1984); with our dependencies this is a short concentrated
likelihood in scipy over about 20 parameters. Cluster the bootstrap by player.

**Use.** A league match's goals count `exp(−γ̂_L)` times: "Premier-League-equivalent"
goals, the NHLe idea. Apply it to the match rows before `run_ranking` sums them,
so that a season split over two leagues is handled. UCL and national-team scopes
stay as they are.

**On-pitch goal difference** is additive and can be negative, so it takes the
RAPM league term instead: `gd_adj = gd + 22·(μ̂_L − μ̂_GB1)`. A team in league `L`
that met average Premier League opposition would face eleven players each better
by `2(μ_GB1 − μ_L)`.

**What identifies `γ_L`.** Only players observed in two or more leagues: 3,664
with ≥ 450 minutes in each. The age curve is identified by everyone. Leagues with
≥ 100 linking players get their own `γ` (19 leagues: every league with a pool
player-season, plus DK1, GR1, SC1, PL1, C1); the other 13 share `γ_REST`.

**What it cannot fix.**
- `γ_L` is the change for the typical mover, and it bundles difficulty with role:
  Desjardins shows the same thing in hockey (less power-play time in the better
  league). A star who moves from a dominant club to a mid-table one also changes
  team context.
- One proportional factor for all players; the pool is the extreme tail.
- Saudi Arabia, MLS and Brazil are identified almost only by players arriving
  from the big 5 at 28 to 29 years (98/7, 38/2, 37/3). There `γ` rests on the age
  curve being right for ageing stars.
- No shots, so finishing luck is not separated from scoring rate.

**Known failure modes.**
- *Selection on form.* Players move up after a good season and down after a bad
  one; regression to the mean then exaggerates the gap in both cases. Check: add
  a dummy for "last season before a cross-league move" (or drop that season). If
  `γ̂` moves materially, report the version with the dummy.
- *Survivors.* Movers who fail get few minutes. The offset keeps them in, with
  low weight; a 900-minute filter would drop them and understate the gap. Do not
  filter.
- *Team context.* Robustness: add the club's Elo **relative to its league-season
  mean** (a within-league quantity, so the artefact of 2.4 cancels).
- *Position.* Robustness: `γ` by position group (forwards vs the rest).
- *Limited mobility*: as in 3.1.

**Validation on a synthetic league.** In the multi-league generator of 3.1, plant
a league factor on the individual shot rate: `shot_rate_p · exp(γ_L)` in league
`L`, with `γ` chosen consistently with `μ` (weaker league, larger `γ`), e.g.
`(0, +0.10, +0.25)`. Then:
1. random movers: `γ̂` unbiased, interval coverage near nominal;
2. movers selected on last season's goals above expectation: measure the bias,
   and check that the pre-move dummy removes it;
3. planted age decline with older players moving down: check the age term;
4. planted role loss for movers up (their shot rate times 0.85): shows that `γ̂`
   absorbs it. This is the limit to state in the paper;
5. a league with 20 movers: interval width.

**Triangulation.** `γ̂_L` (individual rates, movers) and `μ̂_L` (team goal
difference, mostly games between leagues) use different data. Report their rank
correlation across leagues, and compare both with the UEFA coefficient and
Opta's league means as outside benchmarks that enter neither estimate.

### 3.3 Your three candidates

**(i) Opponent-Elo covariate in RAPM — no.**
- *Does it have A12's problem?* Yes, on the defence half. In the row "`s` attacks
  `o`" the opponent's Elo is the own-team Elo of the eleven defenders whose
  `def_q` are in that same row. For the attackers it is a legitimate schedule
  control; for the defenders it absorbs their own contribution, exactly as A12
  feared. The defence rating becomes "defence relative to my team's Elo", and
  `total` inherits half the problem. Putting it on both sides is the Elo
  difference, i.e. the specification A12 already rejected.
- *Would it cure the league problem?* No. Inside a league the opponents are
  already in the model as their own columns. Between leagues it helps only as
  far as Elo levels are comparable, and ours are not (2.4).
- What survives of the idea is a covariate that is zero inside a league and
  varies only between leagues. That is 3.1.

**(ii) Poisson model with player and league effects from movers — yes**, this is
3.2, with four additions: the conditional likelihood, no minutes threshold, an
age term, and the pre-move check. It is the only candidate that reaches Saudi
Arabia and MLS.

**(iii) Match-level opponent adjustment with `opp_elo` — no, not as it stands.**
- Inside a league a double round-robin gives every player nearly the same
  opponents; what is left is "you do not play your own team", an own-team effect
  the context dimension already covers.
- Between leagues the adjustment would be driven by Elo levels that rank Denmark
  above Serie A and put Brazil and MLS at 1,500.
- If the Elo were rebuilt (league and continental games only, lower tiers
  excluded, renormalised each season as in Shelopugin & Sirotkin), (iii) would
  become a second estimate of the league effect. That is a larger change than
  3.2 and I would not make it for this paper.

### 3.4 How to report it

The registered multiverse has run. The honest form is a new amendment (A20):
keep the 5,760-specification result as registered, and add a post-hoc
sensitivity analysis that reruns it with (a) league-mean RAPM and (b)
league-adjusted box rates, reporting rank shifts for the pool. Estimating `μ`
and `γ` does not look at any ranking outcome, so the rule for which leagues get
their own parameter can be fixed now from the counts in section 2.

---

## 4. BibTeX

Same style as `paper-unmeasured/refs.bib`. Already there and reused:
`saebo2015transfer`, `saebo2019financial`, `kharrat2020plusminus`,
`hvattum2020offdef`, `hvattum2021bottomup`. Entries marked `% check` have one
field I could not confirm on the page.

```bibtex
% ---------------------------------------------------------------------------
% League strength (docs/unmeasured/league-strength.md, checked 2026-11-18)
% ---------------------------------------------------------------------------

@article{pantuso2021transfer,
  author  = {Pantuso, Giovanni and Hvattum, Lars Magnus},
  title   = {Maximizing performance with an eye on the finances: a chance-constrained
             model for football transfer market decisions},
  journal = {TOP},
  volume  = {29},
  number  = {2},
  pages   = {583--611},
  year    = {2021},
  note    = {Published online 2020},
  doi     = {10.1007/s11750-020-00584-9}
}

@misc{desjardins2005equivalencies,
  author       = {Desjardins, Gabriel},
  title        = {League Equivalencies},
  howpublished = {hockeyanalytics.com research note},
  year         = {2005},
  note         = {PDF dated 3 January 2005; often cited as 2004},
  url          = {http://hockeyanalytics.com/Research_files/League_Equivalencies.pdf}
}

@misc{bacon2021nhle,
  author       = {Bacon, Patrick},
  title        = {{NHL} Equivalency ({NHLe}) and Prospect Projection Models},
  howpublished = {hockeystats.com methodology page},
  year         = {2021},
  url          = {https://hockeystats.com/methodology/nhle}
}

@misc{dinsdale2022transfer,
  author       = {Dinsdale, Daniel and Gallagher, Joe},
  title        = {Transfer Portal: Accurately Forecasting the Impact of a Player
                  Transfer in Soccer},
  howpublished = {arXiv preprint arXiv:2201.11533},
  year         = {2022},
  url          = {https://arxiv.org/abs/2201.11533}
}

@misc{shelopugin2023leagues,
  author       = {Shelopugin, Andrei and Sirotkin, Alexander},
  title        = {Ratings of {E}uropean and {S}outh {A}merican Football Leagues Based on
                  {G}licko-2 with Modifications},
  howpublished = {arXiv preprint arXiv:2310.11459},
  year         = {2023},
  url          = {https://arxiv.org/abs/2310.11459}
}

@article{csato2024elo,
  author  = {Csat{\'o}, L{\'a}szl{\'o}},
  title   = {Club coefficients in the {UEFA} {C}hampions {L}eague: Time for shift to an
             {E}lo-based formula},
  journal = {International Journal of Performance Analysis in Sport},
  volume  = {24},
  number  = {2},
  pages   = {119--134},
  year    = {2024},
  note    = {Published online 2023},
  doi     = {10.1080/24748668.2023.2274221}
}

@article{aldous2017elo,
  author  = {Aldous, David},
  title   = {Elo Ratings and the Sports Model: A Neglected Topic in Applied
             Probability?},
  journal = {Statistical Science},
  volume  = {32},
  number  = {4},
  pages   = {616--629},
  year    = {2017},
  doi     = {10.1214/17-STS628}
}

% check: descriptive title; the page heading was not recorded
@misc{uefa2026coefficients,
  author       = {{UEFA}},
  title        = {Association club coefficients: how they are calculated},
  howpublished = {uefa.com},
  year         = {2026},
  note         = {Read 2026-11-17},
  url          = {https://www.uefa.com/nationalassociations/uefarankings/country/about/}
}

% check: title taken from the URL slug, not from the headline
@misc{furniss2025leagues,
  author       = {Furniss, Matt},
  title        = {Strongest Leagues in the World: {O}pta {P}ower {R}ankings},
  howpublished = {Opta Analyst (theanalyst.com)},
  year         = {2025},
  note         = {Published 11 June 2025},
  url          = {https://theanalyst.com/articles/strongest-leagues-in-the-world-opta-power-rankings-june-2025}
}

@article{mchale2014goalscoring,
  author  = {McHale, Ian G. and Szczepa{\'n}ski, {\L}ukasz},
  title   = {A mixed effects model for identifying goal scoring ability of footballers},
  journal = {Journal of the Royal Statistical Society: Series A (Statistics in Society)},
  volume  = {177},
  number  = {2},
  pages   = {397--417},
  year    = {2014},
  note    = {Published online 2013},
  doi     = {10.1111/rssa.12015}
}

@article{dixon1997modelling,
  author  = {Dixon, Mark J. and Coles, Stuart G.},
  title   = {Modelling Association Football Scores and Inefficiencies in the Football
             Betting Market},
  journal = {Journal of the Royal Statistical Society: Series C (Applied Statistics)},
  volume  = {46},
  number  = {2},
  pages   = {265--280},
  year    = {1997},
  doi     = {10.1111/1467-9876.00065}
}

% check: Crossref returned only the first page (909)
@article{hausman1984count,
  author  = {Hausman, Jerry and Hall, Bronwyn H. and Griliches, Zvi},
  title   = {Econometric Models for Count Data with an Application to the
             Patents--{R\&D} Relationship},
  journal = {Econometrica},
  volume  = {52},
  number  = {4},
  pages   = {909--938},
  year    = {1984},
  doi     = {10.2307/1911191}
}

@article{abowd1999wage,
  author  = {Abowd, John M. and Kramarz, Francis and Margolis, David N.},
  title   = {High Wage Workers and High Wage Firms},
  journal = {Econometrica},
  volume  = {67},
  number  = {2},
  pages   = {251--333},
  year    = {1999},
  doi     = {10.1111/1468-0262.00020}
}

@article{andrews2008mobility,
  author  = {Andrews, M. J. and Gill, L. and Schank, T. and Upward, R.},
  title   = {High wage workers and low wage firms: negative assortative matching or
             limited mobility bias?},
  journal = {Journal of the Royal Statistical Society: Series A (Statistics in Society)},
  volume  = {171},
  number  = {3},
  pages   = {673--697},
  year    = {2008},
  doi     = {10.1111/j.1467-985X.2007.00533.x}
}

% only if the MLS data is adopted (section 5)
@misc{asa2026api,
  author       = {{American Soccer Analysis}},
  title        = {itscalledsoccer: a wrapper for the {A}merican {S}occer {A}nalysis {API}},
  howpublished = {GitHub repository, MIT; API data CC BY-SA 4.0},
  year         = {2026},
  note         = {Read 2026-11-18},
  url          = {https://github.com/American-Soccer-Analysis/itscalledsoccer}
}
```

---

## 5. Coverage holes and open sources

Not legal advice; quotes are from the pages named, opened 2026-11-18. Nothing was
downloaded.

| Hole | Source | Terms | Offers | Cannot |
|---|---|---|---|---|
| MLS 2023 and 2024; 152 MLS 2025 games without line-ups | American Soccer Analysis API, via `itscalledsoccer` (https://github.com/American-Soccer-Analysis/itscalledsoccer) | README: "Data returned by this API is licensed under CC BY-SA 4.0". Package: MIT. The README describes the API as free, public and without authentication. | Player xG tables for `mls` with `season_name`, `split_by_seasons`, `split_by_games` and a `minimum_minutes` filter; a games table. Enough for player-season minutes, goals and assists **if** those fields are returned: the pages I opened do not list the fields, so check on the first call. | No line-ups or substitutions are documented: no stints, no on-pitch goal difference, no RAPM. Share-alike: tables we publish that contain this data must carry CC BY-SA 4.0 and attribution. |
| MLS 2023 | StatsBomb open data (already on disk) | Public Data User Agreement, see `data-licences.md` §2 | 6 Inter Miami matches with line-ups and events. | The rest of the season. |
| Saudi Pro League, Jan 2023–May 2024 | **None found.** | – | Wikipedia club-season pages (e.g. 2023–24 Al-Nassr) list scorers, assists and appearances; CC BY-SA (below). | No minutes and no line-ups in what I could read, so no per-90 rates and no stints. StatsBomb has no Saudi competition. FBref has the tables, but I did not open Sports Reference's terms: treat as not permitted until someone reads them. |
| Copa América 2021 (28 matches) | Wikipedia group and knockout pages, via the MediaWiki API | Text "can be used under the terms of the Creative Commons Attribution Share-Alike license (CC BY-SA)" (https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content). Scripted reads are allowed: "There is no hard speed limit on read requests, but be considerate and try not to take a site down." (https://www.mediawiki.org/wiki/API:Etiquette) | Both starting elevens, every substitution with its minute, goals with scorer and minute (checked on the Group A page: Argentina–Chile, 14 June 2021). Stints and minutes can be built. | No assists, no stoppage-time lengths, no player ids (names must be matched to Transfermarkt), crowd-edited. The etiquette page requires a User-Agent with contact details, which is Liam's decision. Knockout pages not checked. |
| AFCON 2021 (Jan–Feb 2022, 52 matches) | Same Wikipedia route | Same | Same content (checked on the Group A page: Cameroon–Burkina Faso, 9 January 2022). | Same limits. |
| AFCON 2023 (Jan–Feb 2024, 52 matches) | StatsBomb open data, **already on disk** (`sb_matches`: 52 matches) | Public Data User Agreement, `data-licences.md` §2 | Line-ups, events, xG: stints, goals, assists. | Nothing to acquire. A12 takes StatsBomb stints only for the 2022 World Cup and the Euros, so adding AFCON 2023 is an amendment. |

Checked and ruled out:
- **dcaribou/transfermarkt-datasets**: our raw snapshot has MLS and the Saudi
  league only from the season labelled 2024, one Copa América (2024) and one
  AFCON (2025). The README says "The dataset is current to 6 July 2026" and that
  collection has stopped with no date to resume, so no newer snapshot adds the
  missing seasons. Licence CC0-1.0.
- **StatsBomb open data** (`competitions.json`): Copa América 2024 only, AFCON
  2023 only, MLS 2023 only, no Saudi competition.
