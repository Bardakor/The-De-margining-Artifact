# Taxonomy of football variables — what is measured, proxied, or invisible

Exploratory (not pre-registered), as the pre-registration says. Every citation
is a key in `paper-unmeasured/refs.bib`; summaries and verification links are in
`docs/unmeasured/literature.md`.

**Measurability tiers**

- **Box score**: counted per player-match in Transfermarkt (goals, assists,
  minutes, cards).
- **Public event data**: needs located on-ball events, which our data has only
  through StatsBomb open data (a subset of matches).
- **Tracking or proprietary only**: needs positions of all 22 players, or a
  paid feed.
- **Not measurable today**: no instrument records it at scale.

**Our data**

- **TM**: Transfermarkt lineups, positions, captain flag, minutes, goals and
  assists with minute, substitutions, cards, valuations, date of birth,
  citizenship.
- **ESPN**: withdrawn on 2026-11-15 (terms of use; pre-registration A15). No
  ESPN data is used, so shots, fouls won and offsides exist only on the SB subset.
- **SB**: StatsBomb open events.

**StatsBomb coverage caveat** (checked against the open-data
`competitions.json` and match lists on 2026-11-14). Inside the window the
men's *tournaments* are complete: WC 2022, Euro 2020, Euro 2024, Copa 2024 and
AFCON 2023. The *club* seasons are single-team samples:

- La Liga 2019/20 and 2020/21: Barcelona matches only.
- Ligue 1 2021/22 and 2022/23: PSG only.
- Bundesliga 2023/24: Leverkusen only.
- MLS 2023: 6 Inter Miami matches.

Any SB-based estimate (M7a, the H8 validation) is therefore a
tournament-plus-three-clubs sample, not a league sample.

---

## A. Attacking off-ball

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Runs in behind / depth | Off-ball runs that stretch the line or receive behind it | Tracking only | Weak: SB through-ball passes (subset) | Measured only with tracking (`fernandez2018wide`, `spearman2018beyond`). No league-wide estimate of its share of player value found. **Evidence thin.** | M4 (residual) |
| Space creation for teammates | Dragging defenders to open space for others | Tracking only | None | `fernandez2018wide` quantifies it from tracking with single-match examples. No population effect size. | M4 (residual) |
| Getting into scoring positions | Being where the next shot is likely (off-ball scoring opportunity) | Tracking for the full quantity; its *outcome* is in public data | TM goals as the outcome; SB shots, np xG and touches in box (subset) | `spearman2018beyond` builds OBSO as a leading indicator from tracking. Our proxies see only the occasions that ended in a shot. | M1, M7a |
| Drawing fouls / penalties | Being fouled, winning penalties | Box score | None in TM; SB fouls won (subset) | No verified estimate of its value or stability. **Evidence thin.** | none |
| Making oneself available (receiving options) | Positioning that gives the passer a safe or valuable option | Tracking only | None | `power2017passes` and `fernandez2021epv` model pass options only from tracking. | Tax |

## B. Defensive prevention

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Positional defending (prevention without an event) | Shots and passes that never happen because of positioning | Tracking only | Defensive RAPM on goals against (M3) | Defensive plus-minus is estimable (`hvattum2020offdef`). Event metrics mostly ignore defending (`merhej2021defensive`). By its abstract, event statistics explain 22–38% of plus-minus variance by position, with no correction for noise (`gelade2020events`). | M3, M4, H3 |
| On-ball defensive actions | Tackles, interceptions, blocks, clearances | Public event data | SB counts (subset only); none in TM | `merhej2021defensive` values them by threat stopped, and `vanroy2020xtvaep` notes xT cannot value them at all. Possession depends on location, opponent and score (`lago2009possession`), so event counts partly measure opportunity, not skill. | M7a, M2 |
| Pressing / counterpressing | Pressure on the ball carrier; regaining after loss | Public event (counts); tracking (coordination) | SB `pressures` (subset) | Counterpressing is detected from combined positional and event data (`bauer2021counterpressing`). | M7a |
| Aerial duels | Winning headers in either box | Public event data | None in TM (SB duels not in our contract) | No verified effect size. **Evidence thin.** | none |
| Goalkeeper shot-stopping | Goals prevented given shots faced | Box score + shot quality | Goals conceded only (TM); SB xG faced on the subset | No verified academic estimate in this list. **Evidence thin.** GKs are ranked separately (design §6). | GK ranking |
| Fouls, cards, penalties conceded | Defensive discipline | Box score | TM yellow/red | Cards depend on the setting, not only the player: empty stadiums narrowed the home–away gap in cards (`bryson2021crowd`). | M2 |

## C. On-ball decision quality

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Finishing skill | Converting chances above their expected value | Box score + shot quality | np goals − SB np xG (subset only) | About 46% of Premier League goals involve a chance marker (`wunderlich2021randomness`). Player-level finishing effects persist in Bayesian xG (`scholtes2024bayesxg`), but goals−xG is biased and needs high volume (`davis2024finishing`). | M1 |
| Shot selection / shot quality | Taking good shots rather than many | Public event (location) | SB xG per shot (subset) | Location is among the strongest xG features (`robberechts2020xg`), as in early shot-quality models (`pollard1997strategies`). Tracking-based xG is more accurate (`anzer2021xg`). | M1 (SB subset) |
| Chance creation | Passes that lead to shots and goals | Box score / public event | TM assists; SB key passes and xA (subset) | Action-value models credit creation beyond assists (`decroos2019vaep`, `pappalardo2019playerank`). Assists also carry the teammate's finishing luck. | M2, M4 |
| Ball progression | Passes and carries that move the ball toward goal | Public event data | SB progressive passes and carries, xT added (subset) | xT and VAEP rank players differently (`vanroy2020xtvaep`). xT values zones (`singh2019xt`). EPV needs tracking (`fernandez2021epv`). | M7a |
| Pass choice versus options | Choosing the right pass given what was on | Tracking only | None | Only estimable with all player positions (`power2017passes`, `fernandez2021epv`). | none |
| Dribbling / ball retention | Keeping the ball under pressure | Public event data | SB dribbles completed (subset) | No verified estimate of its stability. **Evidence thin.** | M7a |
| Set pieces and penalties | Taking dead balls; penalty conversion | Box score | TM `penalty_goals` | Shootout outcomes depend on pressure framing (`jordet2008avoidance`). Penalty volume is role-allocated, hence the np-versus-all choice in M5. | M5 |

## D. Physical / availability

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Availability | Being on the pitch: minutes, appearances | Box score | TM minutes | At team level, lower injury burden and higher availability go with higher league rank and more points (`hagglund2013injuries`). | M5 (rate versus volume) |
| Injury history | Time lost to injury | Box score (public records) | None: injuries are on the Transfermarkt site but not in our data contract | Same team-level evidence (`hagglund2013injuries`). No player-value estimate verified. | none |
| Running output | Distance, high-speed running, sprints | Tracking only | None | Very noisy match to match: CV ≈ 16% for high-speed running and ≈ 31% for sprint distance (`gregson2010variability`). Position-dependent (`bradley2009hir`). Hard to interpret without context (`carling2013physical`). | none |
| Age / career stage | Where in the ageing curve a season sits | Box score | TM date of birth | In WhoScored ratings of four top leagues, 2010/11–2014/15, forwards peak at about 25 and defenders at about 27 (`dendir2016peak`). This is why "peak season" and "window mean" can rank players differently. | M5, H5 |

## E. Psychological

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Big-game (knockout) performance | Over-performance in decisive matches | Box score | TM knockout flag × per-90 output (A4: np G+A per 90) | A detectable trait in tennis, where high-stakes points are many (`gonzalezdiaz2012tennis`). In football the only player-level study is cross-sectional (`bransen2019choke`). Performance indicators are unstable in interactive sports (`lames2007reliable`), and streak/clutch tests on short sequences are biased (`miller2018hothand`). | M7b, H7 |
| In-match pressure | Performing when the score or clock raises stakes | Public event + model | Derivable from TM goal minutes (game state); not computed | `bransen2019choke` models pressure from context and reports player differences. No stability test was found. | none |
| Penalty-shootout composure | Converting when a miss loses | Box score in principle | None in our contract | Avoidance under negative-valence kicks (`jordet2008avoidance`). The famous first-kicker advantage (`apesteguia2010pressure`) did **not** replicate in a larger sample (`kocher2012pressure`). | none |
| Form / streakiness ("hot hand") | Short runs of above-baseline output | Box score | TM match sequences | Goals are close to Poisson given team strength (`heuer2010poisson`). The streak test of `gilovich1985hothand` needs the correction of `miller2018hothand`. | M1 |
| Mentality, motivation, resilience | Drive, response to setbacks | Not measurable today | None | No verified effect-size estimate for elite football. | M4 (residual) |

## F. Social / leadership

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Captaincy | Formal leadership role | Box score | TM lineup `captain` flag | The captain is often not perceived as the main leader (`fransen2014captain`), so the flag is a weak proxy for leadership. | Tax (possible M6 covariate) |
| Informal leadership | Task, motivational, social and external leadership by non-captains | Not measurable today | None | Leadership is spread across informal leaders (`fransen2014captain`). | M4 (residual) |
| Chemistry / familiarity | Shared experience with current teammates | Derivable from TM lineups (shared minutes); not computed | Could be built from TM lineups | Prior shared *success* predicts wins (`mukherjee2019shared`), but at match level shared experience has no independent effect once team strength is controlled (`lecoz2025shared`). RAPM's additivity ignores it. | M3 (assumption) |
| Team cohesion | Group attraction and integration | Not measurable today (questionnaires only) | None | Moderate-to-large cohesion–performance association; direction not identified (`carron2002cohesion`). | none |
| Cultural / linguistic fit | Diversity and language overlap of the squad | Derivable from TM citizenship | Not computed | Diverse Champions League squads outperformed after talent controls (`ingersoll2017diversity`). | none |

## G. Tactical / role

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Nominal position | Lineup slot | Box score | TM lineup position → 8 groups (design §4.4) | Nominal and actual roles differ (`aalbers2019roles`, `bialkowski2014roles`). | M2, M4 (H3 groups), M5 |
| Actual role in the system | e.g. false 9, inverted full-back, deep playmaker | Public event (partly); tracking | None (SB subset could cluster roles) | Roles are recoverable from events (`aalbers2019roles`) or tracking (`bialkowski2014roles`). | none |
| Usage / opportunity share | Share of team shots, set pieces and penalties allocated to the player | Box score | Penalties (TM); share of team shots on the SB subset | Possession depends on location, opponent and score, so event counts partly measure opportunity (`lago2009possession`). Usage inflates counting stats independently of skill. | M2 (team ICC), M5 |

## H. Context (team, coach, league, opponents)

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Teammate quality | Who plays alongside | Box score (lineups) | RAPM co-player adjustment (M3) | Collinear lineups make individual effects imprecise. In hockey, most players are not distinguishable from zero (`gramacy2013hockey`, `macdonald2012ridge`). Team and player ratings carry complementary information (`arntzen2021ratings`). | M3, M2 |
| Team strength | Overall club / national-team level | Box score derived | `team_elo` (in-house); squad valuations | Elo is a usable match predictor (`hvattum2010elo`). Squad valuations forecast national-team results better than FIFA ranking and Elo (`peeters2018wisdom`). | M2, M5 (Elo-partialled), M6 |
| Coach / manager | Manager ability and style | Box score (public records) | None: manager tables are not in our data contract | Managers at the top versus bottom of the ability distribution differ considerably in points, and they shape style (`muehlheusser2018managers`). | M3 (absorbed into team terms) |
| League / competition strength | Level of the competition | Box score derived | Competition-season intercepts only in the pre-registered analysis (in-house Elo levels are not comparable across leagues); post hoc: league means in RAPM and factors from players who change league (A20–A24) | League terms inside plus-minus models (`pantuso2021transfer`, `kharrat2020plusminus`). Our levels agree with club results for most leagues and are too low for La Liga. | M3, M5 (competition weights), league-adjusted analysis |
| Opponent quality | Strength of the opposition faced | Box score derived | `opp_elo` | Possession depends on the opponent as well as on location and score (`lago2009possession`). | M3, M5 |
| Home advantage and crowd | Venue, crowd presence | Box score | TM home/away; empty-stadium matches 2020–21 inside the window | Home advantage fell in German top-division ghost games of 2019/20 (`fischer2021ghost`); empty stadiums narrowed the home–away gap in cards (`bryson2021crowd`). A constant home term is mis-specified for 2020–21. | M3 |
| Reputation / visibility / market value | How the market and voters see the player | Box score derived | TM valuations; prior Ballon d'Or placings | Crowd valuations predict fees (`herm2014crowd`). Talent and popularity both raise value (`franck2012superstar`). Award votes show similarity bias, small in aggregate (`coupe2018ballondor`) and rising with cultural similarity (`johnson2022voting`), and growing concentration on certain player types (`anderson2020celebrity`). | M6, H6 |

## I. Randomness

| Variable | What it is | Tier | Best proxy in our data | What the literature says (size / importance) | Module |
|---|---|---|---|---|---|
| Shot-outcome luck | Goals scored or conceded beyond chance quality | Box score + xG | Goals − SB xG (subset) | About 46% of goals involve chance markers (`wunderlich2021randomness`). Short-run results mislead (`brechot2020randomness`). | M1, M8 |
| Match-outcome noise | The better team not winning | Box score | Match results | Typical scorelines are weak evidence of the better team (`skinner2009experiments`). Luck is large in football leagues (`aoki2017luck`). Goals are near-Poisson (`heuer2010poisson`). | M9, M3 |
| Refereeing decisions | Penalties, cards, offsides given | Box score | TM cards and penalty goals | Empty stadiums narrowed the home–away gap in cards (`bryson2021crowd`). | M2 |
| Sampling noise in rates | Finite-minutes error in any per-90 statistic | Box score | Split-half reliability (M1) | Metrics differ greatly in stability (`franks2016metaanalytics`). Interactive-sport indicators are intrinsically unstable (`lames2007reliable`). Physical rates have CVs of 16–31% (`gregson2010variability`). | M1, M8 |

---

## Counts and honest summary

46 variables. By tier, as listed above:

| Tier | Count |
|---|---|
| Box score (including derived, not-yet-computed, or outside our data contract) | 28 |
| Public event data | 6 |
| Mixed (outcome public, process needs tracking) | 3 |
| Tracking or proprietary only | 6 |
| Not measurable today | 3 |

Of the 28 box-score variables, 5 are not in our panel as built: injury
history, coach, penalty shootouts, chemistry and cultural fit.

Where the evidence is solid:

- Randomness in goals.
- Match-to-match noise of physical outputs.
- The limits of plus-minus precision.
- Similarity bias in award voting.
- The weight-versus-importance gap in composite indices (`paruolo2013voodoo`).

Where it is thin:

- The *value* of off-ball movement and of defensive positioning. Tracking
  studies show they exist and can be measured, but none gives a
  population-level share of player impact.
- Aerial play.
- Goalkeeping beyond xG.
- Leadership and mentality.
- League-strength adjustment.

The closest published quantity to "what box-score/event statistics miss" is
`gelade2020events`: by its abstract, event-level statistics explain 22–38% of
plus-minus variance depending on position. The abstract reports no correction
for noise, so it is the closest quantity to M4's uncorrected R², not a
comparison for its corrected share.

Specification choice is not a football variable but is a measured source of
uncertainty in this paper. Many-analyst studies, one of them on football
refereeing data, show it can rival the effect being studied
(`silberzahn2018many`, `breznau2022many`). It is handled by M5.
