# The Unmeasured Remainder — study design

Date: 2026-11-14. Paper 2 of this repository. Paper 1 (`paper/`, the de-margining
artifact) is finished and is not touched by this work.

## 1. Intent

Liam asked for a second research paper, as rigorous as paper 1, that:

1. names the best footballer of the 2020s (1 Jan 2020 onward), and
2. measures, with numbers, how much of the answer the statistics *cannot*
   decide — sampling noise, modelling choices, the parts of football that no
   public number captures (off-ball movement, defensive prevention, leadership,
   tactical role…), team context, and how recognition (Ballon d'Or) departs from
   all of these.

It replaces a "V1" ranking Liam got elsewhere (Rodri 94.7, Mbappé 94.5,
Vinícius 90.9, …; `paper-unmeasured/data/v1_ranking.csv`). That ranking mixed
human ratings, team results and longevity guesses with hand-set weights and
admitted double counting. This paper does not tune those weights. It rebuilds
every player estimate from raw match data, and treats the estimand itself —
what "best" means — as a source of uncertainty to be measured.

The headline is a probability, not a list: P(player is #1), rank intervals, and
the share of pairwise comparisons the data actually decides.

Liam told Claude to make every methodological call ("fonce, choisis toi").
Every call below is Claude's and is marked as such where it matters.

## 2. Goals, non-goals, anti-goals

Goals
- A reproducible pipeline: raw public data → tables → figures → LaTeX macros →
  PDF, with no number typed by hand into the paper (same rule as paper 1).
- A variance accounting of "who is best" into named sources (§5).
- Tests and validation: every estimator is unit-tested on synthetic data with a
  planted truth, and the whole pipeline gets a recovery study (§5, M9).
- A pre-registration committed before any analysis result is computed.

Non-goals
- No UI, no web app, no API (standing rule for this repo).
- No proprietary data (Opta, StatsBomb paid, tracking). The paper states what
  that data would add instead.

Anti-goals (failure even if it "works")
- A single ranked list presented as the answer.
- Any quality counted twice without the overlap being measured (the V1 flaw).
- Any citation not verified against a live source.
- Any number in the paper that is not traceable to a generated file.

## 3. Data (verified reachable on 2026-11-14)

| Source | What | Coverage used | Access |
|---|---|---|---|
| Transfermarkt datasets (dcaribou, R2 mirror `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/<name>.csv.gz`) | games, appearances, lineups (positions, captain), game_events (goals+assists with minute, substitutions with minute, cards), players, player_valuations, club_games, competitions | 2020-01-01 → 2026-08-31; GB1 ES1 L1 IT1 FR1, CL, EL, FIWC EURO COPA AFCN, SA1, MLS1 (+ any other competition a pool player played in) | public CSV, snapshot Last-Modified 2026-09-05 |
| ESPN site API `site.api.espn.com/apis/site/v2/sports/soccer/<lg>/summary?event=<id>`; event lists via `sports.core.api.espn.com/v2/sports/soccer/leagues/<lg>/seasons/<yyyy>/types/1/events?limit=1000` | Opta-style text commentary (every shot with zone, body part, outcome, assister, assist type; fouls, offsides, penalties), rosters with per-player stats | eng.1 esp.1 ger.1 ita.1 fra.1 uefa.champions, plus fifa.world uefa.euro conmebol.america and usa.1 ksa.1 if commentary exists | public, unauthenticated; ≤ 4 req/s, cached, never re-fetched |
| StatsBomb open data (`raw.githubusercontent.com/statsbomb/open-data/master/data/…`) | full event data with location and StatsBomb xG | men's competitions with seasons in 2019/20–2024: WC 2022, Euro 2020, Euro 2024, Copa América 2024, AFCON 2023, La Liga 2019/20 & 2020/21, Ligue 1 2021/22 & 2022/23, 1. Bundesliga 2023/24, MLS 2023 | public, attribution required |
| Ballon d'Or results | points of every ranked player, editions 2021–2025 (2020 cancelled), 2026 only if published before 2026-11-14 and voting covers our window | — | curated by hand from ≥ 2 sources, committed as CSV with source URLs |

Unavailable, checked: Understat ("We will come back soon"), FBref (HTTP 403 to
scripts), ClubElo API (HTTP 502). Team strength is therefore computed in-house
(§4.3). Claude's call: do not work around blocks on sites that refuse scripts.

Study window (Claude's call): matches dated 2020-01-01 to 2026-08-31 inclusive.
The end date is the Transfermarkt snapshot; ESPN is cut at the same date so the
sources agree. The paper says plainly that Sep–Nov 2026 is outside the window.

Season label: club season runs 1 Jul → 30 Jun, labelled `2021/22`. A
national-team tournament belongs to the season in which its final is played.

## 4. Data contracts

Raw caches live under `data/players/raw/<source>/` (gitignored by `/data/`),
each source with a `manifest.json` recording URL, retrieval time and SHA-256 per
file (paper 1's convention, see `src/footy/study/acquire.py`). Processed tables
are `data/players/processed/<name>.csv.gz`. Results are
`results/players/<module>/…`. Paper macros and figures are
`paper-unmeasured/generated/` (gitignored).

Identifiers: `game_id` and `player_id` are Transfermarkt ids and are the keys
of the whole study. ESPN and StatsBomb ids are linked to them (§4.4).

### 4.1 Transfermarkt (`footy/players/transfermarkt.py`, `stints.py`)

- `tm_games`: game_id, competition_id, competition_type, season (label), date,
  round, home_club_id, away_club_id, home_goals, away_goals, home_club_name,
  away_club_name.
- `tm_appearances`: game_id, player_id, club_id, date, competition_id, minutes,
  goals, assists, yellow, red, started (bool), position (lineup string),
  captain (bool), penalty_goals.
- `tm_players`: player_id, name, date_of_birth, position, sub_position,
  citizenship.
- `tm_valuations`: player_id, date, market_value_eur.
- `tm_stints`: stint_id, game_id, start_min, end_min, minutes, home_goals,
  away_goals, home_n, away_n (players on pitch, after red cards). A stint is a
  maximal interval with no substitution or red card. Stoppage time is folded
  into minute 45/90 (Transfermarkt minutes are integers). Goals at the minute
  of a change are assigned to the stint that ends there — stated assumption,
  tested.
- `tm_stint_players`: stint_id, player_id, side (`home`/`away`).

### 4.2 ESPN (`footy/players/espn.py`, `commentary.py`)

- `espn_games`: espn_event_id, league, season, date, home_name, away_name,
  home_score, away_score.
- `espn_shots`: espn_event_id, seq, minute, period, team_name, player_name,
  outcome ∈ {goal, saved, missed, blocked, post}, body ∈ {right_foot,
  left_foot, header, other}, zone (the commentary phrase, normalised, e.g.
  `centre_box`, `outside_box`, `left_box`, `right_box`, `very_close_range`,
  `six_yard_left`, `six_yard_right`, `difficult_angle_left`,
  `difficult_angle_right`, `long_range`, `more_than_35`), assist_player,
  assist_type ∈ {pass, cross, through_ball, headed_pass, corner, set_piece,
  fast_break, none, other}, penalty (bool), own_goal (bool), raw_text.
  Every commentary line starting "Attempt", "Goal!", "Penalty" or "Own Goal" is
  either parsed or counted in an explicit `unparsed` tally — never dropped
  silently. Parse rate is a reported number.
- `espn_player_match`: espn_event_id, espn_player_id, player_name, team_name,
  starter, and the roster stats (totalShots, shotsOnTarget, totalGoals,
  goalAssists, foulsCommitted, foulsSuffered, offsides, saves, goalsConceded,
  shotsFaced, yellowCards, redCards, ownGoals) plus commentary-derived counts:
  fouls_won, fouls_committed, offsides, penalties_won, penalties_conceded,
  handballs.

### 4.3 StatsBomb and team strength

- `sb_shots`: sb_match_id, minute, second, player_id, player_name, team, x, y,
  body_part, shot_type, technique, outcome, sb_xg.
- `sb_player_match`: sb_match_id, player_id, player_name, team, minutes,
  passes, passes_completed, progressive_passes, progressive_carries,
  passes_into_box, key_passes, sb_xa, pressures, tackles, interceptions,
  blocks, clearances, dribbles_completed, touches_in_box, xt_added, shots, sb_xg.
  `xt_added` uses a Markov expected-threat grid fitted on the same open data
  (Singh 2019 formulation; Claude's call over VAEP, simpler and auditable).
- `team_elo`: game_id, home_elo_pre, away_elo_pre. Elo with goal-difference
  multiplier and home advantage, run over **all** Transfermarkt games
  2015-07-01 onward (burn-in) so cross-league European games tie leagues
  together. K and home advantage chosen by minimising one-step-ahead log loss
  on 2015–2019, frozen before 2020. National teams get their own Elo pool.

### 4.4 Linking and panel (`footy/players/linking.py`, `zone_xg.py`, `panel.py`)

- `link_games`: espn_event_id ↔ game_id by (date ± 1 day, normalised team
  names); `link_players`: (espn_event_id, player_name) ↔ player_id by name
  similarity *within that game's two lineups only*. Match rates and the
  ambiguous cases are reported numbers. Same for StatsBomb.
- `zone_xg`: P(goal | zone, body, assist_type, penalty, header-after-cross…)
  fitted on all ESPN non-penalty shots in the window by L2-regularised logistic
  regression, λ by 10-fold CV grouped by match. Penalty xG is the empirical
  conversion rate. Validated against StatsBomb xG on the overlapping matches
  (WC 2022, Euros, Copa 2024, La Liga 2020/21, Ligue 1, Bundesliga 2023/24):
  shot-level and player-season-level correlation, calibration. This gap is
  itself a measured uncertainty source (coarse data).
- `panel_player_match`: one row per (game_id, player_id) with minutes ≥ 1:
  date, season, competition_id, competition_type, club_id, position_group,
  started, minutes, goals, np_goals, assists, shots, shots_on_target, zxg,
  np_zxg, zxa (zone-xG of shots the player assisted), fouls_won,
  fouls_committed, offsides, penalties_won, yellow, red, on_pitch_gf,
  on_pitch_ga, on_pitch_zxg_for, on_pitch_zxg_against, team_elo, opp_elo,
  knockout (bool), espn_linked (bool).
- `panel_player_season`: per (player_id, season, competition scope ∈
  {league, ucl, intl, all}) sums and per-90 rates.

Position groups (Claude's call): GK; CB; FB (left/right-back); DM; CM; AM;
W (left/right winger, left/right midfield); CF (centre-forward, second
striker). A player's group is his minutes-weighted modal lineup position in
the window.

## 5. Methods — the sources of uncertainty

Each module is pure (no I/O, randomness only through an explicit
`numpy.random.Generator` argument), has a synthetic-truth test, and writes its
outputs under `results/players/<module>/`.

- **M1 Sampling noise — `reliability.py`.** Split-half reliability (odd vs even
  matches inside a player-season, averaged over 200 random splits), stepped up
  with Spearman–Brown; stabilisation point per metric (minutes at which
  reliability reaches 0.5 and 0.7); match-level bootstrap intervals for every
  player metric. Test: recovers planted signal/noise ratios.
- **M2 Structure and double counting — `structure.py`** (correlation
  specialist). Spearman and partial correlations (controlling team Elo,
  league-season), hierarchical clustering of metrics, PCA with Horn's parallel
  analysis, maximum-likelihood factor analysis, effective number of
  independent metrics (participation ratio of the eigenvalues), distance
  correlation for non-linear dependence, team intraclass correlation (share of
  a metric's variance that belongs to the team, not the player), and a
  double-counting index for any weighted composite: the share of its variance
  carried by the first factor versus the share its weights claim. Applied to a
  V1-style composite.
- **M3 On-pitch impact — `rapm.py`.** Regularised adjusted plus-minus on
  `tm_stints`: response = goal difference per 90 (and zone-xG difference per 90
  where ESPN is linked), separate offensive and defensive player effects, home
  advantage, competition-season intercepts, Elo difference covariate; ridge
  solved with `scipy.sparse.linalg.lsqr`, λ by match-grouped CV; match bootstrap
  (B = 200) for standard errors; split-half reliability of RAPM. Test: recovers
  planted effects in a simulated league.
- **M4 The unmeasured share — `construct_gap.py`.** Regress RAPM on the
  individual statistics, per position group, with cross-validated ridge.
  Disattenuate: R²_true = R²_obs / (rel_RAPM · rel_pred). Unmeasured share =
  1 − R²_true, with bootstrap CI, per position group. This is the paper's
  core number for "what the stats cannot see".
- **M5 Specification uncertainty — `multiverse.py`.** Full factorial over:
  metric family {box, box+zone-xG, impact (RAPM), hybrid}; weights {equal-z,
  first principal component, regression on RAPM, V1-like expert weights};
  position normalisation {none, within-group z, replacement level};
  context {none, Elo-partialled}; time aggregation {peak season, best three
  seasons, window mean (minutes-weighted), window volume (rate × minutes),
  recency-weighted}; competition weights {equal, UCL & internationals ×1.5,
  league only}; minutes threshold {900, 1800 per season}; shrinkage {raw,
  empirical Bayes}. Outputs: rank distribution per pool player, P(#1),
  specification curve, ANOVA/Sobol first-order and total indices of each
  player's rank per dimension (exact on a full factorial).
- **M6 Recognition — `perception.py`.** Ballon d'Or points per edition against
  blocks: individual statistics (that season), impact (RAPM), team success
  (league title, UCL win/final, international trophy — computed from
  `tm_games`), reputation (prior placings, market value). OLS on log(1+points)
  and a Plackett–Luce model on the ranked lists; Shapley/dominance
  decomposition of explained variance across blocks; residual = what the vote
  rewards that none of the blocks capture.
- **M7 Big games and on-ball value — `tournaments.py`.** (a) On StatsBomb
  events: how much of a player's event value (xT added, progressive actions,
  pressures) is predictable from box-score stats alone — the visibility of
  on-ball value to public box scores. (b) "Big-game player" as a trait:
  knockout over-performance relative to the player's own baseline, correlated
  between disjoint periods (2020–2022 vs 2023–2026). Near-zero correlation
  means the trait is not measurable as a stable quantity.
- **M8 Synthesis — `synthesis.py`.** Draw (specification uniformly from M5) ×
  (match-bootstrap replicate) × (unmeasured-component draw scaled by M4's
  position-specific share). Output: P(#1), P(top-3), P(top-10), 90% rank
  intervals, the share of the pool's pairwise orderings that are decided
  (P ≥ 0.9), and the law-of-total-variance split of each player's rank into
  sampling, specification and unmeasured components. Agreement across the
  stats, impact and recognition views by Kendall's W.
- **M9 Recovery study — `simulation.py`.** Simulate leagues with planted
  player qualities, the observed minutes structure and the noise levels from
  M1/M3; run the estimators; report how often the true #1 is recovered and how
  rank error grows with true-quality gaps. This bounds what *any* method could
  decide with data of this size.

Taxonomy (no code): a table of football variables — measured, proxied,
tracking-only, unobservable — each with our proxy if any and the literature's
best estimate of its size, every entry cited.

## 6. Player pool (fixed before results)

Pool = the 48 players of the V1 list ∪ every player ranked in a Ballon d'Or
edition 2021–2025 (and 2026 under §3's condition). Estimates are produced for
every player in the data (RAPM needs them all); rankings are reported for the
pool. Goalkeepers are ranked separately: no public stat in our data measures
shot-stopping beyond goals prevented versus zone-xG faced, and the paper says
so.

## 7. Repository layout

```
src/footy/players/        package for paper 2 (pure modules + I/O modules)
tests/players/            tests, synthetic-truth fixtures, no network
scripts/players/          acquire.py, build.py, analyse.py, make_figures.py, make_tables.py
paper-unmeasured/         main.tex, refs.bib, preregistration.md, data/ (curated CSVs), generated/ (ignored)
The-Unmeasured-Remainder.pdf   compiled paper at the repo root
```

Make targets: `players-data`, `players-results`, `paper2`. Repo quality bar is
unchanged: ruff, mypy --strict, pytest all green (`make verify`).

Language: English, like paper 1 (Claude's call; a French version can follow).

## 8. Approaches considered

1. **Re-weight V1** (adjust coefficients of the existing ranking). Rejected:
   inherits its double counting and unverifiable inputs; cannot measure
   uncertainty.
2. **One big Bayesian model** (latent quality, all data, one posterior).
   Rejected as the *primary* design: the model's own structure would be a
   hidden specification choice, and fitting it needs a probabilistic-
   programming dependency this repo does not have. Its role is played by M8's
   explicit mixture.
3. **Chosen: triangulated variance accounting.** Three views (individual
   stats, on-pitch impact, recognition), each with measured noise; a
   multiverse for the modelling choices; a disattenuated gap between views as
   the unmeasured share; a recovery study as the upper bound. Each piece is
   simple, separately testable and separately falsifiable.

## 9. Work plan and agents

Wave 1 (parallel, disjoint files): literature + taxonomy; Ballon d'Or data;
Transfermarkt pipeline + stints + Elo; ESPN pipeline + commentary parser;
StatsBomb pipeline + xT. Then pre-registration is committed. Wave 2: linking,
zone-xG, panel. Wave 3 (parallel): M1+M7b, M2, M3+M4, M5, M6, M7a. Wave 4: M8,
M9, figures, tables, paper. Wave 5: adversarial verification of every number,
citation and claim; `make verify`; PDF build.

Agents never commit; Claude reviews and commits per wave. Agents run checks on
their own files only, never on the whole repo while others are working.

## 10. Left to the builder

Exact regex grammar of the commentary parser; Elo K grid; ridge λ grids;
figure styling; ID-matching thresholds (reported, not hidden).
