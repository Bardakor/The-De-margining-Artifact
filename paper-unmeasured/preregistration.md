# Pre-registration — The Unmeasured Remainder

Committed on 2026-11-14, before any player-level statistic, RAPM, ranking or
Ballon d'Or regression has been computed. The data sources had been probed for
availability only (schemas and a 21-match commentary sample). The git commit
that adds this file is the timestamp.

Design: `docs/superpowers/specs/2026-11-14-unmeasured-remainder-design.md`.

## Fixed before results

- Window: matches dated 2020-01-01 to 2026-08-31 inclusive.
- Pool: the 48 players of `data/v1_ranking.csv` ∪ every player ranked in a
  Ballon d'Or edition 2021–2025 (and 2026 only if published before 2026-11-14
  with a voting period inside the window). Goalkeepers ranked separately.
- Minimum minutes for a player-season to enter M1, M2 and M4: 900.
- "Decided" pairwise ordering: P(i ranked above j) ≥ 0.9 or ≤ 0.1 in M8.
- Bootstrap: 200 match-level replicates; 90% intervals.
- Multiverse: the full factorial in design §5 (M5), every specification
  weighted equally in M8.

## Hypotheses

Each is stated with its test and the result that would refute it. All are
reported whatever the outcome.

- **H1 — Most of the ranking is undecided.** Fewer than half of the pool's
  outfield pairwise orderings are decided in M8. Refuted if ≥ 50% are decided.
- **H2 — No consensus number one.** No outfield player has P(#1) ≥ 0.5 in M8.
  Refuted if one does.
- **H3 — The unmeasured share follows position.** The disattenuated unmeasured
  share of on-pitch impact (M4) is larger for CB and DM pooled than for CF and
  W pooled; the 90% bootstrap interval of the difference excludes 0. Refuted if
  the interval includes 0 or the difference is negative.
- **H4 — Modelling choices outweigh sampling noise.** For the ten players
  ranked highest by the median specification, the specification component of
  rank variance exceeds the sampling component (M8 decomposition) on average.
  Refuted if the mean ratio specification/sampling is ≤ 1.
- **H5 — "What best means" is the decisive choice.** Time aggregation has the
  largest total-order index (M5) for the rank of the player with the highest
  M8 P(#1). Refuted if another dimension's index is larger.
- **H6 — Recognition follows the team.** In M6, the Shapley share of the team-
  success block exceeds that of the individual-statistics block. Refuted
  otherwise.
- **H7 — "Big-game player" is not a stable trait.** The correlation of knockout
  over-performance between 2020–2022 and 2023–2026 (M7b) is below 0.2.
  Refuted if ≥ 0.2.
- **H8 — Coarse shot data loses little at season level.** Player-season
  correlation between zone-xG and StatsBomb xG on overlapping matches is
  ≥ 0.9, while the shot-level correlation is lower. Refuted if the season
  correlation is < 0.9.

## Not pre-registered

Everything else — the taxonomy, the descriptive structure of the metrics (M2),
the recovery study (M9), the comparison with V1 — is exploratory and labelled
as such in the paper.

## Amendments

Each is committed before any real-data analysis has run; the commit is the
timestamp.

- **A1 (2026-11-14).** H6 is decided by the OLS Shapley shares (M6); the
  Plackett–Luce shares are reported alongside, and any disagreement between the
  two is reported. Reason: M6 produces both and the original text did not say
  which decides.
- **A2 (2026-11-14).** The 2026 edition exists (ceremony 2026-10-26, voting
  period 2025-08-03 to 2026-07-19, inside the window), so by the rule above its
  ranked players are in the pool.
- **A3 (2026-11-14).** For the 2021 edition, "previous edition" in the
  reputation block means 2019 (2020 was cancelled).
- **A4 (2026-11-14).** H7's primary metric is non-penalty goals plus assists
  per 90 (available for every competition from Transfermarkt); np zone-xG plus
  zone-xA per 90 is reported as secondary where ESPN coverage allows. A player
  enters H7 with at least 360 knockout minutes in each period. Reason: the
  original text named neither.
- **A5 (2026-11-14).** H7 is concluded ("not a stable trait") only if r < 0.2
  *and* the test had power: with the within-period split-half reliabilities of
  the over-performance score, a true correlation of 0.3 would have been
  detected with ≥ 80% power. Otherwise H7 is reported as inconclusive
  (underpowered). Reason: a difference score from a few hundred knockout
  minutes is unreliable, so r < 0.2 alone would be near-automatic.
- **A6 (2026-11-14).** H8 is tested on player-season non-penalty xG *per shot*
  (which removes shot volume); refuted if that correlation is < 0.8. The
  per-90 correlation is descriptive. Reason: season totals are driven by shot
  counts, so the original test was near-automatic. If ESPN data is withdrawn,
  H8 is reported as not tested.
- **A7 (2026-11-14).** In M4 and H3, R²_true is clipped to [0, 1]; intervals
  come from a joint bootstrap over players and RAPM replicates that
  re-estimates both reliabilities in each replicate; raw values are reported.
- **A8 (2026-11-14).** M3 estimates home advantage per competition-season, not
  as one constant (empty stadiums in 2019/20–2020/21).
- **A9 (2026-11-14).** Two corrections found on synthetic leagues, before any
  real-data run. (i) In M4, RAPM reliability is split-half reliability times an
  attribution factor (effects planted on the real schedule and fitted without
  noise): ridge lets teammates leak into each other's estimates, the leak is
  stable across halves, and split-half alone counts it as signal (0.29 vs a
  true 0.15 in one test league). (ii) A7's replicate-based joint bootstrap adds
  a second draw of noise and overstates the spread (synthetic interval
  [−0.08, 0.65] vs a true sampling spread near [0.36, 0.50]). H3 is therefore
  decided by a parametric bootstrap on the real schedule: plant effects with
  the estimated variances and measurable shares on the real stints and
  minutes, re-run M4 end to end, and take the 90% interval of the contrast.
  The player bootstrap and the A7 joint bootstrap are reported alongside.
- **A10 (2026-11-14).** M4's primary estimator is cross-half: with one game
  split shared by both sides, the statistics prediction from one half is
  correlated with RAPM from the other half, and the squared correlation is
  divided by the product of the half-length reliabilities. The within-sample
  estimator is reported as robustness. Reason: on real StatsBomb data (M7a) the
  within-sample correction exceeded 1 because the same matches drive both
  sides; in M4 that shared match noise would bias the unmeasured share down.
- **A11 (2026-11-14).** Primary analyses use only data available for every
  match of the panel: Transfermarkt (plus the StatsBomb-derived stints and
  appearances of WC 2022 and Euros 2020/2024). M4's statistics are per-90
  non-penalty goals, penalty goals, assists, yellow cards and red cards. M5's
  families become: box {goals, non-penalty goals, assists}; box + raw on-pitch
  goal difference per 90 (the unadjusted plus-minus public sites show); impact
  (RAPM); hybrid. Any ESPN-based version (shots, zone-xG, fouls won) is
  secondary, restricted to the leagues ESPN covers, and reported only if ESPN
  is retained. Reason: the primary analysis must not depend on the pending
  ESPN decision, and must be uniform across leagues.
- **A12 (2026-11-14).** M3 primary specification, fixed before its first
  real-data fit: every stint in the panel (Transfermarkt stints of all
  competitions, plus StatsBomb stints of WC 2022 and Euros 2020/2024);
  intercept and home term per competition-season; a player gets his own
  columns with ≥ 900 minutes in the window, otherwise the replacement column;
  **no Elo covariate** — the players on the pitch already are the team's
  strength, so adjusting for Elo would remove part of the signal being
  estimated (the with-Elo fit is reported as robustness); λ by 5-fold
  game-grouped CV; B = 200 game bootstrap; master seed 20261114 for every
  analysis.
- **A13 (2026-11-15).** Implementation choices that A9, A11 and M8 leave
  open, fixed before any real-data RAPM, M4, M5 or M8 result exists.
  (i) *M5 under A11.* Box = per-90 goals, non-penalty goals, assists;
  box + GD adds raw on-pitch goal difference per 90 (minutes in matches with
  stints only); impact = per-season RAPM offence and defence; hybrid = box +
  GD + impact. V1-like weights: goals 0.30, non-penalty goals 0.15, assists
  0.25, on-pitch GD 0.20, RAPM offence 0.05, RAPM defence 0.05 (a stylised
  imitation, as before; on-pitch GD stands in for V1's team-results
  component). (ii) *M8 unmeasured draw.* The share ω of a specification's
  score that sees only individual statistics is the share of its absolute
  weights on the box metrics, taken exactly from M5's weights on the
  original data; on-pitch GD counts with RAPM as an on-pitch measure. The
  share σ per position group is M4's cross-half estimate for the total
  component, clipped to [0, 1]. U = 4 draws per (specification, replicate);
  pairwise probabilities and expected ranks are computed exactly.
  (iii) *M8 replicates.* B = 200 match bootstraps; in each, the same
  resample of games drives pooled RAPM, per-season RAPM and the
  player-season statistics, and M5 is rerun on it. (iv) *A9 calibration.*
  Planted offence and defence SDs per position group are set so that the
  simulated RAPM's split-half covariance per group matches the real one
  (multiplicative updates, a few simulations each); measurable share per
  group = M4's cross estimate (total, clipped); intercept and home term per
  competition-season = the real fit's; no match shock. The simulated box
  score's per-90 rate has the real group mean of non-penalty goals plus
  assists and an SD chosen so that its full-window reliability at the
  group's median minutes equals the real reliability of M4's statistics
  prediction in that group; S = 200, five splits per simulation.
  (v) *Recognition view* for Kendall's W: Ballon d'Or points summed over
  the 2021–2026 editions, 0 for unranked pool players.
- **A14 (2026-11-15).** M6's universe and features, fixed before any M6
  fit. Universe: every outfield pool player with minutes in the season an
  edition votes on (goalkeepers excluded: the statistics block says nothing
  about them), points 0 if unranked. Edition → season: 2022–2026 vote on
  2021/22–2025/26; 2021 (the last calendar-year edition) is mapped to
  2020/21, an approximation. Blocks, all for that season over every
  competition: individual statistics = goals, assists, minutes and
  non-penalty goals per 90; impact = per-season RAPM offence and defence;
  team success = indicators for league title (first in the final table
  computed from the games, 3/1/0 points, then goal difference, then goals),
  UCL winner, UCL losing finalist, and senior international tournament
  winner; reputation = log(1 + previous edition's points) (2019 for 2021,
  A3) and log market value at the last valuation on or before 30 June of
  the edition year. Players without a per-season RAPM estimate are dropped
  and counted.
- **A15 (2026-11-15).** ESPN is withdrawn (terms of use; see
  `docs/unmeasured/data-licences.md`). Its code, cache and tables are
  deleted before any analysis used them. Consequences, by the rules already
  stated: H8 is reported as not tested (A6); A4's secondary zone-xG metric
  and A11's ESPN-based secondary analyses are not run. Shots, fouls won and
  shot quality appear only on the StatsBomb subset, as exploratory analyses.
- **A16 (2026-11-15).** Three records, each stated with when it was made.
  (i) Before any M5 or M8 run: empirical-Bayes shrinkage in M5 uses the
  within-position-group stabilisation constant of each metric (M1, per
  group), because it shrinks toward the group mean; the outfield-pooled k
  would count differences between positions as signal and under-shrink.
  (ii) After M2 ran: the team-ICC interval of M2 is changed from the
  percentile to the basic club-bootstrap interval, because on synthetic data
  the percentile interval was shifted low (a planted ICC sat at its upper
  edge; a null estimate sat outside its own interval). Point estimates are
  unchanged. (iii) After H7 was computed: A5 left the test level open; the
  analysis used a one-sided Pearson test at α = 0.05 and simulated power.
  This is recorded here and labelled as not pre-registered.
- **A17 (2026-11-15), after M2 ran.** A16 (ii) was wrong. On 200 synthetic
  panels the ICC estimate is about unbiased and it is the club bootstrap that
  sits below it, so the basic interval mirrors the shift upward (coverage
  161/200 and 170/200 at nominal 90%, misses mostly above). The team-ICC
  interval is instead the standard bootstrap interval, estimate ± 1.645 ×
  bootstrap SD (coverage 175/200 and 176/200, misses on both sides). On the
  real data the bootstrap's offset is larger (1.2–1.6 bootstrap SEs), so
  these intervals are reported as approximate. Point estimates are unchanged.
- **A18 (2026-11-16), after H3 was computed.** A code review found that the
  A9 parametric bootstrap compared each simulated estimate with the wrong
  planted value. It used the variance-weighted average of the groups'
  planted shares, but M4 on a pool fits one model across its groups, whose
  box-score means differ for reasons unrelated to impact; the pooled
  estimand is therefore larger for each pool, and the contrast smaller,
  than that average. The planted value of each pool in each simulation is
  now the population value of M4's own estimand under the generator:
  1 − R² of the planted component on the true expected statistics, over the
  pool's M4 players, from an OLS with intercept. The basic interval is
  recomputed from these errors. The post-hoc assortative robustness uses the
  same definition. Recorded here and labelled as a correction made after
  the result.
- **A19 (2026-11-17), after M8 ran.** A13 (v) summed raw Ballon d'Or points
  over the 2021–2026 editions. The ballot changed twice (fewer voters from
  2022; ten places instead of five from 2024), so an edition's total points
  range from about 1,500 to 6,700 and the raw sum weights the last three
  editions most: it puts a player with three recent editions first. The recognition view is corrected to the sum
  over editions of the player's share of that edition's points. Kendall's W
  and the recognition column are reported with this view; the raw-sum value
  is reported beside it. Recorded here and labelled as a correction made
  after the result. M6 is unaffected (it models each edition separately,
  with edition fixed effects).
- **A20 (2026-11-18), after M5 and M8 ran: a league-strength-adjusted
  analysis, post hoc.** Motivation, found in the results: nothing in the
  pre-registered analysis adjusts for league strength. In RAPM the players
  of Porto, Benfica and Sporting average the same total as Real Madrid's,
  because a league's level is not identified by its own games and the ridge
  centres every league at zero; in the box families a goal counts the same
  in every league. In-house Elo cannot supply the levels: its league means
  are fed by domestic-cup games against uncovered clubs entering at 1,500
  (`docs/unmeasured/league-strength.md`). The method below is fixed before
  any league-adjusted estimate exists. Verdicts H1–H8 stay decided by the
  pre-registered analysis; both sets of results are reported side by side.
  (i) *RAPM with league means* (the league term of Pantuso and Hvattum
  2021): off_p = δoff_p + Σ_L s_pL μ_L and def_p = δdef_p + Σ_L s_pL μ_L,
  with s_pL the share of the player's league minutes in the window played
  in league L, the ridge on δ only at the primary λ, μ unpenalised and
  μ = 0 for the Premier League. A league has its own μ if it has at least
  100 stint games against clubs of other covered leagues or at least 150
  players with 450 minutes in it and in another league; the remaining
  leagues share one μ. The same 200 game resamples as the primary
  bootstrap. Per-season fits keep the pooled μ. (ii) *Box-score league
  factors from players who change league*: per metric (goals, non-penalty
  goals, assists), a conditional fixed-effects Poisson model on
  player × season × league counts in league matches,
  log E[Y] = log(minutes/90) + α_p + γ_L + b₁·age + b₂·age², γ = 0 for the
  Premier League; a league has its own γ with at least 100 players observed
  in it and in another league, the others share one. League-match counts
  are divided by exp(γ̂_L); European-cup and national-team matches are left
  as they are. Player-bootstrap standard errors are reported; γ̂ is held
  fixed across M8's replicates. (iii) *On-pitch goal difference* in league
  matches is shifted by 22·μ̂_L per 90. (iv) *Context dimension*: team Elo is
  centred within competition-season. (v) M5 and M8 are then rerun exactly
  as the primary (same factorial, σ from the primary M4, ω, U, B, seeds).
  M4, M6 and M7 are not rerun. (vi) *Coverage fill, a separate variant*:
  the MLS seasons before 2025 of pool players, from the American Soccer
  Analysis API (CC BY-SA 4.0), only if it gives minutes, goals, penalty or
  non-penalty goals and primary assists per player-season; box metrics
  only (no stints, so pooled RAPM and the group-season mean of on-pitch
  goal difference, the existing rule for rows without stints). (vii) Both
  estimators must first recover planted league levels on synthetic
  multi-league data, including a league with almost no links.
