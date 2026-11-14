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
