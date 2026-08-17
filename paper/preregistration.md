# Pre-registration — The De-margining Artifact

**Version 2, superseding the version committed at `78bc5c4`.** Section 7
records exactly what changed and why, and the reason the revision is legitimate
rather than a retrofit: **no evaluation result had been produced or read when
it was made.** Version 1 remains in git history. Compare them.

**Committed before the evaluation period was run.** The git timestamp on this
file is the claim: everything below was fixed in advance of seeing any
evaluation result. If this file's history shows it modified after the first
successful `scripts/study.py evaluate`, treat every prediction here as post-hoc.

**Author:** Liam Abourousse
**Repository:** https://github.com/Bardakor/betting-app-yami
**Design spec:** [`docs/superpowers/specs/2026-08-16-demargining-study-design.md`](../docs/superpowers/specs/2026-08-16-demargining-study-design.md)

---

## 1. The question

Empirical football-forecasting papers routinely claim to beat the bookmaker.
Substantiating that claim requires converting offered odds into a benchmark
probability, which means removing the bookmaker's margin. The literature uses
at least four transforms for this and papers typically apply one without
justification.

**Does a model's measured edge over the bookmaker depend materially on which
de-margining transform builds the benchmark?**

## 2. Predictions

Unchanged from version 1. Each is stated with what would falsify it.

| ID | Prediction | Falsified by |
|---|---|---|
| **P1** | Disagreement between transforms scales with the book's margin — negligible on Pinnacle (~102% book), material on Bet365 (~107%) | Transform spread on Bet365 cells not exceeding that on Pinnacle cells |
| **P2** | Disagreement is larger for 3-outcome 1X2 than for 2-outcome Over/Under 2.5 | OU25 spread matching or exceeding 1X2 spread on the same cells |
| **P3** | There exist (league, book) cells where the model's measured edge **changes sign** between proportional and Shin | No cell showing a sign reversal in ROI between those two transforms |

**P3 is the headline.** Sign reversal, not merely magnitude change, is what
would invalidate a published conclusion.

## 3. Reporting commitments

- All three predictions are reported with effect sizes and confidence
  intervals **regardless of outcome**.
- No prediction is dropped, reframed, or replaced after seeing results.
- A null on P3 with P1 and P2 holding is a complete paper and will be written
  as one.
- The coverage table is published as a result, including every excluded cell.
- Cells with fewer than 200 matched fixtures are excluded; the threshold is
  fixed here, not chosen later.

## 4. The frozen decay parameter

```
xi = 0.0017328679513998633
```

Half-life exactly 400 days, so `xi = ln(2) / 400`. Selected by minimising mean
ranked probability score on the **calibration period only**.

The full-precision value is given because it is the value the evaluation runs
against. `scripts/study.py evaluate` parses xi from this line and refuses to
start unless it matches `results/calibration.json` to 1e-9. A rounded
`0.001733` here would be a *different* decay rate from the one selected, and
registering one number while running another defeats the point of registering.

### 4.1 Selection, and how weakly determined it is

Ten candidate half-lives, each scored on the **intersection** of fixtures all
ten priced — 10,567 of 10,933, or 96.6%. Pairing matters because candidates do
not all price the same fixtures: a fit can fail to converge or leave a team
unidentified, and those fixtures are dropped. Scoring arms on different match
sets lets an arm that happens to drop harder fixtures win for reasons
unrelated to forecast quality.

| Half-life (days) | xi | Paired mean RPS |
|---:|---:|---:|
| 30 | 0.023105 | 0.222271 |
| 60 | 0.011552 | 0.211939 |
| 90 | 0.007702 | 0.208691 |
| 150 | 0.004621 | 0.206294 |
| 250 | 0.002773 | 0.205119 |
| **400** | **0.001733** | **0.204889** |
| 700 | 0.000990 | 0.205109 |
| 1200 | 0.000578 | 0.205406 |
| 2000 | 0.000347 | 0.205630 |
| 40000 (no decay) | 0.000017 | 0.206014 |

The optimum is **interior** — 250 and 700 days are both worse — so the grid
contains it rather than truncating it.

**Stated plainly: xi is weakly identified.** 400 and 700 days differ by
0.00022 in paired RPS, and everything from 250 to 1200 days sits within 0.0006
of the optimum. 400 days is not presented as a precisely determined quantity.
What the curve does establish is that both extremes are wrong: a 30-day
half-life scores 0.2223, and no decay at all scores 0.2060 against 0.2049 at
the optimum. Exponential decay earns its place without its exact rate
mattering much.

Because one model is held fixed across all four transforms, a weakly
identified xi shifts every arm together and cannot generate a difference
between them.

## 5. Protocol, fixed in advance

**Data.** football-data.co.uk, 525 files fetched, 514 readable, 176,629
matches. The 11 exclusions are recorded in the coverage table: seven files
whose internal `Div` disagrees with their filename, three carrying corrupted
rows, one declaring the same bookmaker column twice. All 1,393,994 closing-odds
records fall inside evaluation periods; none is stranded in burn-in or
calibration.

**Periods.** Per league, disjoint and in chronological order:

| Period | Span | Use |
|---|---|---|
| Burn-in | first 3 seasons | Fitting only; never forecast or scored |
| Calibration | next 2 seasons | Used **solely** to select xi |
| Evaluation | all remaining seasons | The study |

**Walk-forward.** Before each matchday, refit on every match that kicked off
strictly earlier, weighted by `exp(-xi * age_in_days)`; predict that matchday;
roll forward. Burn-in matches stay in the fit window — excluded from scoring,
not estimation. A matchday whose window holds fewer than 300 matches is
skipped.

**Model.** Dixon–Coles with per-league home advantage, fit by maximum
likelihood with an analytic gradient, identical across all four transforms.
Teams whose total decayed weight falls below one unit — the weight of a single
match played today — are excluded as unidentifiable, along with their matches;
see §7. Fixtures are dropped rather than priced when a team is absent from the
fit, when a fitted rate falls outside (0.05, 6.0) expected goals, or when rho
is inadmissible for that fixture's own (lambda, mu).

**Transforms.** proportional, power, Shin, odds-ratio — one bisection driver at
one tolerance, so none is advantaged by a sloppier solver.

**Markets.** 1X2 and Over/Under 2.5, closing odds only. Opening odds are never
used as a benchmark.

**Books.** Real bookmakers only. The archive's constructed aggregates — market
maximum, market average, and the Betbrain equivalents — are excluded, because
they are not books anyone offers and do not have a margin in the sense P1 is
about. Market maximum is the best price available anywhere, and 39.3% of its
closing books sum below 1: it is an arbitrage by construction that often.
Retaining only its positive-margin rows would bias precisely the quantity P1
measures. Real bookmakers carry at most 0.06% sub-unit books.

**Arbitrage fixtures.** A book summing to 1 or less has no margin to remove and
every transform is undefined on it. Those fixtures are dropped and the count is
reported per cell.

**Inference.** Diebold–Mariano with Newey–West HAC standard errors; stationary
block bootstrap (seeded) for ROI intervals; Benjamini–Hochberg at q = 0.10
across the (league × book × market × transform-pair) family.

## 6. What this pre-registration does not cover

- The model is fixed by choice. This study says nothing about whether a better
  model would beat the market; it asks only whether the *measurement* of any
  model's edge depends on the transform.
- Only two markets and closing odds. Nothing generalises to in-play, Asian
  handicap, or opening lines.
- Backtested returns are not achievable returns: no stake limits, no line
  movement between observation and placement, no account restrictions.
- **The calibration period is early.** A fixed 3-season burn-in and 2-season
  calibration on a 33-season archive puts xi selection in 1996–97 for the
  longest-running leagues, up to 28 years before the evaluation ends. Football
  has changed over that span, so the selected xi may not be optimal late in
  the evaluation period. The split was fixed in the design spec before any
  data was seen and is deliberately not revised here. Its effect on the
  comparison is limited for the reason in §4.1: all four arms share the xi.

## 7. What changed from version 1, and why that is legitimate

Version 1 was committed at `78bc5c4`. Two defects were found afterwards, both
before any evaluation result existed. The first evaluation attempt crashed
before writing `results/cells.csv`; no P1, P2 or P3 outcome has ever been
computed or read. Nothing below could have been shaped by knowing an answer.

**Defect 1 — seasons were partitioned lexicographically, not chronologically.**
Season codes wrap the century, so `"9394"` sorts after `"0001"`. The real
partition came out as burn-in 2000–02, calibration 2023–24, evaluation
1993–99. That selected xi on the *newest* seasons — precisely the leakage the
three-period split exists to prevent — and put the evaluation on years that
predate closing odds almost entirely. The tests could not catch it: every
season fixture was post-2000, which sorts identically either way.

**Defect 2 — no fit converged once the partition was fixed.** With evaluation
running from 1998, fit windows carry 30 years of history. E0 accumulates
around 51 distinct teams while roughly 20 play in any season, and under decay
a side that last played a decade ago carries about 0.002 weight per match. Its
attack and defence parameters are unidentifiable, the likelihood is flat in
those directions, and L-BFGS-B exhausted its iteration budget on every single
matchday. A smoke run priced 7 fixtures where 760 were expected. Teams below
one unit of total decayed weight are now excluded; a test asserts this leaves
the likelihood unchanged when every team is active.

**Effect on xi.** Recalibrating under the corrected pipeline selected the same
400-day half-life, now from a paired comparison covering 96.6% of fixtures
rather than 52%, with near-constant coverage across candidates (10,573 to
10,933 versus 7,804 to 11,152). The agreement is reassuring but incidental —
the procedure, not the answer, is what was broken.

**Defect 3 — the transforms are undefined on a book that carries no margin.**
Sub-unit closing books exist in the archive. For real bookmakers they are
vanishingly rare, but the constructed market-maximum aggregate is sub-unit
39.3% of the time, being the best price across every book. The handling is
recorded under "Books" and "Arbitrage fixtures" in §5. This too was settled
before any result existed.

**Why this is a revision and not a retrofit.** A pre-registration that changes
silently is worthless. One that documents its own correction, keeps the
superseded version in history, and can demonstrate no result was observed in
between is not. The claim is checkable: `git log -p paper/preregistration.md`
against the absence of any `results/cells.csv` in the repository's history.
