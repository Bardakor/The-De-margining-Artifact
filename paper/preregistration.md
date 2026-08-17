# Pre-registration — The De-margining Artifact

**Committed before the evaluation period was run.** The git timestamp on this
file is the claim: everything below was fixed in advance of seeing any
evaluation result. If this file's history shows it was modified after the
first `scripts/study.py evaluate` run, treat every prediction here as
post-hoc.

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

These are the claims. Each is stated with what would falsify it.

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
- Cells with fewer than 200 matched fixtures are excluded, and that threshold
  is fixed here rather than chosen later.

## 4. The frozen decay parameter

```
xi = 0.0017328679513998633
```

Half-life exactly 400 days, so `xi = ln(2) / 400`. Selected by minimising mean
ranked probability score on the **calibration period only**, and frozen here
before the evaluation period was touched.

The full-precision value is given because it is the value the evaluation
actually runs against. `scripts/study.py evaluate` parses xi from this line and
refuses to start unless it matches `results/calibration.json` to 1e-9 — a
rounded `0.001733` in this document would be a *different* decay rate from the
one that was selected, and registering one number while running another would
defeat the point of registering it.

### 4.1 How it was selected, and how honest that number is

Ten candidate half-lives were scored. Candidates do not all price the same
fixtures — a fit can fail to converge, or leave a team's parameters
unidentified, and those fixtures are dropped. The first sweep compared means
over match sets that differed by 43% (7,804 to 11,152 forecasts), which is not
a valid comparison: an arm that happens to drop harder fixtures scores better
for reasons unrelated to forecast quality. That sweep selected 700 days, at
the edge of its grid.

The selection here is **paired**: every candidate is scored on the 5,683
fixtures that all ten candidates forecast. Under pairing the optimum moved to
400 days and became interior, with 250 and 700 days both worse on either side.

| Half-life (days) | xi | Paired mean RPS |
|---:|---:|---:|
| 30 | 0.023105 | 0.219786 |
| 60 | 0.011552 | 0.210578 |
| 90 | 0.007702 | 0.207535 |
| 150 | 0.004621 | 0.205260 |
| 250 | 0.002773 | 0.203972 |
| **400** | **0.00173287** | **0.203460** |
| 700 | 0.000990 | 0.203533 |
| 1200 | 0.000578 | 0.203996 |
| 2000 | 0.000347 | 0.204491 |
| 40000 (no decay) | 0.000017 | 0.205561 |

**Stated plainly: xi is weakly identified.** 400 and 700 days differ by
0.000073 in paired RPS, and anything between roughly 250 and 1200 days scores
within 0.0006 of the optimum. The paper will not present 400 days as a
precisely determined quantity. What the curve does establish is that the
extremes are wrong: a 30-day half-life is far worse (0.2198), and no decay at
all is worse than moderate decay (0.2056 against 0.2035), so exponential decay
earns its place without its exact rate mattering much.

Because the study holds one model fixed across all four transforms, a weakly
identified xi does not threaten the comparison. It shifts every arm together.

## 5. Protocol, fixed in advance

**Data.** football-data.co.uk, 525 files fetched, 514 readable, 176,629
matches. The 11 exclusions are recorded in the coverage table: seven files
whose internal `Div` disagrees with their filename, three carrying corrupted
rows, one declaring the same bookmaker column twice.

**Periods.** Per league, in time order and disjoint:

| Period | Span | Use |
|---|---|---|
| Burn-in | first 3 seasons | Fitting only; never forecast or scored |
| Calibration | next 2 seasons | Used **solely** to select xi above |
| Evaluation | all remaining seasons | The study |

**Walk-forward.** Before each matchday, refit on every match that kicked off
strictly earlier, weighted by `exp(-xi * age_in_days)`; predict that matchday;
roll forward. Burn-in matches remain in the fit window throughout — excluded
from scoring, not from estimation. A matchday whose fit window holds fewer
than 300 matches is skipped.

**Model.** Dixon–Coles with per-league home advantage, fit by maximum
likelihood with an analytic gradient. Identical across all four transforms.
Fixtures are dropped, not priced, when a team is absent from the fit window,
when a fitted rate falls outside (0.05, 6.0) expected goals, or when rho is
inadmissible for that fixture's own (lambda, mu).

**Transforms.** proportional, power, Shin, odds-ratio — sharing one bisection
driver at one tolerance, so none is advantaged by a sloppier solver.

**Markets.** 1X2 and Over/Under 2.5, closing odds only. Opening odds are never
used as a benchmark.

**Inference.** Diebold–Mariano with Newey–West HAC standard errors for
forecast comparison; stationary block bootstrap (10,000 resamples, seeded) for
ROI intervals; Benjamini–Hochberg at q = 0.10 across the
(league × book × market × transform-pair) family.

## 6. What this pre-registration does not cover

- The model is fixed by choice. This study says nothing about whether a better
  model would beat the market; it asks only whether the *measurement* of any
  model's edge depends on the transform.
- Only two markets and closing odds. Nothing here generalises to in-play,
  Asian handicap, or opening lines.
- Backtested returns are not achievable returns: no stake limits, no line
  movement between observation and placement, no account restrictions.
