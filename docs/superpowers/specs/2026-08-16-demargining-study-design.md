# The De-margining Artifact — Research Repository Design

**Date:** 2026-08-16
**Status:** Approved for planning
**Supersedes:** `2026-08-15-quant-rebuild-design.md` (stages 2–5 — API, web, RAG — are cancelled)
**Scope:** Convert the repository from a betting web application into a Python-only
quantitative research repository whose deliverable is a paper.

---

## 1. Motivation

The repository currently contains a Next.js frontend, six Node microservices, a duplicate
application, and a TypeScript pricing engine, totalling roughly 22,000 lines. Only the
pricing engine has research value. Everything else is application scaffolding for a product
that is no longer being built.

The engine itself is a faithful implementation of published models — Dixon–Coles (1997),
Skellam (Karlis & Ntzoufras, 2009), Shin (1993), Murphy (1973). Faithful implementation of
known models is apparatus, not a result. This document specifies the study that turns the
apparatus into a finding, and the repository reorganisation that supports it.

### 1.1 The research question

Empirical football forecasting papers routinely claim to beat the bookmaker. To make that
claim, the author must convert offered odds into a "market probability" benchmark, which
requires removing the bookmaker's margin. The literature uses at least four different
transforms for this, and papers typically apply one without justification, often in a single
unexamined line.

These transforms are not equivalent. They differ most in how they treat longshots, which is
precisely where a model's claimed edge tends to concentrate.

**Claim under test:** measured bookmaker-beating edge is substantially an artifact of the
de-margining transform used to construct the benchmark.

### 1.2 Why this question and not another

It is contested — Shin versus proportional normalisation is unsettled in the literature.
It is cheap to compute, requiring only odds columns already present in the data. And it is
falsifiable in a way that produces a publishable paper whichever way the result goes: if the
transforms agree, that is a useful negative result licensing the field's casual practice.

---

## 2. Pre-registered predictions

These are committed to `paper/preregistration.md` and pushed **before** the full study runs,
so the git history timestamps the hypotheses ahead of the results.

| ID | Prediction | Rationale |
|---|---|---|
| **P1** | Disagreement between transforms scales with the book's margin — negligible at Pinnacle's ~102% book, material at Bet365's ~107% | The transforms coincide in the limit `B → 1`; they can only diverge in proportion to the mass being removed |
| **P2** | Disagreement is larger for 3-outcome 1X2 than for 2-outcome Over/Under 2.5 | The transforms differ chiefly in longshot treatment; a 2-outcome market on a near-even line has no longshot |
| **P3** | There exist (league, book) cells where the model's measured edge **changes sign** between proportional and Shin | The headline claim. Sign reversal, not merely magnitude change, is what would invalidate a published conclusion |

**Reporting commitment.** All three are reported with effect sizes and confidence intervals
regardless of outcome. A null result on P3 with P1 and P2 holding is a complete paper and
will be written as one. No prediction is dropped, reframed, or replaced after seeing results.

---

## 3. Data

**Source:** football-data.co.uk season CSV archives. Free, redistributable, widely used in
this literature, and — critically — they carry both results and multiple bookmakers'
historical odds in the same file.

**Leagues:** the divisions carrying odds columns — E0, E1, D1, D2, SP1, SP2, I1, I2, F1, F2,
N1, B1, P1, T1, SC0, G1.

**Seasons:** every season the archive provides. Coverage is **not** assumed uniform.

### 3.1 Column discovery, not column assumption

Bookmaker coverage and column naming change across eras. Opening odds (`B365H`) and closing
odds (`B365CH`) are different columns, and closing-odds columns were introduced part-way
through the archive's history.

The ingest layer therefore **discovers** available columns per season file rather than
assuming a fixed schema:

1. Parse the header and match it against a registry of known column patterns.
2. Emit a coverage matrix over (league × season × book × market) recording what is actually
   present.
3. Restrict the study to cells with complete closing-odds coverage.
4. **Fail loudly on an unrecognised header.** Never silently mis-map a column.

The coverage matrix is published as a table in the paper. It is a result, not a preliminary.

**Closing odds only.** Opening odds reflect the book's prior, not the market's aggregated
information, and are not a fair benchmark. Where only opening odds exist, the cell is
excluded and recorded as excluded.

### 3.2 Markets

- **1X2** — three outcomes, the primary market
- **Over/Under 2.5 goals** — two outcomes, the contrast case for P2

---

## 4. The instrument: four de-margining transforms

Given offered decimal odds `dᵢ`, raw implied probabilities are `pᵢ = 1/dᵢ` with book sum
`B = Σpᵢ > 1`. Each transform maps `p → π` with `Σπᵢ = 1`.

**Proportional (normalisation).**

```
πᵢ = pᵢ / B
```

Removes margin uniformly in proportion to implied probability. The field's default.

**Power.** Solve for `k` such that `Σ pᵢᵏ = 1`, then `πᵢ = pᵢᵏ`. Since every `pᵢ ∈ (0,1)`,
the sum is strictly decreasing in `k`, so bisection converges. Takes proportionally more
from longshots, consistent with favourite–longshot bias.

**Shin (1993).** Models the book as containing a proportion `z` of insider money:

```
πᵢ = [ √(z² + 4(1−z)·pᵢ²/B) − z ] / (2(1−z))
```

with `z` solved by bisection so `Σπᵢ = 1`.

**Odds-ratio.** Assumes a constant odds ratio `c` between offered and true probabilities,
`pᵢ/(1−pᵢ) = c · πᵢ/(1−πᵢ)`, which rearranges to

```
πᵢ = pᵢ / ( c(1−pᵢ) + pᵢ )
```

with `c ≥ 1` solved by bisection so `Σπᵢ = 1`.

The odds-ratio transform is surveyed and compared against the others by Štrumbelj (2014),
which is the citable peer-reviewed source for it. It is commonly attributed onward to Cheung
(2015), a grey-literature source. **That attribution is unverified** and is to be checked
against primary sources during the literature review; the paper cites Štrumbelj unless the
Cheung provenance is confirmed.

All four share a bisection driver with identical tolerance (`1e-12`) and iteration cap, so
no transform is advantaged by a sloppier solver. Each is tested for the round-trip property
and for the `B → 1` limit, where all four must converge to `pᵢ`.

---

## 5. The model

The model is the *instrument*, not the finding. It must be competent and completely fixed
across all four transforms — every transform is evaluated against forecasts from identical
model fits.

### 5.1 Specification

Dixon–Coles with exponential time decay, per `MODEL.md` §2–§5:

```
λ = α_home · β_away · γ_league
μ = α_away · β_home
P(x,y) ∝ τ(x,y,λ,μ,ρ) · Pois(x;λ) · Pois(y;μ)     0 ≤ x,y ≤ 10
```

renormalised over the 11×11 grid.

### 5.2 Estimation and the analytic gradient

Parameters are fit by maximum likelihood with weights `w_m = exp(−ξ·Δt_m)`:

```
ℓ = Σ_m w_m [ log τ(x_m,y_m,λ_m,μ_m,ρ) + x_m log λ_m − λ_m + y_m log μ_m − μ_m − log x_m! − log y_m! ]
```

**Log parameterisation.** Optimise over `a_i = log α_i`, `b_i = log β_i`, `g = log γ`. This
enforces positivity without bounds and makes the derivatives trivial, since `∂λ/∂a_h = λ`.

**Identifiability.** The likelihood has exactly one flat direction: `α → cα, β → β/c` leaves
both `λ` and `μ` unchanged. One constraint removes it. We reparameterise with
`a_N = −Σ_{i<N} a_i`, which is exact and keeps the gradient chain-differentiable — the free
gradient is `∂ℓ/∂a_i − ∂ℓ/∂a_N`.

This normalises the *geometric* mean of `α` to 1, where `MODEL.md` §2 states the arithmetic
mean. Both identify the same model by fixing the same flat direction; they differ only in
where along it the solution is reported. The difference is documented, not silently adopted.

**Gradient.** Derived by hand rather than finite-differenced. With
`Tλ = (∂τ/∂λ)/τ` and `Tμ = (∂τ/∂μ)/τ`:

```
∂ℓ/∂a_i = Σ_{m: home=i} w_m [ Tλ_m·λ_m + x_m − λ_m ] + Σ_{m: away=i} w_m [ Tμ_m·μ_m + y_m − μ_m ]
∂ℓ/∂b_i = Σ_{m: away=i} w_m [ Tλ_m·λ_m + x_m − λ_m ] + Σ_{m: home=i} w_m [ Tμ_m·μ_m + y_m − μ_m ]
∂ℓ/∂g   = Σ_{m}         w_m [ Tλ_m·λ_m + x_m − λ_m ]
∂ℓ/∂ρ   = Σ_{m}         w_m · (∂τ/∂ρ)/τ
```

with `τ` derivatives non-zero only in the four corrected cells:

| cell | τ | ∂τ/∂λ | ∂τ/∂μ | ∂τ/∂ρ |
|---|---|---|---|---|
| (0,0) | 1 − λμρ | −μρ | −λρ | −λμ |
| (0,1) | 1 + λρ | ρ | 0 | λ |
| (1,0) | 1 + μρ | 0 | ρ | μ |
| (1,1) | 1 − ρ | 0 | 0 | −1 |
| other | 1 | 0 | 0 | 0 |

The per-team sums are scatter-adds (`np.bincount` over the home and away index arrays), so
the whole gradient is computed without a Python-level loop over matches.

**Optimiser.** `scipy.optimize.minimize` with L-BFGS-B, analytic gradient supplied,
`ftol=1e-10`. `ρ` is boxed to `[−0.2, 0.2]`; the per-match admissibility bound
`max(−1/λ, −1/μ) ≤ ρ ≤ min(1/(λμ), 1)` is a function of the parameters and so is verified at
the optimum rather than imposed during it. A violation triggers a refit with a tightened box.

### 5.3 The three periods

Each league's eligible matches are partitioned in time into three disjoint periods. The
partition is defined once, before any fitting, and is fixed.

| Period | Span | Use |
|---|---|---|
| **Burn-in** | first 3 seasons | Fitting only. Never forecast, never scored |
| **Calibration** | next 2 seasons | Walk-forward forecast and scored **solely** to select `ξ`. Never enters the study results |
| **Evaluation** | all remaining seasons | The study. Every reported number comes from here |

`ξ` is selected by minimising mean RPS over the calibration period, then **frozen** and
recorded in `paper/preregistration.md` before the evaluation run. It is never re-tuned.

This matters more than it appears. Tuning `ξ` against the evaluation set would let the model
absorb information about the very matches used to judge it, and would make the comparison
between transforms partly a comparison between overfits. Because a single `ξ` is shared by
all four transforms, a leak would not obviously favour one — but it would inflate every
measured edge, which is exactly the quantity P3 is about.

### 5.4 Walk-forward protocol

Applied identically over the calibration and evaluation periods, for each league
independently:

1. **Refit** — before each matchday, refit on all matches with kickoff strictly earlier than
   that matchday's first kickoff, weighted by `exp(−ξΔt)`. The burn-in matches remain in the
   fit window throughout; they are excluded from scoring, not from estimation.
2. **Predict** — produce 1X2 and Over/Under 2.5 probabilities for that matchday.
3. **Roll forward.**

No future information reaches any forecast. The strict-inequality kickoff filter is the
single point where leakage would enter, and it is asserted by a dedicated test.

Refits across leagues are independent and run in parallel via `multiprocessing`.

---

## 6. Evaluation and inference

### 6.1 Forecast metrics

Per match, model forecasts are scored against the realised outcome, and separately each
de-margined market benchmark is scored the same way:

- **RPS** — distance-sensitive, appropriate to ordered outcomes (Constantinou & Fenton, 2012)
- **Brier** — reported alongside, since RPS is contested (Wheatcroft, 2021)
- **Log loss**
- **Murphy decomposition** — `BS = REL − RES + UNC + WBV`, all four terms, per `MODEL.md` §11.1

### 6.2 Economic metrics

- **ROI**, flat stake, betting whenever model probability exceeds the de-margined market
  probability, at the offered price
- **ROI**, quarter-Kelly staked
- **Closing line value**

Economic metrics are where P3 is measured, since sign reversal of edge is an economic
statement.

### 6.3 Statistical inference

**Diebold–Mariano** on per-match RPS differences, with Newey–West HAC standard errors —
forecasts from overlapping fit windows are serially correlated, and treating them as
independent would materially overstate significance.

**Stationary block bootstrap** (Politis & Romano) for ROI confidence intervals, 10,000
resamples, block length selected from the return autocorrelation. Betting returns are
heavy-tailed and clustered by matchday; a naive bootstrap is invalid here.

**Benjamini–Hochberg** at `q = 0.10` across the (league × book × market × transform-pair)
family. With four transforms over many cells the multiple-comparisons burden is real and is
corrected rather than ignored.

**Effect sizes with intervals are the headline.** P-values are reported but are not the claim.

---

## 7. Architecture

### 7.1 Layout

```
pyproject.toml              uv-managed, locked
Makefile                    terminal entry points: data, fit, study, paper, verify
src/footy/
  core/                     pure: no I/O, no clock, no randomness
    poisson.py              log-space pmf via scipy.special.gammaln
    dixon_coles.py          τ, its derivatives, admissible ρ bounds
    matrix.py               scoreline matrix, vectorised over fixtures
    skellam.py              closed form via scipy.special.ive
    markets.py              marginals: 1X2, O/U, BTTS, correct score, AH, DC
  fit/
    likelihood.py           vectorised weighted log-likelihood + analytic gradient
    mle.py                  L-BFGS-B driver, identifiability, convergence handling
    decay.py                exponential time weights
  market/
    demargin.py             proportional | power | shin | odds_ratio
    overround.py            forward power method (validation utility)
    kelly.py                staking (validation utility)
  eval/
    scoring.py              RPS, Brier, log loss
    murphy.py               REL / RES / UNC / WBV
    inference.py            Diebold-Mariano, block bootstrap, Benjamini-Hochberg
  data/
    football_data.py        ingest with header discovery and validation
    coverage.py             the (league × season × book × market) coverage matrix
  study/
    walkforward.py          the protocol
    run.py                  CLI
tests/
  fixtures/model_md_values.json    numbers transcribed from MODEL.md
paper/
  preregistration.md        committed before the study runs
  main.tex
  figures/                  generated by scripts, never hand-edited
data/                       gitignored; `make data` fetches and checksums
results/                    gitignored; regenerable
```

### 7.2 Purity boundary

`src/footy/core/` contains no I/O, no `datetime`, no `random`. A test scans it and fails the
build on violation, carrying forward the guarantee the TypeScript engine enforced. This is
what makes the invariants in §8 assertable.

### 7.3 Dependencies

NumPy, SciPy, pandas, matplotlib. Managed by `uv` with a committed lockfile. Linted by
`ruff`, type-checked by `mypy --strict`, tested by `pytest` with `hypothesis`.

pandas is used for ingest only. At ~80,000 matches the dataframe is not the bottleneck — the
MLE is — and every hot path operates on NumPy arrays.

**No notebooks.** Notebooks hide execution order and do not reproduce. Every figure and
table in the paper is produced by a script invoked from the Makefile.

---

## 8. Correctness strategy

The port is **clean-room**: the Python core is written from `MODEL.md`, not from the
TypeScript source. An independent implementation that subsequently agrees with the original
is genuine corroboration; a transliteration that agrees proves only that it was copied
accurately. Three layers, in order:

**Layer 1 — spec targets.** `MODEL.md` prints measured values in its prose. These are
transcribed into `tests/fixtures/model_md_values.json` and asserted:

| Source | Value |
|---|---|
| §3.1 ρ table | P(draw) = 0.27257 and fair draw odds 3.669 at ρ = −0.10, λ = 1.6, μ = 1.1 |
| §8.2 | fair-to-offered ratio 1.105 longshot vs 1.055 favourite at B = 1.08 |
| §8.3 | DC:1X price 1.0053 at λ = 3.8, μ = 0.3 |
| §9 | Shin recovers [0.46344, 0.31699, 0.21956] at z = 0.0502 from book sum 1.10 |
| §11.1 | BS = 0.10900, three-way = 0.10875, WBV = 0.00025 |

These come from the document, so using them does not compromise clean-room independence.

**Layer 2 — property tests.** Hypothesis over random `(λ, μ, ρ)` in the admissible region:
matrix sums to 1; all probabilities in [0,1]; `P(Over 2.5) + P(Under 2.5) = 1`;
`P(home) + P(draw) = P(1X)`; correct-score lower triangle equals `P(home win)`; level-ball
Asian handicap equals draw-no-bet; `Σ1/dᵢ` equals the target book sum exactly; all four
de-margin transforms round-trip and converge to `p` as `B → 1`.

**Layer 3 — post-hoc corroboration. VOID, not performed.** The plan was: after layers 1 and 2
pass, dump TypeScript engine outputs over a grid of `(λ, μ, ρ, lines, handicaps)` and diff
against Python at `1e-12`, investigating any divergence on its merits before deleting the
TypeScript.

**This did not happen.** The TypeScript engine was removed from the working tree before the
comparison was run, so the corroboration is no longer available. It remains recoverable from
git history under the `v1-betting-app` tag if the comparison is wanted later.

Consequence, stated plainly: the port's correctness now rests on Layer 1 and Layer 2 alone.
That is weaker than designed. Layer 1 pins exact values against the specification, and Layer 2
pins invariants across the admissible parameter region, but neither is an independent
reimplementation of the same formulas — which was the point of Layer 3.

Two things partially compensate, and both were achieved. The Skellam module reaches the goal
difference by a closed form in the modified Bessel function with no matrix involved, and
agrees with the matrix anti-diagonals to `2.6e-7` at ρ = 0 and with the Asian handicap to
`7.6e-7` across whole, half and quarter lines — both at the grid-truncation floor. That
corroborates the matrix from outside itself. Separately, the clean-room port found a genuine
inconsistency in §8.2 of the specification (see §8 Layer 1), which is evidence the port was
reasoning independently rather than transcribing.

**Gradient verification.** The analytic gradient is checked against
`scipy.optimize.check_grad` on randomised parameter vectors. A hand-derived gradient that is
subtly wrong produces a converged fit to the wrong optimum, silently, so this test is not
optional.

---

## 9. Reproducibility

Every artefact under `results/` carries a manifest recording:

- git commit SHA of the code that produced it
- resolved dependency versions
- RNG seeds (bootstrap only; nothing else is stochastic)
- SHA-256 of every input CSV

`make verify` runs lint, type check, and the full test suite. `make study` reproduces every
number in the paper from raw CSVs. A result that cannot be regenerated by `make` does not go
in the paper.

---

## 10. Repository conversion

### 10.1 Wave 1 — immediate

Tag the current state `v1-betting-app`, then delete:

```
frontend/                    Next.js application
backend/                     six Node microservices
mini-betting-platform/       duplicate application
mongodb-data/                108 tracked WiredTiger binaries
tests/                       HTML API-poking pages and integration scripts
bash/                        shell demo harness
mds/                         API documentation for the deleted API
.code/                       VS Code agent prompt
docker-compose.yml  mongo-init.js  postman-collection-demo.json
demo.sh  quick-test.sh  test_helpers.sh  demo-postman-requests.md
check-bets.js  check-user.js  database-inspector.js  test-mongodb-setup.js
```

History is preserved and no force-push occurs. The `.git` pack is 9.3 MB, so the MongoDB
blobs in history cost essentially nothing to clone; purging them would rewrite 56 commits on
a public remote to no benefit.

The GitHub repository should be renamed to reflect its purpose. GitHub redirects the old URL.

### 10.2 Wave 2 — after Layer 3 corroboration passes

```
packages/quant-engine/       the TypeScript engine
package.json  package-lock.json  node_modules/
.github/workflows/           Node CI, replaced by Python CI
```

`packages/quant-engine/docs/MODEL.md` is **retained** — it is the specification the Python
was written from and a source for the paper's methods section. Before wave 2 deletes its
parent directory, it is moved to `docs/model.md`.

### 10.3 Build order

The work decomposes into five independently verifiable plans, each with its own
implementation plan document:

1. **Conversion + core** — wave 1 deletion, Python scaffolding, `core/` clean-room port
   against Layer 1 and Layer 2 tests
2. **Fitting** — likelihood, analytic gradient, `check_grad` verification, MLE driver
3. **Data** — ingest, header discovery, coverage matrix, de-margining transforms
4. **Study** — walk-forward protocol, leakage test, metrics, inference, wave 2 deletion.
   Runs in two stages: the calibration stage selects `ξ`, then **`paper/preregistration.md`
   is written and committed with that frozen `ξ`**, and only then does the evaluation stage
   run. The commit is the timestamp, so it must land before the evaluation stage, not after
   it.
5. **Paper** — figures, tables, LaTeX, methods and results written against the frozen
   pre-registration

Plan 3's de-margining transforms have no dependency on plans 1–2 and may run in parallel.

---

## 11. Skills

Two capabilities are missing from the environment and are authored as part of this work,
using `superpowers:writing-skills`:

- **Python numerics** — vectorisation patterns, analytic-gradient derivation and
  verification, profiling discipline before optimising, the no-notebooks rule
- **Research writing** — paper structure, claim discipline, figure standards, citation
  handling, pre-registration practice

Both are written early so subsequent sessions inherit them.

---

## 12. Success criteria

1. `make verify` passes: ruff clean, `mypy --strict` clean, all tests green.
2. Every value in `tests/fixtures/model_md_values.json` reproduced.
3. Analytic gradient agrees with numerical differentiation to `1e-6`.
4. ~~Layer 3 corroboration against TypeScript green at `1e-12`.~~ **Void** — the TypeScript
   was removed before the comparison ran. See §8. Recoverable from the `v1-betting-app` tag
   if reinstated.
5. A leakage test proves no forecast uses a match at or after its own kickoff.
6. Full walk-forward across all eligible leagues completes in under one hour on this machine.
7. `paper/preregistration.md` is committed strictly before the first full study run, with the
   frozen `ξ`.
8. P1, P2, P3 each reported with effect size and confidence interval.
9. `make study` regenerates every number in the paper from raw CSVs.

---

## 13. Out of scope

- Any user interface, HTTP API, or database
- Live or in-play modelling
- Real-money betting
- Sports other than football
- New model classes — bivariate Weibull counts, overdispersion, player-level data. The model
  is deliberately fixed so the finding is about the transform, not the model.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| Closing-odds coverage too sparse in early seasons | Coverage matrix computed first and published; study restricted to complete cells. Reduces sample, does not invalidate the design |
| MLE fails to converge for a small league or early window | Documented fallback: widen the fit window; if still failing, exclude the matchday and record the exclusion. Exclusion rules are fixed in advance, not chosen after seeing results |
| Analytic gradient subtly wrong | `check_grad` test, mandatory (§8) |
| ξ selection leaks evaluation information | ξ frozen on a calibration period and recorded in the pre-registration before the study runs (§5.3) |
| P3 does not hold | Pre-registration commits to reporting it either way. A null on P3 with P1 and P2 holding is a complete paper |
| Clean-room port diverges from TypeScript | Layer 3 investigates rather than assumes; divergences are a finding, not a failure |
| Scope creep back toward an application | §13 is binding. No UI, no API, no database |

---

## References

- Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Kelly, J.L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
- Epstein, E.S. (1969). A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology*, 8, 985–987.
- Murphy, A.H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12, 595–600.
- Maher, M.J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.
- Diebold, F.X. and Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Politis, D.N. and Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Shin, H.S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
- Dixon, M.J. and Coles, S.G. (1997). Modelling association football scores and inefficiencies in the football betting market. *JRSS: Series C*, 46(2), 265–280.
- Karlis, D. and Ntzoufras, I. (2009). Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference. *IMA Journal of Management Mathematics*, 20(2), 133–145.
- Constantinou, A.C. and Fenton, N.E. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*, 8(1).
- Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *International Journal of Forecasting*, 30(4), 934–943.
- Cheung, J. (2015). Fixed-odds betting and traditional odds. Grey literature — **attribution unverified**, see §4. To be confirmed or dropped during the literature review.
- Boshnakov, G., Kharrat, T. and McHale, I.G. (2017). A bivariate Weibull count model for forecasting association football scores. *International Journal of Forecasting*, 33(2), 458–466.
- Wheatcroft, E. (2021). Evaluating probabilistic forecasts of football matches: the case against the ranked probability score. *Journal of Quantitative Analysis in Sports*, 17(4), 273–287.
