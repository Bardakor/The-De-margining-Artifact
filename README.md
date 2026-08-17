# The De-margining Artifact

[![verify](https://github.com/Bardakor/betting-app-yami/actions/workflows/verify.yml/badge.svg)](https://github.com/Bardakor/betting-app-yami/actions/workflows/verify.yml)

Reproducible research code testing whether a claimed football-forecasting edge over
bookmakers survives the choice of de-margining transform, or is substantially an artifact
of it.

## Abstract

Empirical football-forecasting papers routinely claim to beat the bookmaker. Substantiating
that claim requires converting offered odds into a fair benchmark probability, which means
removing the bookmaker's margin. The literature uses at least four transforms for this,
typically applying one without justification. They are not equivalent: they diverge most in
how they treat longshots, which is exactly where a model's claimed edge tends to concentrate.

This repository holds a single fixed forecasting model constant across all four transforms
and tests whether the measured edge changes — in size, in significance, or in sign — with
the choice of benchmark. **Plan 1** (implemented) provides the pure pricing and evaluation
apparatus; the empirical study (Plans 2–5) is not yet complete.

## Research question

Does a model's measured edge over the bookmaker depend materially on which de-margining
transform is used to construct the benchmark?

## Pre-registered predictions

Committed ahead of the full study running, so the git history timestamps the hypotheses
before the results. Source: [study design spec](docs/superpowers/specs/2026-08-16-demargining-study-design.md) §2.

| ID | Prediction | Rationale |
|---|---|---|
| **P1** | Disagreement between transforms scales with the book's margin — negligible at Pinnacle's ~102% book, material at Bet365's ~107% | The transforms coincide in the limit $B \to 1$; they can only diverge in proportion to the mass being removed |
| **P2** | Disagreement is larger for 3-outcome 1X2 than for 2-outcome Over/Under 2.5 | The transforms differ chiefly in longshot treatment; a 2-outcome market on a near-even line has no longshot |
| **P3** | There exist (league, book) cells where the model's measured edge **changes sign** between proportional and Shin | Sign reversal, not merely magnitude change, is what would invalidate a published conclusion |

All three will be reported with effect sizes and confidence intervals regardless of outcome.
No prediction is dropped, reframed, or replaced after seeing results.

---

## The instrument: four de-margining transforms

This is the object under study. Given offered decimal odds $d_i$, the raw implied
probabilities are $p_i = 1/d_i$ with book sum $B = \sum_i p_i > 1$. Each transform maps
$p \mapsto \pi$ with $\sum_i \pi_i = 1$.

**Proportional (normalisation)** — the field's default. Removes margin uniformly in
proportion to implied probability:

$$\pi_i = \frac{p_i}{B}$$

**Power** — solve for the exponent $k$ such that the transformed masses sum to one:

$$\sum_i p_i^{\,k} = 1, \qquad \pi_i = p_i^{\,k}$$

Since every $p_i \in (0,1)$ the sum is strictly decreasing in $k$, so bisection converges.
Takes proportionally more from longshots, consistent with favourite–longshot bias.

**Shin (1993)** — models the book as containing a proportion $z$ of insider money:

$$\pi_i = \frac{\sqrt{z^2 + 4(1-z)\dfrac{p_i^2}{B}} - z}{2(1-z)}$$

with $z$ solved by bisection so that $\sum_i \pi_i = 1$.

**Odds-ratio** — assumes a constant odds ratio $c$ between offered and true probabilities,
$\;p_i/(1-p_i) = c \cdot \pi_i/(1-\pi_i)$, which rearranges to

$$\pi_i = \frac{p_i}{c(1-p_i) + p_i}$$

with $c \ge 1$ solved by bisection. Surveyed against the others by Štrumbelj (2014).

All four share one bisection driver at identical tolerance, so no transform is advantaged by
a sloppier solver. Each is tested for the round-trip property and for the $B \to 1$ limit,
where all four must converge to $p_i$.

## The model

The model is the *instrument*, not the finding: it must be competent and completely fixed
across all four transforms. Full specification in [`docs/model.md`](docs/model.md); the
Python core was written clean-room from that document.

Goals arrive as a Poisson process (Maher, 1982), with each side's rate factored into attack
strength, opponent defence weakness, and a per-league home advantage:

$$\lambda = \alpha_{\text{home}} \cdot \beta_{\text{away}} \cdot \gamma, \qquad \mu = \alpha_{\text{away}} \cdot \beta_{\text{home}}$$

Independent Poisson is wrong in one documented way — it under-counts low-scoring draws.
Dixon and Coles (1997) correct exactly the four affected cells:

$$\tau(x,y) = \begin{cases} 1 - \lambda\mu\rho & (0,0) \\\\ 1 + \lambda\rho & (0,1) \\\\ 1 + \mu\rho & (1,0) \\\\ 1 - \rho & (1,1) \\\\ 1 & \text{otherwise} \end{cases}$$

The correction inflates low draws — its entire purpose — only when $\rho < 0$, since both
$\tau(0,0)$ and $\tau(1,1)$ exceed 1 in that case alone. The admissible region is

$$\max\left(-\tfrac{1}{\lambda}, -\tfrac{1}{\mu}\right) \le \rho \le \min\left(\tfrac{1}{\lambda\mu}, 1\right)$$

and is enforced, not merely documented.

### One matrix, every market

$$P(x,y) = \frac{\tau(x,y)\,\mathrm{Pois}(x;\lambda)\,\mathrm{Pois}(y;\mu)}{\sum_{i,j}\tau(i,j)\,\mathrm{Pois}(i;\lambda)\,\mathrm{Pois}(j;\mu)}, \qquad 0 \le x,y \le 10$$

Every market is a sum over regions of this single $11 \times 11$ distribution — 1X2 over
$x>y$, $x=y$, $x<y$; Over/Under $\ell$ over $x+y>\ell$; both-teams-to-score over
$x>0 \wedge y>0$; Asian handicap $h$ over $x+h>y$. Because they are marginals of one
distribution they cannot disagree, and the suite asserts this directly.

Parameters are fit by maximum likelihood with exponential time decay $\varphi(\Delta t) = e^{-\xi \Delta t}$
(Plan 2, not yet implemented).

### An independent check on the matrix

Deriving everything from one matrix is internally consistent *by construction*, which means
a systematic error in the matrix would be invisible to every consistency test above. So the
goal difference is computed a second, independent way — the difference of two Poisson
variables is Skellam-distributed (Karlis & Ntzoufras, 2009), with a closed form in the
modified Bessel function of the first kind and no matrix involved:

$$P(K = k) = e^{-(\lambda+\mu)}\left(\frac{\lambda}{\mu}\right)^{k/2} I_{|k|}\\!\left(2\sqrt{\lambda\mu}\right)$$

At $\rho = 0$ the two routes must agree; at $\rho \neq 0$ they must diverge on the draw,
because Skellam assumes independence and $\tau$ deliberately breaks it. Both are asserted.
Measured agreement at $\rho = 0$ is $2.6 \times 10^{-7}$, the truncation floor of the
$11 \times 11$ grid against Skellam's infinite support.

## Evaluation

Football outcomes are **ordered**, which the Brier score ignores: forecasting a home win
scores identically whether the match was drawn or lost. The ranked probability score
(Epstein, 1969; Constantinou & Fenton, 2012) is distance-sensitive:

$$RPS = \frac{1}{r-1}\sum_{i=1}^{r-1}\left(\sum_{j\le i}(p_j - o_j)\right)^2$$

This is contested — Wheatcroft (2021) argues distance sensitivity is not desirable here — so
both are implemented and the choice stays explicit.

**Murphy's decomposition** turns a score into a statement about *why*:

$$BS = \underbrace{\text{REL}}_{\text{calibration}} - \underbrace{\text{RES}}_{\text{discrimination}} + \underbrace{\text{UNC}}_{\text{irreducible}} + \underbrace{\text{WBV}}_{\text{binning artefact}}$$

The classical three-way identity is exact only when each bin holds a single distinct
forecast. Binning continuous forecasts leaves a residual equal to the within-bin variance,

$$\text{WBV} = \frac{1}{N}\sum_k \sum_{i \in k} (p_i - \bar{p}_k)^2$$

so all four terms are reported, the exact identity is tested to $10^{-12}$, and a separate
test asserts the residual **is** the WBV — so nobody can quietly simplify it back to three.

Staking uses fractional Kelly, $f^{*} = (bp-q)/b$ with $b = d-1$. Note $bp - q = dp - 1$, so
Kelly, edge and expected value can never disagree about whether a bet is worth taking.

---

## Status

| Plan | Scope | State |
|---|---|---|
| **1 — Conversion + core** | Scaffolding, clean-room port of the pricing core | **Complete** |
| **2 — Fitting** | Weighted log-likelihood, analytic gradient, `check_grad`, L-BFGS-B | Not started |
| **3 — Data** | Ingest, header discovery, coverage matrix, remaining three transforms | **Partial** — ingest and coverage complete and tested; the three remaining transforms outstanding |
| **4 — Study** | Walk-forward protocol, leakage test, scoring, inference | Not started |
| **5 — Paper** | Figures, tables, manuscript against the frozen pre-registration | Not started |

## Implemented components

| Component | Module | Tests |
|---|---|---|
| Poisson pmf (log-space, via `scipy.special.gammaln`) | `src/footy/core/poisson.py` | 4 |
| Dixon–Coles $\tau$ and admissible $\rho$ region | `src/footy/core/dixon_coles.py` | 7 |
| Normalised $11\times11$ scoreline matrix | `src/footy/core/matrix.py` | 5 |
| Market marginals (1X2, totals, BTTS, CS, AH, DC) | `src/footy/core/markets.py` | 18 |
| Skellam goal-difference distribution | `src/footy/core/skellam.py` | 6 |
| Power-method overround | `src/footy/market/overround.py` | 8 |
| Shin de-margining inverse | `src/footy/market/demargin.py` | 5 |
| Fractional Kelly staking | `src/footy/market/kelly.py` | 5 |
| RPS, Brier, log loss | `src/footy/eval/scoring.py` | 8 |
| Murphy REL / RES / UNC / WBV | `src/footy/eval/murphy.py` | 8 |
| Column registry (267 odds columns) | `src/footy/data/columns.py` | 12 |
| football-data.co.uk ingest, header discovery | `src/footy/data/football_data.py` | 59 |
| (league × season × book × market) coverage matrix | `src/footy/data/coverage.py` | 15 |

`src/footy/core/` contains no I/O, no clock, and no randomness — enforced by an AST scan in
`tests/test_purity.py` that also catches randomness reached *through* a permitted module
(`np.random.*`), which is what makes the invariants above assertable.

## Validation strategy

| Layer | Mechanism | Location |
|---|---|---|
| **Layer 1** | Reference values transcribed from [`docs/model.md`](docs/model.md) and asserted | `tests/fixtures/model_md_values.json` |
| **Layer 2** | Hypothesis property invariants over the admissible parameter region | `tests/test_properties.py` |
| **Purity** | AST scan banning I/O, clock and randomness in `core/` | `tests/test_purity.py` |

Layer 1 carries one **corrected** and one **derived** value, each with an explicit
provenance note in the fixture. The correction records an inconsistency the clean-room port
found in the source specification: §8.2's printed favourite–longshot pair implies two
different power exponents ($0.928042$ against $0.927518$) and is unreproducible under any
$\rho$. Details in [`docs/model.md`](docs/model.md) and the fixture's `_longshot_provenance`.

> A third validation layer — differential comparison against the original TypeScript
> implementation — was specified but **cannot now be run**: that implementation has been
> removed from the working tree. It remains recoverable from git history under the
> `v1-betting-app` tag should the comparison be wanted later.

## Reproducing

**Prerequisites:** Python ≥ 3.12 (see `.python-version`), [uv](https://github.com/astral-sh/uv).

```bash
make install   # uv sync --extra dev
make verify    # compile, ruff, mypy --strict, pytest
```

`make verify` runs, in order: byte-compilation, `ruff check` and `ruff format --check`,
`mypy --strict`, then `pytest` — **195 tests** at the latest green run, reproduced from a
clean clone.

`make data` and `make study` are specified in the design doc and arrive with Plans 3 and 4.

## Repository layout

| Path | Responsibility |
|---|---|
| `docs/model.md` | Canonical mathematical specification (the clean-room contract) |
| `docs/superpowers/specs/` | Study design and pre-registered predictions |
| `docs/superpowers/plans/` | Implementation plans — research provenance |
| `src/footy/core/` | Pure pricing mathematics: no I/O, no clock, no randomness |
| `src/footy/market/` | Overround, de-margining, Kelly staking |
| `src/footy/eval/` | Scoring rules and the Murphy decomposition |
| `src/footy/data/` | football-data.co.uk ingest and coverage matrix |
| `tests/` | Layer-1 fixtures, Layer-2 properties, purity scan |

## Limitations

- Only the **Shin** inverse is implemented; proportional, power and odds-ratio arrive with
  Plan 3, alongside the harness that compares all four.
- No MLE fitting, walk-forward backtest, or empirical study code exists yet. **No result of
  any kind has been measured.**
- Correlation is corrected in four cells only; no explicit overdispersion (cf. Boshnakov,
  Kharrat & McHale, 2017); no player-level data; pre-match only.
- A dedicated `paper/preregistration.md` with the frozen decay parameter $\xi$ is planned
  for Plan 4, before the evaluation stage runs.

## References

Brier (1950) · Kelly (1956) · Epstein (1969) · Murphy (1973) · Maher (1982) · Shin (1993) ·
Politis & Romano (1994) · Benjamini & Hochberg (1995) · Diebold & Mariano (1995) ·
Dixon & Coles (1997) · Karlis & Ntzoufras (2009) · Constantinou & Fenton (2012) ·
Štrumbelj (2014) · Boshnakov, Kharrat & McHale (2017) · Wheatcroft (2021)

Full citations in [`docs/model.md`](docs/model.md) and the
[study design spec](docs/superpowers/specs/2026-08-16-demargining-study-design.md).

## Licence

MIT. See [LICENSE](LICENSE).
