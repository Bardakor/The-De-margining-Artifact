# The De-margining Artifact

[![verify](https://github.com/Bardakor/betting-app-yami/actions/workflows/verify.yml/badge.svg)](https://github.com/Bardakor/betting-app-yami/actions/workflows/verify.yml)

Reproducible research code testing whether a claimed football-forecasting edge over
bookmakers survives the choice of de-margining transform, or is substantially an artifact
of it.

## Abstract

Empirical football-forecasting papers routinely claim to beat the bookmaker. Substantiating
that claim requires converting offered odds into a fair benchmark probability, which means
removing the bookmaker's margin. The literature uses at least four transforms for this
(proportional normalisation, the power method, Shin's insider-trading model, and the
odds-ratio method), typically applying one without justification. These transforms are not
equivalent: they diverge most in how they treat longshots, which is exactly where a model's
claimed edge tends to concentrate.

This repository holds a single fixed forecasting model constant across all four transforms
and tests whether the measured edge changes — in size, in significance, or in sign — with
the choice of benchmark. **Plan 1** (implemented) provides the pure pricing and evaluation
apparatus; the empirical study (Plans 2–5) is not yet implemented.

## Research question

Does a model's measured edge over the bookmaker depend materially on which de-margining
transform is used to construct the benchmark?

## Pre-registered predictions

These are committed ahead of the full study running, so the git history timestamps the
hypotheses before the results. See
[`docs/superpowers/specs/2026-08-16-demargining-study-design.md`](docs/superpowers/specs/2026-08-16-demargining-study-design.md)
§2 for the source.

| ID | Prediction | Rationale |
|---|---|---|
| **P1** | Disagreement between transforms scales with the book's margin — negligible at Pinnacle's ~102% book, material at Bet365's ~107% | The transforms coincide in the limit `B → 1`; they can only diverge in proportion to the mass being removed |
| **P2** | Disagreement is larger for 3-outcome 1X2 than for 2-outcome Over/Under 2.5 | The transforms differ chiefly in longshot treatment; a 2-outcome market on a near-even line has no longshot |
| **P3** | There exist (league, book) cells where the model's measured edge **changes sign** between proportional and Shin | Sign reversal, not merely magnitude change, is what would invalidate a published conclusion |

All three will be reported with effect sizes and confidence intervals regardless of outcome.
No prediction is dropped, reframed, or replaced after seeing results.

## Status

The work decomposes into five independently verifiable plans.

- [x] **Plan 1 — Conversion + core.** Python scaffolding and a clean-room port of the
  pricing core against Layer-1 (spec-value) and Layer-2 (property) tests.
- [ ] **Plan 2 — Fitting.** Weighted log-likelihood, analytic gradient, `check_grad`
  verification, the L-BFGS-B driver.
- [ ] **Plan 3 — Data.** Ingest with header discovery, the coverage matrix, and the
  remaining three de-margining transforms.
- [ ] **Plan 4 — Study.** The walk-forward protocol, the leakage test, scoring and
  inference.
- [ ] **Plan 5 — Paper.** Figures, tables, and the manuscript, written against the frozen
  pre-registration.

## Methods (implemented)

The forecasting model is Dixon-Coles (1997) with exponential time decay, specified in full in
[`docs/model.md`](docs/model.md). The Python core under `src/footy/core/` was written
clean-room from that document.

| Component | Module | Status |
|---|---|---|
| Poisson pmf (log-space) | `src/footy/core/poisson.py` | Implemented |
| Dixon-Coles τ matrix and derivatives | `src/footy/core/dixon_coles.py` | Implemented |
| Normalised 11×11 scoreline matrix | `src/footy/core/matrix.py` | Implemented |
| Market marginals (1X2, totals, BTTS, AH, DC) | `src/footy/core/markets.py` | Implemented |
| Skellam goal-difference distribution | `src/footy/core/skellam.py` | Implemented |
| Power-method overround | `src/footy/market/overround.py` | Implemented |
| Shin de-margining inverse | `src/footy/market/demargin.py` | Implemented |
| Fractional Kelly staking | `src/footy/market/kelly.py` | Implemented |
| RPS, Brier, log loss | `src/footy/eval/scoring.py` | Implemented |
| Murphy REL / RES / UNC / WBV | `src/footy/eval/murphy.py` | Implemented |

`src/footy/core/` contains no I/O, no clock, and no randomness — enforced by
`tests/test_purity.py` — which is what makes the invariants above assertable.

## Validation strategy

| Layer | Mechanism | Location |
|---|---|---|
| **Layer 1** | Spec-value fixtures transcribed from `docs/model.md` | `tests/fixtures/model_md_values.json` |
| **Layer 2** | Hypothesis property invariants | `tests/test_properties.py` |
| **Purity** | AST scan banning I/O, randomness, and clock access in `core/` | `tests/test_purity.py` |

Continuous integration runs the full verification suite on every push and pull request; the
[Actions log](https://github.com/Bardakor/betting-app-yami/actions/workflows/verify.yml)
is the durable proof of compilation, static analysis, and test passage.

## Reproducing

**Prerequisites:** Python ≥ 3.12 (see `.python-version`), [uv](https://github.com/astral-sh/uv).

```bash
make install   # uv sync --extra dev
make verify    # compile, ruff, mypy --strict, pytest
```

`make verify` runs, in order:

1. **compile** — byte-compilation of all Python source
2. **lint** — `ruff check` and `ruff format --check`
3. **typecheck** — `mypy --strict`
4. **test** — `pytest` (109 tests as of the latest green run)

`make data` and `make study` are specified in the design doc and arrive with Plans 3 and 4
respectively — they do not exist yet.

## Repository layout

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, ruff/mypy/pytest config |
| `Makefile` | Terminal entry points: `verify`, `test`, `lint`, `typecheck`, `compile` |
| `docs/model.md` | Canonical mathematical specification |
| `docs/superpowers/` | Study design, implementation plans (research provenance) |
| `src/footy/core/` | Pure pricing mathematics |
| `src/footy/market/` | Overround, de-margining, Kelly |
| `src/footy/eval/` | Scoring rules and Murphy decomposition |
| `tests/` | Layer-1 fixtures, Layer-2 properties, purity scan |

## Limitations

- Only the **Shin** de-margining inverse is implemented; proportional, power, and
  odds-ratio transforms arrive in Plan 3.
- No MLE fitting, data ingest, walk-forward backtest, or empirical study code exists yet.
- Pre-registration predictions are recorded in this README and the design spec; a dedicated
  `paper/preregistration.md` is planned for Plan 5.

## Licence

MIT. See [LICENSE](LICENSE).
