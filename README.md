# The De-margining Artifact

A research repository testing whether a claimed forecasting edge over football bookmakers
survives the choice of de-margining transform, or is substantially an artifact of it.

## The question

Empirical football-forecasting papers routinely claim to beat the bookmaker. Substantiating
that claim requires converting offered odds into a "fair" benchmark probability, which means
removing the bookmaker's margin — and the literature uses at least four different transforms
for this (proportional normalisation, the power method, Shin's insider-trading model, and the
odds-ratio method), typically applying one without justification, often in a single unexamined
line of code. These transforms are not equivalent: they diverge most in how they treat
longshots, which is exactly where a model's claimed edge tends to concentrate. This repository
holds a single fixed forecasting model constant across all four transforms and tests whether
the measured edge changes — in size, in significance, or in sign — with the choice of
benchmark.

## The three pre-registered predictions

These are committed ahead of the full study running, so the git history timestamps the
hypotheses before the results. See `docs/superpowers/specs/2026-08-16-demargining-study-design.md`
§2 for the source.

| ID | Prediction | Rationale |
|---|---|---|
| **P1** | Disagreement between transforms scales with the book's margin — negligible at Pinnacle's ~102% book, material at Bet365's ~107% | The transforms coincide in the limit `B → 1`; they can only diverge in proportion to the mass being removed |
| **P2** | Disagreement is larger for 3-outcome 1X2 than for 2-outcome Over/Under 2.5 | The transforms differ chiefly in longshot treatment; a 2-outcome market on a near-even line has no longshot |
| **P3** | There exist (league, book) cells where the model's measured edge **changes sign** between proportional and Shin | The headline claim. Sign reversal, not merely magnitude change, is what would invalidate a published conclusion |

All three are reported with effect sizes and confidence intervals regardless of outcome. No
prediction is dropped, reframed, or replaced after seeing results.

## Status

The work decomposes into five independently verifiable plans.

- [x] **Plan 1 — Conversion + core.** Wave-1 deletion of the former application, Python
  scaffolding, and a clean-room port of the pricing core against Layer-1 (spec-value) and
  Layer-2 (property) tests.
- [ ] **Plan 2 — Fitting.** Weighted log-likelihood, analytic gradient, `check_grad`
  verification, the L-BFGS-B driver.
- [ ] **Plan 3 — Data.** Ingest with header discovery, the coverage matrix, and the remaining
  three de-margining transforms.
- [ ] **Plan 4 — Study.** The walk-forward protocol, the leakage test, scoring and inference,
  and the wave-2 deletion of the TypeScript engine once Layer-3 corroboration is green.
- [ ] **Plan 5 — Paper.** Figures, tables, and the manuscript, written against the frozen
  pre-registration.

## Reproducing

```
make install   # uv sync --extra dev
make verify    # ruff check, ruff format --check, mypy --strict, pytest
```

As of this commit, `make verify` is green and `pytest` collects and passes 109 tests.
`make data` and `make study` are specified in the design doc and arrive with Plans 3 and 4
respectively — they do not exist yet.

## The model

The forecasting model is Dixon-Coles (1997) with exponential time decay, specified in full in
[`packages/quant-engine/docs/MODEL.md`](packages/quant-engine/docs/MODEL.md). The Python core
under `src/footy/core/` was written clean-room from that document — without reading the
existing TypeScript implementation — so that a later agreement between the two is genuine
corroboration rather than a proof that one was copied from the other.

## Layout

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, ruff/mypy/pytest config |
| `Makefile` | Terminal entry points: `verify`, `test`, `lint`, `typecheck` |
| `src/footy/core/poisson.py` | Log-space Poisson pmf |
| `src/footy/core/dixon_coles.py` | τ matrix, τ derivatives, admissible ρ bounds |
| `src/footy/core/matrix.py` | Normalised 11×11 scoreline matrix |
| `src/footy/core/markets.py` | Market marginals: 1X2, totals, BTTS, correct score, AH, DC |
| `src/footy/core/skellam.py` | Goal-difference distribution, independent of the matrix |
| `src/footy/market/overround.py` | Power-method margin application |
| `src/footy/market/demargin.py` | Shin inverse (proportional/power/odds-ratio arrive in Plan 3) |
| `src/footy/market/kelly.py` | Fractional Kelly staking |
| `src/footy/eval/scoring.py` | RPS, Brier, log loss |
| `src/footy/eval/murphy.py` | REL / RES / UNC / WBV decomposition |
| `tests/fixtures/model_md_values.json` | Layer-1 targets transcribed from `MODEL.md` |
| `tests/test_purity.py` | Scans `core/` for banned imports |
| `tests/test_properties.py` | Layer-2 Hypothesis invariants |

`src/footy/core/` contains no I/O, no clock, and no randomness — enforced by
`tests/test_purity.py` — which is what makes the invariants above assertable.

The TypeScript pricing engine under `packages/quant-engine/` still exists in this repository.
It is retained deliberately, as the independent oracle that Plan 4's Layer-3 corroboration
diffs the Python core against; it is not scheduled for deletion until that corroboration is
green.

## Licence

MIT. See [LICENSE](LICENSE).
