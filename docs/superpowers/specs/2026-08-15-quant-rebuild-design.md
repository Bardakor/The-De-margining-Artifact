# Yami Betting Platform — Quant Rebuild Design

**Date:** 2026-08-15
**Status:** Approved for planning
**Scope:** Full rebuild — pricing engine, API consolidation, web redesign, RAG layer, repo cleanup

---

## 1. Motivation

The current codebase claims an "Advanced Statistical Model v2.0" with Poisson modelling, xG and
Kelly Criterion. None of those exist. The pricing path is built on `Math.random()` and a hardcoded
2023-24 team dictionary. This document specifies the replacement.

### 1.1 Defects being fixed

| # | Defect | Location | Consequence |
|---|--------|----------|-------------|
| D1 | Draw probability is a residual: `1 - pHome - pAway` | `fixtures.js:71` | Draw odds routinely exceed 6.0; can go negative when the two independent legs sum past 1.0 |
| D2 | Margin applied backwards: `prob * (1 - margin)` | `fixtures.js:75` | Lengthens every price. The book pays out ~105% of fair value instead of retaining 5% |
| D3 | `Math.random()` inside the pricing path | `fixtures.js:114-128`, `odds.js:167-178` | Odds change on every refresh; refresh-arbitrage hole |
| D4 | Unknown teams fall back to a flat rating of `6.0` | `fixtures.js:155,160-164` | Most fixtures price identically — the "generic" feel |
| D5 | Hardcoded 2023-24 ratings (Luton, Sheffield Utd, Burnley) | `fixtures.js:141-153` | Stale by three seasons |
| D6 | `calculateWins() { return matchesPlayed * 0.4 }` | `oddsCalculator.js:194` | Every team reports an identical W/D/L record |
| D7 | Fake Kelly / confidence / xG values | `auth.js:1034-1036` | README claims are unbacked |
| D8 | Three competing odds engines that disagree | `fixtures.js`, `oddsCalculator.js`, `oddsCalculatorMongo.js` | No single source of truth for a price |
| D9 | `weights` object declared, never read | `oddsCalculator.js:16-23` | Dead configuration implying tuning that does not happen |
| D10 | `mongodb-data/` committed to git | repo root | 100+ WiredTiger binaries tracked in version control |
| D11 | No sorting contract between backend and frontend | `fixtures.js:262` vs `live-matches/page.tsx` | Backend sorts, frontend re-renders arbitrarily |

### 1.2 Success criteria

The rebuild is done when all of the following hold:

1. `Σ(1/odds)` across any complete market equals the configured overround to within 1e-9.
2. The scoreline matrix sums to 1.0 within 1e-9.
3. Pricing the same fixture twice with the same parameters returns byte-identical output.
4. A walk-forward backtest over held-out seasons reports a Brier score below 0.60.
5. No market's implied probabilities contradict another market derived from the same matrix.
6. `grep -rn "Math.random" packages/quant-engine/src` returns zero results.

---

## 2. Architecture

### 2.1 Repository layout

```
packages/
  quant-engine/          Pure TypeScript. No I/O, no DB, no HTTP.
    src/ratings/         Elo + Bayesian shrinkage
    src/strengths/       MLE fit of attack/defence/home-advantage/rho
    src/poisson/         Dixon-Coles bivariate distribution, scoreline matrix
    src/markets/         Market derivation from the matrix
    src/pricing/         Overround (Shin), Kelly, CLV
    src/calibration/     Brier, log-loss, reliability, ROI
  data-pipeline/         CSV ingest, parameter fitting, DB writes
apps/
  api/                   Single service replacing all six backends
  web/                   Next.js 15 quant-terminal UI
```

### 2.2 Removed

- `mini-betting-platform/` — duplicate of the main app
- `backend/odds-service/src/services/oddsCalculator.js` and `oddsCalculatorMongo.js`
- Inline pricing helpers in `backend/fixtures-service/src/routes/fixtures.js` (lines 60-260)
- `backend/{api-gateway,main-service,fixtures-service,odds-service,wallet-service,bet-service,result-service}` — folded into `apps/api`
- `mongodb-data/` — deleted from the working tree and added to `.gitignore`

Rationale for collapsing six services into one: the services share a database, are deployed
together, and are called in a strict request-scoped sequence. The separation adds network hops and
three duplicate pricing implementations without buying independent scaling or independent
deployment. One API with clear internal module boundaries is both simpler and more defensible.

### 2.3 Technology

- **Engine:** TypeScript, zero runtime dependencies
- **API:** Fastify (TypeScript)
- **Database:** PostgreSQL 16 + pgvector, via local Docker; connection-string portable to a hosted
  Postgres later
- **Web:** Next.js 15 (App Router), React 19, Tailwind v4
- **LLM:** Claude Opus 5 (`claude-opus-5`) via the Anthropic SDK, with prompt caching

### 2.4 Supabase constraint

The account has exactly one organization, `Amphitrite-Products`, which is explicitly out of bounds.
No Supabase project will be created by this work. Development uses local Postgres in Docker. The
schema and connection layer stay portable so that pointing at a personal hosted Postgres later is a
connection-string change and nothing more.

---

## 3. The quant engine

`quant-engine` is a pure library: functions take data and return numbers. It performs no I/O. This
is what makes the invariants in §1.2 testable.

### 3.1 Ratings layer

Elo per team, with the K-factor scaled by goal margin so a 4-0 moves ratings more than a 1-0.

For teams with few observed matches (promoted sides, new entrants), the rating is shrunk toward the
league mean:

```
rating_shrunk = (n · rating_observed + k · rating_leagueMean) / (n + k)
```

where `n` is matches observed and `k` is the shrinkage constant. This replaces defect D4 correctly:
a low-sample team receives the league prior *with wide uncertainty*, rather than a fabricated
constant. The resulting variance feeds the confidence score, replacing D7.

### 3.2 Strength parameters

For each team `i`, fit attack `α_i` and defence `β_i` by maximum likelihood over historical
results, weighted by exponential time decay:

```
φ(Δt) = exp(−ξ · Δt)
```

so recent matches dominate the fit. Home advantage `γ` is fit **per league** rather than hardcoded,
because home advantage genuinely differs between competitions.

Identifiability constraint: `mean(α) = 1` across each league, preventing the scale degeneracy where
all attacks and all defences drift together.

### 3.3 Expected goals

```
λ_home = α_home · β_away · γ
λ_away = α_away · β_home
```

### 3.4 Dixon-Coles correction

Independent Poisson systematically under-counts low-scoring draws. The Dixon-Coles `τ` term
corrects the four affected cells:

```
τ(0,0) = 1 − λμρ
τ(0,1) = 1 + λρ
τ(1,0) = 1 + μρ
τ(1,1) = 1 − ρ
τ(x,y) = 1        otherwise
```

`ρ` is fit by MLE alongside the strength parameters. This is the direct fix for D1's symptom:
draw probability becomes a modelled quantity rather than arithmetic left over from two independent
estimates.

### 3.5 Scoreline matrix

An 11×11 matrix over 0–10 goals per side:

```
P(x, y) = τ(x, y, λ, μ, ρ) · Poisson(x; λ) · Poisson(y; μ)
```

Renormalised so the matrix sums to exactly 1.0 (absorbing the truncated tail beyond 10 goals and
the `τ` adjustment). **This matrix is the single source of truth for every price.**

### 3.6 Market derivation

Every market is a sum over matrix cells. No market has its own model.

| Market | Derivation |
|--------|------------|
| 1X2 | lower triangle (`x>y`), diagonal (`x=y`), upper triangle (`x<y`) |
| Over/Under, lines 0.5–5.5 | cells where `x + y > line` |
| Both teams to score | cells where `x > 0 ∧ y > 0` |
| Correct score | the individual cell |
| Asian handicap | triangles shifted by the handicap; quarter lines split across the two adjacent whole/half lines |
| Double chance | unions of the 1X2 triangles |

Because all markets are marginals of one distribution, they are mutually consistent by
construction. This is asserted in tests (§6).

### 3.7 Pricing and overround

Fair probabilities convert to fair odds as `1/p`. The book margin is then applied using **Shin's
method**, which distributes the overround according to outcome probability rather than scaling all
legs equally — naive scaling overcharges longshots.

The target overround is configurable (default 1.05). Defect D2 is fixed by construction: the
implementation solves for the margin parameter such that `Σ(1/odds)` equals the target exactly,
and a test asserts this.

### 3.8 Value detection and staking

Fractional Kelly against the model's own price versus an offered price:

```
f* = (b·p − q) / b
```

with `b` the net decimal odds, `p` the model probability, `q = 1 − p`. Fraction defaults to
quarter-Kelly. Closing line value is tracked per settled bet as the real measure of edge.

### 3.9 Calibration harness

Walk-forward backtest: fit on data up to time `t`, predict matches in `(t, t+w]`, roll forward.
Reports Brier score, log-loss, a reliability curve, and ROI against the historical closing odds
included in the source CSVs. Benchmarking against real closing odds is the credibility anchor —
it demonstrates the model's performance relative to the market, not merely relative to chance.

### 3.10 Determinism

No `Math.random()` anywhere in `packages/quant-engine`. An ESLint rule enforces this and CI fails
on violation. Stochastic simulation exists only in the backtest module and takes an explicit seeded
PRNG. Fixes D3.

---

## 4. Data pipeline

**Source:** football-data.co.uk — free CSV archives covering 20+ years across the major European
leagues, containing full results *and* historical closing odds from multiple bookmakers.

**Stages:**

1. **Ingest** — download and parse season CSVs into a normalised `matches` table
2. **Fit** — run the MLE from §3.2/§3.4 per league, producing ~2 parameters per team plus `γ` and
   `ρ` per league
3. **Persist** — write fitted parameters to `model_parameters`, versioned by fit timestamp
4. **Serve** — the API reads parameters; it never fits at request time

API-Football is retained for live fixture listings and in-play state only. It no longer influences
pricing.

Fitting is a batch job. This is why a hosted database is not on the critical path.

---

## 5. API and web

### 5.1 API

A single Fastify service exposing:

- `GET /fixtures` — fixtures with prices, supporting explicit `sort` and `filter` parameters
- `GET /fixtures/:id/markets` — all markets for one fixture, derived from its matrix
- `GET /fixtures/:id/matrix` — the raw 11×11 scoreline distribution
- `GET /fixtures/:id/explain` — the computation trace (§5.3)
- `POST /bets`, `GET /bets` — bet placement and history
- `GET /wallet`, `POST /wallet/deposit` — balance operations

Auth, wallet, bets and results become internal modules with clear boundaries rather than separate
network services.

### 5.2 Sorting contract

Defect D11 is fixed by making sort order a server-side concern with an explicit contract. The API
accepts `sort=kickoff|league|edge|confidence` and `order=asc|desc`, returns fixtures in that exact
order, and the client renders the array as received without re-sorting. League priority is a seeded
database table, not a hardcoded object literal.

### 5.3 RAG layer

**Match briefings.** Retrieve recent form, head-to-head history and news snippets via pgvector
similarity search. Claude Opus 5 generates a pre-match analysis grounded in both the retrieved
documents and the engine's computed `λ` values, citing its sources. Replaces the hardcoded
`reasoning` string arrays.

**Explainability.** `GET /fixtures/:id/explain` returns the real computation trace: the `α`, `β`,
`γ`, `ρ` values used, the resulting `λ_home`/`λ_away`, the matrix regions summed for the requested
market, and the overround applied. Claude renders this trace as prose. Because it reads a
computation rather than generating a narrative, it cannot fabricate a justification.

Static context (model description, market definitions) is prompt-cached to control cost.

### 5.4 Web — quant terminal

Design direction: trading-desk density over consumer sportsbook drama.

- Tabular numerals throughout; monospace for all prices
- Dense fixture grid with real column sorting
- Price movement flashes green/red on change
- Signature element: the 11×11 scoreline matrix rendered as a live heatmap, making the model's
  output directly visible
- Design tokens defined once; no stock shadcn defaults left in place

---

## 6. Testing strategy

The engine is developed test-first. Its purity makes the following assertable, none of which the
current code can express:

**Invariants**
- Scoreline matrix sums to 1.0 (±1e-9)
- `Σ(1/odds)` equals the configured overround (±1e-9)
- All probabilities lie in `[0, 1]`
- 1X2 probabilities sum to 1.0 before margin

**Consistency**
- `P(Over 2.5) + P(Under 2.5) == 1`
- `P(home) + P(draw) == P(double chance 1X)`
- Correct-score cells summed over the lower triangle equal `P(home win)`
- Asian handicap 0.0 equals the draw-no-bet price derived from 1X2

**Regression on the named defects**
- A fixture between two evenly matched teams produces draw odds in a plausible band, never > 6 and
  never negative (D1)
- Applying a 5% margin *shortens* every price relative to fair odds (D2)
- Pricing the same fixture twice returns identical output (D3)
- Two different unknown teams receive different ratings (D4)

**Calibration**
- Walk-forward backtest over held-out seasons reports Brier < 0.60

---

## 7. Out of scope

- Live in-play price updating during a match
- Real money handling, KYC, or payment processing
- Sports other than football
- Deployment to Vercel or any hosted Supabase project (blocked on the org constraint in §2.4)

---

## 8. Risks

| Risk | Mitigation |
|------|-----------|
| MLE fit fails to converge for small leagues | Fall back to league-prior parameters; log and surface as low confidence |
| football-data.co.uk changes CSV schema | Parser validates headers and fails loudly rather than silently mis-mapping columns |
| Rewrite loses working auth/wallet behaviour | Port those modules with their existing behaviour intact; they are not the source of the defects |
| Scope is large | Build order is engine → pipeline → API → web → RAG, each independently verifiable |
