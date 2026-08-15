# Quant Engine Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `packages/quant-engine` — a pure TypeScript library that converts two expected-goal values into a Dixon-Coles scoreline distribution and derives every betting market from it, priced with an exact configurable overround.

**Architecture:** One 11×11 scoreline matrix is the single source of truth. Every market is a sum over cells of that matrix, so all markets are mutually consistent by construction. Fair probabilities are converted to odds, then a margin is applied by the power method solved to hit an exact book sum. No I/O, no database, no randomness.

**Tech Stack:** TypeScript 5 (strict), Vitest, zero runtime dependencies.

**Plan 1 of 5.** Later plans: (2) ratings + MLE fit + backtest, (3) API + Postgres, (4) quant-terminal web, (5) RAG layer.

## Global Constraints

- Package location: `packages/quant-engine`. Node `>=18`.
- TypeScript `strict: true`. No `any`. No non-null assertions (`!`).
- **Zero runtime dependencies.** Dev dependencies (vitest, typescript) only.
- **No `Math.random()` anywhere in `src/`.** Enforced by a test in Task 9.
- All exported functions are pure: same inputs produce identical outputs.
- Floating-point comparisons in tests use tolerance `1e-9`.
- Goal grid is `0..maxGoals` inclusive, `maxGoals` defaults to `10` (an 11×11 matrix).
- Matrix indexing is always `cells[homeGoals][awayGoals]`.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

### Task 1: Package scaffold and Poisson probability mass function

**Files:**
- Create: `packages/quant-engine/package.json`
- Create: `packages/quant-engine/tsconfig.json`
- Create: `packages/quant-engine/vitest.config.ts`
- Create: `packages/quant-engine/src/poisson/poisson.ts`
- Test: `packages/quant-engine/test/poisson.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `poissonPmf(k: number, lambda: number): number` — probability of exactly `k` events given rate `lambda`. Throws `RangeError` if `k < 0`, `k` is not an integer, or `lambda <= 0`.

- [ ] **Step 1: Create the package manifest**

`packages/quant-engine/package.json`:

```json
{
  "name": "@yami/quant-engine",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "./src/index.ts",
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "typecheck": "tsc --noEmit"
  },
  "devDependencies": {
    "typescript": "^5.6.0",
    "vitest": "^2.1.0"
  },
  "engines": { "node": ">=18" }
}
```

- [ ] **Step 2: Create the TypeScript config**

`packages/quant-engine/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "exactOptionalPropertyTypes": true,
    "declaration": true,
    "skipLibCheck": true,
    "types": ["vitest/globals"]
  },
  "include": ["src", "test"]
}
```

- [ ] **Step 3: Create the Vitest config**

`packages/quant-engine/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    include: ["test/**/*.test.ts"],
  },
});
```

- [ ] **Step 4: Install dependencies**

Run: `cd packages/quant-engine && npm install`
Expected: `node_modules` created, no errors.

- [ ] **Step 5: Write the failing test**

`packages/quant-engine/test/poisson.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { poissonPmf } from "../src/poisson/poisson.js";

const TOL = 1e-9;

describe("poissonPmf", () => {
  it("returns e^-lambda for k = 0", () => {
    expect(poissonPmf(0, 1.5)).toBeCloseTo(Math.exp(-1.5), 9);
  });

  it("matches hand-computed values", () => {
    // P(2; 1.5) = 1.5^2 * e^-1.5 / 2! = 1.125 * e^-1.5
    expect(poissonPmf(2, 1.5)).toBeCloseTo(1.125 * Math.exp(-1.5), 9);
  });

  it("sums to approximately 1 over a wide grid", () => {
    let total = 0;
    for (let k = 0; k <= 60; k += 1) total += poissonPmf(k, 2.4);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("stays finite for large k where naive factorial would overflow", () => {
    const p = poissonPmf(170, 2.0);
    expect(Number.isFinite(p)).toBe(true);
    expect(p).toBeGreaterThanOrEqual(0);
  });

  it("rejects invalid input", () => {
    expect(() => poissonPmf(-1, 1)).toThrow(RangeError);
    expect(() => poissonPmf(1.5, 1)).toThrow(RangeError);
    expect(() => poissonPmf(1, 0)).toThrow(RangeError);
  });
});
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/poisson.test.ts`
Expected: FAIL — cannot resolve `../src/poisson/poisson.js`.

- [ ] **Step 7: Write the implementation**

`packages/quant-engine/src/poisson/poisson.ts`:

```ts
/**
 * Log-gamma via the Lanczos approximation. Used so that poissonPmf can be
 * computed in log space, which keeps it finite for large k where a naive
 * lambda^k / k! would overflow to Infinity / Infinity = NaN.
 */
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
] as const;

function logGamma(z: number): number {
  if (z < 0.5) {
    // Reflection formula for the left half-plane.
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  const x = z - 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < LANCZOS.length; i += 1) {
    a += (LANCZOS[i] as number) / (x + i + 1);
  }
  const t = x + LANCZOS.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** log(k!) for non-negative integer k. */
function logFactorial(k: number): number {
  return logGamma(k + 1);
}

/**
 * Poisson probability mass: P(X = k) for rate `lambda`.
 * Computed in log space for numerical stability.
 */
export function poissonPmf(k: number, lambda: number): number {
  if (!Number.isInteger(k) || k < 0) {
    throw new RangeError(`k must be a non-negative integer, received ${k}`);
  }
  if (!(lambda > 0)) {
    throw new RangeError(`lambda must be greater than 0, received ${lambda}`);
  }
  return Math.exp(k * Math.log(lambda) - lambda - logFactorial(k));
}
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/poisson.test.ts`
Expected: PASS, 5 tests.

- [ ] **Step 9: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add numerically stable Poisson PMF

Computed in log space via Lanczos log-gamma so large k stays finite.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Shared types and the Dixon-Coles scoreline matrix

This is the task that fixes defect **D1** (draw probability computed as a residual `1 - pHome - pAway`, which produced draw odds above 6.0 and could go negative).

**Files:**
- Create: `packages/quant-engine/src/types.ts`
- Create: `packages/quant-engine/src/poisson/dixonColes.ts`
- Test: `packages/quant-engine/test/dixonColes.test.ts`

**Interfaces:**
- Consumes: `poissonPmf` from Task 1.
- Produces:
  - `interface ScorelineMatrix { readonly maxGoals: number; readonly cells: readonly (readonly number[])[] }` — `cells[homeGoals][awayGoals]`, sums to 1.
  - `interface MatchLambdas { readonly home: number; readonly away: number }`
  - `dixonColesTau(home: number, away: number, lambdas: MatchLambdas, rho: number): number`
  - `buildScorelineMatrix(lambdas: MatchLambdas, rho: number, maxGoals?: number): ScorelineMatrix`
  - `rhoBounds(lambdas: MatchLambdas): { min: number; max: number }`

- [ ] **Step 1: Create the shared types**

`packages/quant-engine/src/types.ts`:

```ts
/** Expected goals for each side of a single match. */
export interface MatchLambdas {
  readonly home: number;
  readonly away: number;
}

/**
 * Joint distribution over final scores.
 * `cells[h][a]` is P(home scores h AND away scores a). Sums to 1.
 */
export interface ScorelineMatrix {
  readonly maxGoals: number;
  readonly cells: readonly (readonly number[])[];
}

/** One outcome within a market, with its fair (zero-margin) price. */
export interface Selection {
  readonly key: string;
  readonly label: string;
  readonly probability: number;
  readonly fairOdds: number;
}

/** A complete, mutually exclusive and exhaustive set of selections. */
export interface Market {
  readonly key: string;
  readonly label: string;
  readonly selections: readonly Selection[];
}

/** A selection whose price has had the book margin applied. */
export interface PricedSelection extends Selection {
  readonly odds: number;
}

export interface PricedMarket {
  readonly key: string;
  readonly label: string;
  readonly selections: readonly PricedSelection[];
  /** Sum of 1/odds across selections. Equals the configured overround. */
  readonly bookSum: number;
}
```

- [ ] **Step 2: Write the failing test**

`packages/quant-engine/test/dixonColes.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  buildScorelineMatrix,
  dixonColesTau,
  rhoBounds,
} from "../src/poisson/dixonColes.js";
import { poissonPmf } from "../src/poisson/poisson.js";

const TOL = 1e-9;
const LAMBDAS = { home: 1.6, away: 1.1 };

function sum(matrix: { cells: readonly (readonly number[])[] }): number {
  return matrix.cells.reduce(
    (acc, row) => acc + row.reduce((rowAcc, cell) => rowAcc + cell, 0),
    0,
  );
}

describe("dixonColesTau", () => {
  it("returns 1 for scorelines outside the corrected 2x2 block", () => {
    expect(dixonColesTau(2, 1, LAMBDAS, 0.1)).toBe(1);
    expect(dixonColesTau(0, 3, LAMBDAS, 0.1)).toBe(1);
    expect(dixonColesTau(4, 4, LAMBDAS, 0.1)).toBe(1);
  });

  it("applies the four Dixon-Coles corrections", () => {
    const rho = 0.08;
    expect(dixonColesTau(0, 0, LAMBDAS, rho)).toBeCloseTo(
      1 - LAMBDAS.home * LAMBDAS.away * rho, 12,
    );
    expect(dixonColesTau(0, 1, LAMBDAS, rho)).toBeCloseTo(1 + LAMBDAS.home * rho, 12);
    expect(dixonColesTau(1, 0, LAMBDAS, rho)).toBeCloseTo(1 + LAMBDAS.away * rho, 12);
    expect(dixonColesTau(1, 1, LAMBDAS, rho)).toBeCloseTo(1 - rho, 12);
  });

  it("is the identity when rho is 0", () => {
    for (const [h, a] of [[0, 0], [0, 1], [1, 0], [1, 1]] as const) {
      expect(dixonColesTau(h, a, LAMBDAS, 0)).toBeCloseTo(1, 12);
    }
  });
});

describe("buildScorelineMatrix", () => {
  it("produces an 11x11 grid by default", () => {
    const m = buildScorelineMatrix(LAMBDAS, 0.05);
    expect(m.maxGoals).toBe(10);
    expect(m.cells).toHaveLength(11);
    expect(m.cells[0]).toHaveLength(11);
  });

  it("sums to exactly 1", () => {
    expect(Math.abs(sum(buildScorelineMatrix(LAMBDAS, 0.05)) - 1)).toBeLessThan(TOL);
  });

  it("sums to 1 for lopsided and low-scoring fixtures too", () => {
    for (const l of [
      { home: 3.4, away: 0.4 },
      { home: 0.5, away: 0.6 },
      { home: 2.0, away: 2.0 },
    ]) {
      expect(Math.abs(sum(buildScorelineMatrix(l, 0.06)) - 1)).toBeLessThan(TOL);
    }
  });

  it("has no negative cells", () => {
    const m = buildScorelineMatrix(LAMBDAS, 0.05);
    for (const row of m.cells) for (const cell of row) expect(cell).toBeGreaterThanOrEqual(0);
  });

  it("reduces to independent Poisson when rho is 0", () => {
    const m = buildScorelineMatrix(LAMBDAS, 0);
    // Renormalisation only removes the truncated tail beyond maxGoals.
    let mass = 0;
    for (let h = 0; h <= 10; h += 1) {
      for (let a = 0; a <= 10; a += 1) {
        mass += poissonPmf(h, LAMBDAS.home) * poissonPmf(a, LAMBDAS.away);
      }
    }
    const expected =
      (poissonPmf(1, LAMBDAS.home) * poissonPmf(1, LAMBDAS.away)) / mass;
    expect(m.cells[1]?.[1]).toBeCloseTo(expected, 12);
  });

  it("raises low-scoring draw probability relative to independent Poisson", () => {
    // This is the whole point of Dixon-Coles: plain Poisson under-counts 0-0 and 1-1.
    const withRho = buildScorelineMatrix(LAMBDAS, 0.1);
    const withoutRho = buildScorelineMatrix(LAMBDAS, 0);
    expect(withRho.cells[1]?.[1] ?? 0).toBeGreaterThan(withoutRho.cells[1]?.[1] ?? 0);
  });

  it("is deterministic", () => {
    const a = buildScorelineMatrix(LAMBDAS, 0.05);
    const b = buildScorelineMatrix(LAMBDAS, 0.05);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("rejects a rho outside the admissible bounds", () => {
    const { max } = rhoBounds(LAMBDAS);
    expect(() => buildScorelineMatrix(LAMBDAS, max + 0.5)).toThrow(RangeError);
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/dixonColes.test.ts`
Expected: FAIL — cannot resolve `../src/poisson/dixonColes.js`.

- [ ] **Step 4: Write the implementation**

`packages/quant-engine/src/poisson/dixonColes.ts`:

```ts
import type { MatchLambdas, ScorelineMatrix } from "../types.js";
import { poissonPmf } from "./poisson.js";

const DEFAULT_MAX_GOALS = 10;

/**
 * Admissible range for rho. Outside this range the Dixon-Coles correction
 * drives one of the four adjusted cells negative.
 */
export function rhoBounds(lambdas: MatchLambdas): { min: number; max: number } {
  const { home, away } = lambdas;
  return {
    min: Math.max(-1 / home, -1 / away),
    max: Math.min(1 / (home * away), 1),
  };
}

/**
 * Dixon-Coles dependence correction. Independent Poisson under-counts the four
 * low-scoring results; tau adjusts exactly those cells and leaves the rest at 1.
 */
export function dixonColesTau(
  home: number,
  away: number,
  lambdas: MatchLambdas,
  rho: number,
): number {
  if (home === 0 && away === 0) return 1 - lambdas.home * lambdas.away * rho;
  if (home === 0 && away === 1) return 1 + lambdas.home * rho;
  if (home === 1 && away === 0) return 1 + lambdas.away * rho;
  if (home === 1 && away === 1) return 1 - rho;
  return 1;
}

/**
 * Joint scoreline distribution over 0..maxGoals for each side.
 *
 * Renormalised so the matrix sums to exactly 1, absorbing both the truncated
 * tail beyond maxGoals and the mass shifted by the tau correction. Every market
 * in this engine is derived from this matrix, which is what makes the markets
 * mutually consistent.
 */
export function buildScorelineMatrix(
  lambdas: MatchLambdas,
  rho: number,
  maxGoals: number = DEFAULT_MAX_GOALS,
): ScorelineMatrix {
  if (!(lambdas.home > 0) || !(lambdas.away > 0)) {
    throw new RangeError("both lambdas must be greater than 0");
  }
  if (!Number.isInteger(maxGoals) || maxGoals < 1) {
    throw new RangeError(`maxGoals must be a positive integer, received ${maxGoals}`);
  }
  const { min, max } = rhoBounds(lambdas);
  if (rho < min || rho > max) {
    throw new RangeError(
      `rho ${rho} outside admissible bounds [${min}, ${max}] for these lambdas`,
    );
  }

  const homePmf = Array.from({ length: maxGoals + 1 }, (_, k) =>
    poissonPmf(k, lambdas.home),
  );
  const awayPmf = Array.from({ length: maxGoals + 1 }, (_, k) =>
    poissonPmf(k, lambdas.away),
  );

  const raw: number[][] = [];
  let total = 0;
  for (let h = 0; h <= maxGoals; h += 1) {
    const row: number[] = [];
    for (let a = 0; a <= maxGoals; a += 1) {
      const value =
        dixonColesTau(h, a, lambdas, rho) *
        (homePmf[h] as number) *
        (awayPmf[a] as number);
      row.push(value);
      total += value;
    }
    raw.push(row);
  }

  const cells = raw.map((row) => row.map((value) => value / total));
  return { maxGoals, cells };
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/dixonColes.test.ts`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add Dixon-Coles scoreline matrix

Replaces the residual draw calculation that produced draw odds above 6.0.
The draw is now a modelled quantity summed from the matrix diagonal rather
than arithmetic left over from two independent estimates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: 1X2, double chance and draw-no-bet from the matrix

**Files:**
- Create: `packages/quant-engine/src/markets/matchOdds.ts`
- Test: `packages/quant-engine/test/matchOdds.test.ts`

**Interfaces:**
- Consumes: `ScorelineMatrix`, `Market`, `Selection` from Task 2.
- Produces:
  - `matchOddsMarket(matrix: ScorelineMatrix): Market` — key `"1X2"`, selections keyed `"1X2:HOME"`, `"1X2:DRAW"`, `"1X2:AWAY"`.
  - `doubleChanceMarket(matrix: ScorelineMatrix): Market` — key `"DC"`, selections `"DC:1X"`, `"DC:12"`, `"DC:X2"`.
  - `drawNoBetMarket(matrix: ScorelineMatrix): Market` — key `"DNB"`, selections `"DNB:HOME"`, `"DNB:AWAY"`.
  - `toFairOdds(probability: number): number` — `1/p`, or `Infinity` when `p === 0`.

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/matchOdds.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import {
  doubleChanceMarket,
  drawNoBetMarket,
  matchOddsMarket,
} from "../src/markets/matchOdds.js";

const TOL = 1e-9;
const MATRIX = buildScorelineMatrix({ home: 1.6, away: 1.1 }, 0.06);

function prob(market: { selections: readonly { key: string; probability: number }[] }, key: string): number {
  const found = market.selections.find((s) => s.key === key);
  if (!found) throw new Error(`missing selection ${key}`);
  return found.probability;
}

describe("matchOddsMarket", () => {
  it("produces three selections summing to 1", () => {
    const m = matchOddsMarket(MATRIX);
    expect(m.selections).toHaveLength(3);
    const total = m.selections.reduce((acc, s) => acc + s.probability, 0);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("favours the home side when its lambda is higher", () => {
    const m = matchOddsMarket(MATRIX);
    expect(prob(m, "1X2:HOME")).toBeGreaterThan(prob(m, "1X2:AWAY"));
  });

  it("gives a plausible draw probability, never the runaway price of the old model", () => {
    // Regression guard for defect D1. An evenly matched fixture should land
    // near a real-world draw rate, roughly 24-30%, never below 15%.
    const even = buildScorelineMatrix({ home: 1.35, away: 1.25 }, 0.06);
    const p = prob(matchOddsMarket(even), "1X2:DRAW");
    expect(p).toBeGreaterThan(0.15);
    expect(p).toBeLessThan(0.40);
    // Fair draw odds must therefore stay well under the old model's 6.0+.
    expect(1 / p).toBeLessThan(6);
  });

  it("never yields a negative probability, even for extreme mismatches", () => {
    const lopsided = buildScorelineMatrix({ home: 4.2, away: 0.3 }, 0.02);
    for (const s of matchOddsMarket(lopsided).selections) {
      expect(s.probability).toBeGreaterThanOrEqual(0);
      expect(s.probability).toBeLessThanOrEqual(1);
    }
  });

  it("sets fairOdds to the reciprocal of probability", () => {
    for (const s of matchOddsMarket(MATRIX).selections) {
      expect(s.fairOdds).toBeCloseTo(1 / s.probability, 9);
    }
  });
});

describe("doubleChanceMarket", () => {
  it("derives each selection as the sum of its two 1X2 legs", () => {
    const x2 = matchOddsMarket(MATRIX);
    const dc = doubleChanceMarket(MATRIX);
    expect(prob(dc, "DC:1X")).toBeCloseTo(prob(x2, "1X2:HOME") + prob(x2, "1X2:DRAW"), 9);
    expect(prob(dc, "DC:12")).toBeCloseTo(prob(x2, "1X2:HOME") + prob(x2, "1X2:AWAY"), 9);
    expect(prob(dc, "DC:X2")).toBeCloseTo(prob(x2, "1X2:AWAY") + prob(x2, "1X2:DRAW"), 9);
  });

  it("has selections summing to 2, since each outcome appears twice", () => {
    const total = doubleChanceMarket(MATRIX).selections.reduce((a, s) => a + s.probability, 0);
    expect(Math.abs(total - 2)).toBeLessThan(TOL);
  });
});

describe("drawNoBetMarket", () => {
  it("renormalises the 1X2 market with the draw removed", () => {
    const x2 = matchOddsMarket(MATRIX);
    const dnb = drawNoBetMarket(MATRIX);
    const total = dnb.selections.reduce((a, s) => a + s.probability, 0);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
    const expectedHome =
      prob(x2, "1X2:HOME") / (prob(x2, "1X2:HOME") + prob(x2, "1X2:AWAY"));
    expect(prob(dnb, "DNB:HOME")).toBeCloseTo(expectedHome, 9);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/matchOdds.test.ts`
Expected: FAIL — cannot resolve `../src/markets/matchOdds.js`.

- [ ] **Step 3: Write the implementation**

`packages/quant-engine/src/markets/matchOdds.ts`:

```ts
import type { Market, ScorelineMatrix, Selection } from "../types.js";

/** Fair decimal odds for a probability. */
export function toFairOdds(probability: number): number {
  return probability === 0 ? Number.POSITIVE_INFINITY : 1 / probability;
}

function selection(key: string, label: string, probability: number): Selection {
  return { key, label, probability, fairOdds: toFairOdds(probability) };
}

/**
 * Sums matrix cells for which `predicate(homeGoals, awayGoals)` holds.
 * Every market in this engine is expressed as one of these sums, which is what
 * guarantees the markets agree with one another.
 */
function sumWhere(
  matrix: ScorelineMatrix,
  predicate: (home: number, away: number) => boolean,
): number {
  let total = 0;
  for (let h = 0; h <= matrix.maxGoals; h += 1) {
    const row = matrix.cells[h];
    if (!row) continue;
    for (let a = 0; a <= matrix.maxGoals; a += 1) {
      if (predicate(h, a)) total += row[a] ?? 0;
    }
  }
  return total;
}

export function matchOddsMarket(matrix: ScorelineMatrix): Market {
  const home = sumWhere(matrix, (h, a) => h > a);
  const draw = sumWhere(matrix, (h, a) => h === a);
  const away = sumWhere(matrix, (h, a) => h < a);
  return {
    key: "1X2",
    label: "Match Result",
    selections: [
      selection("1X2:HOME", "Home", home),
      selection("1X2:DRAW", "Draw", draw),
      selection("1X2:AWAY", "Away", away),
    ],
  };
}

export function doubleChanceMarket(matrix: ScorelineMatrix): Market {
  return {
    key: "DC",
    label: "Double Chance",
    selections: [
      selection("DC:1X", "Home or Draw", sumWhere(matrix, (h, a) => h >= a)),
      selection("DC:12", "Home or Away", sumWhere(matrix, (h, a) => h !== a)),
      selection("DC:X2", "Draw or Away", sumWhere(matrix, (h, a) => h <= a)),
    ],
  };
}

export function drawNoBetMarket(matrix: ScorelineMatrix): Market {
  const home = sumWhere(matrix, (h, a) => h > a);
  const away = sumWhere(matrix, (h, a) => h < a);
  const decisive = home + away;
  return {
    key: "DNB",
    label: "Draw No Bet",
    selections: [
      selection("DNB:HOME", "Home", home / decisive),
      selection("DNB:AWAY", "Away", away / decisive),
    ],
  };
}

export { sumWhere };
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/matchOdds.test.ts`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): derive 1X2, double chance and DNB from the matrix

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Totals, both-teams-to-score and correct score

**Files:**
- Create: `packages/quant-engine/src/markets/totals.ts`
- Create: `packages/quant-engine/src/markets/btts.ts`
- Create: `packages/quant-engine/src/markets/correctScore.ts`
- Test: `packages/quant-engine/test/derivedMarkets.test.ts`

**Interfaces:**
- Consumes: `sumWhere`, `toFairOdds` from Task 3.
- Produces:
  - `totalsMarket(matrix: ScorelineMatrix, line: number): Market` — key `"OU_<line>"`, selections `"OU:<line>:OVER"` / `"OU:<line>:UNDER"`. Throws `RangeError` unless `line` is a half-integer (`x.5`).
  - `bttsMarket(matrix: ScorelineMatrix): Market` — key `"BTTS"`, selections `"BTTS:YES"` / `"BTTS:NO"`.
  - `correctScoreMarket(matrix: ScorelineMatrix, maxDisplayGoals?: number): Market` — key `"CS"`, selections `"CS:<h>-<a>"` plus an `"CS:OTHER"` bucket.

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/derivedMarkets.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { matchOddsMarket } from "../src/markets/matchOdds.js";
import { totalsMarket } from "../src/markets/totals.js";
import { bttsMarket } from "../src/markets/btts.js";
import { correctScoreMarket } from "../src/markets/correctScore.js";

const TOL = 1e-9;
const MATRIX = buildScorelineMatrix({ home: 1.6, away: 1.1 }, 0.06);

function prob(m: { selections: readonly { key: string; probability: number }[] }, key: string): number {
  const found = m.selections.find((s) => s.key === key);
  if (!found) throw new Error(`missing selection ${key}`);
  return found.probability;
}

describe("totalsMarket", () => {
  it("has over and under summing to 1 on every line", () => {
    for (const line of [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]) {
      const m = totalsMarket(MATRIX, line);
      const total = m.selections.reduce((a, s) => a + s.probability, 0);
      expect(Math.abs(total - 1)).toBeLessThan(TOL);
    }
  });

  it("makes over less likely as the line rises", () => {
    const overs = [0.5, 1.5, 2.5, 3.5, 4.5].map((l) =>
      prob(totalsMarket(MATRIX, l), `OU:${l}:OVER`),
    );
    for (let i = 1; i < overs.length; i += 1) {
      expect(overs[i] as number).toBeLessThan(overs[i - 1] as number);
    }
  });

  it("computes Under 0.5 as exactly the 0-0 cell", () => {
    expect(prob(totalsMarket(MATRIX, 0.5), "OU:0.5:UNDER")).toBeCloseTo(
      MATRIX.cells[0]?.[0] ?? 0, 12,
    );
  });

  it("rejects whole-number lines, which would allow a push", () => {
    expect(() => totalsMarket(MATRIX, 2)).toThrow(RangeError);
  });
});

describe("bttsMarket", () => {
  it("has yes and no summing to 1", () => {
    const total = bttsMarket(MATRIX).selections.reduce((a, s) => a + s.probability, 0);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("computes NO as the union of the zero row and zero column", () => {
    let expected = 0;
    for (let a = 0; a <= MATRIX.maxGoals; a += 1) expected += MATRIX.cells[0]?.[a] ?? 0;
    for (let h = 1; h <= MATRIX.maxGoals; h += 1) expected += MATRIX.cells[h]?.[0] ?? 0;
    expect(prob(bttsMarket(MATRIX), "BTTS:NO")).toBeCloseTo(expected, 12);
  });
});

describe("correctScoreMarket", () => {
  it("sums to 1 including the OTHER bucket", () => {
    const total = correctScoreMarket(MATRIX).selections.reduce((a, s) => a + s.probability, 0);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("agrees with the 1X2 market when its cells are aggregated", () => {
    // The critical consistency property: correct score and 1X2 are marginals of
    // the same distribution, so they cannot disagree.
    const cs = correctScoreMarket(MATRIX, MATRIX.maxGoals);
    let homeWin = 0;
    for (const s of cs.selections) {
      const parsed = /^CS:(\d+)-(\d+)$/.exec(s.key);
      if (!parsed) continue;
      if (Number(parsed[1]) > Number(parsed[2])) homeWin += s.probability;
    }
    expect(homeWin).toBeCloseTo(prob(matchOddsMarket(MATRIX), "1X2:HOME"), 9);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/derivedMarkets.test.ts`
Expected: FAIL — cannot resolve the three new modules.

- [ ] **Step 3: Write the totals implementation**

`packages/quant-engine/src/markets/totals.ts`:

```ts
import type { Market, ScorelineMatrix } from "../types.js";
import { sumWhere, toFairOdds } from "./matchOdds.js";

/**
 * Over/Under total goals. Only half-integer lines are supported, because a
 * whole-number line admits a push, which is a different settlement rule than
 * the two-way market this returns.
 */
export function totalsMarket(matrix: ScorelineMatrix, line: number): Market {
  if (!Number.isFinite(line) || (line * 2) % 2 !== 1) {
    throw new RangeError(`line must be a half-integer such as 2.5, received ${line}`);
  }
  const over = sumWhere(matrix, (h, a) => h + a > line);
  const under = 1 - over;
  return {
    key: `OU_${line}`,
    label: `Total Goals ${line}`,
    selections: [
      { key: `OU:${line}:OVER`, label: `Over ${line}`, probability: over, fairOdds: toFairOdds(over) },
      { key: `OU:${line}:UNDER`, label: `Under ${line}`, probability: under, fairOdds: toFairOdds(under) },
    ],
  };
}
```

- [ ] **Step 4: Write the BTTS implementation**

`packages/quant-engine/src/markets/btts.ts`:

```ts
import type { Market, ScorelineMatrix } from "../types.js";
import { sumWhere, toFairOdds } from "./matchOdds.js";

export function bttsMarket(matrix: ScorelineMatrix): Market {
  const yes = sumWhere(matrix, (h, a) => h > 0 && a > 0);
  const no = 1 - yes;
  return {
    key: "BTTS",
    label: "Both Teams To Score",
    selections: [
      { key: "BTTS:YES", label: "Yes", probability: yes, fairOdds: toFairOdds(yes) },
      { key: "BTTS:NO", label: "No", probability: no, fairOdds: toFairOdds(no) },
    ],
  };
}
```

- [ ] **Step 5: Write the correct score implementation**

`packages/quant-engine/src/markets/correctScore.ts`:

```ts
import type { Market, ScorelineMatrix, Selection } from "../types.js";
import { toFairOdds } from "./matchOdds.js";

const DEFAULT_DISPLAY_GOALS = 5;

/**
 * Correct score. Scorelines above `maxDisplayGoals` for either side are
 * collapsed into a single OTHER selection so the market stays sized for a UI
 * while still summing to 1.
 */
export function correctScoreMarket(
  matrix: ScorelineMatrix,
  maxDisplayGoals: number = DEFAULT_DISPLAY_GOALS,
): Market {
  const selections: Selection[] = [];
  let displayed = 0;

  for (let h = 0; h <= Math.min(maxDisplayGoals, matrix.maxGoals); h += 1) {
    for (let a = 0; a <= Math.min(maxDisplayGoals, matrix.maxGoals); a += 1) {
      const p = matrix.cells[h]?.[a] ?? 0;
      displayed += p;
      selections.push({
        key: `CS:${h}-${a}`,
        label: `${h} - ${a}`,
        probability: p,
        fairOdds: toFairOdds(p),
      });
    }
  }

  const other = Math.max(0, 1 - displayed);
  selections.push({
    key: "CS:OTHER",
    label: "Any other score",
    probability: other,
    fairOdds: toFairOdds(other),
  });

  return { key: "CS", label: "Correct Score", selections };
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/derivedMarkets.test.ts`
Expected: PASS, 8 tests.

- [ ] **Step 7: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add totals, BTTS and correct score markets

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Asian handicap with quarter-line splitting

**Files:**
- Create: `packages/quant-engine/src/markets/asianHandicap.ts`
- Test: `packages/quant-engine/test/asianHandicap.test.ts`

**Interfaces:**
- Consumes: `ScorelineMatrix`, `sumWhere`, `toFairOdds`.
- Produces: `asianHandicapMarket(matrix: ScorelineMatrix, handicap: number): AsianHandicapMarket`, where

```ts
export interface AsianHandicapOutcome {
  readonly key: string;
  readonly label: string;
  /** Probability the stake is won outright. */
  readonly win: number;
  /** Probability the stake is returned (push). */
  readonly push: number;
  /** Probability the stake is lost. */
  readonly lose: number;
  /** Fair odds accounting for the push refund. */
  readonly fairOdds: number;
}

export interface AsianHandicapMarket {
  readonly key: string;
  readonly label: string;
  readonly handicap: number;
  readonly selections: readonly AsianHandicapOutcome[];
}
```

`handicap` is applied to the home side and must be a multiple of `0.25`.

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/asianHandicap.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { drawNoBetMarket } from "../src/markets/matchOdds.js";
import { asianHandicapMarket } from "../src/markets/asianHandicap.js";

const TOL = 1e-9;
const MATRIX = buildScorelineMatrix({ home: 1.6, away: 1.1 }, 0.06);

function leg(h: number, key: string) {
  const found = asianHandicapMarket(MATRIX, h).selections.find((s) => s.key === key);
  if (!found) throw new Error(`missing ${key}`);
  return found;
}

describe("asianHandicapMarket", () => {
  it("has win, push and lose summing to 1 for each side", () => {
    for (const h of [-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1]) {
      for (const s of asianHandicapMarket(MATRIX, h).selections) {
        expect(Math.abs(s.win + s.push + s.lose - 1)).toBeLessThan(TOL);
      }
    }
  });

  it("mirrors the two sides: home win equals away lose", () => {
    const home = leg(-0.5, "AH:-0.5:HOME");
    const away = leg(-0.5, "AH:-0.5:AWAY");
    expect(home.win).toBeCloseTo(away.lose, 9);
    expect(home.push).toBeCloseTo(away.push, 9);
  });

  it("reduces to draw-no-bet at handicap 0", () => {
    // The level-ball handicap refunds on a draw, exactly like DNB.
    const dnb = drawNoBetMarket(MATRIX);
    const dnbHome = dnb.selections.find((s) => s.key === "DNB:HOME");
    const ah = leg(0, "AH:0:HOME");
    const impliedHome = ah.win / (ah.win + ah.lose);
    expect(impliedHome).toBeCloseTo(dnbHome?.probability ?? 0, 9);
  });

  it("never pushes on a half-goal line", () => {
    for (const s of asianHandicapMarket(MATRIX, -0.5).selections) {
      expect(s.push).toBeCloseTo(0, 12);
    }
  });

  it("splits a quarter line across its two neighbours", () => {
    // -0.75 is half the stake on -0.5 and half on -1.0.
    const quarter = leg(-0.75, "AH:-0.75:HOME");
    const half = leg(-0.5, "AH:-0.5:HOME");
    const whole = leg(-1, "AH:-1:HOME");
    expect(quarter.win).toBeCloseTo((half.win + whole.win) / 2, 9);
    expect(quarter.push).toBeCloseTo((half.push + whole.push) / 2, 9);
  });

  it("makes the favoured side less likely to win as its handicap deepens", () => {
    expect(leg(-1.5, "AH:-1.5:HOME").win).toBeLessThan(leg(-0.5, "AH:-0.5:HOME").win);
  });

  it("rejects handicaps that are not a multiple of 0.25", () => {
    expect(() => asianHandicapMarket(MATRIX, -0.3)).toThrow(RangeError);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/asianHandicap.test.ts`
Expected: FAIL — cannot resolve `../src/markets/asianHandicap.js`.

- [ ] **Step 3: Write the implementation**

`packages/quant-engine/src/markets/asianHandicap.ts`:

```ts
import type { ScorelineMatrix } from "../types.js";
import { sumWhere } from "./matchOdds.js";

export interface AsianHandicapOutcome {
  readonly key: string;
  readonly label: string;
  readonly win: number;
  readonly push: number;
  readonly lose: number;
  readonly fairOdds: number;
}

export interface AsianHandicapMarket {
  readonly key: string;
  readonly label: string;
  readonly handicap: number;
  readonly selections: readonly AsianHandicapOutcome[];
}

interface Legs {
  readonly win: number;
  readonly push: number;
  readonly lose: number;
}

/**
 * Fair odds for a bet that can push. A push returns the stake, so only the
 * non-push mass is at risk: fair = 1 + lose/win, capped at Infinity when win is 0.
 */
function fairOddsWithPush(legs: Legs): number {
  if (legs.win === 0) return Number.POSITIVE_INFINITY;
  return 1 + legs.lose / legs.win;
}

/** Win/push/lose for the home side on a whole or half handicap line. */
function homeLegsForWholeLine(matrix: ScorelineMatrix, handicap: number): Legs {
  const win = sumWhere(matrix, (h, a) => h + handicap > a);
  const push = sumWhere(matrix, (h, a) => h + handicap === a);
  return { win, push, lose: 1 - win - push };
}

function averageLegs(first: Legs, second: Legs): Legs {
  return {
    win: (first.win + second.win) / 2,
    push: (first.push + second.push) / 2,
    lose: (first.lose + second.lose) / 2,
  };
}

function invert(legs: Legs): Legs {
  return { win: legs.lose, push: legs.push, lose: legs.win };
}

/**
 * Asian handicap applied to the home side. Quarter lines (x.25, x.75) split the
 * stake evenly across the two adjacent lines, which is how they settle in
 * practice.
 */
export function asianHandicapMarket(
  matrix: ScorelineMatrix,
  handicap: number,
): AsianHandicapMarket {
  if (!Number.isFinite(handicap) || (handicap * 4) % 1 !== 0) {
    throw new RangeError(
      `handicap must be a multiple of 0.25, received ${handicap}`,
    );
  }

  const isQuarter = (handicap * 2) % 1 !== 0;
  const homeLegs = isQuarter
    ? averageLegs(
        homeLegsForWholeLine(matrix, handicap - 0.25),
        homeLegsForWholeLine(matrix, handicap + 0.25),
      )
    : homeLegsForWholeLine(matrix, handicap);
  const awayLegs = invert(homeLegs);

  const label = handicap > 0 ? `+${handicap}` : `${handicap}`;
  return {
    key: `AH_${handicap}`,
    label: `Asian Handicap ${label}`,
    handicap,
    selections: [
      {
        key: `AH:${handicap}:HOME`,
        label: `Home ${label}`,
        ...homeLegs,
        fairOdds: fairOddsWithPush(homeLegs),
      },
      {
        key: `AH:${handicap}:AWAY`,
        label: `Away ${handicap > 0 ? `-${handicap}` : `+${Math.abs(handicap)}`}`,
        ...awayLegs,
        fairOdds: fairOddsWithPush(awayLegs),
      },
    ],
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/asianHandicap.test.ts`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add Asian handicap with quarter-line splitting

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Overround via the power method

This task fixes defect **D2** (`prob * (1 - margin)`, which lengthened every price and made the book pay out roughly 105% of fair value).

**Files:**
- Create: `packages/quant-engine/src/pricing/overround.ts`
- Test: `packages/quant-engine/test/overround.test.ts`

**Interfaces:**
- Consumes: `Market`, `PricedMarket`, `PricedSelection` from Task 2.
- Produces:
  - `solvePowerExponent(probabilities: readonly number[], targetBookSum: number): number` — the `k` such that `Σ pᵢ^k = target`.
  - `applyOverround(market: Market, targetBookSum?: number): PricedMarket` — default target `1.05`.
  - `removeOverroundShin(impliedProbabilities: readonly number[]): number[]` — recovers true probabilities from a bookmaker's implied probabilities using Shin's insider-trading model. Used by the backtest in Plan 2 to de-margin historical closing odds.

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/overround.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { matchOddsMarket } from "../src/markets/matchOdds.js";
import {
  applyOverround,
  removeOverroundShin,
  solvePowerExponent,
} from "../src/pricing/overround.js";

const TOL = 1e-9;
const MARKET = matchOddsMarket(buildScorelineMatrix({ home: 1.6, away: 1.1 }, 0.06));

describe("solvePowerExponent", () => {
  it("returns 1 when the target equals the fair book sum", () => {
    expect(solvePowerExponent([0.5, 0.3, 0.2], 1)).toBeCloseTo(1, 9);
  });

  it("returns an exponent below 1 for a target above 1", () => {
    expect(solvePowerExponent([0.5, 0.3, 0.2], 1.05)).toBeLessThan(1);
  });
});

describe("applyOverround", () => {
  it("makes the book sum exactly the target", () => {
    for (const target of [1.02, 1.05, 1.08, 1.12]) {
      const priced = applyOverround(MARKET, target);
      const bookSum = priced.selections.reduce((a, s) => a + 1 / s.odds, 0);
      expect(Math.abs(bookSum - target)).toBeLessThan(TOL);
      expect(Math.abs(priced.bookSum - target)).toBeLessThan(TOL);
    }
  });

  it("SHORTENS every price relative to fair odds", () => {
    // Regression guard for defect D2. The old code multiplied probability by
    // (1 - margin), which lengthened prices and handed value to the bettor.
    const priced = applyOverround(MARKET, 1.05);
    for (const s of priced.selections) {
      expect(s.odds).toBeLessThan(s.fairOdds);
    }
  });

  it("leaves prices unchanged at a target of 1", () => {
    for (const s of applyOverround(MARKET, 1).selections) {
      expect(s.odds).toBeCloseTo(s.fairOdds, 9);
    }
  });

  it("preserves the favourite ordering", () => {
    const priced = applyOverround(MARKET, 1.05);
    const byOdds = [...priced.selections].sort((a, b) => a.odds - b.odds);
    const byProb = [...priced.selections].sort((a, b) => b.probability - a.probability);
    expect(byOdds.map((s) => s.key)).toEqual(byProb.map((s) => s.key));
  });

  it("takes proportionally more margin from the longshot than the favourite", () => {
    // The favourite-longshot bias the power method exists to reproduce.
    const priced = applyOverround(MARKET, 1.08);
    const withRatio = priced.selections.map((s) => ({
      probability: s.probability,
      ratio: s.fairOdds / s.odds,
    }));
    const favourite = withRatio.reduce((a, b) => (a.probability > b.probability ? a : b));
    const longshot = withRatio.reduce((a, b) => (a.probability < b.probability ? a : b));
    expect(longshot.ratio).toBeGreaterThan(favourite.ratio);
  });

  it("rejects a target below 1", () => {
    expect(() => applyOverround(MARKET, 0.98)).toThrow(RangeError);
  });
});

describe("removeOverroundShin", () => {
  it("returns probabilities summing to 1", () => {
    const recovered = removeOverroundShin([0.42, 0.31, 0.32]);
    const total = recovered.reduce((a, b) => a + b, 0);
    expect(Math.abs(total - 1)).toBeLessThan(1e-8);
  });

  it("is the identity for an already-fair book", () => {
    const recovered = removeOverroundShin([0.5, 0.3, 0.2]);
    expect(recovered[0]).toBeCloseTo(0.5, 7);
    expect(recovered[2]).toBeCloseTo(0.2, 7);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/overround.test.ts`
Expected: FAIL — cannot resolve `../src/pricing/overround.js`.

- [ ] **Step 3: Write the implementation**

`packages/quant-engine/src/pricing/overround.ts`:

```ts
import type { Market, PricedMarket, PricedSelection } from "../types.js";

const DEFAULT_BOOK_SUM = 1.05;
const BISECTION_ITERATIONS = 200;

/**
 * Solves for the exponent k such that the sum of p^k equals `targetBookSum`.
 *
 * Since every p is in (0, 1), p^k is strictly decreasing in k, so the sum is
 * monotonic and bisection converges. Using an exponent rather than a constant
 * multiplier reproduces the favourite-longshot bias: the margin taken from a
 * longshot is proportionally larger than from a favourite.
 */
export function solvePowerExponent(
  probabilities: readonly number[],
  targetBookSum: number,
): number {
  const positive = probabilities.filter((p) => p > 0);
  const sumAt = (k: number): number =>
    positive.reduce((acc, p) => acc + Math.pow(p, k), 0);

  let low = 0.01;
  let high = 1;
  // Expand downwards until the sum overshoots the target.
  while (sumAt(low) < targetBookSum && low > 1e-6) low /= 2;
  // Expand upwards in case the target is below the fair sum.
  while (sumAt(high) > targetBookSum && high < 64) high *= 2;

  for (let i = 0; i < BISECTION_ITERATIONS; i += 1) {
    const mid = (low + high) / 2;
    if (sumAt(mid) > targetBookSum) low = mid;
    else high = mid;
  }
  return (low + high) / 2;
}

/**
 * Applies the book margin to a market so that the sum of 1/odds equals
 * `targetBookSum` exactly.
 *
 * Note the direction: a target above 1 SHORTENS every price. The previous
 * implementation multiplied probability by (1 - margin), which lengthened
 * prices and produced a book paying out more than fair value.
 */
export function applyOverround(
  market: Market,
  targetBookSum: number = DEFAULT_BOOK_SUM,
): PricedMarket {
  if (!(targetBookSum >= 1)) {
    throw new RangeError(
      `targetBookSum must be at least 1, received ${targetBookSum}`,
    );
  }

  const probabilities = market.selections.map((s) => s.probability);
  const k = solvePowerExponent(probabilities, targetBookSum);

  const selections: PricedSelection[] = market.selections.map((s) => {
    const implied = s.probability > 0 ? Math.pow(s.probability, k) : 0;
    return {
      ...s,
      odds: implied > 0 ? 1 / implied : Number.POSITIVE_INFINITY,
    };
  });

  const bookSum = selections.reduce(
    (acc, s) => acc + (Number.isFinite(s.odds) ? 1 / s.odds : 0),
    0,
  );

  return { key: market.key, label: market.label, selections, bookSum };
}

/**
 * Shin's method: recovers true probabilities from a bookmaker's implied
 * probabilities by modelling the proportion z of insider money.
 *
 * This is the INVERSE of applying a margin, and is used to de-margin historical
 * closing odds so the model can be benchmarked against the market on equal
 * terms. It is not used to price our own markets.
 */
export function removeOverroundShin(
  impliedProbabilities: readonly number[],
): number[] {
  const bookSum = impliedProbabilities.reduce((a, b) => a + b, 0);
  if (bookSum <= 1) return impliedProbabilities.map((p) => p / bookSum);

  const trueProbs = (z: number): number[] =>
    impliedProbabilities.map((p) => {
      const root = Math.sqrt(z * z + 4 * (1 - z) * ((p * p) / bookSum));
      return (root - z) / (2 * (1 - z));
    });

  let low = 0;
  let high = 0.99;
  for (let i = 0; i < BISECTION_ITERATIONS; i += 1) {
    const mid = (low + high) / 2;
    const total = trueProbs(mid).reduce((a, b) => a + b, 0);
    if (total > 1) low = mid;
    else high = mid;
  }

  const result = trueProbs((low + high) / 2);
  const total = result.reduce((a, b) => a + b, 0);
  return result.map((p) => p / total);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/overround.test.ts`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/quant-engine
git commit -m "fix(engine): apply book margin in the correct direction

The previous implementation multiplied probability by (1 - margin), which
lengthened every price and left the book paying out above fair value. Margin
is now applied by the power method, solved so the book sum hits the target
exactly and longshots carry proportionally more margin than favourites.

Shin's method is included as the inverse, for de-margining historical closing
odds during backtesting.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Kelly staking and value detection

**Files:**
- Create: `packages/quant-engine/src/pricing/kelly.ts`
- Test: `packages/quant-engine/test/kelly.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `kellyFraction(modelProbability: number, offeredOdds: number): number` — full-Kelly stake fraction, floored at 0 when there is no edge.
  - `expectedValue(modelProbability: number, offeredOdds: number): number` — EV per unit staked.
  - `assessValue(modelProbability: number, offeredOdds: number, kellyMultiplier?: number): ValueAssessment`, where

```ts
export interface ValueAssessment {
  readonly modelProbability: number;
  readonly offeredOdds: number;
  readonly impliedProbability: number;
  /** Model probability minus implied probability. Positive means value. */
  readonly edge: number;
  readonly expectedValue: number;
  readonly fullKelly: number;
  /** fullKelly scaled by kellyMultiplier, default 0.25. */
  readonly recommendedStakeFraction: number;
  readonly hasValue: boolean;
}
```

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/kelly.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { assessValue, expectedValue, kellyFraction } from "../src/pricing/kelly.js";

describe("kellyFraction", () => {
  it("matches the textbook formula on a known case", () => {
    // p = 0.6, decimal 2.0 so b = 1: f* = (1*0.6 - 0.4)/1 = 0.2
    expect(kellyFraction(0.6, 2.0)).toBeCloseTo(0.2, 12);
  });

  it("returns 0 when there is no edge", () => {
    expect(kellyFraction(0.5, 2.0)).toBeCloseTo(0, 12);
  });

  it("returns 0 rather than a negative stake when the price is bad", () => {
    expect(kellyFraction(0.4, 2.0)).toBe(0);
  });

  it("stakes more as the offered price improves", () => {
    expect(kellyFraction(0.6, 3.0)).toBeGreaterThan(kellyFraction(0.6, 2.2));
  });

  it("never recommends staking the whole bankroll below certainty", () => {
    expect(kellyFraction(0.99, 10)).toBeLessThan(1);
  });

  it("rejects invalid input", () => {
    expect(() => kellyFraction(1.2, 2)).toThrow(RangeError);
    expect(() => kellyFraction(0.5, 1)).toThrow(RangeError);
  });
});

describe("expectedValue", () => {
  it("is zero at a fair price", () => {
    expect(expectedValue(0.5, 2.0)).toBeCloseTo(0, 12);
  });

  it("is positive when the offered price beats the model", () => {
    expect(expectedValue(0.5, 2.2)).toBeCloseTo(0.1, 12);
  });
});

describe("assessValue", () => {
  it("applies a quarter-Kelly multiplier by default", () => {
    const a = assessValue(0.6, 2.0);
    expect(a.recommendedStakeFraction).toBeCloseTo(a.fullKelly * 0.25, 12);
    expect(a.hasValue).toBe(true);
  });

  it("reports no value when the model agrees with the price", () => {
    const a = assessValue(0.5, 2.0);
    expect(a.hasValue).toBe(false);
    expect(a.edge).toBeCloseTo(0, 12);
  });

  it("computes edge as model minus implied probability", () => {
    const a = assessValue(0.55, 2.0);
    expect(a.impliedProbability).toBeCloseTo(0.5, 12);
    expect(a.edge).toBeCloseTo(0.05, 12);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/kelly.test.ts`
Expected: FAIL — cannot resolve `../src/pricing/kelly.js`.

- [ ] **Step 3: Write the implementation**

`packages/quant-engine/src/pricing/kelly.ts`:

```ts
const DEFAULT_KELLY_MULTIPLIER = 0.25;

export interface ValueAssessment {
  readonly modelProbability: number;
  readonly offeredOdds: number;
  readonly impliedProbability: number;
  readonly edge: number;
  readonly expectedValue: number;
  readonly fullKelly: number;
  readonly recommendedStakeFraction: number;
  readonly hasValue: boolean;
}

function assertInputs(modelProbability: number, offeredOdds: number): void {
  if (!(modelProbability >= 0 && modelProbability <= 1)) {
    throw new RangeError(
      `modelProbability must be within [0, 1], received ${modelProbability}`,
    );
  }
  if (!(offeredOdds > 1)) {
    throw new RangeError(
      `offeredOdds must be greater than 1, received ${offeredOdds}`,
    );
  }
}

/**
 * Full-Kelly stake fraction: f* = (b*p - q) / b, with b the net decimal odds.
 * Floored at 0, because a negative Kelly means "do not bet", not "bet the
 * other side" (the other side has its own price and its own assessment).
 */
export function kellyFraction(
  modelProbability: number,
  offeredOdds: number,
): number {
  assertInputs(modelProbability, offeredOdds);
  const b = offeredOdds - 1;
  const q = 1 - modelProbability;
  return Math.max(0, (b * modelProbability - q) / b);
}

/** Expected profit per unit staked. */
export function expectedValue(
  modelProbability: number,
  offeredOdds: number,
): number {
  assertInputs(modelProbability, offeredOdds);
  return modelProbability * offeredOdds - 1;
}

/**
 * Full value assessment of an offered price against the model's own number.
 * Fractional Kelly is the default because full Kelly is intolerably volatile
 * when the probability estimate itself carries error.
 */
export function assessValue(
  modelProbability: number,
  offeredOdds: number,
  kellyMultiplier: number = DEFAULT_KELLY_MULTIPLIER,
): ValueAssessment {
  assertInputs(modelProbability, offeredOdds);
  if (!(kellyMultiplier > 0 && kellyMultiplier <= 1)) {
    throw new RangeError(
      `kellyMultiplier must be within (0, 1], received ${kellyMultiplier}`,
    );
  }
  const impliedProbability = 1 / offeredOdds;
  const fullKelly = kellyFraction(modelProbability, offeredOdds);
  return {
    modelProbability,
    offeredOdds,
    impliedProbability,
    edge: modelProbability - impliedProbability,
    expectedValue: expectedValue(modelProbability, offeredOdds),
    fullKelly,
    recommendedStakeFraction: fullKelly * kellyMultiplier,
    hasValue: fullKelly > 0,
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/kelly.test.ts`
Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add real Kelly staking and value detection

Replaces the placeholder that returned Math.floor(Math.random() * 10) + 5.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Public API — price a full fixture

**Files:**
- Create: `packages/quant-engine/src/priceFixture.ts`
- Create: `packages/quant-engine/src/index.ts`
- Test: `packages/quant-engine/test/priceFixture.test.ts`

**Interfaces:**
- Consumes: every module from Tasks 2 to 7.
- Produces:

```ts
export interface PricingConfig {
  readonly rho: number;
  readonly targetBookSum: number;
  readonly totalsLines: readonly number[];
  readonly handicaps: readonly number[];
  readonly maxGoals: number;
}

export interface FixturePricing {
  readonly lambdas: MatchLambdas;
  readonly matrix: ScorelineMatrix;
  readonly markets: readonly PricedMarket[];
  readonly asianHandicaps: readonly AsianHandicapMarket[];
  readonly config: PricingConfig;
}

export const DEFAULT_PRICING_CONFIG: PricingConfig;
export function priceFixture(lambdas: MatchLambdas, config?: Partial<PricingConfig>): FixturePricing;
```

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/priceFixture.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { DEFAULT_PRICING_CONFIG, priceFixture } from "../src/priceFixture.js";

const TOL = 1e-9;
const LAMBDAS = { home: 1.6, away: 1.1 };

function market(pricing: ReturnType<typeof priceFixture>, key: string) {
  const found = pricing.markets.find((m) => m.key === key);
  if (!found) throw new Error(`missing market ${key}`);
  return found;
}

describe("priceFixture", () => {
  it("prices every configured market", () => {
    const p = priceFixture(LAMBDAS);
    const keys = p.markets.map((m) => m.key);
    expect(keys).toContain("1X2");
    expect(keys).toContain("BTTS");
    expect(keys).toContain("CS");
    expect(keys).toContain("DC");
    expect(keys).toContain("OU_2.5");
    expect(p.asianHandicaps.length).toBe(DEFAULT_PRICING_CONFIG.handicaps.length);
  });

  it("gives every exhaustive market the configured book sum", () => {
    for (const m of priceFixture(LAMBDAS).markets) {
      // Double chance is excluded deliberately and asserted separately below:
      // its selections each cover two outcomes, so a fair DC book sums to 2.
      if (m.key === "DC") continue;
      expect(Math.abs(m.bookSum - DEFAULT_PRICING_CONFIG.targetBookSum)).toBeLessThan(TOL);
    }
  });

  it("gives double chance exactly twice the target book sum", () => {
    // Each of 1X, 12 and X2 covers two of the three outcomes, so the DC book
    // is 2 x (the 1X2 book). Deriving DC from the margined 1X2 rather than
    // applying a margin to it directly is what makes this exact.
    const dc = market(priceFixture(LAMBDAS), "DC");
    expect(Math.abs(dc.bookSum - 2 * DEFAULT_PRICING_CONFIG.targetBookSum)).toBeLessThan(TOL);
  });

  it("prices double chance consistently with the margined 1X2 book", () => {
    const p = priceFixture(LAMBDAS);
    const x2 = market(p, "1X2");
    const dc = market(p, "DC");
    const impl = (m: typeof x2, key: string): number => {
      const found = m.selections.find((s) => s.key === key);
      return found ? 1 / found.odds : 0;
    };
    expect(impl(dc, "DC:1X")).toBeCloseTo(impl(x2, "1X2:HOME") + impl(x2, "1X2:DRAW"), 9);
    expect(impl(dc, "DC:X2")).toBeCloseTo(impl(x2, "1X2:DRAW") + impl(x2, "1X2:AWAY"), 9);
  });

  it("keeps derived markets consistent with one another", () => {
    const p = priceFixture(LAMBDAS);
    const x2 = market(p, "1X2");
    const dc = market(p, "DC");
    const home = x2.selections.find((s) => s.key === "1X2:HOME")?.probability ?? 0;
    const draw = x2.selections.find((s) => s.key === "1X2:DRAW")?.probability ?? 0;
    const oneX = dc.selections.find((s) => s.key === "DC:1X")?.probability ?? 0;
    expect(oneX).toBeCloseTo(home + draw, 9);
  });

  it("produces identical output when called twice", () => {
    expect(JSON.stringify(priceFixture(LAMBDAS))).toBe(
      JSON.stringify(priceFixture(LAMBDAS)),
    );
  });

  it("never returns a draw price above 6 for an evenly matched fixture", () => {
    // End-to-end regression guard for defect D1.
    const p = priceFixture({ home: 1.35, away: 1.25 });
    const draw = market(p, "1X2").selections.find((s) => s.key === "1X2:DRAW");
    expect(draw?.odds).toBeLessThan(6);
    expect(draw?.odds).toBeGreaterThan(1);
  });

  it("never returns a negative or non-finite price anywhere", () => {
    for (const lambdas of [
      { home: 0.4, away: 0.5 },
      { home: 3.8, away: 0.3 },
      { home: 2.2, away: 2.4 },
    ]) {
      for (const m of priceFixture(lambdas).markets) {
        for (const s of m.selections) {
          expect(s.probability).toBeGreaterThanOrEqual(0);
          if (s.probability > 1e-6) {
            expect(s.odds).toBeGreaterThan(1);
            expect(Number.isFinite(s.odds)).toBe(true);
          }
        }
      }
    }
  });

  it("honours a config override", () => {
    const p = priceFixture(LAMBDAS, { targetBookSum: 1.02, totalsLines: [1.5] });
    expect(Math.abs(market(p, "1X2").bookSum - 1.02)).toBeLessThan(TOL);
    expect(p.markets.filter((m) => m.key.startsWith("OU_"))).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/priceFixture.test.ts`
Expected: FAIL — cannot resolve `../src/priceFixture.js`.

- [ ] **Step 3: Write the implementation**

`packages/quant-engine/src/priceFixture.ts`:

```ts
import type {
  Market,
  MatchLambdas,
  PricedMarket,
  PricedSelection,
  ScorelineMatrix,
} from "./types.js";
import { buildScorelineMatrix } from "./poisson/dixonColes.js";
import {
  doubleChanceMarket,
  drawNoBetMarket,
  matchOddsMarket,
} from "./markets/matchOdds.js";
import { totalsMarket } from "./markets/totals.js";
import { bttsMarket } from "./markets/btts.js";
import { correctScoreMarket } from "./markets/correctScore.js";
import {
  asianHandicapMarket,
  type AsianHandicapMarket,
} from "./markets/asianHandicap.js";
import { applyOverround } from "./pricing/overround.js";

export interface PricingConfig {
  readonly rho: number;
  readonly targetBookSum: number;
  readonly totalsLines: readonly number[];
  readonly handicaps: readonly number[];
  readonly maxGoals: number;
}

export interface FixturePricing {
  readonly lambdas: MatchLambdas;
  readonly matrix: ScorelineMatrix;
  readonly markets: readonly PricedMarket[];
  readonly asianHandicaps: readonly AsianHandicapMarket[];
  readonly config: PricingConfig;
}

export const DEFAULT_PRICING_CONFIG: PricingConfig = {
  rho: 0.06,
  targetBookSum: 1.05,
  totalsLines: [0.5, 1.5, 2.5, 3.5, 4.5],
  handicaps: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
  maxGoals: 10,
};

/** Which 1X2 outcomes each double-chance selection covers. */
const DOUBLE_CHANCE_LEGS: Readonly<Record<string, readonly string[]>> = {
  "DC:1X": ["1X2:HOME", "1X2:DRAW"],
  "DC:12": ["1X2:HOME", "1X2:AWAY"],
  "DC:X2": ["1X2:DRAW", "1X2:AWAY"],
};

/**
 * Double chance cannot carry a book margin of its own. Each of its selections
 * covers two of the three outcomes, so a fair DC book sums to 2, and solving
 * the power method against a target of 1.05 would produce meaningless prices.
 *
 * Instead the margined 1X2 implied probabilities are summed pairwise. The DC
 * book therefore lands at exactly twice the 1X2 book, and the two markets
 * cannot drift apart.
 */
function priceDoubleChance(
  fair: Market,
  pricedMatchOdds: PricedMarket,
): PricedMarket {
  const impliedOf = (key: string): number => {
    const found = pricedMatchOdds.selections.find((s) => s.key === key);
    return found && Number.isFinite(found.odds) ? 1 / found.odds : 0;
  };

  const selections: PricedSelection[] = fair.selections.map((s) => {
    const implied = (DOUBLE_CHANCE_LEGS[s.key] ?? []).reduce(
      (acc, key) => acc + impliedOf(key),
      0,
    );
    return {
      ...s,
      odds: implied > 0 ? 1 / implied : Number.POSITIVE_INFINITY,
    };
  });

  return {
    key: fair.key,
    label: fair.label,
    selections,
    bookSum: selections.reduce(
      (acc, s) => acc + (Number.isFinite(s.odds) ? 1 / s.odds : 0),
      0,
    ),
  };
}

/**
 * Prices every supported market for one fixture from its expected goals.
 *
 * All markets are derived from a single scoreline matrix, so they cannot
 * disagree with one another. Nothing here is stochastic: the same lambdas and
 * config always produce byte-identical output.
 */
export function priceFixture(
  lambdas: MatchLambdas,
  config: Partial<PricingConfig> = {},
): FixturePricing {
  const resolved: PricingConfig = { ...DEFAULT_PRICING_CONFIG, ...config };
  const matrix = buildScorelineMatrix(lambdas, resolved.rho, resolved.maxGoals);

  const pricedMatchOdds = applyOverround(
    matchOddsMarket(matrix),
    resolved.targetBookSum,
  );

  const markets: PricedMarket[] = [
    pricedMatchOdds,
    priceDoubleChance(doubleChanceMarket(matrix), pricedMatchOdds),
    applyOverround(drawNoBetMarket(matrix), resolved.targetBookSum),
    applyOverround(bttsMarket(matrix), resolved.targetBookSum),
    applyOverround(correctScoreMarket(matrix), resolved.targetBookSum),
    ...resolved.totalsLines.map((line) =>
      applyOverround(totalsMarket(matrix, line), resolved.targetBookSum),
    ),
  ];

  const asianHandicaps = resolved.handicaps.map((h) =>
    asianHandicapMarket(matrix, h),
  );

  return { lambdas, matrix, markets, asianHandicaps, config: resolved };
}
```

- [ ] **Step 4: Write the package entry point**

`packages/quant-engine/src/index.ts`:

```ts
export type {
  Market,
  MatchLambdas,
  PricedMarket,
  PricedSelection,
  ScorelineMatrix,
  Selection,
} from "./types.js";
export { poissonPmf } from "./poisson/poisson.js";
export {
  buildScorelineMatrix,
  dixonColesTau,
  rhoBounds,
} from "./poisson/dixonColes.js";
export {
  doubleChanceMarket,
  drawNoBetMarket,
  matchOddsMarket,
  toFairOdds,
} from "./markets/matchOdds.js";
export { totalsMarket } from "./markets/totals.js";
export { bttsMarket } from "./markets/btts.js";
export { correctScoreMarket } from "./markets/correctScore.js";
export {
  asianHandicapMarket,
  type AsianHandicapMarket,
  type AsianHandicapOutcome,
} from "./markets/asianHandicap.js";
export {
  applyOverround,
  removeOverroundShin,
  solvePowerExponent,
} from "./pricing/overround.js";
export {
  assessValue,
  expectedValue,
  kellyFraction,
  type ValueAssessment,
} from "./pricing/kelly.js";
export {
  DEFAULT_PRICING_CONFIG,
  priceFixture,
  type FixturePricing,
  type PricingConfig,
} from "./priceFixture.js";
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/priceFixture.test.ts`
Expected: PASS, 7 tests.

- [ ] **Step 6: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add priceFixture entry point

One call turns two expected-goal values into every priced market, all derived
from a single scoreline matrix.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Determinism and purity guards

**Files:**
- Create: `packages/quant-engine/test/purity.test.ts`
- Modify: `packages/quant-engine/package.json` (add the `verify` script)

**Interfaces:**
- Consumes: the whole `src/` tree as text.
- Produces: no runtime exports. A test that fails the build if randomness or I/O enters the engine.

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/purity.test.ts`:

```ts
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { priceFixture } from "../src/priceFixture.js";

const SRC = fileURLToPath(new URL("../src", import.meta.url));

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return full.endsWith(".ts") ? [full] : [];
  });
}

describe("engine purity", () => {
  const files = sourceFiles(SRC);

  it("finds source files to scan", () => {
    expect(files.length).toBeGreaterThan(5);
  });

  it("contains no Math.random anywhere in src", () => {
    // Defect D3: the previous engine had Math.random in the pricing path, so
    // odds changed on every refresh and could be re-rolled by the bettor.
    const offenders = files.filter((f) => readFileSync(f, "utf8").includes("Math.random"));
    expect(offenders).toEqual([]);
  });

  it("contains no Date.now or new Date in src", () => {
    const offenders = files.filter((f) => {
      const text = readFileSync(f, "utf8");
      return text.includes("Date.now(") || text.includes("new Date(");
    });
    expect(offenders).toEqual([]);
  });

  it("imports no node builtins or network clients in src", () => {
    const banned = ["node:fs", "node:http", "node:https", "axios", "node-fetch", "fetch("];
    const offenders = files.filter((f) => {
      const text = readFileSync(f, "utf8");
      return banned.some((b) => text.includes(b));
    });
    expect(offenders).toEqual([]);
  });

  it("prices identically across 100 repeated calls", () => {
    const first = JSON.stringify(priceFixture({ home: 1.72, away: 1.03 }));
    for (let i = 0; i < 100; i += 1) {
      expect(JSON.stringify(priceFixture({ home: 1.72, away: 1.03 }))).toBe(first);
    }
  });
});
```

- [ ] **Step 2: Run the test to verify the behaviour**

Run: `cd packages/quant-engine && npx vitest run test/purity.test.ts`
Expected: PASS. If any test fails, remove the offending construct from `src/` — do not weaken the test.

- [ ] **Step 3: Add the verify script**

In `packages/quant-engine/package.json`, replace the `"scripts"` block with:

```json
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "typecheck": "tsc --noEmit",
    "verify": "npm run typecheck && npm run test"
  },
```

- [ ] **Step 4: Run the full verification**

Run: `cd packages/quant-engine && npm run verify`
Expected: typecheck clean, all tests pass across every test file.

- [ ] **Step 5: Commit**

```bash
git add packages/quant-engine
git commit -m "test(engine): guard determinism and purity

Fails the build if Math.random, wall-clock time, or I/O enters the engine.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Golden-file snapshot and README

**Files:**
- Create: `packages/quant-engine/test/golden.test.ts`
- Create: `packages/quant-engine/README.md`

**Interfaces:**
- Consumes: `priceFixture`.
- Produces: a committed snapshot that fails on any unintended change to pricing output.

- [ ] **Step 1: Write the snapshot test**

`packages/quant-engine/test/golden.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { priceFixture } from "../src/priceFixture.js";

const FIXTURES = [
  { name: "home favourite", lambdas: { home: 1.9, away: 0.9 } },
  { name: "even match", lambdas: { home: 1.35, away: 1.25 } },
  { name: "away favourite", lambdas: { home: 0.95, away: 1.75 } },
  { name: "low scoring", lambdas: { home: 0.7, away: 0.6 } },
  { name: "high scoring", lambdas: { home: 2.6, away: 2.2 } },
];

describe("golden pricing snapshots", () => {
  for (const { name, lambdas } of FIXTURES) {
    it(`prices "${name}" stably`, () => {
      const pricing = priceFixture(lambdas);
      const summary = pricing.markets.map((m) => ({
        key: m.key,
        bookSum: Number(m.bookSum.toFixed(6)),
        odds: m.selections.map((s) => ({
          key: s.key,
          odds: Number(s.odds.toFixed(4)),
        })),
      }));
      expect(summary).toMatchSnapshot();
    });
  }
});
```

- [ ] **Step 2: Generate the snapshot**

Run: `cd packages/quant-engine && npx vitest run test/golden.test.ts`
Expected: PASS, and `test/__snapshots__/golden.test.ts.snap` is written.

- [ ] **Step 3: Inspect the snapshot by hand**

Open `packages/quant-engine/test/__snapshots__/golden.test.ts.snap` and confirm:
- Every `bookSum` reads `1.05`.
- In "even match", the `1X2:DRAW` price is below 6 — this is the defect the rebuild exists to fix.
- In "home favourite", `1X2:HOME` is the shortest of the three prices.
- No price is negative, zero, or `null`.

If any of these fail, the bug is in the implementation, not the snapshot. Fix the source and regenerate.

- [ ] **Step 4: Write the README**

`packages/quant-engine/README.md`:

```markdown
# @yami/quant-engine

Pure TypeScript football pricing engine. No I/O, no database, no randomness.

## Model

    ratings -> attack/defence strengths -> lambda -> Dixon-Coles matrix -> markets -> prices

Independent Poisson under-counts low-scoring draws, which is why the previous
implementation produced draw odds above 6.0. The Dixon-Coles `tau` correction
adjusts the 0-0, 1-0, 0-1 and 1-1 cells to fix exactly that.

## Why one matrix

Every market is a sum over cells of a single 11x11 scoreline distribution:

| Market | Cells summed |
|---|---|
| 1X2 | `h>a`, `h==a`, `h<a` |
| Over/Under | `h+a > line` |
| BTTS | `h>0 && a>0` |
| Correct score | the single cell |
| Asian handicap | triangles shifted by the handicap |
| Double chance | unions of the 1X2 triangles |

Because they are all marginals of the same distribution, the markets cannot
contradict one another. The test suite asserts this directly.

## Margin

Fair odds are `1/p`. Margin is applied by the power method: solve for `k` such
that the sum of `p^k` equals the target book sum. A target above 1 shortens
every price, and longshots carry proportionally more margin than favourites.

Shin's method is provided as the inverse, for de-margining historical closing
odds when backtesting.

## Usage

```ts
import { priceFixture } from "@yami/quant-engine";

const pricing = priceFixture({ home: 1.6, away: 1.1 });
const matchResult = pricing.markets.find((m) => m.key === "1X2");
```

## Verify

    npm run verify
```

- [ ] **Step 5: Run the full suite**

Run: `cd packages/quant-engine && npm run verify`
Expected: typecheck clean, all tests pass.

- [ ] **Step 6: Commit**

```bash
git add packages/quant-engine
git commit -m "test(engine): add golden pricing snapshots and README

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Definition of done

- [ ] `cd packages/quant-engine && npm run verify` passes
- [ ] `grep -rn "Math.random" packages/quant-engine/src` returns nothing
- [ ] Every priced two-way and three-way market has `bookSum` of exactly 1.05
- [ ] An evenly matched fixture prices the draw below 6.0
- [ ] Applying margin shortens prices relative to fair odds
- [ ] Correct-score cells aggregated over the lower triangle equal the 1X2 home probability

## Self-review notes

**Spec coverage.** Spec §3.3 to §3.8 are covered by Tasks 2 to 8. §3.10 by Task 9.
§3.1, §3.2 (ratings and MLE fit) and §3.9 (calibration) are deliberately deferred
to Plan 2, because they need the historical data pipeline. Until then callers
supply lambdas directly, which is what makes this plan independently testable.

**Deviation from the spec, recorded deliberately.** Spec §3.7 names Shin's method
for applying the margin. That is wrong: Shin is an inverse method that recovers
true probabilities from bookmaker odds. Task 6 applies margin by the power method
and keeps Shin for de-margining historical closing odds in Plan 2's backtest.
Spec §3.7 should be amended to match.

**Pre-flight fix, recorded.** The first draft of Task 8 called
`applyOverround(doubleChanceMarket(matrix), 1.05)`, and its test skipped DC and CS
from the book-sum assertion. That was wrong twice over: a fair double-chance book
sums to 2, not 1, so solving the power method against 1.05 produces meaningless
prices — and skipping DC in the assertion hid the error rather than catching it.
Task 8 now derives DC by summing the margined 1X2 implied probabilities pairwise,
which puts the DC book at exactly twice the 1X2 book, and the test asserts that
relationship explicitly. CS is exhaustive, so it is now included in the 1.05
assertion as it always should have been.

**Type consistency.** `MatchLambdas`, `ScorelineMatrix`, `Market`, `Selection`,
`PricedMarket` and `PricedSelection` are defined once in Task 2's `types.ts` and
imported unchanged thereafter. `sumWhere` and `toFairOdds` are defined in Task 3
and reused by Tasks 4 and 5. Selection key formats are fixed in Tasks 3 to 5 and
asserted in Task 8.
