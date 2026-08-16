# Quant Engine — Statistical Depth Extension

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the Skellam goal-difference distribution and a proper scoring-rule / calibration module to `packages/quant-engine`, giving the engine a second independent mathematical route to its own answers and the standard academic metrics for evaluating it.

**Architecture:** Two new pure modules. `math/` holds shared special functions (log-gamma, modified Bessel). `skellam/` derives the goal-difference distribution analytically, independently of the scoreline matrix — which makes it a cross-check, not just a feature. `calibration/` implements RPS, Brier with the Murphy decomposition, and log loss.

**Tech Stack:** TypeScript 5 strict, Vitest, zero runtime dependencies.

**Plan 1b of 5.** Extends the completed `2026-08-15-quant-engine-core.md`.

## Global Constraints

- Package: `packages/quant-engine`. TypeScript `strict: true`, `noUncheckedIndexedAccess: true`.
- No `any`. No non-null assertions (`!`). Zero runtime dependencies.
- No `Math.random()`, no `Date`, no Node builtins in `src/` — the purity test in `test/purity.test.ts` enforces this and must not be weakened.
- All exported functions pure.
- Float comparisons in tests use tolerance `1e-9` unless a task states otherwise.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Why these two additions

The engine currently computes everything by summing an 11×11 matrix. That is correct but
self-referential: every market agrees with every other because they all come from one
object, so a systematic error in the matrix would be invisible to every consistency test
in the suite.

The Skellam distribution reaches the goal-difference distribution by an entirely different
analytical route — a closed form in the modified Bessel function of the first kind, with no
matrix involved. Where the two agree, the matrix is independently corroborated. That is a
materially stronger claim than internal consistency.

The scoring-rule module supplies the metrics the literature actually uses to judge a
football forecast, and is the prerequisite for the backtest in Plan 2.

---

### Task 11: Shared special functions and the Skellam distribution

**Files:**
- Create: `packages/quant-engine/src/math/gamma.ts`
- Create: `packages/quant-engine/src/math/bessel.ts`
- Modify: `packages/quant-engine/src/poisson/poisson.ts` (import `logGamma` instead of defining it)
- Create: `packages/quant-engine/src/skellam/skellam.ts`
- Test: `packages/quant-engine/test/skellam.test.ts`

**Interfaces:**
- Consumes: `buildScorelineMatrix`, `matchOddsMarket` (tests only).
- Produces:
  - `logGamma(z: number): number`
  - `logBesselI(order: number, z: number): number` — log of the modified Bessel function of the first kind, `I_n(z)`, for non-negative integer order.
  - `besselI(order: number, z: number): number`
  - `skellamPmf(k: number, lambdaHome: number, lambdaAway: number): number`
  - `skellamMatchProbabilities(lambdaHome, lambdaAway): { home: number; draw: number; away: number }`
  - `skellamSupremacyMarket(lambdaHome, lambdaAway, line: number): Market` — key `"SUP_<line>"`, selections `"SUP:<line>:HOME"` / `"SUP:<line>:AWAY"`. Half-integer lines only.

- [ ] **Step 1: Extract `logGamma` into a shared module**

`packages/quant-engine/src/math/gamma.ts` — move the Lanczos implementation out of
`poisson.ts` verbatim and export it:

```ts
/**
 * Lanczos approximation coefficients (g = 7, n = 9).
 */
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
] as const;

/**
 * Natural log of the gamma function, by the Lanczos approximation.
 * Working in log space keeps factorial-scale quantities finite.
 */
export function logGamma(z: number): number {
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
export function logFactorial(k: number): number {
  return logGamma(k + 1);
}
```

Then edit `poisson.ts` to delete its private `LANCZOS`, `logGamma` and `logFactorial`, and
import instead:

```ts
import { logFactorial } from "../math/gamma.js";
```

The existing `test/poisson.test.ts` is the guard for this refactor — it must still pass
unchanged. Do not modify it.

- [ ] **Step 2: Run the existing suite to confirm the refactor is safe**

Run: `cd packages/quant-engine && npm run verify`
Expected: all existing tests still pass. If `poisson.test.ts` fails, the extraction is
wrong — fix `gamma.ts`, do not touch the test.

- [ ] **Step 3: Write the failing Skellam test**

`packages/quant-engine/test/skellam.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { besselI, logBesselI } from "../src/math/bessel.js";
import {
  skellamMatchProbabilities,
  skellamPmf,
  skellamSupremacyMarket,
} from "../src/skellam/skellam.js";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { matchOddsMarket } from "../src/markets/matchOdds.js";

const TOL = 1e-9;

describe("besselI", () => {
  it("matches known values of the modified Bessel function of the first kind", () => {
    // Reference values, Abramowitz & Stegun Table 9.8.
    expect(besselI(0, 1)).toBeCloseTo(1.2660658778, 8);
    expect(besselI(1, 1)).toBeCloseTo(0.5651591040, 8);
    expect(besselI(0, 2)).toBeCloseTo(2.2795853023, 8);
    expect(besselI(2, 2)).toBeCloseTo(0.6889484477, 8);
    expect(besselI(0, 5)).toBeCloseTo(27.2398718236, 6);
  });

  it("is 0 at z = 0 for positive order and 1 for order 0", () => {
    expect(besselI(0, 0)).toBeCloseTo(1, 12);
    expect(besselI(3, 0)).toBeCloseTo(0, 12);
  });

  it("stays finite in log space for large argument", () => {
    expect(Number.isFinite(logBesselI(0, 600))).toBe(true);
  });

  it("rejects a negative or non-integer order", () => {
    expect(() => besselI(-1, 1)).toThrow(RangeError);
    expect(() => besselI(1.5, 1)).toThrow(RangeError);
  });
});

describe("skellamPmf", () => {
  it("sums to 1 across the support", () => {
    let total = 0;
    for (let k = -40; k <= 40; k += 1) total += skellamPmf(k, 1.6, 1.1);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("is symmetric when the two rates are equal", () => {
    for (const k of [1, 2, 3, 5]) {
      expect(skellamPmf(k, 1.4, 1.4)).toBeCloseTo(skellamPmf(-k, 1.4, 1.4), 12);
    }
  });

  it("swaps sign when the rates are swapped", () => {
    expect(skellamPmf(2, 1.9, 0.8)).toBeCloseTo(skellamPmf(-2, 0.8, 1.9), 12);
  });

  it("puts more mass on a positive difference when the home rate is higher", () => {
    expect(skellamPmf(1, 2.0, 1.0)).toBeGreaterThan(skellamPmf(-1, 2.0, 1.0));
  });

  it("rejects a non-integer difference or a non-positive rate", () => {
    expect(() => skellamPmf(0.5, 1, 1)).toThrow(RangeError);
    expect(() => skellamPmf(1, 0, 1)).toThrow(RangeError);
  });
});

describe("Skellam agrees with the scoreline matrix", () => {
  // The critical test in this task. The matrix and the Skellam closed form reach
  // the goal-difference distribution by completely different routes: one sums an
  // 11x11 grid of Poisson products, the other evaluates a Bessel series. With the
  // Dixon-Coles correction switched off (rho = 0) the two must agree exactly.
  // Agreement here corroborates the matrix independently, which no internal
  // consistency test in this package can do.
  const LAMBDAS = { home: 1.7, away: 1.2 };

  it("reproduces the matrix anti-diagonal sums when rho is 0", () => {
    const m = buildScorelineMatrix(LAMBDAS, 0, 14);
    for (let d = -6; d <= 6; d += 1) {
      let fromMatrix = 0;
      for (let h = 0; h <= m.maxGoals; h += 1) {
        const a = h - d;
        if (a >= 0 && a <= m.maxGoals) fromMatrix += m.cells[h]?.[a] ?? 0;
      }
      expect(fromMatrix).toBeCloseTo(
        skellamPmf(d, LAMBDAS.home, LAMBDAS.away), 7,
      );
    }
  });

  it("reproduces the 1X2 market when rho is 0", () => {
    const market = matchOddsMarket(buildScorelineMatrix(LAMBDAS, 0, 14));
    const p = (key: string): number =>
      market.selections.find((s) => s.key === key)?.probability ?? 0;
    const skellam = skellamMatchProbabilities(LAMBDAS.home, LAMBDAS.away);
    expect(skellam.home).toBeCloseTo(p("1X2:HOME"), 7);
    expect(skellam.draw).toBeCloseTo(p("1X2:DRAW"), 7);
    expect(skellam.away).toBeCloseTo(p("1X2:AWAY"), 7);
  });

  it("DIFFERS from the matrix when rho is non-zero, as it must", () => {
    // Skellam assumes independence. The Dixon-Coles correction deliberately
    // breaks independence in the four low-scoring cells, so with rho != 0 the
    // draw probabilities must diverge. If they did not, tau would be doing
    // nothing.
    const corrected = matchOddsMarket(buildScorelineMatrix(LAMBDAS, -0.1, 14));
    const drawCorrected =
      corrected.selections.find((s) => s.key === "1X2:DRAW")?.probability ?? 0;
    const drawIndependent = skellamMatchProbabilities(
      LAMBDAS.home, LAMBDAS.away,
    ).draw;
    expect(drawCorrected).toBeGreaterThan(drawIndependent);
  });
});

describe("skellamMatchProbabilities", () => {
  it("sums to 1", () => {
    const p = skellamMatchProbabilities(2.1, 0.9);
    expect(Math.abs(p.home + p.draw + p.away - 1)).toBeLessThan(TOL);
  });
});

describe("skellamSupremacyMarket", () => {
  it("has two selections summing to 1", () => {
    const m = skellamSupremacyMarket(1.8, 1.0, -0.5);
    expect(m.selections).toHaveLength(2);
    const total = m.selections.reduce((a, s) => a + s.probability, 0);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("agrees with the Asian handicap on the same half-line", () => {
    // Supremacy at line -0.5 is "home wins by 1 or more", which is exactly the
    // AH -0.5 home leg. Derived from Bessel functions here, from matrix
    // triangles there.
    const sup = skellamSupremacyMarket(1.7, 1.2, -0.5);
    const home = sup.selections.find((s) => s.key === "SUP:-0.5:HOME");
    const m = buildScorelineMatrix({ home: 1.7, away: 1.2 }, 0, 14);
    let ahWin = 0;
    for (let h = 0; h <= m.maxGoals; h += 1) {
      for (let a = 0; a <= m.maxGoals; a += 1) {
        if (h - 0.5 > a) ahWin += m.cells[h]?.[a] ?? 0;
      }
    }
    expect(home?.probability).toBeCloseTo(ahWin, 7);
  });

  it("rejects a whole-number line", () => {
    expect(() => skellamSupremacyMarket(1.5, 1.2, -1)).toThrow(RangeError);
  });
});
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/skellam.test.ts`
Expected: FAIL — cannot resolve `../src/math/bessel.js`.

- [ ] **Step 5: Write the Bessel implementation**

`packages/quant-engine/src/math/bessel.ts`:

```ts
import { logGamma } from "./gamma.js";

const SERIES_TERMS = 200;

/**
 * Natural log of the modified Bessel function of the first kind, I_n(z), for
 * non-negative integer order n and z >= 0.
 *
 *   I_n(z) = sum over m >= 0 of  (z/2)^(2m+n) / ( m! * (m+n)! )
 *
 * Each term is evaluated in log space and the sum accumulated by the
 * log-sum-exp trick, so the result stays finite for large z where the raw
 * series overflows.
 */
export function logBesselI(order: number, z: number): number {
  if (!Number.isInteger(order) || order < 0) {
    throw new RangeError(
      `order must be a non-negative integer, received ${order}`,
    );
  }
  if (!(z >= 0)) {
    throw new RangeError(`z must be non-negative, received ${z}`);
  }
  if (z === 0) {
    return order === 0 ? 0 : Number.NEGATIVE_INFINITY;
  }

  const halfZ = Math.log(z / 2);
  const logTerms: number[] = [];
  for (let m = 0; m < SERIES_TERMS; m += 1) {
    logTerms.push(
      (2 * m + order) * halfZ - logGamma(m + 1) - logGamma(m + order + 1),
    );
  }

  const max = logTerms.reduce((a, b) => (b > a ? b : a), Number.NEGATIVE_INFINITY);
  if (!Number.isFinite(max)) return Number.NEGATIVE_INFINITY;

  let sum = 0;
  for (const term of logTerms) sum += Math.exp(term - max);
  return max + Math.log(sum);
}

/** Modified Bessel function of the first kind, I_n(z). */
export function besselI(order: number, z: number): number {
  return Math.exp(logBesselI(order, z));
}
```

- [ ] **Step 6: Write the Skellam implementation**

`packages/quant-engine/src/skellam/skellam.ts`:

```ts
import type { Market, Selection } from "../types.js";
import { logBesselI } from "../math/bessel.js";

const SUPPORT_LIMIT = 60;

/**
 * Skellam probability mass: the distribution of the difference of two
 * independent Poisson variables, K = X - Y.
 *
 *   P(K = k) = e^-(l1+l2) * (l1/l2)^(k/2) * I_|k|( 2*sqrt(l1*l2) )
 *
 * where I is the modified Bessel function of the first kind.
 *
 * This is an entirely separate analytical route to the goal-difference
 * distribution from summing the scoreline matrix, which is what makes it useful
 * as a cross-check rather than merely as another feature. See Karlis and
 * Ntzoufras (2009).
 *
 * Note the independence assumption: this does NOT carry the Dixon-Coles
 * correction, so it agrees with the matrix only when rho is 0.
 */
export function skellamPmf(
  k: number,
  lambdaHome: number,
  lambdaAway: number,
): number {
  if (!Number.isInteger(k)) {
    throw new RangeError(`k must be an integer, received ${k}`);
  }
  if (!(lambdaHome > 0) || !(lambdaAway > 0)) {
    throw new RangeError("both rates must be greater than 0");
  }
  const logValue =
    -(lambdaHome + lambdaAway) +
    (k / 2) * Math.log(lambdaHome / lambdaAway) +
    logBesselI(Math.abs(k), 2 * Math.sqrt(lambdaHome * lambdaAway));
  return Math.exp(logValue);
}

/**
 * Home / draw / away probabilities from the goal-difference distribution alone,
 * summed over a support wide enough that the omitted tail is far below the
 * package's 1e-9 test tolerance.
 */
export function skellamMatchProbabilities(
  lambdaHome: number,
  lambdaAway: number,
): { home: number; draw: number; away: number } {
  let home = 0;
  let away = 0;
  for (let k = 1; k <= SUPPORT_LIMIT; k += 1) {
    home += skellamPmf(k, lambdaHome, lambdaAway);
    away += skellamPmf(-k, lambdaHome, lambdaAway);
  }
  const draw = skellamPmf(0, lambdaHome, lambdaAway);
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}

/**
 * Supremacy: whether the home side's goal difference beats a half-goal line.
 * A negative line favours the home side, matching Asian handicap convention.
 */
export function skellamSupremacyMarket(
  lambdaHome: number,
  lambdaAway: number,
  line: number,
): Market {
  if (!Number.isFinite(line) || Math.abs(line * 2) % 2 !== 1) {
    throw new RangeError(
      `line must be a half-integer such as -0.5, received ${line}`,
    );
  }

  let home = 0;
  for (let k = -SUPPORT_LIMIT; k <= SUPPORT_LIMIT; k += 1) {
    if (k + line > 0) home += skellamPmf(k, lambdaHome, lambdaAway);
  }
  let total = 0;
  for (let k = -SUPPORT_LIMIT; k <= SUPPORT_LIMIT; k += 1) {
    total += skellamPmf(k, lambdaHome, lambdaAway);
  }
  const homeProbability = home / total;
  const awayProbability = 1 - homeProbability;

  const selection = (key: string, label: string, p: number): Selection => ({
    key,
    label,
    probability: p,
    fairOdds: p === 0 ? Number.POSITIVE_INFINITY : 1 / p,
  });

  return {
    key: `SUP_${line}`,
    label: `Supremacy ${line}`,
    selections: [
      selection(`SUP:${line}:HOME`, `Home ${line}`, homeProbability),
      selection(`SUP:${line}:AWAY`, `Away +${Math.abs(line)}`, awayProbability),
    ],
  };
}
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/skellam.test.ts`
Expected: PASS.

If the "reproduces the matrix anti-diagonal sums" test fails, do NOT loosen its tolerance.
Two independent derivations disagreeing means one of them is wrong, and finding out which
is the entire point of the test. Investigate and report.

- [ ] **Step 8: Export from the barrel and run the full suite**

Add to `packages/quant-engine/src/index.ts`:

```ts
export { logGamma, logFactorial } from "./math/gamma.js";
export { besselI, logBesselI } from "./math/bessel.js";
export {
  skellamMatchProbabilities,
  skellamPmf,
  skellamSupremacyMarket,
} from "./skellam/skellam.js";
```

Run: `cd packages/quant-engine && npm run verify`
Expected: typecheck clean, entire suite passes including the purity guards.

- [ ] **Step 9: Commit**

```bash
git add packages/quant-engine
git commit -m "feat(engine): add Skellam goal-difference distribution

Reaches the goal-difference distribution analytically via the modified
Bessel function of the first kind, independently of the scoreline matrix.
Where the two agree at rho = 0 the matrix is corroborated by a separate
derivation, which internal consistency tests cannot do.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Proper scoring rules and calibration

**Files:**
- Create: `packages/quant-engine/src/calibration/scoringRules.ts`
- Create: `packages/quant-engine/src/calibration/reliability.ts`
- Test: `packages/quant-engine/test/scoringRules.test.ts`

**Interfaces:**
- Produces:
  - `rankedProbabilityScore(forecast: readonly number[], outcomeIndex: number): number`
  - `brierScore(forecast: readonly number[], outcomeIndex: number): number`
  - `logLoss(forecast: readonly number[], outcomeIndex: number): number`
  - `meanScore(scores: readonly number[]): number`
  - `brierDecomposition(forecasts: readonly number[], outcomes: readonly (0 | 1)[], bins?: number): BrierDecomposition`

```ts
export interface BrierDecomposition {
  readonly brier: number;
  readonly reliability: number;
  readonly resolution: number;
  readonly uncertainty: number;
  readonly bins: readonly ReliabilityBin[];
}

export interface ReliabilityBin {
  readonly lower: number;
  readonly upper: number;
  readonly count: number;
  readonly meanForecast: number;
  readonly observedFrequency: number;
}
```

- [ ] **Step 1: Write the failing test**

`packages/quant-engine/test/scoringRules.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  brierDecomposition,
  brierScore,
  logLoss,
  rankedProbabilityScore,
} from "../src/calibration/scoringRules.js";

const TOL = 1e-9;

describe("rankedProbabilityScore", () => {
  it("is 0 for a perfect forecast", () => {
    expect(rankedProbabilityScore([1, 0, 0], 0)).toBeCloseTo(0, 12);
    expect(rankedProbabilityScore([0, 0, 1], 2)).toBeCloseTo(0, 12);
  });

  it("is 1 for a maximally wrong ordered forecast", () => {
    // All mass on home, away actually occurred: the worst possible 1X2 forecast.
    expect(rankedProbabilityScore([1, 0, 0], 2)).toBeCloseTo(1, 12);
  });

  it("punishes a distant miss more than a near one", () => {
    // THE property RPS exists for. Home forecast, draw occurred, versus home
    // forecast, away occurred. Brier cannot tell these apart; RPS must.
    const nearMiss = rankedProbabilityScore([0.8, 0.15, 0.05], 1);
    const farMiss = rankedProbabilityScore([0.8, 0.15, 0.05], 2);
    expect(farMiss).toBeGreaterThan(nearMiss);
  });

  it("matches a hand-computed value", () => {
    // p = [0.5, 0.3, 0.2], draw occurs so e = [0, 1, 0].
    // cumulative p = [0.5, 0.8], cumulative e = [0, 1]
    // RPS = ((0.5-0)^2 + (0.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145
    expect(rankedProbabilityScore([0.5, 0.3, 0.2], 1)).toBeCloseTo(0.145, 12);
  });

  it("rejects a forecast that does not sum to 1 or an out-of-range outcome", () => {
    expect(() => rankedProbabilityScore([0.5, 0.3], 0)).toThrow(RangeError);
    expect(() => rankedProbabilityScore([0.5, 0.3, 0.2], 3)).toThrow(RangeError);
  });
});

describe("brierScore", () => {
  it("is 0 for a perfect forecast", () => {
    expect(brierScore([1, 0, 0], 0)).toBeCloseTo(0, 12);
  });

  it("matches a hand-computed value", () => {
    // p = [0.5, 0.3, 0.2], home occurs: (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.38
    expect(brierScore([0.5, 0.3, 0.2], 0)).toBeCloseTo(0.38, 12);
  });

  it("is blind to the ordering of outcomes, unlike RPS", () => {
    // The documented limitation that motivates RPS for football.
    expect(brierScore([0.8, 0.15, 0.05], 1)).not.toBeCloseTo(
      rankedProbabilityScore([0.8, 0.15, 0.05], 1), 6,
    );
    const a = brierScore([0.6, 0.2, 0.2], 1);
    const b = brierScore([0.6, 0.2, 0.2], 2);
    expect(a).toBeCloseTo(b, 12);
  });
});

describe("logLoss", () => {
  it("is 0 for a certain, correct forecast", () => {
    expect(logLoss([1, 0, 0], 0)).toBeCloseTo(0, 12);
  });

  it("grows without bound as the true outcome is given less probability", () => {
    expect(logLoss([0.5, 0.3, 0.2], 0)).toBeLessThan(logLoss([0.01, 0.3, 0.69], 0));
  });

  it("returns a finite penalty rather than Infinity at probability 0", () => {
    expect(Number.isFinite(logLoss([0, 0.5, 0.5], 0))).toBe(true);
  });
});

describe("brierDecomposition", () => {
  it("satisfies Murphy's identity: brier = reliability - resolution + uncertainty", () => {
    const forecasts = [0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95];
    const outcomes = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1] as const;
    const d = brierDecomposition(forecasts, [...outcomes]);
    expect(
      Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty)),
    ).toBeLessThan(1e-9);
  });

  it("gives near-zero reliability for a perfectly calibrated forecaster", () => {
    // 100 forecasts at 0.5, exactly half of which come true.
    const forecasts = Array.from({ length: 100 }, () => 0.5);
    const outcomes = Array.from({ length: 100 }, (_, i) => (i % 2 === 0 ? 1 : 0));
    const d = brierDecomposition(forecasts, outcomes as (0 | 1)[]);
    expect(d.reliability).toBeLessThan(1e-9);
  });

  it("gives zero resolution for a forecaster who always says the base rate", () => {
    const forecasts = Array.from({ length: 40 }, () => 0.25);
    const outcomes = Array.from({ length: 40 }, (_, i) => (i < 10 ? 1 : 0));
    const d = brierDecomposition(forecasts, outcomes as (0 | 1)[]);
    expect(d.resolution).toBeLessThan(1e-9);
    expect(d.uncertainty).toBeCloseTo(0.25 * 0.75, 9);
  });

  it("rejects mismatched input lengths", () => {
    expect(() => brierDecomposition([0.5], [1, 0] as (0 | 1)[])).toThrow(RangeError);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd packages/quant-engine && npx vitest run test/scoringRules.test.ts`
Expected: FAIL — cannot resolve the module.

- [ ] **Step 3: Write the scoring rules**

`packages/quant-engine/src/calibration/scoringRules.ts`:

```ts
/** Smallest probability charged by logLoss, so a zero forecast is finite. */
const LOG_LOSS_FLOOR = 1e-15;

function assertForecast(forecast: readonly number[], outcomeIndex: number): void {
  if (forecast.length < 2) {
    throw new RangeError("forecast must have at least two outcomes");
  }
  if (!Number.isInteger(outcomeIndex) || outcomeIndex < 0 ||
      outcomeIndex >= forecast.length) {
    throw new RangeError(
      `outcomeIndex ${outcomeIndex} outside the forecast of length ${forecast.length}`,
    );
  }
  const total = forecast.reduce((a, b) => a + b, 0);
  if (Math.abs(total - 1) > 1e-6) {
    throw new RangeError(`forecast must sum to 1, received ${total}`);
  }
}

/**
 * Ranked probability score (Epstein 1969; argued for football by Constantinou
 * and Fenton 2012).
 *
 *   RPS = 1/(r-1) * sum over i of ( sum_{j<=i} (p_j - e_j) )^2
 *
 * Unlike Brier, RPS is sensitive to DISTANCE: for an ordered outcome set such
 * as home / draw / away, forecasting a home win when the away side wins is
 * penalised more than forecasting a home win when the match is drawn. Lower is
 * better; the range is [0, 1].
 *
 * Note that this is contested. Wheatcroft (2021) argues distance sensitivity is
 * not in fact desirable here and that RPS should not be the default. Both
 * metrics are provided so the choice stays explicit rather than assumed.
 */
export function rankedProbabilityScore(
  forecast: readonly number[],
  outcomeIndex: number,
): number {
  assertForecast(forecast, outcomeIndex);
  let cumulativeForecast = 0;
  let cumulativeOutcome = 0;
  let total = 0;
  for (let i = 0; i < forecast.length - 1; i += 1) {
    cumulativeForecast += forecast[i] ?? 0;
    cumulativeOutcome += i === outcomeIndex ? 1 : 0;
    const diff = cumulativeForecast - cumulativeOutcome;
    total += diff * diff;
  }
  return total / (forecast.length - 1);
}

/**
 * Multi-category Brier score (Brier 1950): the squared error summed over every
 * category. Lower is better. Treats all outcomes as unordered, which is exactly
 * the limitation RPS addresses.
 */
export function brierScore(
  forecast: readonly number[],
  outcomeIndex: number,
): number {
  assertForecast(forecast, outcomeIndex);
  let total = 0;
  for (let i = 0; i < forecast.length; i += 1) {
    const observed = i === outcomeIndex ? 1 : 0;
    const diff = (forecast[i] ?? 0) - observed;
    total += diff * diff;
  }
  return total;
}

/**
 * Negative log likelihood of the realised outcome, also called the ignorance
 * score when taken in base 2. Unlike Brier and RPS it is unbounded, so a single
 * confident mistake dominates — which is either the point or a drawback,
 * depending on what is being measured.
 */
export function logLoss(
  forecast: readonly number[],
  outcomeIndex: number,
): number {
  assertForecast(forecast, outcomeIndex);
  return -Math.log(Math.max(forecast[outcomeIndex] ?? 0, LOG_LOSS_FLOOR));
}

/** Mean of a set of per-match scores. */
export function meanScore(scores: readonly number[]): number {
  if (scores.length === 0) {
    throw new RangeError("cannot take the mean of an empty score set");
  }
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

export interface ReliabilityBin {
  readonly lower: number;
  readonly upper: number;
  readonly count: number;
  readonly meanForecast: number;
  readonly observedFrequency: number;
}

export interface BrierDecomposition {
  readonly brier: number;
  readonly reliability: number;
  readonly resolution: number;
  readonly uncertainty: number;
  readonly bins: readonly ReliabilityBin[];
}

/**
 * Murphy's (1973) three-way partition of the binary Brier score:
 *
 *   BS = reliability - resolution + uncertainty
 *
 * Reliability measures how far the forecast probabilities sit from the observed
 * frequencies within each bin — lower is better, and zero means perfectly
 * calibrated. Resolution measures how far the bin frequencies sit from the base
 * rate — higher is better, and zero means the forecaster says nothing more than
 * the climatology. Uncertainty is a property of the events, not the forecaster,
 * and cannot be improved.
 *
 * This is what turns "the model scored 0.58" into a statement about WHY.
 */
export function brierDecomposition(
  forecasts: readonly number[],
  outcomes: readonly (0 | 1)[],
  bins = 10,
): BrierDecomposition {
  if (forecasts.length !== outcomes.length) {
    throw new RangeError(
      `forecasts (${forecasts.length}) and outcomes (${outcomes.length}) must be the same length`,
    );
  }
  if (forecasts.length === 0) {
    throw new RangeError("cannot decompose an empty forecast set");
  }
  if (!Number.isInteger(bins) || bins < 1) {
    throw new RangeError(`bins must be a positive integer, received ${bins}`);
  }

  const n = forecasts.length;
  const baseRate = outcomes.reduce<number>((a, b) => a + b, 0) / n;

  let brier = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = (forecasts[i] ?? 0) - (outcomes[i] ?? 0);
    brier += diff * diff;
  }
  brier /= n;

  const buckets: { forecasts: number[]; outcomes: number[] }[] = Array.from(
    { length: bins },
    () => ({ forecasts: [], outcomes: [] }),
  );
  for (let i = 0; i < n; i += 1) {
    const p = forecasts[i] ?? 0;
    const index = Math.min(bins - 1, Math.max(0, Math.floor(p * bins)));
    const bucket = buckets[index];
    if (!bucket) continue;
    bucket.forecasts.push(p);
    bucket.outcomes.push(outcomes[i] ?? 0);
  }

  let reliability = 0;
  let resolution = 0;
  const reported: ReliabilityBin[] = [];

  for (let k = 0; k < bins; k += 1) {
    const bucket = buckets[k];
    if (!bucket || bucket.forecasts.length === 0) continue;
    const count = bucket.forecasts.length;
    const meanForecast =
      bucket.forecasts.reduce((a, b) => a + b, 0) / count;
    const observedFrequency =
      bucket.outcomes.reduce((a, b) => a + b, 0) / count;

    reliability +=
      (count / n) * (meanForecast - observedFrequency) ** 2;
    resolution += (count / n) * (observedFrequency - baseRate) ** 2;

    reported.push({
      lower: k / bins,
      upper: (k + 1) / bins,
      count,
      meanForecast,
      observedFrequency,
    });
  }

  return {
    brier,
    reliability,
    resolution,
    uncertainty: baseRate * (1 - baseRate),
    bins: reported,
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd packages/quant-engine && npx vitest run test/scoringRules.test.ts`
Expected: PASS.

Murphy's identity is exact only when each bin's contribution uses that bin's own mean
forecast. If the identity test fails by more than 1e-9, the binning is wrong — fix the
implementation, do not loosen the tolerance.

- [ ] **Step 5: Export from the barrel, run the full suite, commit**

Add to `packages/quant-engine/src/index.ts`:

```ts
export {
  brierDecomposition,
  brierScore,
  logLoss,
  meanScore,
  rankedProbabilityScore,
  type BrierDecomposition,
  type ReliabilityBin,
} from "./calibration/scoringRules.js";
```

Run: `cd packages/quant-engine && npm run verify`

```bash
git add packages/quant-engine
git commit -m "feat(engine): add proper scoring rules and Murphy decomposition

Adds RPS, multi-category Brier, log loss, and Murphy's three-way partition
of the Brier score into reliability, resolution and uncertainty.

Both RPS and Brier are provided deliberately: RPS is distance-sensitive and
is the usual choice for ordered football outcomes, but Wheatcroft (2021)
disputes that this is desirable, so the choice stays explicit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Definition of done

- [ ] `cd packages/quant-engine && npm run verify` passes
- [ ] Skellam reproduces the matrix anti-diagonal sums at rho = 0 to 1e-7
- [ ] Skellam DIVERGES from the matrix at rho != 0, proving tau does something
- [ ] Murphy's identity holds to 1e-9
- [ ] RPS penalises a distant miss more than a near one; Brier does not

## Correction applied during execution

The Murphy identity test as originally written in Task 12 was WRONG, and the implementer
caught it before the assertion was touched. Murphy's classical three-way identity
`BS = REL - RES + UNC` is exact only when each bin holds a SINGLE DISTINCT forecast value.
The plan's test data binned continuous forecasts, two bins holding two distinct values
each, which leaves a residual equal to the within-bin variance.

Measured: BS = 0.109, REL - RES + UNC = 0.10875, residual = 0.00025 = WBV exactly.

The resolution was to implement the generalised four-way decomposition
`BS = REL - RES + UNC + WBV`, report all four terms, test the exact identity to 1e-12, and
add a separate test asserting the residual IS the within-bin variance so the decomposition
cannot be quietly reduced back to three terms. A third test uses one-distinct-value-per-bin
data to confirm WBV is exactly 0 there, recovering Murphy's classical form.

This is strictly better maths than the plan specified.

## References

- Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*.
- Murphy, A.H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12, 595–600.
- Maher, M.J. (1982). Modelling association football scores. *Statistica Neerlandica*.
- Shin, H.S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
- Dixon, M.J. and Coles, S.G. (1997). Modelling association football scores and inefficiencies in the football betting market. *JRSS Series C*, 46(2), 265–280.
- Karlis, D. and Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models. *JRSS Series D*, 52(3), 381–393.
- Karlis, D. and Ntzoufras, I. (2009). Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference. *IMA Journal of Management Mathematics*, 20(2), 133–145.
- Constantinou, A.C. and Fenton, N.E. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*, 8(1).
- Wheatcroft, E. (2021). Evaluating probabilistic forecasts of football matches: the case against the ranked probability score. *Journal of Quantitative Analysis in Sports*.
