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
