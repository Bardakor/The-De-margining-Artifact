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
    // With the tau sign convention locked in above (tau(1,1) = 1 - rho, tau(0,0) = 1 - home*away*rho,
    // matching Dixon & Coles 1997), it is a NEGATIVE rho that inflates the low-scoring cells —
    // exactly as fitted rho values in the literature are typically negative (~-0.1).
    const withRho = buildScorelineMatrix(LAMBDAS, -0.1);
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
