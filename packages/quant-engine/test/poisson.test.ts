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
