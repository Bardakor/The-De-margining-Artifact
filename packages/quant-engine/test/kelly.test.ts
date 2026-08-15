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
