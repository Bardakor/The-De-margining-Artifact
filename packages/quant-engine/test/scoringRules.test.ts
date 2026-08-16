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
  const CONTINUOUS = [0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95];
  const CONTINUOUS_OUTCOMES: (0 | 1)[] = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1];

  it("satisfies the EXACT identity brier = reliability - resolution + uncertainty + withinBinVariance", () => {
    // The identity that holds for arbitrary continuous forecasts. Murphy's
    // classical three-way form is exact only when every bin holds a single
    // distinct forecast value; binning continuous forecasts leaves a residual
    // equal to the within-bin variance.
    const d = brierDecomposition(CONTINUOUS, CONTINUOUS_OUTCOMES);
    expect(
      Math.abs(
        d.brier -
          (d.reliability - d.resolution + d.uncertainty + d.withinBinVariance),
      ),
    ).toBeLessThan(1e-12);
  });

  it("shows the classical three-way identity is NOT exact here, and the gap IS the within-bin variance", () => {
    // Guards against anyone "simplifying" the decomposition back to three terms.
    const d = brierDecomposition(CONTINUOUS, CONTINUOUS_OUTCOMES);
    const threeWay = d.reliability - d.resolution + d.uncertainty;
    const residual = d.brier - threeWay;
    expect(residual).toBeGreaterThan(1e-6);
    expect(residual).toBeCloseTo(d.withinBinVariance, 12);
    // Hand-computed: bins [0.2,0.25] and [0.8,0.85] each contribute
    // 2 * 0.025^2 = 0.00125, so WBV = 0.0025 / 10 = 0.00025.
    expect(d.withinBinVariance).toBeCloseTo(0.00025, 12);
    expect(d.brier).toBeCloseTo(0.109, 12);
    expect(threeWay).toBeCloseTo(0.10875, 12);
  });

  it("makes withinBinVariance exactly 0 when every bin holds one distinct value, recovering Murphy's classical identity", () => {
    // Each forecast lands in its own bin, so the three-way form is exact.
    const forecasts = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    const outcomes: (0 | 1)[] = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
    const d = brierDecomposition(forecasts, outcomes);
    expect(d.withinBinVariance).toBeCloseTo(0, 15);
    expect(
      Math.abs(d.brier - (d.reliability - d.resolution + d.uncertainty)),
    ).toBeLessThan(1e-12);
  });

  it("gives near-zero reliability for a perfectly calibrated forecaster", () => {
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
