import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { matchOddsMarket } from "../src/markets/matchOdds.js";
import {
  applyOverround,
  removeOverroundShin,
  solvePowerExponent,
} from "../src/pricing/overround.js";

const TOL = 1e-9;
const MARKET = matchOddsMarket(buildScorelineMatrix({ home: 1.6, away: 1.1 }, -0.06));

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
