import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import {
  doubleChanceMarket,
  drawNoBetMarket,
  matchOddsMarket,
} from "../src/markets/matchOdds.js";

const TOL = 1e-9;
const MATRIX = buildScorelineMatrix({ home: 1.6, away: 1.1 }, -0.06);

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
    const even = buildScorelineMatrix({ home: 1.35, away: 1.25 }, -0.06);
    const p = prob(matchOddsMarket(even), "1X2:DRAW");
    expect(p).toBeGreaterThan(0.15);
    expect(p).toBeLessThan(0.40);
    // Fair draw odds must therefore stay well under the old model's 6.0+.
    expect(1 / p).toBeLessThan(6);
  });

  it("never yields a negative probability, even for extreme mismatches", () => {
    const lopsided = buildScorelineMatrix({ home: 4.2, away: 0.3 }, -0.02);
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
