import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { matchOddsMarket, toFairOdds } from "../src/markets/matchOdds.js";
import {
  applyOverround,
  removeOverroundShin,
  solvePowerExponent,
} from "../src/pricing/overround.js";
import type { Market } from "../src/types.js";

const TOL = 1e-9;
const MARKET = matchOddsMarket(buildScorelineMatrix({ home: 1.6, away: 1.1 }, -0.06));

describe("solvePowerExponent", () => {
  it("returns 1 when the target equals the fair book sum", () => {
    expect(solvePowerExponent([0.5, 0.3, 0.2], 1)).toBeCloseTo(1, 9);
  });

  it("returns an exponent below 1 for a target above 1", () => {
    expect(solvePowerExponent([0.5, 0.3, 0.2], 1.05)).toBeLessThan(1);
  });

  it("still solves normally for a reachable multi-selection target (regression guard for the postcondition check)", () => {
    expect(() => solvePowerExponent([0.5, 0.3, 0.2], 1.05)).not.toThrow();
  });

  it("throws when the target is unreachable for a single-selection book", () => {
    // For n=1, sumAt(k) = p^k has supremum 1 as k -> 0+ and never exceeds it,
    // so any target above 1 is mathematically unreachable. Before the
    // postcondition guard, both bracket-expansion loops left low and high
    // sitting below the target, the bisection's else-branch fired every
    // iteration, and k silently collapsed toward 1e-6 -- returning odds near
    // 1.0 and a bookSum nowhere near 1.05 with no error at all.
    expect(() => solvePowerExponent([0.6], 1.05)).toThrow(RangeError);
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

  it("throws rather than silently mispricing a one-selection market", () => {
    // Regression guard for the solvePowerExponent postcondition: a book with
    // a single selection can never reach a target above 1 (its probability
    // sum caps out at 1 as k -> 0+), so applying a normal 1.05 margin to one
    // must fail loudly instead of returning odds close to fair with a
    // silently wrong bookSum.
    const singleSelectionMarket: Market = {
      key: "TEST",
      label: "Single Selection Test Market",
      selections: [
        { key: "ONLY", label: "Only", probability: 0.6, fairOdds: toFairOdds(0.6) },
      ],
    };
    expect(() => applyOverround(singleSelectionMarket, 1.05)).toThrow(RangeError);
  });

  it("still succeeds for a normal multi-selection market (guard is not over-eager)", () => {
    expect(() => applyOverround(MARKET, 1.05)).not.toThrow();
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

  it("round-trips through Shin's forward relation when the bisection is actually exercised", () => {
    // The two tests above are vacuous with respect to Shin's mathematics:
    // the "sums to 1" test can't fail because the function's last line
    // unconditionally renormalises to 1, and the "identity" test uses a book
    // that already sums to 1, which takes the `bookSum <= 1` fast path and
    // never reaches the bisection at all. This test forces the bisection
    // branch (bookSum > 1) and checks the actual formula, not just the
    // renormalisation.
    //
    // The inverse formula implemented in the `trueProbs` closure in
    // overround.ts is:
    //   pi = (sqrt(z^2 + 4*(1-z)*p^2/B) - z) / (2*(1-z))
    // where p is the original implied probability, B is the book's implied
    // probabilities summed, z is the insider-money proportion, and pi is the
    // recovered true probability. Solving that same equation for p (isolate
    // the sqrt, square both sides, and simplify) gives the forward relation:
    //   p^2 = B * ((1-z)*pi^2 + z*pi)
    //   p   = sqrt(B * pi * ((1-z)*pi + z))
    // and solving it instead for z from a single known (p, pi) pair gives:
    //   z = (p^2/(B*pi) - pi) / (1 - pi)
    //
    // z isn't exposed by removeOverroundShin, so it's inferred here from one
    // selection using that last identity, then substituted into the forward
    // relation for every selection. If the recovered probabilities didn't
    // come from a real, consistently-converged z, this would not reproduce
    // the other two selections' implied probabilities -- renormalisation
    // alone cannot satisfy this, since it has no dependency on z at all.
    const implied = [0.5, 0.35, 0.25]; // sums to 1.10 > 1: exercises the bisection.
    const bookSum = implied.reduce((a, b) => a + b, 0);
    const recovered = removeOverroundShin(implied);

    const p0 = implied[0] ?? 0;
    const pi0 = recovered[0] ?? 0;
    const z = ((p0 * p0) / (bookSum * pi0) - pi0) / (1 - pi0);

    for (let i = 0; i < implied.length; i += 1) {
      const pi = recovered[i] ?? 0;
      const forwardImplied = Math.sqrt(bookSum * pi * ((1 - z) * pi + z));
      expect(Math.abs(forwardImplied - (implied[i] ?? 0))).toBeLessThan(1e-6);
    }
  });
});
