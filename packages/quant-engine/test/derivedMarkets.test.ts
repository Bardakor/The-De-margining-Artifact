import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { matchOddsMarket } from "../src/markets/matchOdds.js";
import { totalsMarket } from "../src/markets/totals.js";
import { bttsMarket } from "../src/markets/btts.js";
import { correctScoreMarket } from "../src/markets/correctScore.js";
import { applyOverround } from "../src/pricing/overround.js";

const TOL = 1e-9;
const MATRIX = buildScorelineMatrix({ home: 1.6, away: 1.1 }, -0.06);

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

  const MIN_SELECTION_PROBABILITY = 1e-4; // matches the default in correctScoreMarket

  it("still sums to 1 with the minSelectionProbability threshold applied", () => {
    // Passing the full matrix width so folding is driven purely by the
    // probability floor, not by maxDisplayGoals truncation.
    const cs = correctScoreMarket(MATRIX, MATRIX.maxGoals, MIN_SELECTION_PROBABILITY);
    const total = cs.selections.reduce((a, s) => a + s.probability, 0);
    expect(Math.abs(total - 1)).toBeLessThan(TOL);
  });

  it("publishes no CS:h-a selection with probability below the threshold", () => {
    const cs = correctScoreMarket(MATRIX, MATRIX.maxGoals, MIN_SELECTION_PROBABILITY);
    const belowThreshold = cs.selections.filter(
      (s) => s.key !== "CS:OTHER" && s.probability < MIN_SELECTION_PROBABILITY,
    );
    expect(belowThreshold).toEqual([]);
    // Sanity check that the threshold is actually doing something for this
    // fixture and this isn't a vacuously-true assertion.
    const folded = cs.selections.find((s) => s.key === "CS:OTHER");
    expect(folded?.probability).toBeGreaterThan(0);
  });

  it("keeps the book sum after margin exactly the target with the threshold applied", () => {
    const cs = correctScoreMarket(MATRIX, MATRIX.maxGoals, MIN_SELECTION_PROBABILITY);
    const priced = applyOverround(cs, 1.05);
    expect(Math.abs(priced.bookSum - 1.05)).toBeLessThan(TOL);
  });

  it("agrees with the 1X2 market when its cells are aggregated, including cells folded into OTHER", () => {
    // The critical consistency property: correct score and 1X2 are marginals of
    // the same distribution, so they cannot disagree. With the
    // minSelectionProbability floor (Finding 3) some home-win cells for this
    // fixture (e.g. 8-0, 9-2, ...) are no longer published as their own
    // CS:h-a selection -- they're folded into CS:OTHER instead. The
    // probability mass isn't lost, just relabelled, so this test independently
    // re-derives which cells fold (same rule correctScoreMarket applies) and
    // adds their home-win share back in directly from the raw matrix before
    // comparing to 1X2:HOME.
    const cs = correctScoreMarket(MATRIX, MATRIX.maxGoals, MIN_SELECTION_PROBABILITY);
    let homeWin = 0;
    for (const s of cs.selections) {
      const parsed = /^CS:(\d+)-(\d+)$/.exec(s.key);
      if (!parsed) continue;
      if (Number(parsed[1]) > Number(parsed[2])) homeWin += s.probability;
    }
    for (let h = 0; h <= MATRIX.maxGoals; h += 1) {
      for (let a = 0; a <= MATRIX.maxGoals; a += 1) {
        const p = MATRIX.cells[h]?.[a] ?? 0;
        if (p < MIN_SELECTION_PROBABILITY && h > a) homeWin += p;
      }
    }
    expect(homeWin).toBeCloseTo(prob(matchOddsMarket(MATRIX), "1X2:HOME"), 9);
  });
});
