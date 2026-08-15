import { describe, expect, it } from "vitest";
import { buildScorelineMatrix } from "../src/poisson/dixonColes.js";
import { drawNoBetMarket } from "../src/markets/matchOdds.js";
import { asianHandicapMarket } from "../src/markets/asianHandicap.js";

const TOL = 1e-9;
const MATRIX = buildScorelineMatrix({ home: 1.6, away: 1.1 }, -0.06);

function leg(h: number, key: string) {
  const found = asianHandicapMarket(MATRIX, h).selections.find((s) => s.key === key);
  if (!found) throw new Error(`missing ${key}`);
  return found;
}

describe("asianHandicapMarket", () => {
  it("has win, push and lose summing to 1 for each side", () => {
    for (const h of [-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1]) {
      for (const s of asianHandicapMarket(MATRIX, h).selections) {
        expect(Math.abs(s.win + s.push + s.lose - 1)).toBeLessThan(TOL);
      }
    }
  });

  it("mirrors the two sides: home win equals away lose", () => {
    const home = leg(-0.5, "AH:-0.5:HOME");
    const away = leg(-0.5, "AH:-0.5:AWAY");
    expect(home.win).toBeCloseTo(away.lose, 9);
    expect(home.push).toBeCloseTo(away.push, 9);
  });

  it("mirrors the two sides on an integer line with genuine push mass", () => {
    // -0.5 always has push === 0, which makes the mirror assertion above
    // trivially 0 ≈ 0. Use an integer line, where a draw at exactly the
    // handicap is possible, to exercise invert()'s push-mirroring for real.
    const home = leg(-1, "AH:-1:HOME");
    const away = leg(-1, "AH:-1:AWAY");
    expect(home.push).toBeGreaterThan(0);
    expect(home.push).toBeCloseTo(away.push, 9);
    expect(home.win).toBeCloseTo(away.lose, 9);
    expect(home.lose).toBeCloseTo(away.win, 9);
  });

  it("reduces to draw-no-bet at handicap 0", () => {
    // The level-ball handicap refunds on a draw, exactly like DNB.
    const dnb = drawNoBetMarket(MATRIX);
    const dnbHome = dnb.selections.find((s) => s.key === "DNB:HOME");
    const ah = leg(0, "AH:0:HOME");
    const impliedHome = ah.win / (ah.win + ah.lose);
    expect(impliedHome).toBeCloseTo(dnbHome?.probability ?? 0, 9);
  });

  it("never pushes on a half-goal line", () => {
    for (const s of asianHandicapMarket(MATRIX, -0.5).selections) {
      expect(s.push).toBeCloseTo(0, 12);
    }
  });

  it("splits a quarter line across its two neighbours", () => {
    // -0.75 is half the stake on -0.5 and half on -1.0.
    const quarter = leg(-0.75, "AH:-0.75:HOME");
    const half = leg(-0.5, "AH:-0.5:HOME");
    const whole = leg(-1, "AH:-1:HOME");
    expect(quarter.win).toBeCloseTo((half.win + whole.win) / 2, 9);
    expect(quarter.push).toBeCloseTo((half.push + whole.push) / 2, 9);
  });

  it("makes the favoured side less likely to win as its handicap deepens", () => {
    expect(leg(-1.5, "AH:-1.5:HOME").win).toBeLessThan(leg(-0.5, "AH:-0.5:HOME").win);
  });

  it("rejects handicaps that are not a multiple of 0.25", () => {
    expect(() => asianHandicapMarket(MATRIX, -0.3)).toThrow(RangeError);
  });
});
