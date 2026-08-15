import { describe, expect, it } from "vitest";
import { DEFAULT_PRICING_CONFIG, priceFixture } from "../src/priceFixture.js";

const TOL = 1e-9;
const LAMBDAS = { home: 1.6, away: 1.1 };

function market(pricing: ReturnType<typeof priceFixture>, key: string) {
  const found = pricing.markets.find((m) => m.key === key);
  if (!found) throw new Error(`missing market ${key}`);
  return found;
}

describe("priceFixture", () => {
  it("prices every configured market", () => {
    const p = priceFixture(LAMBDAS);
    const keys = p.markets.map((m) => m.key);
    expect(keys).toContain("1X2");
    expect(keys).toContain("BTTS");
    expect(keys).toContain("CS");
    expect(keys).toContain("DC");
    expect(keys).toContain("OU_2.5");
    expect(p.asianHandicaps.length).toBe(DEFAULT_PRICING_CONFIG.handicaps.length);
  });

  it("gives every exhaustive market the configured book sum", () => {
    for (const m of priceFixture(LAMBDAS).markets) {
      // Double chance is excluded deliberately and asserted separately below:
      // its selections each cover two outcomes, so a fair DC book sums to 2.
      if (m.key === "DC") continue;
      expect(Math.abs(m.bookSum - DEFAULT_PRICING_CONFIG.targetBookSum)).toBeLessThan(TOL);
    }
  });

  it("gives double chance exactly twice the target book sum", () => {
    // Each of 1X, 12 and X2 covers two of the three outcomes, so the DC book
    // is 2 x (the 1X2 book). Deriving DC from the margined 1X2 rather than
    // applying a margin to it directly is what makes this exact.
    const dc = market(priceFixture(LAMBDAS), "DC");
    expect(Math.abs(dc.bookSum - 2 * DEFAULT_PRICING_CONFIG.targetBookSum)).toBeLessThan(TOL);
  });

  it("prices double chance consistently with the margined 1X2 book", () => {
    const p = priceFixture(LAMBDAS);
    const x2 = market(p, "1X2");
    const dc = market(p, "DC");
    const impl = (m: typeof x2, key: string): number => {
      const found = m.selections.find((s) => s.key === key);
      return found ? 1 / found.odds : 0;
    };
    expect(impl(dc, "DC:1X")).toBeCloseTo(impl(x2, "1X2:HOME") + impl(x2, "1X2:DRAW"), 9);
    expect(impl(dc, "DC:X2")).toBeCloseTo(impl(x2, "1X2:DRAW") + impl(x2, "1X2:AWAY"), 9);
  });

  it("keeps derived markets consistent with one another", () => {
    const p = priceFixture(LAMBDAS);
    const x2 = market(p, "1X2");
    const dc = market(p, "DC");
    const home = x2.selections.find((s) => s.key === "1X2:HOME")?.probability ?? 0;
    const draw = x2.selections.find((s) => s.key === "1X2:DRAW")?.probability ?? 0;
    const oneX = dc.selections.find((s) => s.key === "DC:1X")?.probability ?? 0;
    expect(oneX).toBeCloseTo(home + draw, 9);
  });

  it("produces identical output when called twice", () => {
    expect(JSON.stringify(priceFixture(LAMBDAS))).toBe(
      JSON.stringify(priceFixture(LAMBDAS)),
    );
  });

  it("never returns a draw price above 6 for an evenly matched fixture", () => {
    // End-to-end regression guard for defect D1.
    const p = priceFixture({ home: 1.35, away: 1.25 });
    const draw = market(p, "1X2").selections.find((s) => s.key === "1X2:DRAW");
    expect(draw?.odds).toBeLessThan(6);
    expect(draw?.odds).toBeGreaterThan(1);
  });

  it("never returns a negative or non-finite price anywhere", () => {
    for (const lambdas of [
      { home: 0.4, away: 0.5 },
      { home: 3.8, away: 0.3 },
      { home: 2.2, away: 2.4 },
    ]) {
      for (const m of priceFixture(lambdas).markets) {
        for (const s of m.selections) {
          expect(s.probability).toBeGreaterThanOrEqual(0);
          if (s.probability > 1e-6) {
            expect(s.odds).toBeGreaterThan(1);
            expect(Number.isFinite(s.odds)).toBe(true);
          }
        }
      }
    }
  });

  it("honours a config override", () => {
    const p = priceFixture(LAMBDAS, { targetBookSum: 1.02, totalsLines: [1.5] });
    expect(Math.abs(market(p, "1X2").bookSum - 1.02)).toBeLessThan(TOL);
    expect(p.markets.filter((m) => m.key.startsWith("OU_"))).toHaveLength(1);
  });
});
