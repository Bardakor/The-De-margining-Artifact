import { describe, expect, it } from "vitest";
import { priceFixture } from "../src/priceFixture.js";

const FIXTURES = [
  { name: "home favourite", lambdas: { home: 1.9, away: 0.9 } },
  { name: "even match", lambdas: { home: 1.35, away: 1.25 } },
  { name: "away favourite", lambdas: { home: 0.95, away: 1.75 } },
  { name: "low scoring", lambdas: { home: 0.7, away: 0.6 } },
  { name: "high scoring", lambdas: { home: 2.6, away: 2.2 } },
];

describe("golden pricing snapshots", () => {
  for (const { name, lambdas } of FIXTURES) {
    it(`prices "${name}" stably`, () => {
      const pricing = priceFixture(lambdas);
      const summary = pricing.markets.map((m) => ({
        key: m.key,
        bookSum: Number(m.bookSum.toFixed(6)),
        odds: m.selections.map((s) => ({
          key: s.key,
          odds: Number(s.odds.toFixed(4)),
        })),
      }));
      expect(summary).toMatchSnapshot();
    });
  }
});
