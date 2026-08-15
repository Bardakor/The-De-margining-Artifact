import type {
  Market,
  MatchLambdas,
  PricedMarket,
  PricedSelection,
  ScorelineMatrix,
} from "./types.js";
import { buildScorelineMatrix } from "./poisson/dixonColes.js";
import {
  doubleChanceMarket,
  drawNoBetMarket,
  matchOddsMarket,
} from "./markets/matchOdds.js";
import { totalsMarket } from "./markets/totals.js";
import { bttsMarket } from "./markets/btts.js";
import { correctScoreMarket } from "./markets/correctScore.js";
import {
  asianHandicapMarket,
  type AsianHandicapMarket,
} from "./markets/asianHandicap.js";
import { applyOverround } from "./pricing/overround.js";

export interface PricingConfig {
  readonly rho: number;
  readonly targetBookSum: number;
  readonly totalsLines: readonly number[];
  readonly handicaps: readonly number[];
  readonly maxGoals: number;
}

export interface FixturePricing {
  readonly lambdas: MatchLambdas;
  readonly matrix: ScorelineMatrix;
  readonly markets: readonly PricedMarket[];
  readonly asianHandicaps: readonly AsianHandicapMarket[];
  readonly config: PricingConfig;
}

export const DEFAULT_PRICING_CONFIG: PricingConfig = {
  rho: -0.10,
  targetBookSum: 1.05,
  totalsLines: [0.5, 1.5, 2.5, 3.5, 4.5],
  handicaps: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
  maxGoals: 10,
};

/** Which 1X2 outcomes each double-chance selection covers. */
const DOUBLE_CHANCE_LEGS: Readonly<Record<string, readonly string[]>> = {
  "DC:1X": ["1X2:HOME", "1X2:DRAW"],
  "DC:12": ["1X2:HOME", "1X2:AWAY"],
  "DC:X2": ["1X2:DRAW", "1X2:AWAY"],
};

/**
 * Double chance cannot carry a book margin of its own. Each of its selections
 * covers two of the three outcomes, so a fair DC book sums to 2, and solving
 * the power method against a target of 1.05 would produce meaningless prices.
 *
 * Instead the margined 1X2 implied probabilities are summed pairwise. The DC
 * book therefore lands at exactly twice the 1X2 book, and the two markets
 * cannot drift apart.
 */
function priceDoubleChance(
  fair: Market,
  pricedMatchOdds: PricedMarket,
): PricedMarket {
  const impliedOf = (key: string): number => {
    const found = pricedMatchOdds.selections.find((s) => s.key === key);
    return found && Number.isFinite(found.odds) ? 1 / found.odds : 0;
  };

  const selections: PricedSelection[] = fair.selections.map((s) => {
    const rawImplied = (DOUBLE_CHANCE_LEGS[s.key] ?? []).reduce(
      (acc, key) => acc + impliedOf(key),
      0,
    );
    // Each individual 1X2 leg's margined implied probability is guaranteed
    // < 1 (applyOverround raises a probability in (0,1) to a positive power,
    // which cannot reach or exceed 1). Summing two such legs has no such
    // guarantee: the power method's favourite-longshot bias inflates small
    // probabilities (e.g. a longshot draw) proportionally far more than
    // large ones, so for a strongly lopsided fixture the pair can sum to
    // just over 1, which would price the DC selection at odds <= 1. Clamp
    // just under 1 so DC odds stay strictly > 1 like every other market;
    // this is a no-op for any fixture where the pairwise sum is < 1.
    const implied = Math.min(rawImplied, 1 - 1e-9);
    return {
      ...s,
      odds: implied > 0 ? 1 / implied : Number.POSITIVE_INFINITY,
    };
  });

  return {
    key: fair.key,
    label: fair.label,
    selections,
    bookSum: selections.reduce(
      (acc, s) => acc + (Number.isFinite(s.odds) ? 1 / s.odds : 0),
      0,
    ),
  };
}

/**
 * Prices every supported market for one fixture from its expected goals.
 *
 * All markets are derived from a single scoreline matrix, so they cannot
 * disagree with one another. Nothing here is stochastic: the same lambdas and
 * config always produce byte-identical output.
 */
export function priceFixture(
  lambdas: MatchLambdas,
  config: Partial<PricingConfig> = {},
): FixturePricing {
  const resolved: PricingConfig = { ...DEFAULT_PRICING_CONFIG, ...config };
  const matrix = buildScorelineMatrix(lambdas, resolved.rho, resolved.maxGoals);

  const pricedMatchOdds = applyOverround(
    matchOddsMarket(matrix),
    resolved.targetBookSum,
  );

  const markets: PricedMarket[] = [
    pricedMatchOdds,
    priceDoubleChance(doubleChanceMarket(matrix), pricedMatchOdds),
    applyOverround(drawNoBetMarket(matrix), resolved.targetBookSum),
    applyOverround(bttsMarket(matrix), resolved.targetBookSum),
    applyOverround(correctScoreMarket(matrix), resolved.targetBookSum),
    ...resolved.totalsLines.map((line) =>
      applyOverround(totalsMarket(matrix, line), resolved.targetBookSum),
    ),
  ];

  const asianHandicaps = resolved.handicaps.map((h) =>
    asianHandicapMarket(matrix, h),
  );

  return { lambdas, matrix, markets, asianHandicaps, config: resolved };
}
