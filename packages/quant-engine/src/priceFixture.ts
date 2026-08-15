import type {
  MatchLambdas,
  PricedMarket,
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
  /**
   * Floor on a published correct-score selection's fair probability. Any
   * scoreline cell below this is folded into CS:OTHER rather than published
   * on its own, capping the maximum published correct-score price at roughly
   * `1 / minSelectionProbability` without touching the market's exact sum-to-1
   * invariant. See correctScoreMarket in markets/correctScore.ts.
   */
  readonly minSelectionProbability: number;
}

export interface FixturePricing {
  readonly lambdas: MatchLambdas;
  readonly matrix: ScorelineMatrix;
  readonly markets: readonly PricedMarket[];
  /**
   * Asian handicap markets, carrying FAIR odds ONLY -- NO margin has been
   * applied, unlike every market in `markets`.
   *
   * This is deliberate, not an oversight: AH legs are not a probability
   * simplex (win + push + lose = 1 per side, not across both sides), so the
   * power-method margin in pricing/overround.ts -- which solves for book sum
   * over a set of selections that partition probability 1 -- does not apply
   * to it. By construction, 1/fairHome + 1/fairAway equals exactly 1 for
   * every handicap line, so leaving these unmargined is what makes them
   * fair, not what makes them wrong.
   *
   * A consumer looping this array and publishing `fairOdds` directly would
   * ship the highest-volume market in the book at 0.00% margin. Callers MUST
   * apply their own margin (e.g. shorten each side's odds proportionally
   * before publishing) before quoting these prices to a bettor.
   */
  readonly asianHandicaps: readonly AsianHandicapMarket[];
  readonly config: PricingConfig;
}

export const DEFAULT_PRICING_CONFIG: PricingConfig = {
  rho: -0.10,
  targetBookSum: 1.05,
  totalsLines: [0.5, 1.5, 2.5, 3.5, 4.5],
  handicaps: [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
  maxGoals: 10,
  minSelectionProbability: 1e-4,
};

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

  // Double chance cannot be margined against a target of 1: each selection
  // covers two of the three outcomes, so a fair DC book already sums to 2.
  // It is margined independently, against twice the configured book sum.
  //
  // The rejected alternative was deriving DC by summing the already-margined
  // 1X2 implied probabilities pairwise. That breaks down on heavy favourites
  // (lambdas 2.5 vs 0.3 already produce implied probabilities that sum past
  // 1, i.e. odds below 1) because summing two margined legs has no upper
  // bound, whereas a single margined leg is always < 1 by construction.
  //
  // Margining DC directly avoids that structurally: every fair DC
  // probability is strictly below 1, and p^k < 1 for any k > 0, so the
  // margined implied probability can never reach 1 and the odds can never
  // fall to or below 1, for any fixture.
  //
  // DC still stays probability-consistent with 1X2: both are marginals of
  // the same scoreline matrix, so their FAIR probabilities agree exactly
  // (DC:1X fair probability equals 1X2:HOME + 1X2:DRAW fair probability).
  // Only the margin is applied to each market separately.
  const pricedDoubleChance = applyOverround(
    doubleChanceMarket(matrix),
    2 * resolved.targetBookSum,
  );

  const markets: PricedMarket[] = [
    pricedMatchOdds,
    pricedDoubleChance,
    applyOverround(drawNoBetMarket(matrix), resolved.targetBookSum),
    applyOverround(bttsMarket(matrix), resolved.targetBookSum),
    applyOverround(
      correctScoreMarket(matrix, undefined, resolved.minSelectionProbability),
      resolved.targetBookSum,
    ),
    ...resolved.totalsLines.map((line) =>
      applyOverround(totalsMarket(matrix, line), resolved.targetBookSum),
    ),
  ];

  const asianHandicaps = resolved.handicaps.map((h) =>
    asianHandicapMarket(matrix, h),
  );

  return { lambdas, matrix, markets, asianHandicaps, config: resolved };
}
