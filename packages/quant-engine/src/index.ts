export type {
  Market,
  MatchLambdas,
  PricedMarket,
  PricedSelection,
  ScorelineMatrix,
  Selection,
} from "./types.js";
export { poissonPmf } from "./poisson/poisson.js";
export {
  buildScorelineMatrix,
  dixonColesTau,
  rhoBounds,
} from "./poisson/dixonColes.js";
export {
  doubleChanceMarket,
  drawNoBetMarket,
  matchOddsMarket,
  toFairOdds,
} from "./markets/matchOdds.js";
export { totalsMarket } from "./markets/totals.js";
export { bttsMarket } from "./markets/btts.js";
export { correctScoreMarket } from "./markets/correctScore.js";
export {
  asianHandicapMarket,
  type AsianHandicapMarket,
  type AsianHandicapOutcome,
} from "./markets/asianHandicap.js";
export {
  applyOverround,
  removeOverroundShin,
  solvePowerExponent,
} from "./pricing/overround.js";
export {
  assessValue,
  expectedValue,
  kellyFraction,
  type ValueAssessment,
} from "./pricing/kelly.js";
export {
  DEFAULT_PRICING_CONFIG,
  priceFixture,
  type FixturePricing,
  type PricingConfig,
} from "./priceFixture.js";
