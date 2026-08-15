import type { Market, PricedMarket, PricedSelection } from "../types.js";

const DEFAULT_BOOK_SUM = 1.05;
const BISECTION_ITERATIONS = 200;
const POSTCONDITION_TOLERANCE = 1e-9;

/**
 * Solves for the exponent k such that the sum of p^k equals `targetBookSum`.
 *
 * Since every p is in (0, 1), p^k is strictly decreasing in k, so the sum is
 * monotonic and bisection converges. Using an exponent rather than a constant
 * multiplier reproduces the favourite-longshot bias: the margin taken from a
 * longshot is proportionally larger than from a favourite.
 *
 * Not every target is reachable, though: as k -> 0+, the sum of p^k tends to
 * `probabilities.length` (each term tends to 1), so any target above that
 * count (most notably a book of a single selection, where the supremum is 1
 * and a target above 1 is never attained) leaves the bracket-expansion loops
 * unable to find a k where sumAt(k) actually clears the target. The loop
 * below still runs to completion in that case, but it silently converges
 * toward the low end of the bracket instead of a real root. The postcondition
 * check catches that and fails loudly instead of returning a k whose implied
 * book sum is nowhere near what was asked for.
 *
 * Every probability must be a legitimate value in [0, 1]. NaN, negative, and
 * above-1 values are rejected explicitly here rather than being silently
 * dropped by the `p > 0` filter below: without this check, a NaN or negative
 * probability would simply fail to appear in `positive`, and the caller
 * (applyOverround) would go on to map it to Infinity odds as if it were a
 * legitimate impossible outcome, fabricating a healthy-looking book out of
 * invalid input instead of failing loudly like every other entry point in
 * this package. A probability of exactly 0 is legitimate (an impossible
 * outcome, which correctly maps to Infinity odds) and is not rejected here.
 */
export function solvePowerExponent(
  probabilities: readonly number[],
  targetBookSum: number,
): number {
  for (const p of probabilities) {
    if (Number.isNaN(p) || p < 0 || p > 1) {
      throw new RangeError(
        `solvePowerExponent: probability must be a number in [0, 1], received ${p}`,
      );
    }
  }

  const positive = probabilities.filter((p) => p > 0);
  const sumAt = (k: number): number =>
    positive.reduce((acc, p) => acc + Math.pow(p, k), 0);

  let low = 0.01;
  let high = 1;
  // Expand downwards until the sum overshoots the target.
  while (sumAt(low) < targetBookSum && low > 1e-6) low /= 2;
  // Expand upwards in case the target is below the fair sum.
  while (sumAt(high) > targetBookSum && high < 64) high *= 2;

  for (let i = 0; i < BISECTION_ITERATIONS; i += 1) {
    const mid = (low + high) / 2;
    if (sumAt(mid) > targetBookSum) low = mid;
    else high = mid;
  }
  const k = (low + high) / 2;
  const achieved = sumAt(k);
  if (Math.abs(achieved - targetBookSum) > POSTCONDITION_TOLERANCE) {
    throw new RangeError(
      `solvePowerExponent: target book sum ${targetBookSum} is unreachable ` +
        `for the given probabilities (achieved ${achieved} at k=${k})`,
    );
  }
  return k;
}

/**
 * Applies the book margin to a market so that the sum of 1/odds equals
 * `targetBookSum` exactly.
 *
 * Note the direction: a target above 1 SHORTENS every price. The previous
 * implementation multiplied probability by (1 - margin), which lengthened
 * prices and produced a book paying out more than fair value.
 */
export function applyOverround(
  market: Market,
  targetBookSum: number = DEFAULT_BOOK_SUM,
): PricedMarket {
  if (!(targetBookSum >= 1)) {
    throw new RangeError(
      `targetBookSum must be at least 1, received ${targetBookSum}`,
    );
  }

  const probabilities = market.selections.map((s) => s.probability);
  const k = solvePowerExponent(probabilities, targetBookSum);

  const selections: PricedSelection[] = market.selections.map((s) => {
    const implied = s.probability > 0 ? Math.pow(s.probability, k) : 0;
    return {
      ...s,
      odds: implied > 0 ? 1 / implied : Number.POSITIVE_INFINITY,
    };
  });

  const bookSum = selections.reduce(
    (acc, s) => acc + (Number.isFinite(s.odds) ? 1 / s.odds : 0),
    0,
  );

  return { key: market.key, label: market.label, selections, bookSum };
}

/**
 * Shin's method: recovers true probabilities from a bookmaker's implied
 * probabilities by modelling the proportion z of insider money.
 *
 * This is the INVERSE of applying a margin, and is used to de-margin historical
 * closing odds so the model can be benchmarked against the market on equal
 * terms. It is not used to price our own markets.
 */
export function removeOverroundShin(
  impliedProbabilities: readonly number[],
): number[] {
  const bookSum = impliedProbabilities.reduce((a, b) => a + b, 0);
  if (bookSum <= 1) return impliedProbabilities.map((p) => p / bookSum);

  const trueProbs = (z: number): number[] =>
    impliedProbabilities.map((p) => {
      const root = Math.sqrt(z * z + 4 * (1 - z) * ((p * p) / bookSum));
      return (root - z) / (2 * (1 - z));
    });

  let low = 0;
  let high = 0.99;
  for (let i = 0; i < BISECTION_ITERATIONS; i += 1) {
    const mid = (low + high) / 2;
    const total = trueProbs(mid).reduce((a, b) => a + b, 0);
    if (total > 1) low = mid;
    else high = mid;
  }

  const result = trueProbs((low + high) / 2);
  const total = result.reduce((a, b) => a + b, 0);
  return result.map((p) => p / total);
}
