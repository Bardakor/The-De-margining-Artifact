import type { Market, Selection } from "../types.js";
import { logBesselI } from "../math/bessel.js";

const SUPPORT_LIMIT = 60;

/**
 * Skellam probability mass: the distribution of the difference of two
 * independent Poisson variables, K = X - Y.
 *
 *   P(K = k) = e^-(l1+l2) * (l1/l2)^(k/2) * I_|k|( 2*sqrt(l1*l2) )
 *
 * where I is the modified Bessel function of the first kind.
 *
 * This is an entirely separate analytical route to the goal-difference
 * distribution from summing the scoreline matrix, which is what makes it useful
 * as a cross-check rather than merely as another feature. See Karlis and
 * Ntzoufras (2009).
 *
 * Note the independence assumption: this does NOT carry the Dixon-Coles
 * correction, so it agrees with the matrix only when rho is 0.
 */
export function skellamPmf(
  k: number,
  lambdaHome: number,
  lambdaAway: number,
): number {
  if (!Number.isInteger(k)) {
    throw new RangeError(`k must be an integer, received ${k}`);
  }
  if (!(lambdaHome > 0) || !(lambdaAway > 0)) {
    throw new RangeError("both rates must be greater than 0");
  }
  const logValue =
    -(lambdaHome + lambdaAway) +
    (k / 2) * Math.log(lambdaHome / lambdaAway) +
    logBesselI(Math.abs(k), 2 * Math.sqrt(lambdaHome * lambdaAway));
  return Math.exp(logValue);
}

/**
 * Home / draw / away probabilities from the goal-difference distribution alone,
 * summed over a support wide enough that the omitted tail is far below the
 * package's 1e-9 test tolerance.
 */
export function skellamMatchProbabilities(
  lambdaHome: number,
  lambdaAway: number,
): { home: number; draw: number; away: number } {
  let home = 0;
  let away = 0;
  for (let k = 1; k <= SUPPORT_LIMIT; k += 1) {
    home += skellamPmf(k, lambdaHome, lambdaAway);
    away += skellamPmf(-k, lambdaHome, lambdaAway);
  }
  const draw = skellamPmf(0, lambdaHome, lambdaAway);
  const total = home + draw + away;
  return { home: home / total, draw: draw / total, away: away / total };
}

/**
 * Supremacy: whether the home side's goal difference beats a half-goal line.
 * A negative line favours the home side, matching Asian handicap convention.
 */
export function skellamSupremacyMarket(
  lambdaHome: number,
  lambdaAway: number,
  line: number,
): Market {
  if (!Number.isFinite(line) || Math.abs(line * 2) % 2 !== 1) {
    throw new RangeError(
      `line must be a half-integer such as -0.5, received ${line}`,
    );
  }

  let home = 0;
  for (let k = -SUPPORT_LIMIT; k <= SUPPORT_LIMIT; k += 1) {
    if (k + line > 0) home += skellamPmf(k, lambdaHome, lambdaAway);
  }
  let total = 0;
  for (let k = -SUPPORT_LIMIT; k <= SUPPORT_LIMIT; k += 1) {
    total += skellamPmf(k, lambdaHome, lambdaAway);
  }
  const homeProbability = home / total;
  const awayProbability = 1 - homeProbability;

  const selection = (key: string, label: string, p: number): Selection => ({
    key,
    label,
    probability: p,
    fairOdds: p === 0 ? Number.POSITIVE_INFINITY : 1 / p,
  });

  return {
    key: `SUP_${line}`,
    label: `Supremacy ${line}`,
    selections: [
      selection(`SUP:${line}:HOME`, `Home ${line}`, homeProbability),
      selection(`SUP:${line}:AWAY`, `Away +${Math.abs(line)}`, awayProbability),
    ],
  };
}
