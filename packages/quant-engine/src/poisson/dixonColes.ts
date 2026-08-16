import type { MatchLambdas, ScorelineMatrix } from "../types.js";
import { poissonPmf } from "./poisson.js";

const DEFAULT_MAX_GOALS = 10;

/**
 * Admissible range for rho. Outside this range the Dixon-Coles correction
 * drives one of the four adjusted cells negative.
 */
export function rhoBounds(lambdas: MatchLambdas): { min: number; max: number } {
  const { home, away } = lambdas;
  return {
    min: Math.max(-1 / home, -1 / away),
    max: Math.min(1 / (home * away), 1),
  };
}

/**
 * Dixon-Coles dependence correction. Independent Poisson under-counts the four
 * low-scoring results; tau adjusts exactly those cells and leaves the rest at 1.
 */
export function dixonColesTau(
  home: number,
  away: number,
  lambdas: MatchLambdas,
  rho: number,
): number {
  if (home === 0 && away === 0) return 1 - lambdas.home * lambdas.away * rho;
  if (home === 0 && away === 1) return 1 + lambdas.home * rho;
  if (home === 1 && away === 0) return 1 + lambdas.away * rho;
  if (home === 1 && away === 1) return 1 - rho;
  return 1;
}

/**
 * Joint scoreline distribution over 0..maxGoals for each side.
 *
 * Renormalised so the matrix sums to exactly 1, absorbing both the truncated
 * tail beyond maxGoals and the mass shifted by the tau correction. Every market
 * in this engine is derived from this matrix, which is what makes the markets
 * mutually consistent.
 */
export function buildScorelineMatrix(
  lambdas: MatchLambdas,
  rho: number,
  maxGoals: number = DEFAULT_MAX_GOALS,
): ScorelineMatrix {
  if (!(lambdas.home > 0) || !(lambdas.away > 0)) {
    throw new RangeError("both lambdas must be greater than 0");
  }
  if (!Number.isInteger(maxGoals) || maxGoals < 1) {
    throw new RangeError(`maxGoals must be a positive integer, received ${maxGoals}`);
  }
  const { min, max } = rhoBounds(lambdas);
  if (rho < min || rho > max) {
    throw new RangeError(
      `rho ${rho} outside admissible bounds [${min}, ${max}] for these lambdas`,
    );
  }

  const homePmf = Array.from({ length: maxGoals + 1 }, (_, k) =>
    poissonPmf(k, lambdas.home),
  );
  const awayPmf = Array.from({ length: maxGoals + 1 }, (_, k) =>
    poissonPmf(k, lambdas.away),
  );

  const raw: number[][] = [];
  let total = 0;
  for (let h = 0; h <= maxGoals; h += 1) {
    const row: number[] = [];
    for (let a = 0; a <= maxGoals; a += 1) {
      const value =
        dixonColesTau(h, a, lambdas, rho) *
        (homePmf[h] as number) *
        (awayPmf[a] as number);
      row.push(value);
      total += value;
    }
    raw.push(row);
  }

  const cells = raw.map((row) => row.map((value) => value / total));
  return { maxGoals, cells };
}
