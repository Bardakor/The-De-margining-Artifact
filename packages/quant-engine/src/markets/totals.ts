import type { Market, ScorelineMatrix } from "../types.js";
import { sumWhere, toFairOdds } from "./matchOdds.js";

/**
 * Over/Under total goals. Only POSITIVE half-integer lines (0.5, 1.5, 2.5,
 * ...) are accepted. A whole-number line is rejected because it admits a
 * push (total goals can land exactly on the line), which is a different
 * settlement rule than the two-way, no-push market this function returns.
 * Negative half-integer lines (-0.5, -1.5, ...) are also rejected: they are
 * not meaningful for a non-negative total-goals count, and JavaScript's `%`
 * operator keeps the sign of the dividend, so this validation correctly
 * excludes them as well.
 */
export function totalsMarket(matrix: ScorelineMatrix, line: number): Market {
  if (!Number.isFinite(line) || (line * 2) % 2 !== 1) {
    throw new RangeError(`line must be a half-integer such as 2.5, received ${line}`);
  }
  const over = sumWhere(matrix, (h, a) => h + a > line);
  const under = 1 - over;
  return {
    key: `OU_${line}`,
    label: `Total Goals ${line}`,
    selections: [
      { key: `OU:${line}:OVER`, label: `Over ${line}`, probability: over, fairOdds: toFairOdds(over) },
      { key: `OU:${line}:UNDER`, label: `Under ${line}`, probability: under, fairOdds: toFairOdds(under) },
    ],
  };
}
