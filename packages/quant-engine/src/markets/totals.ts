import type { Market, ScorelineMatrix } from "../types.js";
import { sumWhere, toFairOdds } from "./matchOdds.js";

/**
 * Over/Under total goals. Only half-integer lines are supported, because a
 * whole-number line admits a push, which is a different settlement rule than
 * the two-way market this returns.
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
