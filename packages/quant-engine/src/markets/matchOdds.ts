import type { Market, ScorelineMatrix, Selection } from "../types.js";

/** Fair decimal odds for a probability. */
export function toFairOdds(probability: number): number {
  return probability === 0 ? Number.POSITIVE_INFINITY : 1 / probability;
}

function selection(key: string, label: string, probability: number): Selection {
  return { key, label, probability, fairOdds: toFairOdds(probability) };
}

/**
 * Sums matrix cells for which `predicate(homeGoals, awayGoals)` holds.
 * Every market in this engine is expressed as one of these sums, which is what
 * guarantees the markets agree with one another.
 */
function sumWhere(
  matrix: ScorelineMatrix,
  predicate: (home: number, away: number) => boolean,
): number {
  let total = 0;
  for (let h = 0; h <= matrix.maxGoals; h += 1) {
    const row = matrix.cells[h];
    if (!row) continue;
    for (let a = 0; a <= matrix.maxGoals; a += 1) {
      if (predicate(h, a)) total += row[a] ?? 0;
    }
  }
  return total;
}

export function matchOddsMarket(matrix: ScorelineMatrix): Market {
  const home = sumWhere(matrix, (h, a) => h > a);
  const draw = sumWhere(matrix, (h, a) => h === a);
  const away = sumWhere(matrix, (h, a) => h < a);
  return {
    key: "1X2",
    label: "Match Result",
    selections: [
      selection("1X2:HOME", "Home", home),
      selection("1X2:DRAW", "Draw", draw),
      selection("1X2:AWAY", "Away", away),
    ],
  };
}

export function doubleChanceMarket(matrix: ScorelineMatrix): Market {
  return {
    key: "DC",
    label: "Double Chance",
    selections: [
      selection("DC:1X", "Home or Draw", sumWhere(matrix, (h, a) => h >= a)),
      selection("DC:12", "Home or Away", sumWhere(matrix, (h, a) => h !== a)),
      selection("DC:X2", "Draw or Away", sumWhere(matrix, (h, a) => h <= a)),
    ],
  };
}

export function drawNoBetMarket(matrix: ScorelineMatrix): Market {
  const home = sumWhere(matrix, (h, a) => h > a);
  const away = sumWhere(matrix, (h, a) => h < a);
  const decisive = home + away;
  return {
    key: "DNB",
    label: "Draw No Bet",
    selections: [
      selection("DNB:HOME", "Home", home / decisive),
      selection("DNB:AWAY", "Away", away / decisive),
    ],
  };
}

export { sumWhere };
