import type { Market, ScorelineMatrix, Selection } from "../types.js";
import { toFairOdds } from "./matchOdds.js";

const DEFAULT_DISPLAY_GOALS = 5;

/**
 * Correct score. Scorelines above `maxDisplayGoals` for either side are
 * collapsed into a single OTHER selection so the market stays sized for a UI
 * while still summing to 1.
 */
export function correctScoreMarket(
  matrix: ScorelineMatrix,
  maxDisplayGoals: number = DEFAULT_DISPLAY_GOALS,
): Market {
  const selections: Selection[] = [];
  let displayed = 0;

  for (let h = 0; h <= Math.min(maxDisplayGoals, matrix.maxGoals); h += 1) {
    for (let a = 0; a <= Math.min(maxDisplayGoals, matrix.maxGoals); a += 1) {
      const p = matrix.cells[h]?.[a] ?? 0;
      displayed += p;
      selections.push({
        key: `CS:${h}-${a}`,
        label: `${h} - ${a}`,
        probability: p,
        fairOdds: toFairOdds(p),
      });
    }
  }

  const other = Math.max(0, 1 - displayed);
  selections.push({
    key: "CS:OTHER",
    label: "Any other score",
    probability: other,
    fairOdds: toFairOdds(other),
  });

  return { key: "CS", label: "Correct Score", selections };
}
