import type { Market, ScorelineMatrix, Selection } from "../types.js";
import { toFairOdds } from "./matchOdds.js";

const DEFAULT_DISPLAY_GOALS = 5;

/**
 * Default floor on a published correct-score selection's probability. Any
 * scoreline cell below this is folded into CS:OTHER instead of being
 * published on its own, which caps the maximum published correct-score price
 * at roughly `1 / minSelectionProbability` (fair, before margin shortens it
 * further). Without this floor, independent Poisson tails compound: two
 * individually-unlikely-but-plausible goal counts (e.g. an 8-6 scoreline)
 * multiply into a cell probability small enough to price above 1,000,000 to
 * 1, which is correct arithmetic but an unbounded, unstakeable price that
 * nothing downstream should have to defend against.
 */
const DEFAULT_MIN_SELECTION_PROBABILITY = 1e-4;

/**
 * Correct score. Scorelines above `maxDisplayGoals` for either side, and any
 * scoreline whose probability falls below `minSelectionProbability`, are
 * collapsed into a single OTHER selection so the market stays sized for a UI
 * (and bounded in price) while still summing to 1.
 */
export function correctScoreMarket(
  matrix: ScorelineMatrix,
  maxDisplayGoals: number = DEFAULT_DISPLAY_GOALS,
  minSelectionProbability: number = DEFAULT_MIN_SELECTION_PROBABILITY,
): Market {
  const selections: Selection[] = [];
  let inRange = 0;
  let folded = 0;

  for (let h = 0; h <= Math.min(maxDisplayGoals, matrix.maxGoals); h += 1) {
    for (let a = 0; a <= Math.min(maxDisplayGoals, matrix.maxGoals); a += 1) {
      const p = matrix.cells[h]?.[a] ?? 0;
      inRange += p;
      if (p < minSelectionProbability) {
        folded += p;
        continue;
      }
      selections.push({
        key: `CS:${h}-${a}`,
        label: `${h} - ${a}`,
        probability: p,
        fairOdds: toFairOdds(p),
      });
    }
  }

  // OTHER absorbs both the cells folded in for being below the probability
  // floor and the truncated tail beyond maxDisplayGoals (1 - inRange), so the
  // market still sums to exactly 1 regardless of how many cells were folded.
  const other = folded + Math.max(0, 1 - inRange);
  selections.push({
    key: "CS:OTHER",
    label: "Any other score",
    probability: other,
    fairOdds: toFairOdds(other),
  });

  return { key: "CS", label: "Correct Score", selections };
}
