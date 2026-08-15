import type { ScorelineMatrix } from "../types.js";
import { sumWhere } from "./matchOdds.js";

export interface AsianHandicapOutcome {
  readonly key: string;
  readonly label: string;
  readonly win: number;
  readonly push: number;
  readonly lose: number;
  readonly fairOdds: number;
}

export interface AsianHandicapMarket {
  readonly key: string;
  readonly label: string;
  readonly handicap: number;
  readonly selections: readonly AsianHandicapOutcome[];
}

interface Legs {
  readonly win: number;
  readonly push: number;
  readonly lose: number;
}

/**
 * Fair odds for a bet that can push. A push returns the stake, so only the
 * non-push mass is at risk: fair = 1 + lose/win, capped at Infinity when win is 0.
 */
function fairOddsWithPush(legs: Legs): number {
  if (legs.win === 0) return Number.POSITIVE_INFINITY;
  return 1 + legs.lose / legs.win;
}

/** Win/push/lose for the home side on a whole or half handicap line. */
function homeLegsForWholeLine(matrix: ScorelineMatrix, handicap: number): Legs {
  const win = sumWhere(matrix, (h, a) => h + handicap > a);
  const push = sumWhere(matrix, (h, a) => h + handicap === a);
  return { win, push, lose: 1 - win - push };
}

function averageLegs(first: Legs, second: Legs): Legs {
  return {
    win: (first.win + second.win) / 2,
    push: (first.push + second.push) / 2,
    lose: (first.lose + second.lose) / 2,
  };
}

function invert(legs: Legs): Legs {
  return { win: legs.lose, push: legs.push, lose: legs.win };
}

/**
 * Asian handicap applied to the home side. Quarter lines (x.25, x.75) split the
 * stake evenly across the two adjacent lines, which is how they settle in
 * practice.
 */
export function asianHandicapMarket(
  matrix: ScorelineMatrix,
  handicap: number,
): AsianHandicapMarket {
  if (!Number.isFinite(handicap) || (handicap * 4) % 1 !== 0) {
    throw new RangeError(
      `handicap must be a multiple of 0.25, received ${handicap}`,
    );
  }

  const isQuarter = (handicap * 2) % 1 !== 0;
  const homeLegs = isQuarter
    ? averageLegs(
        homeLegsForWholeLine(matrix, handicap - 0.25),
        homeLegsForWholeLine(matrix, handicap + 0.25),
      )
    : homeLegsForWholeLine(matrix, handicap);
  const awayLegs = invert(homeLegs);

  const label = handicap > 0 ? `+${handicap}` : `${handicap}`;
  return {
    key: `AH_${handicap}`,
    label: `Asian Handicap ${label}`,
    handicap,
    selections: [
      {
        key: `AH:${handicap}:HOME`,
        label: `Home ${label}`,
        ...homeLegs,
        fairOdds: fairOddsWithPush(homeLegs),
      },
      {
        key: `AH:${handicap}:AWAY`,
        label: `Away ${handicap > 0 ? `-${handicap}` : `+${Math.abs(handicap)}`}`,
        ...awayLegs,
        fairOdds: fairOddsWithPush(awayLegs),
      },
    ],
  };
}
