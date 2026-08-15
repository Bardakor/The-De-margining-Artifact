const DEFAULT_KELLY_MULTIPLIER = 0.25;

export interface ValueAssessment {
  readonly modelProbability: number;
  readonly offeredOdds: number;
  readonly impliedProbability: number;
  readonly edge: number;
  readonly expectedValue: number;
  readonly fullKelly: number;
  readonly recommendedStakeFraction: number;
  readonly hasValue: boolean;
}

function assertInputs(modelProbability: number, offeredOdds: number): void {
  if (!(modelProbability >= 0 && modelProbability <= 1)) {
    throw new RangeError(
      `modelProbability must be within [0, 1], received ${modelProbability}`,
    );
  }
  if (!(offeredOdds > 1)) {
    throw new RangeError(
      `offeredOdds must be greater than 1, received ${offeredOdds}`,
    );
  }
}

/**
 * Full-Kelly stake fraction: f* = (b*p - q) / b, with b the net decimal odds.
 * Floored at 0, because a negative Kelly means "do not bet", not "bet the
 * other side" (the other side has its own price and its own assessment).
 */
export function kellyFraction(
  modelProbability: number,
  offeredOdds: number,
): number {
  assertInputs(modelProbability, offeredOdds);
  const b = offeredOdds - 1;
  const q = 1 - modelProbability;
  return Math.max(0, (b * modelProbability - q) / b);
}

/** Expected profit per unit staked. */
export function expectedValue(
  modelProbability: number,
  offeredOdds: number,
): number {
  assertInputs(modelProbability, offeredOdds);
  return modelProbability * offeredOdds - 1;
}

/**
 * Full value assessment of an offered price against the model's own number.
 * Fractional Kelly is the default because full Kelly is intolerably volatile
 * when the probability estimate itself carries error.
 */
export function assessValue(
  modelProbability: number,
  offeredOdds: number,
  kellyMultiplier: number = DEFAULT_KELLY_MULTIPLIER,
): ValueAssessment {
  assertInputs(modelProbability, offeredOdds);
  if (!(kellyMultiplier > 0 && kellyMultiplier <= 1)) {
    throw new RangeError(
      `kellyMultiplier must be within (0, 1], received ${kellyMultiplier}`,
    );
  }
  const impliedProbability = 1 / offeredOdds;
  const fullKelly = kellyFraction(modelProbability, offeredOdds);
  return {
    modelProbability,
    offeredOdds,
    impliedProbability,
    edge: modelProbability - impliedProbability,
    expectedValue: expectedValue(modelProbability, offeredOdds),
    fullKelly,
    recommendedStakeFraction: fullKelly * kellyMultiplier,
    hasValue: fullKelly > 0,
  };
}
