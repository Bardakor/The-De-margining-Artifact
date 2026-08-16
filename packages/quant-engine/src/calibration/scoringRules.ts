/** Smallest probability charged by logLoss, so a zero forecast is finite. */
const LOG_LOSS_FLOOR = 1e-15;

function assertForecast(forecast: readonly number[], outcomeIndex: number): void {
  if (forecast.length < 2) {
    throw new RangeError("forecast must have at least two outcomes");
  }
  if (!Number.isInteger(outcomeIndex) || outcomeIndex < 0 ||
      outcomeIndex >= forecast.length) {
    throw new RangeError(
      `outcomeIndex ${outcomeIndex} outside the forecast of length ${forecast.length}`,
    );
  }
  const total = forecast.reduce((a, b) => a + b, 0);
  if (Math.abs(total - 1) > 1e-6) {
    throw new RangeError(`forecast must sum to 1, received ${total}`);
  }
}

/**
 * Ranked probability score (Epstein 1969; argued for football by Constantinou
 * and Fenton 2012).
 *
 *   RPS = 1/(r-1) * sum over i of ( sum_{j<=i} (p_j - e_j) )^2
 *
 * Unlike Brier, RPS is sensitive to DISTANCE: for an ordered outcome set such
 * as home / draw / away, forecasting a home win when the away side wins is
 * penalised more than forecasting a home win when the match is drawn. Lower is
 * better; the range is [0, 1].
 *
 * Note that this is contested. Wheatcroft (2021) argues distance sensitivity is
 * not in fact desirable here and that RPS should not be the default. Both
 * metrics are provided so the choice stays explicit rather than assumed.
 */
export function rankedProbabilityScore(
  forecast: readonly number[],
  outcomeIndex: number,
): number {
  assertForecast(forecast, outcomeIndex);
  let cumulativeForecast = 0;
  let cumulativeOutcome = 0;
  let total = 0;
  for (let i = 0; i < forecast.length - 1; i += 1) {
    cumulativeForecast += forecast[i] ?? 0;
    cumulativeOutcome += i === outcomeIndex ? 1 : 0;
    const diff = cumulativeForecast - cumulativeOutcome;
    total += diff * diff;
  }
  return total / (forecast.length - 1);
}

/**
 * Multi-category Brier score (Brier 1950): the squared error summed over every
 * category. Lower is better. Treats all outcomes as unordered, which is exactly
 * the limitation RPS addresses.
 */
export function brierScore(
  forecast: readonly number[],
  outcomeIndex: number,
): number {
  assertForecast(forecast, outcomeIndex);
  let total = 0;
  for (let i = 0; i < forecast.length; i += 1) {
    const observed = i === outcomeIndex ? 1 : 0;
    const diff = (forecast[i] ?? 0) - observed;
    total += diff * diff;
  }
  return total;
}

/**
 * Negative log likelihood of the realised outcome, also called the ignorance
 * score when taken in base 2. Unlike Brier and RPS it is unbounded, so a single
 * confident mistake dominates — which is either the point or a drawback,
 * depending on what is being measured.
 */
export function logLoss(
  forecast: readonly number[],
  outcomeIndex: number,
): number {
  assertForecast(forecast, outcomeIndex);
  return -Math.log(Math.max(forecast[outcomeIndex] ?? 0, LOG_LOSS_FLOOR));
}

/** Mean of a set of per-match scores. */
export function meanScore(scores: readonly number[]): number {
  if (scores.length === 0) {
    throw new RangeError("cannot take the mean of an empty score set");
  }
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

export interface ReliabilityBin {
  readonly lower: number;
  readonly upper: number;
  readonly count: number;
  readonly meanForecast: number;
  readonly observedFrequency: number;
}

export interface BrierDecomposition {
  readonly brier: number;
  readonly reliability: number;
  readonly resolution: number;
  readonly uncertainty: number;
  /**
   * Variance of the forecasts WITHIN each bin. Zero when every bin holds a
   * single distinct forecast value, which is the only case in which Murphy's
   * classical three-way identity is exact. See the note on the identity below.
   */
  readonly withinBinVariance: number;
  readonly bins: readonly ReliabilityBin[];
}

/**
 * Partition of the binary Brier score into interpretable components, after
 * Murphy (1973).
 *
 * Reliability measures how far the forecast probabilities sit from the observed
 * frequencies within each bin — lower is better, and zero means perfectly
 * calibrated. Resolution measures how far the bin frequencies sit from the base
 * rate — higher is better, and zero means the forecaster says nothing beyond the
 * climatology. Uncertainty is a property of the events, not the forecaster, and
 * cannot be improved on.
 *
 * This is what turns "the model scored 0.58" into a statement about WHY.
 *
 * ON THE IDENTITY — this is easy to get wrong, so it is stated precisely.
 * Murphy's classical form
 *
 *   BS = REL - RES + UNC
 *
 * is exact only when each bin contains a SINGLE DISTINCT forecast value, which
 * holds when forecasts are discrete and binned onto their own values. Binning
 * CONTINUOUS forecasts leaves a residual, because REL compares each bin's MEAN
 * forecast against its observed frequency while BS uses each individual
 * forecast. The residual is exactly the within-bin variance
 *
 *   WBV = (1/N) * sum over bins k, members i of (p_i - mean(p_k))^2
 *
 * so the identity that holds for arbitrary continuous forecasts is
 *
 *   BS = REL - RES + UNC + WBV
 *
 * Both are reported. A worked case: forecasts
 * [0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95] over 10 bins give
 * BS = 0.109 but REL - RES + UNC = 0.10875, a residual of 0.00025 arising
 * entirely from the two bins holding two distinct values each. That residual is
 * WBV, not an error.
 */
export function brierDecomposition(
  forecasts: readonly number[],
  outcomes: readonly (0 | 1)[],
  bins = 10,
): BrierDecomposition {
  if (forecasts.length !== outcomes.length) {
    throw new RangeError(
      `forecasts (${forecasts.length}) and outcomes (${outcomes.length}) must be the same length`,
    );
  }
  if (forecasts.length === 0) {
    throw new RangeError("cannot decompose an empty forecast set");
  }
  if (!Number.isInteger(bins) || bins < 1) {
    throw new RangeError(`bins must be a positive integer, received ${bins}`);
  }

  const n = forecasts.length;
  const baseRate = outcomes.reduce<number>((a, b) => a + b, 0) / n;

  let brier = 0;
  for (let i = 0; i < n; i += 1) {
    const diff = (forecasts[i] ?? 0) - (outcomes[i] ?? 0);
    brier += diff * diff;
  }
  brier /= n;

  const buckets: { forecasts: number[]; outcomes: number[] }[] = Array.from(
    { length: bins },
    () => ({ forecasts: [], outcomes: [] }),
  );
  for (let i = 0; i < n; i += 1) {
    const p = forecasts[i] ?? 0;
    const index = Math.min(bins - 1, Math.max(0, Math.floor(p * bins)));
    const bucket = buckets[index];
    if (!bucket) continue;
    bucket.forecasts.push(p);
    bucket.outcomes.push(outcomes[i] ?? 0);
  }

  let reliability = 0;
  let resolution = 0;
  let withinBinVariance = 0;
  const reported: ReliabilityBin[] = [];

  for (let k = 0; k < bins; k += 1) {
    const bucket = buckets[k];
    if (!bucket || bucket.forecasts.length === 0) continue;
    const count = bucket.forecasts.length;
    const meanForecast =
      bucket.forecasts.reduce((a, b) => a + b, 0) / count;
    const observedFrequency =
      bucket.outcomes.reduce((a, b) => a + b, 0) / count;

    reliability +=
      (count / n) * (meanForecast - observedFrequency) ** 2;
    resolution += (count / n) * (observedFrequency - baseRate) ** 2;

    // The term that makes the identity exact for continuous forecasts. It is
    // zero exactly when this bin holds a single distinct forecast value.
    for (const p of bucket.forecasts) {
      withinBinVariance += (p - meanForecast) ** 2 / n;
    }

    reported.push({
      lower: k / bins,
      upper: (k + 1) / bins,
      count,
      meanForecast,
      observedFrequency,
    });
  }

  return {
    brier,
    reliability,
    resolution,
    uncertainty: baseRate * (1 - baseRate),
    withinBinVariance,
    bins: reported,
  };
}
