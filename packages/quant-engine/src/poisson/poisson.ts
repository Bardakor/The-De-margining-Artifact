import { logFactorial } from "../math/gamma.js";

/**
 * Poisson probability mass: P(X = k) for rate `lambda`.
 * Computed in log space for numerical stability.
 */
export function poissonPmf(k: number, lambda: number): number {
  if (!Number.isInteger(k) || k < 0) {
    throw new RangeError(`k must be a non-negative integer, received ${k}`);
  }
  if (!(lambda > 0)) {
    throw new RangeError(`lambda must be greater than 0, received ${lambda}`);
  }
  return Math.exp(k * Math.log(lambda) - lambda - logFactorial(k));
}
