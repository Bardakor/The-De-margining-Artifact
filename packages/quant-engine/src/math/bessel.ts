import { logGamma } from "./gamma.js";

const SERIES_TERMS = 200;

/**
 * Natural log of the modified Bessel function of the first kind, I_n(z), for
 * non-negative integer order n and z >= 0.
 *
 *   I_n(z) = sum over m >= 0 of  (z/2)^(2m+n) / ( m! * (m+n)! )
 *
 * Each term is evaluated in log space and the sum accumulated by the
 * log-sum-exp trick, so the result stays finite for large z where the raw
 * series overflows.
 */
export function logBesselI(order: number, z: number): number {
  if (!Number.isInteger(order) || order < 0) {
    throw new RangeError(
      `order must be a non-negative integer, received ${order}`,
    );
  }
  if (!(z >= 0)) {
    throw new RangeError(`z must be non-negative, received ${z}`);
  }
  if (z === 0) {
    return order === 0 ? 0 : Number.NEGATIVE_INFINITY;
  }

  const halfZ = Math.log(z / 2);
  const logTerms: number[] = [];
  for (let m = 0; m < SERIES_TERMS; m += 1) {
    logTerms.push(
      (2 * m + order) * halfZ - logGamma(m + 1) - logGamma(m + order + 1),
    );
  }

  const max = logTerms.reduce((a, b) => (b > a ? b : a), Number.NEGATIVE_INFINITY);
  if (!Number.isFinite(max)) return Number.NEGATIVE_INFINITY;

  let sum = 0;
  for (const term of logTerms) sum += Math.exp(term - max);
  return max + Math.log(sum);
}

/** Modified Bessel function of the first kind, I_n(z). */
export function besselI(order: number, z: number): number {
  return Math.exp(logBesselI(order, z));
}
