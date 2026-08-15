/**
 * Log-gamma via the Lanczos approximation. Used so that poissonPmf can be
 * computed in log space, which keeps it finite for large k where a naive
 * lambda^k / k! would overflow to Infinity / Infinity = NaN.
 */
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
] as const;

function logGamma(z: number): number {
  if (z < 0.5) {
    // Reflection formula for the left half-plane.
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  const x = z - 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < LANCZOS.length; i += 1) {
    a += (LANCZOS[i] as number) / (x + i + 1);
  }
  const t = x + LANCZOS.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** log(k!) for non-negative integer k. */
function logFactorial(k: number): number {
  return logGamma(k + 1);
}

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
