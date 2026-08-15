/**
 * Lanczos approximation coefficients (g = 7, n = 9).
 */
const LANCZOS = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
] as const;

/**
 * Natural log of the gamma function, by the Lanczos approximation.
 * Working in log space keeps factorial-scale quantities finite.
 */
export function logGamma(z: number): number {
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
export function logFactorial(k: number): number {
  return logGamma(k + 1);
}
