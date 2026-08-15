# @yami/quant-engine

Pure TypeScript football pricing engine. No I/O, no database, no randomness.

## Model

    ratings -> attack/defence strengths -> lambda -> Dixon-Coles matrix -> markets -> prices

Independent Poisson under-counts low-scoring draws, which is why the previous
implementation produced draw odds above 6.0. The Dixon-Coles `tau` correction
adjusts the 0-0, 1-0, 0-1 and 1-1 cells to fix exactly that. The correction
only inflates those cells when `rho` is **negative** — the engine's default
is `rho: -0.10`. A positive rho pushes draw probability the wrong way, which
is why the sign matters more than the magnitude.

## Why one matrix

Every market is a sum over cells of a single 11x11 scoreline distribution:

| Market | Cells summed |
|---|---|
| 1X2 | `h>a`, `h==a`, `h<a` |
| Over/Under | `h+a > line` |
| BTTS | `h>0 && a>0` |
| Correct score | the single cell |
| Asian handicap | triangles shifted by the handicap |
| Double chance | unions of the 1X2 triangles |

Because they are all marginals of the same distribution, the markets cannot
contradict one another (their fair probabilities agree exactly). The test
suite asserts this directly.

## Margin

Fair odds are `1/p`. Margin is applied by the power method: solve for `k`
such that the sum of `p^k` equals the target book sum. A target above 1
shortens every price, and longshots carry proportionally more margin than
favourites.

Double chance is the one exception to "margin against the target book sum."
A fair DC book sums to 2, not 1, because each DC selection covers two of the
three 1X2 outcomes. Deriving DC by summing the already-margined 1X2 legs
looks tidy but is wrong: for a heavy favourite the summed legs can exceed
implied probability 1, producing decimal odds below 1, which is not a
payable price. Instead DC is margined independently, against twice the
configured target book sum. Every fair DC probability is strictly below 1,
and `p^k < 1` for any `k > 0`, so valid odds are guaranteed structurally
rather than by clamping. DC still stays probability-consistent with 1X2
because both are marginals of the same matrix — only the margin is applied
separately, which is also what real books do.

Shin's method is provided as the inverse, for de-margining historical closing
odds when backtesting.

## Usage

```ts
import { priceFixture } from "@yami/quant-engine";

const pricing = priceFixture({ home: 1.6, away: 1.1 });
const matchResult = pricing.markets.find((m) => m.key === "1X2");
```

## Verify

    npm run verify
