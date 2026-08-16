# @yami/quant-engine

Football pricing engine built on the Dixon–Coles bivariate Poisson model. Pure TypeScript,
zero runtime dependencies, no I/O, no randomness. **137 tests.**

> 📐 **[Full mathematical specification → `docs/MODEL.md`](docs/MODEL.md)** — every formula,
> derivation, measured result, and the 15 papers behind them.

---

## The model

$$
\text{ratings} \to (\alpha_i, \beta_i, \gamma) \to (\lambda, \mu) \to P(x,y) \to \text{markets} \to \text{prices}
$$

Goals arrive as a Poisson process (Maher, 1982), with each side's rate factored into attack
strength, opponent defence weakness, and a per-league home advantage:

$$
\lambda = \alpha_{\text{home}} \cdot \beta_{\text{away}} \cdot \gamma
\qquad
\mu = \alpha_{\text{away}} \cdot \beta_{\text{home}}
$$

Independent Poisson is wrong in one documented way: it **under-counts low-scoring draws**.
Dixon and Coles (1997) correct exactly the four affected cells:

$$
\tau(x,y) =
\begin{cases}
1 - \lambda\mu\rho & (0,0)\\
1 + \lambda\rho & (0,1)\\
1 + \mu\rho & (1,0)\\
1 - \rho & (1,1)\\
1 & \text{otherwise}
\end{cases}
$$

### The sign of ρ decides whether the correction helps or hurts

Since $\tau(0,0) = 1 - \lambda\mu\rho$ and $\tau(1,1) = 1 - \rho$, both exceed 1 — inflating
the low draws — **only when $\rho < 0$**. Measured on a $\lambda{=}1.6,\ \mu{=}1.1$ fixture:

| ρ | P(0-0) | P(1-1) | **P(draw)** | fair draw odds |
|---:|---:|---:|---:|---:|
| −0.20 | 0.09086 | 0.14194 | 0.29622 | 3.376 |
| **−0.10** (default) | 0.07903 | 0.13011 | **0.27257** | **3.669** |
| 0.00 | 0.06721 | 0.11828 | 0.24891 | 4.017 |
| +0.06 | 0.06011 | 0.11118 | 0.23472 | 4.260 |

A positive ρ *suppresses* the very draws the correction exists to raise. Dixon and Coles'
own fitted value is negative, around −0.13.

---

## One matrix, every market

$$
P(x,y) = \frac{\tau(x,y)\,\mathrm{Pois}(x;\lambda)\,\mathrm{Pois}(y;\mu)}{\sum_{i,j}\tau(i,j)\,\mathrm{Pois}(i;\lambda)\,\mathrm{Pois}(j;\mu)}
$$

Every market is a sum over regions of this single 11×11 distribution:

| Market | Region summed |
|---|---|
| Home / Draw / Away | $x>y$ , $x=y$ , $x<y$ |
| Over/Under ℓ | $x+y > \ell$ |
| Both teams to score | $x>0 \wedge y>0$ |
| Correct score | the single cell |
| Asian handicap *h* | $x+h>y$ (win), $x+h=y$ (push) |
| Double chance | unions of the 1X2 regions |
| Supremacy | via Skellam, see below |

Because they are marginals of one distribution they cannot disagree on fair probabilities.
The suite asserts this directly — correct-score cells summed over the lower triangle equal
$P(\text{home win})$, and the level-ball handicap equals draw-no-bet.

---

## Skellam: checking the model from outside itself

Deriving everything from one matrix is internally consistent **by construction** — which
means a systematic error in the matrix would be invisible to every consistency test above.

So the goal difference is computed a second, independent way. The difference of two Poisson
variables is Skellam-distributed (Karlis & Ntzoufras, 2009), with a closed form in the
modified Bessel function of the first kind:

$$
P(K = k) = e^{-(\lambda+\mu)}\left(\frac{\lambda}{\mu}\right)^{k/2} I_{|k|}\!\left(2\sqrt{\lambda\mu}\right)
$$

No matrix involved. Two assertions follow, and both are tested:

1. At **ρ = 0** the Bessel series and the matrix anti-diagonals **must agree** — corroborating
   the matrix from outside.
2. At **ρ ≠ 0** they **must diverge** on the draw, because Skellam assumes independence and τ
   deliberately breaks it. *If they agreed, τ would be doing nothing.*

---

## Margin

Fair odds are $1/p$. Margin solves for the exponent $k$ with

$$
\sum_i p_i^{\,k} = B, \qquad d_i = p_i^{-k}
$$

A constant multiplier takes the same proportional margin from every outcome; the exponent
does not, reproducing the **favourite–longshot bias**. Measured at $B{=}1.08$: fair-to-offered
ratio 1.105 for the longshot against 1.055 for the favourite.

Book sums land on target to $10^{-16}$, and a postcondition throws if a target is unreachable
rather than returning a plausible-looking wrong book.

> **The previous implementation had this backwards** — `p * (1 - margin)` *lengthens* prices,
> so the book paid out ~105% of fair value. The bettor had the edge.

**Double chance** is margined independently against $2B$, since each selection covers two of
three outcomes and a fair DC book already sums to 2. Summing the already-margined 1X2 legs
looks tidier but yields odds **below 1** for heavy favourites (break point: λ=2.5, μ=0.3).
Margining independently guarantees validity structurally, because $p^k<1$ for any $p<1,\,k>0$.

**Shin's method** (1993) is the inverse — recovering true probabilities *from* bookmaker odds
by modelling insider proportion $z$. It de-margins historical closing odds for backtesting; it
is never used to price our markets.

---

## ⚠️ Asian handicap carries NO margin

`FixturePricing.asianHandicaps` sits **outside** the `markets` array and its `fairOdds` are
exactly that — zero-margin. `applyOverround` is never run on it.

This is deliberate: AH legs are not a probability simplex (win + push + lose = 1 *per side*),
so a book-sum margin does not apply, and `1/fairHome + 1/fairAway` equals 1 by construction.

**A consumer rendering `fairOdds` directly would publish the highest-volume market at 0.00%
margin.** Applying a margin there is the caller's responsibility.

---

## Staking

$$
f^{*} = \frac{bp-q}{b}, \qquad b = d-1
$$

Note $bp - q = dp - 1$, so Kelly, edge and expected value can never disagree about whether a
bet is worth taking. Default is **quarter Kelly** — full Kelly is intolerably volatile once
the probability estimate itself carries error.

---

## Evaluation

Football outcomes are **ordered**, which Brier ignores: forecasting a home win scores the same
whether the match was drawn or lost.

$$
RPS = \frac{1}{r-1}\sum_{i=1}^{r-1}\left(\sum_{j\le i}(p_j - o_j)\right)^2
$$

RPS is distance-sensitive (Constantinou & Fenton, 2012). **This is contested** — Wheatcroft
(2021) argues distance sensitivity is not desirable here. Both are implemented so the choice
stays explicit.

**Murphy's decomposition** turns *"the model scored 0.58"* into a statement about **why**:

$$
BS = \underbrace{\text{REL}}_{\text{calibration}} - \underbrace{\text{RES}}_{\text{discrimination}} + \underbrace{\text{UNC}}_{\text{irreducible}} + \underbrace{\text{WBV}}_{\text{binning artefact}}
$$

The classical three-way identity is exact only when each bin holds a single distinct forecast.
Binning continuous forecasts leaves a residual equal to the within-bin variance — measured at
0.00025 on a sample where BS = 0.109 and the three-way form gives 0.10875. All four terms are
reported, the exact identity is tested to $10^{-12}$, and a separate test asserts the residual
**is** the WBV.

---

## Determinism

No `Math.random`, no wall-clock, no I/O in `src/` — enforced by a test that scans the source
tree and fails the build. Same inputs, byte-identical output, always.

The implementation this replaced had `Math.random()` **inside the pricing path**: odds changed
on every refresh, so a bettor could re-roll a price until it suited them.

---

## Usage

```ts
import { priceFixture, skellamSupremacyMarket, rankedProbabilityScore } from "@yami/quant-engine";

const pricing = priceFixture({ home: 1.6, away: 1.1 });
const matchResult = pricing.markets.find((m) => m.key === "1X2");

// Independent route to the goal difference
const supremacy = skellamSupremacyMarket(1.6, 1.1, -0.5);

// Evaluate a forecast against a realised outcome (0 = home, 1 = draw, 2 = away)
const score = rankedProbabilityScore([0.5, 0.3, 0.2], 1);
```

## Verify

```bash
npm run verify
```

## Limitations

Pre-match only; no in-play. Correlation corrected in four cells only. No explicit
overdispersion (see Boshnakov et al., 2017). No player-level data. Margined implied
probabilities are *not* consistent across markets — see [`docs/MODEL.md`](docs/MODEL.md) §8.3
for why that is forced rather than overlooked.
