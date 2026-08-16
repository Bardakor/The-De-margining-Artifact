# Yami — Football Betting Platform

A betting platform built around a tested implementation of the **Dixon–Coles bivariate
Poisson model**. The pricing engine is pure TypeScript, has zero runtime dependencies,
137 tests, and no randomness anywhere in the pricing path.

📐 **[Full mathematical specification → `packages/quant-engine/docs/MODEL.md`](packages/quant-engine/docs/MODEL.md)**

> **Status: mid-rebuild.** An earlier version of this README claimed Poisson modelling,
> Expected Goals and a Kelly Criterion implementation. **None existed.** The pricing path ran
> on `Math.random()`, the draw probability was a residual (`1 − home − away`) which sent draw
> odds past 6.0, and the margin was applied backwards so the book paid out ~105% of fair
> value. Those defects are catalogued with file/line references in the
> [rebuild spec](docs/superpowers/specs/2026-08-15-quant-rebuild-design.md). Stage 1 of 5 —
> the engine — is complete and is what this document describes. `backend/*` and `frontend/`
> are still legacy.

---

# The Model

$$
\text{ratings} \to (\alpha_i, \beta_i, \gamma) \to (\lambda, \mu) \to P(x,y) \to \text{markets} \to \text{prices}
$$

## 1. Goals as a Poisson process

Maher (1982) established the working model: goals arrive as a Poisson process, with each
side's rate factored into attack strength, opponent defence weakness, and home advantage.

$$
\lambda = \alpha_{\text{home}}\,\beta_{\text{away}}\,\gamma
\qquad
\mu = \alpha_{\text{away}}\,\beta_{\text{home}}
$$

Identifiability requires $\overline{\alpha} = 1$ per league, preventing the degeneracy where
all attacks and defences drift together. Home advantage $\gamma$ is fit **per league** — it
genuinely differs between competitions.

The mass function is evaluated in log space,

$$
P(X=k) = \exp\big(k\ln\lambda - \lambda - \ln\Gamma(k{+}1)\big)
$$

with $\ln\Gamma$ by Lanczos approximation. The naive $\lambda^k/k!$ overflows to
$\infty/\infty = \mathrm{NaN}$.

## 2. The Dixon–Coles dependence correction

Independent Poisson is wrong in one documented way: it **under-counts low-scoring draws**.
Dixon and Coles (1997) correct exactly the four affected cells.

$$
\tau(x,y)=
\begin{cases}
1-\lambda\mu\rho & (0,0)\\
1+\lambda\rho & (0,1)\\
1+\mu\rho & (1,0)\\
1-\rho & (1,1)\\
1 & \text{otherwise}
\end{cases}
$$

### 2.1 The sign of ρ

Since $\tau(0,0)=1-\lambda\mu\rho$ and $\tau(1,1)=1-\rho$, both exceed 1 — inflating the low
draws — **only when $\rho<0$**. Measured on $\lambda{=}1.6,\ \mu{=}1.1$:

| ρ | P(0-0) | P(1-1) | **P(draw)** | fair draw odds |
|---:|---:|---:|---:|---:|
| −0.20 | 0.09086 | 0.14194 | 0.29622 | 3.376 |
| **−0.10** (default) | 0.07903 | 0.13011 | **0.27257** | **3.669** |
| 0.00 | 0.06721 | 0.11828 | 0.24891 | 4.017 |
| +0.06 | 0.06011 | 0.11118 | 0.23472 | 4.260 |

A positive ρ *suppresses* the draws the correction exists to raise. Dixon and Coles' fitted
value is negative, ≈ −0.13. Admissibility requires

$$
\max\!\left(-\tfrac1\lambda,-\tfrac1\mu\right) \le \rho \le \min\!\left(\tfrac{1}{\lambda\mu},1\right)
$$

which the engine validates rather than silently emitting a negative probability.

## 3. Estimation

Parameters are fit by maximum likelihood with **exponential time decay**, so recent matches
dominate — Dixon and Coles' second contribution after $\tau$:

$$
\mathcal{L} = \prod_m \Big[\tau(x_m,y_m)\,e^{-\lambda_m}\lambda_m^{x_m}\,e^{-\mu_m}\mu_m^{y_m}\Big]^{\varphi(t-t_m)},
\qquad \varphi(\Delta t)=e^{-\xi\Delta t}
$$

*Fitting lands in Plan 2. The engine consumes fitted parameters; it does not fit at request
time.*

## 4. The scoreline matrix

$$
P(x,y)=\frac{\tau(x,y)\,\mathrm{Pois}(x;\lambda)\,\mathrm{Pois}(y;\mu)}{\sum_{i,j}\tau(i,j)\,\mathrm{Pois}(i;\lambda)\,\mathrm{Pois}(j;\mu)}
$$

over an $11\times11$ grid. Renormalisation absorbs the truncated tail and the mass shifted
by $\tau$. **Tested: sums to 1 within $10^{-9}$.**

## 5. Markets as marginals

| Market | Region summed |
|---|---|
| Home / Draw / Away | $x>y$ , $x=y$ , $x<y$ |
| Over/Under $\ell$ | $x+y>\ell$ |
| Both teams to score | $x>0 \wedge y>0$ |
| Correct score | the single cell |
| Asian handicap $h$ | $x+h>y$ (win), $x+h=y$ (push) |
| Double chance | unions of the 1X2 regions |
| Supremacy | via Skellam (§6) |

Marginals of one distribution cannot disagree. Asserted directly: correct-score cells over
the lower triangle equal $P(\text{home win})$; the level-ball handicap equals draw-no-bet.

An Asian handicap can push, so one probability cannot describe it. Each side carries
$(\text{win},\text{push},\text{lose})$ with

$$
d_{\text{fair}} = 1+\frac{P(\text{lose})}{P(\text{win})}
$$

Quarter lines split the stake across the two adjacent lines.

## 6. Skellam — checking the model from outside itself

Deriving everything from one matrix is internally consistent **by construction**, so a
systematic matrix error would be invisible to every consistency test above.

The goal difference is therefore computed a second, independent way. The difference of two
Poisson variables is Skellam-distributed (Karlis & Ntzoufras, 2009), with a closed form in
the modified Bessel function of the first kind:

$$
P(K=k)=e^{-(\lambda+\mu)}\left(\frac{\lambda}{\mu}\right)^{k/2} I_{|k|}\!\left(2\sqrt{\lambda\mu}\right),
\qquad
I_n(z)=\sum_{m\ge0}\frac{(z/2)^{2m+n}}{m!\,(m+n)!}
$$

evaluated by log-sum-exp. **No matrix involved.** Two assertions, both tested:

1. At $\rho=0$ the Bessel series and the matrix anti-diagonals **must agree** — corroborating
   the matrix from outside.
2. At $\rho\neq0$ they **must diverge** on the draw, since Skellam assumes independence and
   $\tau$ deliberately breaks it. *If they agreed, $\tau$ would be doing nothing.*

## 7. Margin

Fair odds are $1/p$. The margin solves for exponent $k$:

$$
\sum_i p_i^{\,k}=B,\qquad d_i=p_i^{-k}
$$

Monotonic in $k$ since $p_i\in(0,1)$, so bisection converges. A constant multiplier takes the
same proportional margin from every outcome; the exponent does not, reproducing the
**favourite–longshot bias**. Measured at $B{=}1.08$: fair-to-offered ratio **1.105** for the
longshot vs **1.055** for the favourite. Book sums land on target to $10^{-16}$; a
postcondition throws if the target is unreachable rather than returning a plausible-looking
wrong book.

> The previous implementation computed `p * (1 - margin)`, which **lengthens** prices — the
> book paid out ~105% of fair value and the bettor held the edge.

**Double chance** is margined independently against $2B$, since each selection covers two of
three outcomes and a fair DC book sums to 2. Summing the already-margined 1X2 legs looks
tidier but yields odds **below 1** for heavy favourites (break point $\lambda{=}2.5,\ \mu{=}0.3$).
Margining independently guarantees validity structurally, as $p^k<1$ for any $p<1,k>0$.

## 8. Shin's method — the inverse

Shin (1993) models bookmaker prices as containing a proportion $z$ of insider money:

$$
\pi_i=\frac{\sqrt{z^2+4(1-z)p_i^2/B}-z}{2(1-z)}
$$

with $z$ solved so $\sum\pi_i=1$. **Not used to price our markets** — it runs the other way,
de-margining *historical closing odds* so the model can be benchmarked against the market on
equal terms. Measured round trip: implied $[0.5,0.35,0.25]$ recovers
$[0.46344,0.31699,0.21956]$ at $z=0.0502$, reproducing the inputs to $1.1\times10^{-16}$.

## 9. Staking

Kelly (1956):

$$
f^{*}=\frac{bp-q}{b},\qquad b=d-1,\ q=1-p
$$

Note $bp-q = dp-1$, so Kelly, edge and expected value can never disagree about whether a bet
is worth taking. Negative $f^*$ means *do not bet*, not *bet the other side*. Default is
**quarter Kelly** — full Kelly is intolerably volatile once the probability estimate itself
carries error.

## 10. Evaluation

Football outcomes are **ordered**, which Brier (1950) ignores — forecasting a home win scores
the same whether the match was drawn or lost.

$$
BS=\sum_j (p_j-o_j)^2
\qquad
RPS=\frac{1}{r-1}\sum_{i=1}^{r-1}\left(\sum_{j\le i}(p_j-o_j)\right)^2
$$

RPS (Epstein, 1969) is distance-sensitive; Constantinou and Fenton (2012) argue it is
therefore the right metric for football. **This is contested** — Wheatcroft (2021) argues
distance sensitivity is not desirable here. Both are implemented so the choice stays explicit.

**Murphy's (1973) decomposition** turns *"the model scored 0.58"* into a statement about
**why**:

$$
BS=\underbrace{\text{REL}}_{\text{calibration}}-\underbrace{\text{RES}}_{\text{discrimination}}+\underbrace{\text{UNC}}_{\text{irreducible}}+\underbrace{\text{WBV}}_{\text{binning artefact}}
$$

$$
\text{REL}=\tfrac1N\textstyle\sum_k n_k(\bar p_k-\bar o_k)^2,\quad
\text{RES}=\tfrac1N\textstyle\sum_k n_k(\bar o_k-\bar o)^2,\quad
\text{UNC}=\bar o(1-\bar o)
$$

The classical three-way identity is exact **only when each bin holds a single distinct
forecast value**. Binning continuous forecasts leaves a residual equal to the within-bin
variance $\text{WBV}=\frac1N\sum_k\sum_{i\in k}(p_i-\bar p_k)^2$. Measured: $BS=0.10900$
against $\text{REL}-\text{RES}+\text{UNC}=0.10875$, a gap of $0.00025$ — exactly the WBV. All
four terms are reported, the exact identity tested to $10^{-12}$, with a separate test
asserting the residual **is** the WBV.

## 11. Determinism

No `Math.random`, no wall-clock, no I/O in `src/` — enforced by a test that scans the source
tree and fails the build. Same inputs, byte-identical output, always. The implementation this
replaced had `Math.random()` *inside the pricing path*: odds changed on every refresh, so a
bettor could re-roll a price until it suited them.

---

# Verified Output

Actual output of `priceFixture({ home: 1.62, away: 1.18 })`. Reproduce with `npm run verify`.

```jsonc
// 1X2 — book sum exactly 1.05
{ "key": "1X2:HOME", "probability": 0.4644, "fairOdds": 2.153, "odds": 2.079 }
{ "key": "1X2:DRAW", "probability": 0.2692, "fairOdds": 3.715, "odds": 3.498 }
{ "key": "1X2:AWAY", "probability": 0.2664, "fairOdds": 3.754, "odds": 3.533 }

// Over/Under 2.5                            // Both teams to score
{ "OU:2.5:OVER":  0.5305, "odds": 1.802 }    { "BTTS:YES": 0.5673, "odds": 1.693 }
{ "OU:2.5:UNDER": 0.4695, "odds": 2.019 }    { "BTTS:NO":  0.4327, "odds": 2.177 }

// Double chance — book sum exactly 2.10
{ "key": "DC:1X", "probability": 0.7336, "fairOdds": 1.363, "odds": 1.312 }

// assessValue(modelProbability: 0.4471, offeredOdds: 2.45)
{ "edge": 0.04895, "expectedValue": 0.09540, "fullKelly": 0.06579, "stake": 0.01645 }
```

Three properties the old implementation violated: the **draw prices at 3.50**, not past 6.0;
**every offered price is shorter than its fair price**; and **`DC:1X` = 0.7336 = 0.4644 +
0.2692 exactly**, because the markets are marginals of one distribution.

---

# Architecture

```
packages/quant-engine/     ← the real work. Pure TS, 0 deps, 137 tests
  math/                    log-gamma, modified Bessel
  poisson/                 log-space Poisson, Dixon-Coles matrix
  markets/                 1X2, totals, BTTS, correct score, AH, DC
  pricing/                 power-method overround, Shin inverse, Kelly
  skellam/                 goal difference (independent cross-check)
  calibration/             RPS, Brier, log loss, Murphy decomposition

backend/                   ← LEGACY. Six Express services, being consolidated
frontend/                  ← LEGACY. Next.js 15, still on the old endpoints
```

The six backend services share a database, deploy together, and are called in a strict
request-scoped sequence. That separation bought no independent scaling and produced three
competing, mutually inconsistent odds implementations. The
[rebuild spec](docs/superpowers/specs/2026-08-15-quant-rebuild-design.md) §2.2 collapses them
into one API. Stack: Node 18+, Express, MongoDB, Next.js 15, React 19, Tailwind.

## Quick start

```bash
npm install
```

Engine only — no database or API keys needed:

```bash
cd packages/quant-engine && npm run verify
```

Full stack (starts MongoDB via Docker, then all services):

```bash
chmod +x bash/clean-and-dev.sh && ./bash/clean-and-dev.sh
```

Frontend `:3000` · Gateway `:8080` · Main API `:3001` · MongoDB `:27017` · Mongo Express `:8081`.
Seeded accounts: `admin@admin.com` / `admin123`, and `user@demo.com` / `demo123`.

---

# Limitations

Stated plainly, because a model's limits are part of its specification.

- **Pre-match only.** No in-play modelling, no within-match $\lambda$ decay, no red-card or
  score-state adjustment.
- **Independence beyond $\tau$.** Correlation is corrected in four cells; real dependence
  extends further.
- **No explicit overdispersion.** Real goal counts are slightly overdispersed relative to
  Poisson; bivariate Weibull counts (Boshnakov et al., 2017) address this, this engine does not.
- **No player-level data.** Injuries and rotation enter only via fitted team strength, with a lag.
- **Ratings and fitting are not yet implemented** — Plan 2. The engine consumes $\lambda,\mu$.
- **No backtest yet**, so no Brier or ROI figure is claimed. Plan 2.
- **Margined implied probabilities are not consistent across markets.** Fair probabilities
  agree exactly; margined ones do not, because each book carries its own exponent. This is
  forced — cross-market implied consistency and valid double-chance prices are mutually
  exclusive. See [`MODEL.md`](packages/quant-engine/docs/MODEL.md) §8.3.

---

# References

1. **Brier, G.W.** (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
2. **Kelly, J.L.** (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
3. **Epstein, E.S.** (1969). A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology*, 8, 985–987.
4. **Murphy, A.H.** (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12, 595–600.
5. **Maher, M.J.** (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.
6. **Shin, H.S.** (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
7. **Dixon, M.J. and Coles, S.G.** (1997). Modelling association football scores and inefficiencies in the football betting market. *JRSS Series C*, 46(2), 265–280.
8. **Rue, H. and Salvesen, Ø.** (2000). Prediction and retrospective analysis of soccer matches in a league. *JRSS Series D*, 49(3), 399–418.
9. **Karlis, D. and Ntzoufras, I.** (2003). Analysis of sports data by using bivariate Poisson models. *JRSS Series D*, 52(3), 381–393.
10. **Karlis, D. and Ntzoufras, I.** (2009). Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference. *IMA Journal of Management Mathematics*, 20(2), 133–145.
11. **Constantinou, A.C. and Fenton, N.E.** (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*, 8(1).
12. **Boshnakov, G., Kharrat, T. and McHale, I.G.** (2017). A bivariate Weibull count model for forecasting association football scores. *International Journal of Forecasting*, 33(2), 458–466.
13. **Ley, C., Van de Wiele, T. and Van Eetvelde, H.** (2019). Ranking soccer teams on the basis of their current strength: a comparison of maximum likelihood approaches. *Statistical Modelling*, 19(1), 55–73.
14. **Wheatcroft, E.** (2021). Evaluating probabilistic forecasts of football matches: the case against the ranked probability score. *Journal of Quantitative Analysis in Sports*, 17(4), 273–287.
15. **Abramowitz, M. and Stegun, I.A.** (1964). *Handbook of Mathematical Functions*. Table 9.8 — reference values for the modified Bessel function, used in the test suite.

---

## Documentation

- **[Mathematical specification](packages/quant-engine/docs/MODEL.md)** — full derivations
- **[Engine README](packages/quant-engine/README.md)** — API and design notes
- **[Rebuild spec](docs/superpowers/specs/2026-08-15-quant-rebuild-design.md)** — the 11 catalogued defects and the plan
- **[Implementation plans](docs/superpowers/plans/)** — stage-by-stage

## Licence

MIT. Educational project — not licensed gambling software, and no real-money wagering.
