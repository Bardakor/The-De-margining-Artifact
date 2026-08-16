# The Model

Complete mathematical specification of `@yami/quant-engine`. Every formula here is
implemented, tested, and cited. Numbers quoted as "measured" are actual outputs of this
engine, not illustrations.

---

## 1. The chain

$$
\text{ratings} \;\longrightarrow\; (\alpha_i, \beta_i, \gamma) \;\longrightarrow\; (\lambda, \mu) \;\longrightarrow\; P(x,y) \;\longrightarrow\; \text{markets} \;\longrightarrow\; \text{prices}
$$

One joint distribution over scorelines sits at the centre. Every market is a sum over
regions of it, so no two markets can disagree about the same event. Margin is applied
last, per market, and never mixed into the probability model.

---

## 2. Goals as a Poisson process

Maher (1982) established the working model: goals arrive as a Poisson process, with each
side's rate factored into an attack strength, an opponent's defence weakness, and a home
advantage.

$$
\lambda = \alpha_{\text{home}} \cdot \beta_{\text{away}} \cdot \gamma
\qquad
\mu = \alpha_{\text{away}} \cdot \beta_{\text{home}}
$$

$\lambda$ and $\mu$ are the expected goals for the home and away sides. The
identifiability constraint $\overline{\alpha} = 1$ across each league prevents the scale
degeneracy in which all attacks and all defences drift together.

Home advantage $\gamma$ is fit **per league**, not fixed globally. It genuinely differs
between competitions, and hardcoding it is one of the errors this engine was built to
correct.

The Poisson mass function is evaluated in log space,

$$
P(X = k) = \exp\!\big(k \ln \lambda - \lambda - \ln k!\big),
\qquad
\ln k! = \ln \Gamma(k+1)
$$

with $\ln\Gamma$ by the Lanczos approximation ($g=7$, $n=9$). The naive form
$\lambda^k / k!$ overflows to $\infty/\infty = \text{NaN}$ for large $k$; the log form
stays finite.

---

## 3. Dixon–Coles: the dependence correction

Independent Poisson is wrong in a specific, well-documented way: it **under-counts
low-scoring draws**. Dixon and Coles (1997) correct exactly the four affected cells and
leave the rest untouched.

$$
\tau(x,y) =
\begin{cases}
1 - \lambda\mu\rho & x=0,\, y=0\\[2pt]
1 + \lambda\rho & x=0,\, y=1\\[2pt]
1 + \mu\rho & x=1,\, y=0\\[2pt]
1 - \rho & x=1,\, y=1\\[2pt]
1 & \text{otherwise}
\end{cases}
$$

### 3.1 The sign of $\rho$ — where this is easy to get wrong

Read the formula: $\tau(0,0) = 1 - \lambda\mu\rho$ and $\tau(1,1) = 1 - \rho$. Both exceed
1 — that is, both *inflate* the low draws — **only when $\rho < 0$**.

A positive $\rho$ does the opposite of what the correction exists for. An earlier draft of
this engine shipped $\rho = +0.06$ as the default. Measured effect on a
$\lambda = 1.6,\ \mu = 1.1$ fixture:

| $\rho$ | $P(0\text{–}0)$ | $P(1\text{–}1)$ | $P(\text{draw})$ | fair draw odds |
|---:|---:|---:|---:|---:|
| $-0.20$ | 0.09086 | 0.14194 | 0.29622 | 3.376 |
| $-0.10$ | 0.07903 | 0.13011 | **0.27257** | **3.669** |
| $0.00$ | 0.06721 | 0.11828 | 0.24891 | 4.017 |
| $+0.06$ | 0.06011 | 0.11118 | 0.23472 | 4.260 |

The default is $\rho = -0.10$, giving a 27.3% draw rate — consistent with observed
football. Dixon and Coles' own fitted value is negative, around $-0.13$.

### 3.2 Admissible range

$\tau$ must leave all four cells non-negative, which bounds $\rho$:

$$
\max\!\left(-\tfrac{1}{\lambda},\, -\tfrac{1}{\mu}\right) \;\le\; \rho \;\le\; \min\!\left(\tfrac{1}{\lambda\mu},\, 1\right)
$$

The engine validates against this and throws rather than silently producing a negative
probability.

---

## 4. Estimation

Parameters are fit by maximum likelihood over historical results, with **exponential time
decay** so recent matches dominate:

$$
\mathcal{L}(\alpha,\beta,\gamma,\rho) = \prod_{m} \Big[ \tau(x_m, y_m)\, e^{-\lambda_m}\lambda_m^{x_m}\, e^{-\mu_m}\mu_m^{y_m} \Big]^{\varphi(t - t_m)}
$$

$$
\varphi(\Delta t) = e^{-\xi \Delta t}
$$

$\xi$ controls how fast the past is forgotten. Dixon and Coles' second contribution, after
$\tau$, was recognising that team strength is not static and that an unweighted likelihood
therefore mis-states it.

*Fitting is implemented in the data pipeline (Plan 2). The engine consumes fitted
parameters; it does not fit at request time.*

---

## 5. The scoreline matrix

$$
P(x,y) = \frac{\tau(x,y,\lambda,\mu,\rho)\; \mathrm{Pois}(x;\lambda)\; \mathrm{Pois}(y;\mu)}{\displaystyle\sum_{i=0}^{G}\sum_{j=0}^{G} \tau(i,j)\,\mathrm{Pois}(i;\lambda)\,\mathrm{Pois}(j;\mu)}
$$

over $0 \le x,y \le G$ with $G = 10$, an $11\times11$ grid. Renormalisation absorbs both
the truncated tail beyond $G$ and the mass shifted by $\tau$.

**Tested invariant:** the matrix sums to $1$ within $10^{-9}$.

---

## 6. Markets as marginals

Every market is a sum of matrix cells. No market has a model of its own.

| Market | Region summed |
|---|---|
| Home / Draw / Away | $x>y$ , $x=y$ , $x<y$ |
| Over/Under $\ell$ | $x + y > \ell$ |
| Both teams to score | $x>0 \wedge y>0$ |
| Correct score | the single cell $(x,y)$ |
| Asian handicap $h$ | $x + h > y$ (win), $x + h = y$ (push) |
| Double chance | unions of the 1X2 regions |

Because they are marginals of one distribution, they cannot contradict one another. This
is asserted directly — for instance, correct-score cells summed over the lower triangle
equal $P(\text{home win})$, and the level-ball Asian handicap equals draw-no-bet.

### 6.1 Asian handicap and the push

A handicap bet can push, so a single probability cannot describe it. Each side carries
$(\text{win}, \text{push}, \text{lose})$, and the fair price accounts for the stake being
returned on a push:

$$
d_{\text{fair}} = 1 + \frac{P(\text{lose})}{P(\text{win})}
$$

Quarter lines ($\pm0.25$, $\pm0.75$) split the stake evenly across the two adjacent lines,
which is how they settle in practice.

---

## 7. Skellam: an independent check on the whole model

Everything above derives from one matrix. That is internally consistent by construction —
which means a systematic error in the matrix would be invisible to every consistency test.

So the goal difference is *also* computed a completely different way. The difference of
two independent Poisson variables follows a **Skellam distribution** (Karlis and Ntzoufras,
2009), which has a closed form in the modified Bessel function of the first kind:

$$
P(K = k) = e^{-(\lambda+\mu)} \left(\frac{\lambda}{\mu}\right)^{k/2} I_{|k|}\!\left(2\sqrt{\lambda\mu}\right)
$$

$$
I_n(z) = \sum_{m=0}^{\infty} \frac{1}{m!\,(m+n)!} \left(\frac{z}{2}\right)^{2m+n}
$$

evaluated by log-sum-exp so it stays finite for large argument.

No matrix is involved. Where the Bessel series and the matrix anti-diagonals agree, the
matrix is corroborated **from outside itself**. That is a materially stronger claim than
internal consistency, and it is the only test in the package that makes it.

Two consequences are asserted:

1. At $\rho = 0$ the two derivations **must agree**.
2. At $\rho \neq 0$ they **must diverge** on the draw — because Skellam assumes
   independence and $\tau$ deliberately breaks it. If they agreed, $\tau$ would be doing
   nothing.

Skellam also gives supremacy markets directly, and provides a second route to the Asian
handicap.

---

## 8. Margin

### 8.1 The direction, which the previous implementation had backwards

The old code computed `probability * (1 - margin)`. That *lengthens* every price: the book
pays out roughly 105% of fair value rather than retaining 5%. The bettor had the edge.

### 8.2 The power method

Fair odds are $1/p$. To apply a margin, solve for the exponent $k$ such that

$$
\sum_i p_i^{\,k} = B
$$

where $B$ is the target book sum (default $1.05$). Since every $p_i \in (0,1)$, the sum is
strictly decreasing in $k$, so bisection converges. The offered price is

$$
d_i = p_i^{-k}
$$

A constant multiplier would take the same proportional margin from every outcome. The
exponent does not: it reproduces the **favourite–longshot bias**, taking proportionally
more from the longshot. Measured at $B = 1.08$ on a $\lambda=1.6,\ \mu=1.1$ fixture, the
ratio of fair to offered odds was 1.105 for the longshot against 1.055 for the favourite.

**Tested invariants:** $\sum_i 1/d_i = B$ exactly (measured to $10^{-16}$), and every
offered price is strictly shorter than its fair price. A postcondition throws if the
target is unreachable rather than returning a plausible-looking wrong book.

### 8.3 Double chance cannot be margined to 1

Each double-chance selection covers two of the three outcomes, so a *fair* DC book already
sums to 2. It is margined independently to $2B$.

The tempting alternative — sum the already-margined 1X2 legs, so the DC book lands at
exactly $2B$ — is wrong. For a heavy favourite the summed legs exceed implied probability
1, giving a decimal odd **below 1**, which is not a payable price. Measured break point:
$\lambda = 2.5,\ \mu = 0.3$, an ordinary heavy favourite.

Margining independently guarantees validity structurally, since $p^k < 1$ for any $p<1$
and $k>0$. Measured: DC:1X prices at 1.0053 for $\lambda=3.8,\ \mu=0.3$.

**The trade-off, stated honestly.** DC and 1X2 agree exactly on *fair* probabilities — both
are marginals of the same matrix — but their *margined* implied probabilities differ,
because the exponent is applied to each book separately. Cross-market implied consistency
and valid double-chance prices are mutually exclusive under this architecture. Validity
wins; the inconsistency is a documented non-goal, not an oversight.

---

## 9. Shin's method — the inverse

Shin (1993) models a bookmaker's prices as containing a proportion $z$ of insider money,
and inverts to recover the true probabilities from the offered ones:

$$
\pi_i = \frac{\sqrt{z^2 + 4(1-z)\dfrac{p_i^2}{B}} - z}{2(1-z)}
$$

with $z$ solved by bisection so that $\sum_i \pi_i = 1$.

**This is not used to price our markets.** It runs in the opposite direction: it de-margins
*historical closing odds* so the model can be benchmarked against the market on equal
terms. Conflating the two — using an inverse method to apply a margin — is a real and
easy mistake.

Measured round trip: implied $[0.5, 0.35, 0.25]$ (book sum 1.10) recovers
$[0.46344, 0.31699, 0.21956]$ at $z = 0.0502$, and substituting back through the forward
relation reproduces the inputs to within $1.1\times10^{-16}$.

---

## 10. Staking

Kelly (1956) gives the bankroll fraction maximising long-run logarithmic growth:

$$
f^{*} = \frac{bp - q}{b}, \qquad b = d - 1,\; q = 1 - p
$$

Note the identity $bp - q = dp - 1$, so $f^{*} > 0$ exactly when the expected value
$dp - 1$ is positive. Edge, EV and Kelly can never disagree about whether a bet is worth
taking.

A negative $f^{*}$ means *do not bet* — not *bet the other side*, which has its own price
and its own assessment. The default is **quarter Kelly**: full Kelly is intolerably
volatile once the probability estimate itself carries error.

---

## 11. Evaluation

A model is only as good as the metric judging it, and football has an awkward property:
its outcomes are **ordered**. Home, draw, away is not an arbitrary set of three labels.

**Brier score** (Brier, 1950), summed over categories:

$$
BS = \sum_{j} (p_j - o_j)^2
$$

Brier is blind to ordering. Forecasting a home win when the match is drawn scores the same
as forecasting a home win when the away side wins.

**Ranked probability score** (Epstein, 1969; argued for football by Constantinou and
Fenton, 2012) is *distance-sensitive*:

$$
RPS = \frac{1}{r-1} \sum_{i=1}^{r-1} \left( \sum_{j=1}^{i} (p_j - o_j) \right)^{2}
$$

A near miss is punished less than a distant one.

**This is contested.** Wheatcroft (2021) argues distance sensitivity is not in fact
desirable here and that RPS should not be the default choice. Both metrics are implemented
so the decision stays explicit rather than assumed by whoever wrote the harness first.

**Murphy's decomposition** (Murphy, 1973) partitions the Brier score:

$$
BS = \underbrace{\text{REL}}_{\text{calibration}} - \underbrace{\text{RES}}_{\text{discrimination}} + \underbrace{\text{UNC}}_{\text{irreducible}}
$$

$$
\text{REL} = \frac{1}{N}\sum_k n_k (\bar{p}_k - \bar{o}_k)^2,\quad
\text{RES} = \frac{1}{N}\sum_k n_k (\bar{o}_k - \bar{o})^2,\quad
\text{UNC} = \bar{o}(1-\bar{o})
$$

Reliability asks whether the stated probabilities match observed frequencies — zero means
perfectly calibrated. Resolution asks whether the model says anything beyond the base rate
— zero means it does not. Uncertainty belongs to the sport, not the model.

This is what turns *"the model scored 0.58"* into a statement about **why**.

### 11.1 The identity is not exact for continuous forecasts

Stated carefully, because it is easy to get wrong and this engine got it wrong first time.

The three-way identity above is exact **only when each bin contains a single distinct
forecast value** — the discrete case Murphy was writing about. Bin *continuous* forecasts
and a residual appears, because REL compares each bin's **mean** forecast against its
observed frequency while BS uses each **individual** forecast. The residual is exactly the
within-bin variance:

$$
\text{WBV} = \frac{1}{N}\sum_k \sum_{i \in k} (p_i - \bar{p}_k)^2
$$

so the identity that holds for arbitrary forecasts is

$$
BS = \text{REL} - \text{RES} + \text{UNC} + \text{WBV}
$$

Measured on forecasts $[0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95]$ over 10 bins:

| quantity | value |
|---|---:|
| $BS$ | 0.10900 |
| $\text{REL} - \text{RES} + \text{UNC}$ | 0.10875 |
| residual | 0.00025 |
| $\text{WBV}$ | **0.00025** |

The gap comes entirely from the two bins holding two distinct values each,
$[0.2, 0.25]$ and $[0.8, 0.85]$. The engine reports all four terms and tests the exact
identity to $10^{-12}$, plus a separate test asserting that the residual **is** the WBV —
so nobody can quietly "simplify" it back to three terms.

---

## 12. Determinism

No `Math.random`, no wall-clock time, no I/O anywhere in `src/`. Enforced by a test that
scans the source tree and fails the build on violation. Same inputs, byte-identical output,
always.

This is not fastidiousness. The implementation this replaced had `Math.random()` inside the
pricing path, so odds changed on every page refresh — a bettor could re-roll a price until
it suited them.

---

## 13. What this model does not do

Stated plainly, because a model's limits are part of its specification:

- **No in-play modelling.** Pre-match only; no time-decay of $\lambda$ within a match, no
  red-card or score-state adjustment.
- **Independence beyond $\tau$.** Correlation is corrected in four cells. Real dependence
  extends further — a side chasing a game concedes differently.
- **No explicit overdispersion.** Real goal counts are slightly overdispersed relative to
  Poisson. Bivariate Weibull counts (Boshnakov, Kharrat and McHale, 2017) address this;
  this engine does not.
- **No player-level information.** Injuries, suspensions and rotation enter only through
  their effect on fitted team strength, with a lag.
- **Static within a matchday.** Parameters are fit in batch, not updated on news.
- **Cross-market margin inconsistency**, as described in §8.3.

---

## References

- Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Kelly, J.L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4), 917–926.
- Epstein, E.S. (1969). A scoring system for probability forecasts of ranked categories. *Journal of Applied Meteorology*, 8, 985–987.
- Murphy, A.H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12, 595–600.
- Maher, M.J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.
- Shin, H.S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
- Dixon, M.J. and Coles, S.G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265–280.
- Rue, H. and Salvesen, Ø. (2000). Prediction and retrospective analysis of soccer matches in a league. *Journal of the Royal Statistical Society: Series D*, 49(3), 399–418.
- Karlis, D. and Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models. *Journal of the Royal Statistical Society: Series D*, 52(3), 381–393.
- Karlis, D. and Ntzoufras, I. (2009). Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference. *IMA Journal of Management Mathematics*, 20(2), 133–145.
- Constantinou, A.C. and Fenton, N.E. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast models. *Journal of Quantitative Analysis in Sports*, 8(1).
- Boshnakov, G., Kharrat, T. and McHale, I.G. (2017). A bivariate Weibull count model for forecasting association football scores. *International Journal of Forecasting*, 33(2), 458–466.
- Ley, C., Van de Wiele, T. and Van Eetvelde, H. (2019). Ranking soccer teams on the basis of their current strength: a comparison of maximum likelihood approaches. *Statistical Modelling*, 19(1), 55–73.
- Wheatcroft, E. (2021). Evaluating probabilistic forecasts of football matches: the case against the ranked probability score. *Journal of Quantitative Analysis in Sports*, 17(4), 273–287.
