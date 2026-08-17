# Paper Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Raise the manuscript from "rigorous but the headline failed" to a defensible submission, without touching the pre-registration or altering a single reported outcome.

**Architecture:** Everything new is generated from `results/`, exactly like the existing tables. Figures come from a new `scripts/make_figures.py` writing PDFs into `paper/generated/`; the sensitivity sweep comes from a new `scripts/sensitivity.py` writing `results/sensitivity.csv`. The paper gains prose and positioning; the science it reports does not move.

**Tech Stack:** matplotlib (new dependency), existing NumPy/SciPy/pandas, LaTeX.

**Plan 6 of 6.** Follows the completed study.

## Global Constraints

- **The pre-registration is immutable.** `paper/preregistration.md` is not edited by any task in this plan. Its value rests on not having changed after the evaluation ran.
- **No reported outcome may change.** P1, P2 and P3 are reported exactly as they came out. The reframe changes framing, title and emphasis — never a number, never a verdict.
- Every quantity in the paper stays generated. No task may type a result into `main.tex`.
- `make verify` green before every commit: ruff, `mypy --strict`, pytest.
- New code carries tests. Figures are exempt from unit tests but their **data preparation** is not.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/make_figures.py` | Five figures, from `results/` into `paper/generated/` |
| `src/footy/eval/effect.py` | Effect-size translation and bootstrap CI for a correlation |
| `scripts/sensitivity.py` | Re-runs P1/P2 across threshold settings |
| `tests/test_effect.py` | Tests for the new evaluation helpers |
| `paper/main.tex` | Figures, sensitivity section, reframed front matter, Štrumbelj engagement |
| `paper/refs.bib` | Any citations added by Task 6 |

---

### Task 1: Effect-size translation and correlation intervals

The paper reports an RPS spread of ~1e-4 and never says whether that is a lot. It reports ρ = 0.594 on 14 books with no interval. Both are reviewer questions with cheap answers.

**Files:**
- Create: `src/footy/eval/effect.py`, `tests/test_effect.py`

**Interfaces:**
- Produces:
  - `spearman(x, y) -> float`
  - `bootstrap_correlation(x, y, *, rng, n_resamples=10_000, level=0.95) -> tuple[float, float, float]` — (rho, low, high)
  - `spread_as_share_of_gap(spread: float, model_gap: float) -> float`
  - `equivalent_sample_size(spread: float, per_match_sd: float) -> float` — matches needed for the spread to equal one standard error

- [ ] **Step 1: Write the failing test**

```python
def test_spearman_matches_a_known_ranking() -> None:
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 0.7 * x + rng.normal(size=60)
    rho, low, high = bootstrap_correlation(x, y, rng=rng, n_resamples=2000)
    assert low < rho < high
    assert -1.0 <= low and high <= 1.0


def test_a_small_sample_gives_a_wide_interval() -> None:
    """The point of reporting it: rho on 14 books is not a precise quantity."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=14)
    y = 0.6 * x + rng.normal(size=14)
    _, low, high = bootstrap_correlation(x, y, rng=rng, n_resamples=2000)
    assert high - low > 0.4


def test_spread_expressed_against_the_model_gap() -> None:
    """An RPS spread of 1e-4 against a model-market gap of 6.5e-3 is 1.5% of
    the quantity a paper would be claiming."""
    assert spread_as_share_of_gap(1e-4, 6.5e-3) == pytest.approx(0.01538, abs=1e-5)


def test_equivalent_sample_size_grows_as_the_spread_shrinks() -> None:
    big = equivalent_sample_size(1e-3, per_match_sd=0.2)
    small = equivalent_sample_size(1e-4, per_match_sd=0.2)
    assert small > big
```

- [ ] **Step 2: Run and confirm failure.** `uv run pytest tests/test_effect.py -v` → module not found.
- [ ] **Step 3: Implement `src/footy/eval/effect.py`.** `spearman` via `scipy.stats.spearmanr`; `bootstrap_correlation` resampling pairs with the supplied generator; `spread_as_share_of_gap` a ratio with a zero guard; `equivalent_sample_size` returning `(per_match_sd / spread) ** 2`, documented as the sample at which the spread equals one standard error of a mean.
- [ ] **Step 4: Run and confirm pass.**
- [ ] **Step 5: Commit.**

---

### Task 2: The five figures

**Files:**
- Create: `scripts/make_figures.py`
- Modify: `pyproject.toml` (add `matplotlib`), `Makefile` (add `figures` target)

**REQUIRED SUB-SKILL: invoke the `dataviz` skill before writing any plotting code.** It sets the palette, mark specs and axis rules. Do not improvise chart design.

Figures, each written to `paper/generated/fig-*.pdf`, vector, greyscale-safe:

| Figure | Content | Argument it carries |
|---|---|---|
| `fig-margin-spread` | Scatter: mean book sum (x) against mean transform spread (y), one point per bookmaker, labelled, with the fitted trend and ρ with its bootstrap CI | P1, currently a 14-row table |
| `fig-calibration` | Paired mean RPS against half-life, log x, optimum marked, no-decay baseline as a horizontal rule | The optimum is interior and the curve is flat around it |
| `fig-roi` | Distribution of per-cell ROI across all 91 cells, one panel per transform, zero line drawn | The model has no edge — the finding that governs P3 |
| `fig-divergence` | For one worked closing book, the four transforms' recovered probabilities against raw implied, annotated at the longshot | *Why* the transforms differ, which no table shows |
| `fig-cell-scatter` | Per-cell model RPS against benchmark RPS, diagonal drawn | Near-universal loss to the market, at a glance |

- [ ] **Step 1: Invoke the `dataviz` skill.** Follow its palette and form guidance.
- [ ] **Step 2: Add matplotlib to `pyproject.toml`, run `uv sync --extra dev`.**
- [ ] **Step 3: Write `scripts/make_figures.py`.** Reads `results/cells.csv`, `results/calibration.json`, `results/book_margin.csv`. Each figure in its own function returning a `Figure`. A `main()` writing all five. Must fail loudly if an input is missing, never emit an empty axis.
- [ ] **Step 4: Add a `figures` target to the Makefile** and make `paper` depend on it.
- [ ] **Step 5: Run and inspect every figure.** Check: axis labels carry units, no clipped text, readable at print size, legible in greyscale.
- [ ] **Step 6: Reference all five from `main.tex`** with `\IfFileExists` fallbacks matching the existing table pattern, each with a caption stating what the reader should take from it.
- [ ] **Step 7: `make paper`, confirm 0 missing-figure warnings. Commit.**

---

### Task 3: Sensitivity analysis

Every threshold in the study is defensible and arbitrary. A reviewer will ask whether P1 and P2 survive other choices. The apparatus already exists, so this is a sweep, not new science.

**Files:**
- Create: `scripts/sensitivity.py`
- Modify: `scripts/make_tables.py` (a `tab-sensitivity.tex`), `paper/main.tex`

Settings to vary, one at a time from the registered baseline:

| Parameter | Registered | Also try |
|---|---|---|
| Minimum fixtures per cell | 200 | 100, 400 |
| Minimum fit-window matches | 300 | 200, 500 |
| Burn-in / calibration seasons | 3 / 2 | 5 / 3 |
| Team weight floor | 1.0 | 0.5, 2.0 |

- [ ] **Step 1: Write `scripts/sensitivity.py`.** For each setting, recompute P1's ρ and P2's direction and share. Reuse cached forecasts where the setting does not change them; only the split and the weight floor force a re-run of the walk-forward, and those two are the expensive ones.
- [ ] **Step 2: Run it, write `results/sensitivity.csv`.**
- [ ] **Step 3: Generate `tab-sensitivity.tex` from that file.**
- [ ] **Step 4: Add a Sensitivity subsection to Results**, stating plainly whether the conclusions move. **If P1 or P2 flips under any setting, that is reported, not buried** — a conclusion that survives only its registered thresholds is a finding about the thresholds.
- [ ] **Step 5: `make verify`, `make paper`, commit.**

---

### Task 4: State what the effect sizes mean

**Files:**
- Modify: `scripts/make_tables.py` (new macros), `paper/main.tex`

- [ ] **Step 1: Emit macros** for: the RPS spread as a share of the model–market gap; the equivalent sample size at which the spread equals one standard error; ρ with its bootstrap interval.
- [ ] **Step 2: Add a paragraph to the P1 subsection** translating the spread into terms a reader can judge. The honest framing: the disagreement is small in absolute RPS but is a non-trivial fraction of the quantity a paper claiming an edge would be reporting — and that ratio, not the raw spread, is the reviewer-relevant number.
- [ ] **Step 3: Report ρ with its interval everywhere ρ appears**, including the abstract. On 14 books the interval will be wide; that is the point.
- [ ] **Step 4: Commit.**

---

### Task 5: Reframe as a measurement paper

The data strongly supports *how much do these transforms disagree, and where does the disagreement concentrate*. It does not support *the artifact overturns conclusions*. The paper should lead with what it measured.

**This task changes framing only. P1, P2 and P3 are still reported exactly as registered, with the same verdicts. `paper/preregistration.md` is not touched.**

**Files:**
- Modify: `paper/main.tex`

- [ ] **Step 1: Retitle.** Something in the register of *"How much do de-margining transforms disagree, and when does it matter? A pre-registered study on 176,000 matches"*. The current title asks a question the study could not answer.
- [ ] **Step 2: Rewrite the abstract** to lead with the measurement result (P1, P2, the magnitudes) and report the P3 null as what it is — including that the model's absence of edge made P3 untestable here.
- [ ] **Step 3: Rewrite the introduction's contribution paragraph** to claim the measurement, not the artifact.
- [ ] **Step 4: Add a short subsection, "What would settle P3"**, specifying the conditions a follow-up needs: a forecaster competitive with closing prices in some segment, or a slower market, or opening lines. State that the apparatus runs it unchanged.
- [ ] **Step 5: Verify no verdict changed.** `git diff` must show no edit to any P1/P2/P3 outcome sentence, and `paper/preregistration.md` must be untouched. **This check is the task's gate.**
- [ ] **Step 6: Commit.**

---

### Task 6: Engage the prior literature

Štrumbelj (2014) is the closest prior work and is currently cited without being engaged. A reviewer who knows the field will notice immediately.

**Files:**
- Modify: `paper/main.tex`, `paper/refs.bib`

- [ ] **Step 1: Read Štrumbelj (2014) properly** and establish what it concluded about the relative accuracy of these transforms — it compares them as probability estimators, which is adjacent to but distinct from this study's question.
- [ ] **Step 2: Add a Related Work subsection** stating what is already known, what this study adds (the pre-registered walk-forward design and the margin-scaling relationship), and where the two agree or differ.
- [ ] **Step 3: Verify every claim about prior work against the source.** Do not characterise a paper from its abstract. If a claim cannot be checked, do not make it.
- [ ] **Step 4: Check the four unused bib entries** — cite them where they belong or delete them.
- [ ] **Step 5: Commit.**

---

## Success criteria

1. `make verify` green: ruff, `mypy --strict`, all tests.
2. `make figures && make tables && make paper` reproduces every figure, table and number from `results/`.
3. Five figures present, referenced, captioned, and legible in greyscale.
4. Sensitivity table present, with the conclusions' stability stated either way.
5. ρ reported with an interval everywhere it appears.
6. Effect sizes translated into a quantity a reader can judge.
7. Title and abstract claim only what was measured.
8. `paper/preregistration.md` byte-identical to its committed state — verified by `git log --oneline -- paper/preregistration.md` showing no commit from this plan.
9. No P1/P2/P3 verdict altered.

## Out of scope

- Re-running the study, changing ξ, or altering any registered threshold as the *primary* analysis. Sensitivity is reported alongside the registered result, never in place of it.
- A second forecasting model. That is the follow-up study, not this paper.
- Submission itself.

## Risks

| Risk | Mitigation |
|---|---|
| The reframe drifts into softening the P3 null | Task 5 Step 5 is an explicit diff gate on outcome sentences |
| Sensitivity shows P1 or P2 is fragile | Report it. A conclusion that holds only at its registered thresholds is a finding, and burying it would be the one unrecoverable error here |
| Figures invent an effect the data does not support | Axes start at zero or say why not; every figure shows the sample it rests on |
| matplotlib pulls a heavy dependency chain into a reproducibility-critical env | Pin it in the lockfile; it is a dev-time concern only, never imported by `src/footy/core/` |
