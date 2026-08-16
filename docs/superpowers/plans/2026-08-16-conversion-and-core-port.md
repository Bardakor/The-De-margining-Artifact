# Conversion and Clean-Room Core Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the betting application, stand up a Python research repository, and port every model in `MODEL.md` to Python clean-room, verified against numbers printed in that document.

**Architecture:** A pure-math package `src/footy/core/` with no I/O, no clock and no randomness, wrapped by `market/` (margin and staking) and `eval/` (scoring rules). Everything is a NumPy array operation over an 11×11 scoreline matrix. Special functions come from SciPy rather than hand-rolled approximations, which is what makes this an independent implementation rather than a transliteration.

**Tech Stack:** Python 3.12, NumPy, SciPy, pytest, Hypothesis, `uv`, ruff, mypy.

**Plan 1 of 5.** Implements §7, §8 (Layers 1 and 2) and §10.1 of `docs/superpowers/specs/2026-08-16-demargining-study-design.md`.

## Global Constraints

- **CLEAN-ROOM: do not open `packages/quant-engine/src/**` at any point in this plan.** The Python is written from `MODEL.md` only. Reading the TypeScript destroys the independence that makes the Plan 4 corroboration meaningful. `MODEL.md` and this plan are the sole sources.
- Python 3.12. Dependencies: `numpy`, `scipy` only in `src/` (pandas arrives in Plan 3).
- `src/footy/core/` must contain no I/O, no `datetime`, no `random`, no file access. Enforced by a test.
- All public functions are pure: same inputs, same outputs, no mutation of arguments.
- `mypy --strict` clean. `ruff` clean. No `# type: ignore` without an inline reason.
- Float assertions use `pytest.approx(..., abs=1e-9)` unless a step states otherwise.
- No notebooks anywhere in the repository.
- Every commit message ends with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Work happens on branch `research/demargining-study`.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, ruff/mypy/pytest config |
| `Makefile` | Terminal entry points: `verify`, `test`, `lint`, `typecheck` |
| `src/footy/core/poisson.py` | Log-space Poisson pmf |
| `src/footy/core/dixon_coles.py` | τ matrix, τ derivatives, admissible ρ bounds |
| `src/footy/core/matrix.py` | Normalised 11×11 scoreline matrix |
| `src/footy/core/markets.py` | Market marginals: 1X2, totals, BTTS, correct score, AH, DC |
| `src/footy/core/skellam.py` | Goal-difference distribution, independent of the matrix |
| `src/footy/market/overround.py` | Power-method margin application |
| `src/footy/market/demargin.py` | Shin inverse (proportional/power/odds-ratio arrive in Plan 3) |
| `src/footy/market/kelly.py` | Fractional Kelly staking |
| `src/footy/eval/scoring.py` | RPS, Brier, log loss |
| `src/footy/eval/murphy.py` | REL / RES / UNC / WBV decomposition |
| `tests/fixtures/model_md_values.json` | Layer-1 targets transcribed from `MODEL.md` |
| `tests/test_purity.py` | Scans `core/` for banned imports |
| `tests/test_properties.py` | Layer-2 Hypothesis invariants |

---

### Task 1: Wave-1 deletion

**Files:**
- Delete: see step 2
- Create: `.gitignore` (replace)

**Interfaces:**
- Consumes: nothing
- Produces: a repository containing only `packages/quant-engine/`, `docs/`, `README.md`, `LICENSE`, `.github/`

- [ ] **Step 1: Tag the current state so nothing is lost**

```bash
git tag -a v1-betting-app -m "Final state of the betting application before the research pivot"
git tag -l v1-betting-app
```

Expected: prints `v1-betting-app`. Do **not** push the tag yet; Task 16 pushes branch and tag together.

- [ ] **Step 2: Delete the application**

```bash
git rm -r --quiet frontend backend mini-betting-platform mongodb-data tests bash mds .code
git rm --quiet docker-compose.yml mongo-init.js postman-collection-demo.json \
  demo.sh quick-test.sh test_helpers.sh demo-postman-requests.md \
  check-bets.js check-user.js database-inspector.js test-mongodb-setup.js
```

- [ ] **Step 3: Verify only intended paths survive**

Run: `git ls-files | awk -F/ '{print $1}' | sort -u`
Expected exactly: `.github`, `.gitignore`, `LICENSE`, `README.md`, `docs`, `package-lock.json`, `package.json`, `packages`

`package.json`, `package-lock.json` and `packages/` are deliberate survivors — the TypeScript engine is needed for Plan 4 corroboration and is deleted in wave 2.

- [ ] **Step 4: Replace .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/

# Node (transitional — removed with the TypeScript engine in wave 2)
node_modules/

# Study artefacts: regenerable, never committed
data/
results/
paper/figures/*.pdf
paper/figures/*.png

# OS
.DS_Store
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete the betting application

Removes the Next.js frontend, six Node microservices, the duplicate
mini-betting-platform, 108 tracked WiredTiger binaries, the HTML API
harness and the shell demo scripts — roughly 22,000 lines of application
scaffolding for a product that is no longer being built.

The TypeScript pricing engine survives temporarily: it is the corroboration
oracle for the clean-room Python port and is removed in wave 2.

State before this commit is tagged v1-betting-app.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Python scaffolding

**Files:**
- Create: `pyproject.toml`, `Makefile`, `src/footy/__init__.py`, `src/footy/core/__init__.py`, `src/footy/market/__init__.py`, `src/footy/eval/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`
- Delete: `.github/workflows/` contents (Node CI), replace with Python CI

**Interfaces:**
- Consumes: Task 1's cleaned tree
- Produces: `make verify` as the single verification command for every later task

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "footy"
version = "0.1.0"
description = "Dixon-Coles football pricing and the de-margining artifact study"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "scipy>=1.14"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "hypothesis>=6.100", "ruff>=0.6", "mypy>=1.11"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/footy"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "NPY"]

[tool.mypy]
strict = true
files = ["src", "tests"]

[[tool.mypy.overrides]]
module = "scipy.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write the Makefile**

Use real tabs for the recipe lines, not spaces.

```makefile
.PHONY: install test lint typecheck verify clean

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

verify: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
```

- [ ] **Step 3: Create the package skeleton**

```bash
mkdir -p src/footy/core src/footy/market src/footy/eval tests/fixtures
touch src/footy/__init__.py src/footy/core/__init__.py \
      src/footy/market/__init__.py src/footy/eval/__init__.py tests/__init__.py
```

- [ ] **Step 4: Write a smoke test**

`tests/test_smoke.py`:

```python
"""The package imports and the toolchain runs."""


def test_package_imports() -> None:
    import footy

    assert footy is not None
```

- [ ] **Step 5: Install and verify the toolchain**

Run: `make install && make verify`
Expected: ruff clean, mypy clean, 1 test passing.

- [ ] **Step 6: Replace Node CI with Python CI**

```bash
rm -rf .github/workflows
mkdir -p .github/workflows
```

`.github/workflows/verify.yml`:

```yaml
name: verify
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv python install 3.12
      - run: make install
      - run: make verify
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "build: stand up the Python research package

uv-managed, ruff + mypy --strict + pytest, with make verify as the single
verification command. Node CI replaced by Python CI.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The Layer-1 fixture, transcribed from MODEL.md

Every later task asserts against this file. It is written **first**, from the document, so that no implementation can be quietly tuned to match itself.

**Files:**
- Create: `tests/fixtures/model_md_values.json`

**Interfaces:**
- Consumes: nothing
- Produces: `tests/fixtures/model_md_values.json`, loaded by every Layer-1 test via the `model_md` fixture defined in Task 4

**One derivation is required.** `MODEL.md` §11.1 prints the forecast vector and `BS = 0.10900` but **never states the realised outcomes**, without which the Brier score cannot be recomputed. The outcome vector is recovered by solving for it: with forecasts `[0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95]`, the vector `[0, 0, 0, 1, 0, 1, 1, 1, 1, 1]` gives a squared-error sum of exactly `1.09`, hence `BS = 0.109`. It is the only assignment consistent with the printed value, and it also reproduces the printed `WBV = 0.00025` (bins `{0.2, 0.25}` and `{0.8, 0.85}` each contribute `2 × 0.025² = 0.00125`, totalling `0.0025`, divided by `N = 10`). This is recorded in the fixture as a derived value, not a transcribed one.

- [ ] **Step 1: Write the fixture**

```json
{
  "_source": "packages/quant-engine/docs/MODEL.md",
  "_note": "Values transcribed from the specification prose. Nothing here was read from the TypeScript implementation.",

  "dixon_coles_rho_table": {
    "_source": "MODEL.md §3.1",
    "lam": 1.6,
    "mu": 1.1,
    "tolerance": 5e-5,
    "rows": [
      {"rho": -0.20, "p_0_0": 0.09086, "p_1_1": 0.14194, "p_draw": 0.29622, "fair_draw_odds": 3.376},
      {"rho": -0.10, "p_0_0": 0.07903, "p_1_1": 0.13011, "p_draw": 0.27257, "fair_draw_odds": 3.669},
      {"rho":  0.00, "p_0_0": 0.06721, "p_1_1": 0.11828, "p_draw": 0.24891, "fair_draw_odds": 4.017},
      {"rho":  0.06, "p_0_0": 0.06011, "p_1_1": 0.11118, "p_draw": 0.23472, "fair_draw_odds": 4.260}
    ]
  },

  "favourite_longshot_bias": {
    "_source": "MODEL.md §8.2",
    "lam": 1.6, "mu": 1.1, "book_sum": 1.08,
    "tolerance": 5e-4,
    "fair_to_offered_ratio_longshot": 1.105,
    "fair_to_offered_ratio_favourite": 1.055
  },

  "double_chance_validity": {
    "_source": "MODEL.md §8.3",
    "lam": 3.8, "mu": 0.3, "book_sum": 1.05,
    "tolerance": 5e-4,
    "dc_1x_price": 1.0053,
    "naive_method_breaks_at": {"lam": 2.5, "mu": 0.3}
  },

  "shin_round_trip": {
    "_source": "MODEL.md §9",
    "implied": [0.5, 0.35, 0.25],
    "book_sum": 1.10,
    "tolerance": 5e-5,
    "recovered": [0.46344, 0.31699, 0.21956],
    "z": 0.0502
  },

  "murphy_decomposition": {
    "_source": "MODEL.md §11.1",
    "forecasts": [0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.95],
    "outcomes": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
    "_outcomes_provenance": "DERIVED, not transcribed. MODEL.md §11.1 omits the outcome vector. This is the assignment that reproduces the printed BS = 0.10900 exactly (squared-error sum 1.09) and the printed WBV = 0.00025.",
    "n_bins": 10,
    "tolerance": 5e-6,
    "brier": 0.10900,
    "three_way_sum": 0.10875,
    "wbv": 0.00025,
    "uncertainty": 0.24
  }
}
```

- [ ] **Step 2: Verify the derived outcome vector by hand before trusting it**

Run:

```bash
python3 -c "
p=[0.1,0.2,0.25,0.4,0.55,0.6,0.7,0.8,0.85,0.95]
o=[0,0,0,1,0,1,1,1,1,1]
print('BS =', sum((a-b)**2 for a,b in zip(p,o))/len(p))
"
```

Expected: `BS = 0.109`. If this prints anything else, stop and re-derive — every Murphy assertion downstream depends on it.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/model_md_values.json
git commit -m "test: transcribe MODEL.md reference values as Layer-1 targets

Written from the specification before any implementation exists, so no
implementation can be tuned to agree with itself.

The Murphy outcome vector is derived rather than transcribed: MODEL.md
§11.1 omits it. Recorded as derived in the fixture.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Poisson pmf in log space

**Files:**
- Create: `src/footy/core/poisson.py`, `tests/conftest.py`, `tests/test_poisson.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `poisson_pmf(k: npt.NDArray[np.int_] | int, lam: float) -> npt.NDArray[np.float64]`
  - `tests/conftest.py` exposing a session fixture `model_md` returning the parsed Layer-1 JSON

- [ ] **Step 1: Write conftest.py**

```python
"""Shared test fixtures."""

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model_md_values.json"


@pytest.fixture(scope="session")
def model_md() -> dict[str, Any]:
    """Reference values transcribed from MODEL.md. See Task 3."""
    with FIXTURE_PATH.open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data
```

- [ ] **Step 2: Write the failing test**

`tests/test_poisson.py`:

```python
"""MODEL.md §2 — goals as a Poisson process."""

import math

import numpy as np
import pytest

from footy.core.poisson import poisson_pmf


def test_matches_the_naive_formula_for_small_k() -> None:
    lam = 1.6
    k = np.arange(6)
    expected = np.array([math.exp(-lam) * lam**i / math.factorial(i) for i in range(6)])
    assert poisson_pmf(k, lam) == pytest.approx(expected, abs=1e-12)


def test_stays_finite_where_the_naive_form_overflows() -> None:
    """MODEL.md §2: lam**k / k! overflows to NaN for large k; the log form does not."""
    result = poisson_pmf(np.arange(200), 1.6)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


def test_sums_to_one_over_a_wide_support() -> None:
    assert poisson_pmf(np.arange(300), 12.0).sum() == pytest.approx(1.0, abs=1e-12)


def test_rejects_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        poisson_pmf(np.arange(5), 0.0)
```

- [ ] **Step 3: Run and confirm it fails**

Run: `uv run pytest tests/test_poisson.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'footy.core.poisson'`

- [ ] **Step 4: Implement**

`src/footy/core/poisson.py`:

```python
"""Poisson probability mass, evaluated in log space.

MODEL.md §2:

    P(X = k) = exp(k ln lam - lam - ln k!),  ln k! = ln Gamma(k + 1)

The naive form lam**k / k! overflows to inf/inf = NaN for large k. Working in
logs keeps every intermediate finite.

MODEL.md specifies a Lanczos approximation for ln Gamma. We use
scipy.special.gammaln instead: same function, an implementation we did not
write, and one whose accuracy is independently maintained.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln


def poisson_pmf(k: npt.ArrayLike, lam: float) -> npt.NDArray[np.float64]:
    """Poisson pmf at each k for rate lam.

    Args:
        k: Non-negative integer counts.
        lam: Rate, strictly positive.

    Returns:
        Probability mass at each k, same shape as k.

    Raises:
        ValueError: If lam is not strictly positive, or any k is negative.
    """
    if not lam > 0.0:
        raise ValueError(f"rate must be strictly positive, got {lam}")
    counts = np.asarray(k, dtype=np.float64)
    if np.any(counts < 0.0):
        raise ValueError("counts must be non-negative")
    log_pmf = counts * np.log(lam) - lam - gammaln(counts + 1.0)
    return np.exp(log_pmf)
```

- [ ] **Step 5: Run and confirm it passes**

Run: `uv run pytest tests/test_poisson.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/footy/core/poisson.py tests/conftest.py tests/test_poisson.py
git commit -m "feat(core): Poisson pmf in log space

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Dixon-Coles tau and the admissible rho region

**Files:**
- Create: `src/footy/core/dixon_coles.py`, `tests/test_dixon_coles.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `GRID: int = 11`
  - `tau_matrix(lam: float, mu: float, rho: float) -> npt.NDArray[np.float64]` — shape `(GRID, GRID)`
  - `rho_bounds(lam: float, mu: float) -> tuple[float, float]`
  - `validate_rho(lam: float, mu: float, rho: float) -> None` — raises `ValueError`

- [ ] **Step 1: Write the failing test**

`tests/test_dixon_coles.py`:

```python
"""MODEL.md §3 — the dependence correction."""

import numpy as np
import pytest

from footy.core.dixon_coles import GRID, rho_bounds, tau_matrix, validate_rho


def test_only_the_four_low_scoring_cells_are_corrected() -> None:
    tau = tau_matrix(1.6, 1.1, -0.10)
    assert tau.shape == (GRID, GRID)
    corrected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for x in range(GRID):
        for y in range(GRID):
            if (x, y) not in corrected:
                assert tau[x, y] == 1.0


def test_the_four_cells_take_their_specified_values() -> None:
    lam, mu, rho = 1.6, 1.1, -0.10
    tau = tau_matrix(lam, mu, rho)
    assert tau[0, 0] == pytest.approx(1.0 - lam * mu * rho, abs=1e-12)
    assert tau[0, 1] == pytest.approx(1.0 + lam * rho, abs=1e-12)
    assert tau[1, 0] == pytest.approx(1.0 + mu * rho, abs=1e-12)
    assert tau[1, 1] == pytest.approx(1.0 - rho, abs=1e-12)


def test_negative_rho_inflates_the_low_draws() -> None:
    """MODEL.md §3.1: tau(0,0) and tau(1,1) exceed 1 only when rho < 0."""
    negative = tau_matrix(1.6, 1.1, -0.10)
    positive = tau_matrix(1.6, 1.1, 0.06)
    assert negative[0, 0] > 1.0 and negative[1, 1] > 1.0
    assert positive[0, 0] < 1.0 and positive[1, 1] < 1.0


def test_rho_zero_is_the_identity() -> None:
    assert np.all(tau_matrix(1.6, 1.1, 0.0) == 1.0)


def test_bounds_match_the_specified_region() -> None:
    """MODEL.md §3.2."""
    lam, mu = 1.6, 1.1
    lo, hi = rho_bounds(lam, mu)
    assert lo == pytest.approx(max(-1.0 / lam, -1.0 / mu), abs=1e-12)
    assert hi == pytest.approx(min(1.0 / (lam * mu), 1.0), abs=1e-12)


def test_validate_rejects_rho_outside_the_region() -> None:
    lam, mu = 1.6, 1.1
    lo, hi = rho_bounds(lam, mu)
    validate_rho(lam, mu, (lo + hi) / 2.0)  # must not raise
    with pytest.raises(ValueError, match="admissible"):
        validate_rho(lam, mu, lo - 0.01)
    with pytest.raises(ValueError, match="admissible"):
        validate_rho(lam, mu, hi + 0.01)


def test_tau_is_non_negative_throughout_the_admissible_region() -> None:
    lam, mu = 1.6, 1.1
    lo, hi = rho_bounds(lam, mu)
    for rho in np.linspace(lo, hi, 50):
        assert np.all(tau_matrix(lam, mu, float(rho)) >= 0.0)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_dixon_coles.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/core/dixon_coles.py`:

```python
"""The Dixon-Coles dependence correction.

MODEL.md §3. Independent Poisson under-counts low-scoring draws. Dixon and
Coles (1997) correct exactly four cells and leave the rest alone:

    tau(0,0) = 1 - lam*mu*rho
    tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1               otherwise

Sign matters. tau(0,0) and tau(1,1) exceed 1 -- inflating the low draws, which
is the whole point of the correction -- only when rho < 0. A positive rho does
the opposite of what the correction exists for.
"""

import numpy as np
import numpy.typing as npt

GRID = 11
"""Scoreline grid is 0..10 goals per side (MODEL.md §5)."""


def rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Admissible range for rho, keeping all four corrected cells non-negative.

    MODEL.md §3.2:

        max(-1/lam, -1/mu) <= rho <= min(1/(lam*mu), 1)
    """
    if not lam > 0.0 or not mu > 0.0:
        raise ValueError(f"rates must be strictly positive, got lam={lam}, mu={mu}")
    return max(-1.0 / lam, -1.0 / mu), min(1.0 / (lam * mu), 1.0)


def validate_rho(lam: float, mu: float, rho: float) -> None:
    """Raise if rho would drive a corrected cell negative."""
    lo, hi = rho_bounds(lam, mu)
    if not lo <= rho <= hi:
        raise ValueError(
            f"rho={rho} outside the admissible region [{lo}, {hi}] for lam={lam}, mu={mu}"
        )


def tau_matrix(lam: float, mu: float, rho: float) -> npt.NDArray[np.float64]:
    """The correction factor for every cell of the scoreline grid."""
    validate_rho(lam, mu, rho)
    tau = np.ones((GRID, GRID), dtype=np.float64)
    tau[0, 0] = 1.0 - lam * mu * rho
    tau[0, 1] = 1.0 + lam * rho
    tau[1, 0] = 1.0 + mu * rho
    tau[1, 1] = 1.0 - rho
    return tau
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_dixon_coles.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/core/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat(core): Dixon-Coles tau correction and admissible rho region

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The scoreline matrix, against the MODEL.md rho table

This is the first Layer-1 checkpoint. If the ρ table reproduces, the Poisson pmf, τ and the renormalisation are all correct together.

**Files:**
- Create: `src/footy/core/matrix.py`, `tests/test_matrix.py`

**Interfaces:**
- Consumes: `poisson_pmf`, `tau_matrix`, `validate_rho`, `GRID`
- Produces: `scoreline_matrix(lam: float, mu: float, rho: float = -0.10) -> npt.NDArray[np.float64]` — shape `(GRID, GRID)`, `[x, y]` is P(home scores x, away scores y), summing to 1

- [ ] **Step 1: Write the failing test**

`tests/test_matrix.py`:

```python
"""MODEL.md §5 — the scoreline matrix, and §3.1 as a Layer-1 target."""

from typing import Any

import numpy as np
import pytest

from footy.core.dixon_coles import GRID
from footy.core.matrix import scoreline_matrix


def test_sums_to_one() -> None:
    assert scoreline_matrix(1.6, 1.1, -0.10).sum() == pytest.approx(1.0, abs=1e-12)


def test_shape_and_non_negativity() -> None:
    m = scoreline_matrix(1.6, 1.1, -0.10)
    assert m.shape == (GRID, GRID)
    assert np.all(m >= 0.0)


def test_reproduces_the_model_md_rho_table(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §3.1."""
    spec = model_md["dixon_coles_rho_table"]
    lam, mu, tol = spec["lam"], spec["mu"], spec["tolerance"]

    for row in spec["rows"]:
        m = scoreline_matrix(lam, mu, row["rho"])
        draw = float(np.trace(m))
        assert m[0, 0] == pytest.approx(row["p_0_0"], abs=tol), f"P(0-0) at rho={row['rho']}"
        assert m[1, 1] == pytest.approx(row["p_1_1"], abs=tol), f"P(1-1) at rho={row['rho']}"
        assert draw == pytest.approx(row["p_draw"], abs=tol), f"P(draw) at rho={row['rho']}"
        assert 1.0 / draw == pytest.approx(row["fair_draw_odds"], abs=1e-3)


def test_draw_probability_falls_as_rho_rises(model_md: dict[str, Any]) -> None:
    """MODEL.md §3.1: a positive rho suppresses the draws the correction exists to raise."""
    spec = model_md["dixon_coles_rho_table"]
    draws = [
        float(np.trace(scoreline_matrix(spec["lam"], spec["mu"], row["rho"])))
        for row in spec["rows"]
    ]
    assert draws == sorted(draws, reverse=True)


def test_independence_when_rho_is_zero() -> None:
    """At rho = 0 the matrix is exactly the outer product of two Poisson margins."""
    m = scoreline_matrix(2.0, 1.0, 0.0)
    row_margin = m.sum(axis=1)
    col_margin = m.sum(axis=0)
    assert m == pytest.approx(np.outer(row_margin, col_margin), abs=1e-9)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/core/matrix.py`:

```python
"""The joint distribution over scorelines.

MODEL.md §5. One matrix is the single source of truth for every price:

    P(x,y) = tau(x,y) * Pois(x; lam) * Pois(y; mu) / Z

over 0 <= x, y <= 10. Renormalisation absorbs both the truncated tail beyond
10 goals and the mass shifted by tau.
"""

import numpy as np
import numpy.typing as npt

from footy.core.dixon_coles import GRID, tau_matrix
from footy.core.poisson import poisson_pmf

DEFAULT_RHO = -0.10
"""MODEL.md §3.1. Dixon and Coles' own fitted value is near -0.13."""


def scoreline_matrix(lam: float, mu: float, rho: float = DEFAULT_RHO) -> npt.NDArray[np.float64]:
    """Joint distribution over scorelines.

    Args:
        lam: Expected goals, home.
        mu: Expected goals, away.
        rho: Dependence parameter; must lie in the admissible region.

    Returns:
        An 11x11 array where [x, y] is P(home scores x, away scores y),
        summing to 1.
    """
    goals = np.arange(GRID)
    joint = np.outer(poisson_pmf(goals, lam), poisson_pmf(goals, mu))
    joint = joint * tau_matrix(lam, mu, rho)
    total = joint.sum()
    if not total > 0.0:
        raise ValueError(f"degenerate matrix for lam={lam}, mu={mu}, rho={rho}")
    return joint / total
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_matrix.py -v`
Expected: 5 passed. **If `test_reproduces_the_model_md_rho_table` fails, stop.** That is the clean-room port disagreeing with its specification, and it must be resolved before anything downstream is built.

- [ ] **Step 5: Commit**

```bash
git add src/footy/core/matrix.py tests/test_matrix.py
git commit -m "feat(core): scoreline matrix, reproducing the MODEL.md rho table

First Layer-1 checkpoint: the four-row rho table from MODEL.md §3.1 is
reproduced to 5e-5, which exercises the Poisson pmf, tau and the
renormalisation together.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Market marginals — 1X2, totals, BTTS, correct score

**Files:**
- Create: `src/footy/core/markets.py`, `tests/test_markets.py`

**Interfaces:**
- Consumes: `scoreline_matrix`, `GRID`
- Produces:
  - `match_odds(m) -> tuple[float, float, float]` — (home, draw, away)
  - `totals(m, line: float) -> tuple[float, float]` — (over, under)
  - `both_teams_to_score(m) -> tuple[float, float]` — (yes, no)
  - `correct_score(m, x: int, y: int) -> float`
  - `double_chance(m) -> tuple[float, float, float]` — (1X, 12, X2)

- [ ] **Step 1: Write the failing test**

`tests/test_markets.py`:

```python
"""MODEL.md §6 — every market is a sum over regions of one matrix."""

import numpy as np
import pytest

from footy.core.markets import (
    both_teams_to_score,
    correct_score,
    double_chance,
    match_odds,
    totals,
)
from footy.core.matrix import scoreline_matrix

MATRIX = scoreline_matrix(1.6, 1.1, -0.10)


def test_match_odds_partition_the_matrix() -> None:
    home, draw, away = match_odds(MATRIX)
    assert home + draw + away == pytest.approx(1.0, abs=1e-12)
    assert home > away, "lam > mu should favour the home side"


def test_totals_are_complementary() -> None:
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        over, under = totals(MATRIX, line)
        assert over + under == pytest.approx(1.0, abs=1e-12)
    assert totals(MATRIX, 0.5)[0] > totals(MATRIX, 5.5)[0]


def test_btts_is_complementary() -> None:
    yes, no = both_teams_to_score(MATRIX)
    assert yes + no == pytest.approx(1.0, abs=1e-12)


def test_correct_score_cells_sum_to_the_whole() -> None:
    total = sum(
        correct_score(MATRIX, x, y) for x in range(MATRIX.shape[0]) for y in range(MATRIX.shape[1])
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_correct_score_triangle_equals_home_win() -> None:
    """MODEL.md §6: markets are marginals of one matrix, so they cannot disagree."""
    home, _, _ = match_odds(MATRIX)
    triangle = sum(
        correct_score(MATRIX, x, y)
        for x in range(MATRIX.shape[0])
        for y in range(MATRIX.shape[1])
        if x > y
    )
    assert triangle == pytest.approx(home, abs=1e-12)


def test_double_chance_is_the_union_of_1x2_regions() -> None:
    home, draw, away = match_odds(MATRIX)
    dc_1x, dc_12, dc_x2 = double_chance(MATRIX)
    assert dc_1x == pytest.approx(home + draw, abs=1e-12)
    assert dc_12 == pytest.approx(home + away, abs=1e-12)
    assert dc_x2 == pytest.approx(draw + away, abs=1e-12)
    assert dc_1x + dc_12 + dc_x2 == pytest.approx(2.0, abs=1e-12)


def test_btts_agrees_with_the_correct_score_cells() -> None:
    yes, _ = both_teams_to_score(MATRIX)
    cells = sum(
        correct_score(MATRIX, x, y)
        for x in range(1, MATRIX.shape[0])
        for y in range(1, MATRIX.shape[1])
    )
    assert cells == pytest.approx(yes, abs=1e-12)


def test_rejects_a_whole_number_total_line() -> None:
    """A whole line can push, so a two-way over/under cannot describe it."""
    with pytest.raises(ValueError, match="half-integer"):
        totals(MATRIX, 2.0)


def test_correct_score_rejects_out_of_grid() -> None:
    with pytest.raises(IndexError):
        correct_score(MATRIX, 11, 0)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_markets.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/core/markets.py`:

```python
"""Markets as marginals of the scoreline matrix.

MODEL.md §6. Every market is a sum over a region of one distribution. No
market has a model of its own, so no two markets can disagree about the same
event.

Indexing convention throughout: m[x, y] is P(home scores x, away scores y).
A home win is therefore x > y, which is strictly below the main diagonal.
"""

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]


def match_odds(m: Matrix) -> tuple[float, float, float]:
    """1X2 probabilities: (home, draw, away)."""
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return home, draw, away


def totals(m: Matrix, line: float) -> tuple[float, float]:
    """Over/under a half-integer goal line: (over, under).

    Whole lines are rejected: they can push, so two probabilities cannot
    describe the market.
    """
    if float(line * 2).is_integer() and float(line).is_integer():
        raise ValueError(f"line must be a half-integer, got {line}")
    x = np.arange(m.shape[0])[:, None]
    y = np.arange(m.shape[1])[None, :]
    over = float(m[(x + y) > line].sum())
    return over, 1.0 - over


def both_teams_to_score(m: Matrix) -> tuple[float, float]:
    """(yes, no) — yes is every cell with x > 0 and y > 0."""
    yes = float(m[1:, 1:].sum())
    return yes, 1.0 - yes


def correct_score(m: Matrix, x: int, y: int) -> float:
    """The single cell (x, y)."""
    if not (0 <= x < m.shape[0] and 0 <= y < m.shape[1]):
        raise IndexError(f"scoreline ({x}, {y}) outside the {m.shape} grid")
    return float(m[x, y])


def double_chance(m: Matrix) -> tuple[float, float, float]:
    """(1X, 12, X2) — unions of the 1X2 regions. Sums to 2 by construction."""
    home, draw, away = match_odds(m)
    return home + draw, home + away, draw + away
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_markets.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/core/markets.py tests/test_markets.py
git commit -m "feat(core): 1X2, totals, BTTS, correct score and double chance

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Asian handicap with the push

**Files:**
- Modify: `src/footy/core/markets.py`
- Create: `tests/test_asian_handicap.py`

**Interfaces:**
- Consumes: `match_odds`, `Matrix`
- Produces:
  - `AsianOutcome` — a `NamedTuple` with fields `win: float`, `push: float`, `lose: float`
  - `asian_handicap(m, handicap: float) -> tuple[AsianOutcome, AsianOutcome]` — (home side, away side)
  - `asian_fair_odds(outcome: AsianOutcome) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_asian_handicap.py`:

```python
"""MODEL.md §6.1 — a handicap bet can push, so one probability cannot describe it."""

import pytest

from footy.core.markets import asian_fair_odds, asian_handicap, match_odds
from footy.core.matrix import scoreline_matrix

MATRIX = scoreline_matrix(1.6, 1.1, -0.10)


def test_each_side_partitions_into_win_push_lose() -> None:
    home, away = asian_handicap(MATRIX, 0.0)
    for side in (home, away):
        assert side.win + side.push + side.lose == pytest.approx(1.0, abs=1e-12)


def test_the_two_sides_mirror_each_other() -> None:
    home, away = asian_handicap(MATRIX, -0.5)
    assert home.win == pytest.approx(away.lose, abs=1e-12)
    assert home.lose == pytest.approx(away.win, abs=1e-12)
    assert home.push == pytest.approx(away.push, abs=1e-12)


def test_level_ball_push_is_the_draw() -> None:
    _, draw, _ = match_odds(MATRIX)
    home, _ = asian_handicap(MATRIX, 0.0)
    assert home.push == pytest.approx(draw, abs=1e-12)


def test_level_ball_equals_draw_no_bet() -> None:
    """MODEL.md §6: the level-ball handicap must equal the draw-no-bet price."""
    p_home, draw, p_away = match_odds(MATRIX)
    dnb_home = p_home / (p_home + p_away)
    home, _ = asian_handicap(MATRIX, 0.0)
    assert asian_fair_odds(home) == pytest.approx(1.0 / dnb_home, abs=1e-12)


def test_fair_odds_account_for_the_returned_stake() -> None:
    """MODEL.md §6.1: d_fair = 1 + P(lose)/P(win)."""
    home, _ = asian_handicap(MATRIX, -0.5)
    assert asian_fair_odds(home) == pytest.approx(1.0 + home.lose / home.win, abs=1e-12)


def test_half_line_cannot_push() -> None:
    home, away = asian_handicap(MATRIX, -0.5)
    assert home.push == 0.0
    assert away.push == 0.0


def test_quarter_line_splits_the_two_adjacent_lines() -> None:
    """MODEL.md §6.1: quarter lines split the stake evenly, which is how they settle."""
    quarter, _ = asian_handicap(MATRIX, -0.25)
    level, _ = asian_handicap(MATRIX, 0.0)
    half, _ = asian_handicap(MATRIX, -0.5)
    assert quarter.win == pytest.approx((level.win + half.win) / 2.0, abs=1e-12)
    assert quarter.push == pytest.approx((level.push + half.push) / 2.0, abs=1e-12)
    assert quarter.lose == pytest.approx((level.lose + half.lose) / 2.0, abs=1e-12)


def test_a_bigger_handicap_against_the_home_side_lowers_its_win_probability() -> None:
    assert asian_handicap(MATRIX, -1.5)[0].win < asian_handicap(MATRIX, -0.5)[0].win


def test_rejects_an_eighth_line() -> None:
    with pytest.raises(ValueError, match="quarter"):
        asian_handicap(MATRIX, -0.125)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_asian_handicap.py -v`
Expected: FAIL — `ImportError: cannot import name 'asian_handicap'`

- [ ] **Step 3: Implement — append to `src/footy/core/markets.py`**

```python
class AsianOutcome(NamedTuple):
    """One side of a handicap bet. Sums to 1."""

    win: float
    push: float
    lose: float


def _whole_or_half_handicap(m: Matrix, handicap: float) -> tuple[AsianOutcome, AsianOutcome]:
    """Handicap applied to the home side: win when x + h > y, push when equal."""
    x = np.arange(m.shape[0])[:, None]
    y = np.arange(m.shape[1])[None, :]
    adjusted = x + handicap - y

    home_win = float(m[adjusted > 0].sum())
    home_push = float(m[adjusted == 0].sum())
    home_lose = float(m[adjusted < 0].sum())

    home = AsianOutcome(home_win, home_push, home_lose)
    away = AsianOutcome(home_lose, home_push, home_win)
    return home, away


def asian_handicap(m: Matrix, handicap: float) -> tuple[AsianOutcome, AsianOutcome]:
    """Asian handicap applied to the home side: (home, away).

    Quarter lines split the stake evenly across the two adjacent lines, which
    is how they settle in practice (MODEL.md §6.1).

    Raises:
        ValueError: If the handicap is not a multiple of 0.25.
    """
    quarters = handicap * 4.0
    if not float(quarters).is_integer():
        raise ValueError(f"handicap must be a multiple of a quarter goal, got {handicap}")

    is_quarter_line = int(quarters) % 2 != 0
    if not is_quarter_line:
        return _whole_or_half_handicap(m, handicap)

    lower_home, lower_away = _whole_or_half_handicap(m, handicap - 0.25)
    upper_home, upper_away = _whole_or_half_handicap(m, handicap + 0.25)
    blend = lambda a, b: AsianOutcome(  # noqa: E731
        (a.win + b.win) / 2.0, (a.push + b.push) / 2.0, (a.lose + b.lose) / 2.0
    )
    return blend(lower_home, upper_home), blend(lower_away, upper_away)


def asian_fair_odds(outcome: AsianOutcome) -> float:
    """Fair decimal price, accounting for the stake being returned on a push.

    MODEL.md §6.1:  d_fair = 1 + P(lose) / P(win)
    """
    if not outcome.win > 0.0:
        raise ValueError("cannot price a handicap the side can never win")
    return 1.0 + outcome.lose / outcome.win
```

Add `from typing import NamedTuple` to the imports at the top of the file.

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_asian_handicap.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/core/markets.py tests/test_asian_handicap.py
git commit -m "feat(core): Asian handicap with push handling and quarter lines

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Skellam — the independent check on the matrix

**Files:**
- Create: `src/footy/core/skellam.py`, `tests/test_skellam.py`

**Interfaces:**
- Consumes: `scoreline_matrix`
- Produces:
  - `skellam_pmf(k: npt.ArrayLike, lam: float, mu: float) -> npt.NDArray[np.float64]`
  - `skellam_supremacy(lam: float, mu: float, handicap: float) -> tuple[float, float, float]` — (home covers, push, away covers)

- [ ] **Step 1: Write the failing test**

`tests/test_skellam.py`:

```python
"""MODEL.md §7 — corroborating the matrix from outside itself."""

import numpy as np
import pytest

from footy.core.matrix import scoreline_matrix
from footy.core.skellam import skellam_pmf


def matrix_goal_difference(m: np.ndarray, k: int) -> float:
    """Sum the anti-diagonal x - y == k."""
    x = np.arange(m.shape[0])[:, None]
    y = np.arange(m.shape[1])[None, :]
    return float(m[(x - y) == k].sum())


def test_sums_to_one_over_a_wide_support() -> None:
    assert skellam_pmf(np.arange(-40, 41), 1.6, 1.1).sum() == pytest.approx(1.0, abs=1e-10)


def test_agrees_with_the_matrix_when_rho_is_zero() -> None:
    """MODEL.md §7 consequence 1: at rho = 0 the two derivations must agree.

    This is the only test in the suite that corroborates the matrix by a route
    that does not use the matrix.
    """
    lam, mu = 1.6, 1.1
    m = scoreline_matrix(lam, mu, 0.0)
    for k in range(-5, 6):
        assert skellam_pmf(k, lam, mu) == pytest.approx(
            matrix_goal_difference(m, k), abs=1e-6
        ), f"goal difference {k}"


def test_diverges_from_the_matrix_when_rho_is_non_zero() -> None:
    """MODEL.md §7 consequence 2: if they agreed, tau would be doing nothing."""
    lam, mu = 1.6, 1.1
    m = scoreline_matrix(lam, mu, -0.10)
    assert abs(float(skellam_pmf(0, lam, mu)) - matrix_goal_difference(m, 0)) > 1e-3


def test_stays_finite_for_large_rates() -> None:
    """The Bessel series must be evaluated stably, not by naive summation."""
    result = skellam_pmf(np.arange(-60, 61), 40.0, 35.0)
    assert np.all(np.isfinite(result))
    assert result.sum() == pytest.approx(1.0, abs=1e-8)


def test_symmetric_rates_give_a_symmetric_distribution() -> None:
    values = skellam_pmf(np.arange(-8, 9), 1.5, 1.5)
    assert values == pytest.approx(values[::-1], abs=1e-12)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_skellam.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/core/skellam.py`:

```python
"""The goal difference, derived without the matrix.

MODEL.md §7. Everything else in this package derives from one 11x11 matrix,
which is internally consistent by construction -- meaning a systematic error
in the matrix would be invisible to every consistency test.

The difference of two independent Poisson variables is Skellam-distributed
(Karlis and Ntzoufras, 2009), with a closed form in the modified Bessel
function of the first kind:

    P(K = k) = exp(-(lam + mu)) * (lam/mu)^(k/2) * I_|k|(2*sqrt(lam*mu))

No matrix is involved. Where this agrees with the matrix anti-diagonals, the
matrix is corroborated from outside itself.

Numerically we use scipy.special.ive, the exponentially scaled Bessel
function ive(v, z) = I_v(z) * exp(-|z|). Folding exp(z) back in analytically
keeps every intermediate finite for large rates, where I_v(z) alone overflows.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import ive


def skellam_pmf(k: npt.ArrayLike, lam: float, mu: float) -> npt.NDArray[np.float64]:
    """Probability that the goal difference (home minus away) equals k."""
    if not lam > 0.0 or not mu > 0.0:
        raise ValueError(f"rates must be strictly positive, got lam={lam}, mu={mu}")
    diff = np.asarray(k, dtype=np.float64)
    z = 2.0 * np.sqrt(lam * mu)
    # exp(-(lam+mu)) * I_v(z) == exp(-(lam+mu) + z) * ive(v, z)
    scale = np.exp(-(lam + mu) + z)
    return scale * (lam / mu) ** (diff / 2.0) * ive(np.abs(diff), z)


def skellam_supremacy(lam: float, mu: float, handicap: float) -> tuple[float, float, float]:
    """(home covers, push, away covers) for a supremacy line, via Skellam.

    A second, independent route to the Asian handicap. The handicap is applied
    to the home side, so the home side covers when K + handicap > 0.
    """
    quarters = handicap * 4.0
    if not float(quarters).is_integer():
        raise ValueError(f"handicap must be a multiple of a quarter goal, got {handicap}")

    support = np.arange(-60, 61)
    mass = skellam_pmf(support, lam, mu)
    adjusted = support + handicap

    home = float(mass[adjusted > 0].sum())
    push = float(mass[adjusted == 0].sum())
    away = float(mass[adjusted < 0].sum())
    return home, push, away
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_skellam.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/core/skellam.py tests/test_skellam.py
git commit -m "feat(core): Skellam goal difference as an external check on the matrix

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: The power method for applying margin

**Files:**
- Create: `src/footy/market/overround.py`, `tests/test_overround.py`

**Interfaces:**
- Consumes: `match_odds`, `double_chance`, `scoreline_matrix`
- Produces:
  - `solve_power_exponent(probabilities, target_sum) -> float`
  - `apply_overround(probabilities, target_sum) -> npt.NDArray[np.float64]` — returns offered decimal odds
  - `margin_double_chance(m, target_sum) -> npt.NDArray[np.float64]`

- [ ] **Step 1: Write the failing test**

`tests/test_overround.py`:

```python
"""MODEL.md §8 — margin, in the correct direction."""

from typing import Any

import numpy as np
import pytest

from footy.core.markets import double_chance, match_odds
from footy.core.matrix import scoreline_matrix
from footy.market.overround import apply_overround, margin_double_chance, solve_power_exponent

FAIR = np.array(match_odds(scoreline_matrix(1.6, 1.1, -0.10)))


def test_book_sum_lands_exactly_on_target() -> None:
    for target in (1.02, 1.05, 1.08, 1.15):
        odds = apply_overround(FAIR, target)
        assert float(np.sum(1.0 / odds)) == pytest.approx(target, abs=1e-12)


def test_margin_shortens_every_price() -> None:
    """MODEL.md §8.1: the old implementation had this backwards."""
    odds = apply_overround(FAIR, 1.05)
    assert np.all(odds < 1.0 / FAIR)


def test_exponent_is_one_when_no_margin_is_taken() -> None:
    assert solve_power_exponent(FAIR, 1.0) == pytest.approx(1.0, abs=1e-9)


def test_reproduces_the_favourite_longshot_bias(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §8.2."""
    spec = model_md["favourite_longshot_bias"]
    fair = np.array(match_odds(scoreline_matrix(spec["lam"], spec["mu"], -0.10)))
    offered = apply_overround(fair, spec["book_sum"])
    ratio = (1.0 / fair) / offered

    longshot, favourite = int(np.argmin(fair)), int(np.argmax(fair))
    assert ratio[longshot] == pytest.approx(
        spec["fair_to_offered_ratio_longshot"], abs=spec["tolerance"]
    )
    assert ratio[favourite] == pytest.approx(
        spec["fair_to_offered_ratio_favourite"], abs=spec["tolerance"]
    )
    assert ratio[longshot] > ratio[favourite]


def test_rejects_an_unreachable_target() -> None:
    """MODEL.md §8.2: throw rather than return a plausible-looking wrong book."""
    with pytest.raises(ValueError, match="unreachable"):
        apply_overround(FAIR, float(len(FAIR)) + 0.1)
    with pytest.raises(ValueError, match="at least 1"):
        apply_overround(FAIR, 0.95)


def test_double_chance_is_margined_against_twice_the_book(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §8.3."""
    spec = model_md["double_chance_validity"]
    m = scoreline_matrix(spec["lam"], spec["mu"], -0.10)
    odds = margin_double_chance(m, spec["book_sum"])
    assert float(np.sum(1.0 / odds)) == pytest.approx(2.0 * spec["book_sum"], abs=1e-12)
    assert np.min(odds) == pytest.approx(spec["dc_1x_price"], abs=spec["tolerance"])


def test_double_chance_stays_payable_for_a_heavy_favourite(model_md: dict[str, Any]) -> None:
    """MODEL.md §8.3: summing the already-margined 1X2 legs yields odds below 1."""
    spec = model_md["double_chance_validity"]
    break_point = spec["naive_method_breaks_at"]
    m = scoreline_matrix(break_point["lam"], break_point["mu"], -0.10)

    assert np.all(margin_double_chance(m, 1.05) > 1.0)

    legs = 1.0 / apply_overround(np.array(match_odds(m)), 1.05)
    home, draw, away = legs
    naive_1x = 1.0 / (home + draw)
    assert naive_1x < 1.0, "the naive method should be demonstrably broken here"


def test_double_chance_agrees_with_1x2_on_fair_probabilities() -> None:
    """MODEL.md §8.3: fair probabilities agree; only the margined ones differ."""
    m = scoreline_matrix(1.6, 1.1, -0.10)
    home, draw, away = match_odds(m)
    dc_1x, _, _ = double_chance(m)
    assert dc_1x == pytest.approx(home + draw, abs=1e-12)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_overround.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/market/overround.py`:

```python
"""Applying a bookmaker margin by the power method.

MODEL.md §8. Fair odds are 1/p. To apply a margin, solve for the exponent k
such that

    sum_i p_i**k = B

where B is the target book sum. Every p_i lies in (0, 1), so the sum is
strictly decreasing in k and bisection converges. The offered price is
d_i = p_i**(-k).

An exponent rather than a constant multiplier reproduces favourite-longshot
bias: it takes proportionally more from the longshot.

Direction matters. The implementation this replaced computed p * (1 - margin),
which *lengthens* every price -- the book paid out roughly 105% of fair value.
"""

import numpy as np
import numpy.typing as npt

from footy.core.markets import Matrix, double_chance

_TOLERANCE = 1e-13
_MAX_ITERATIONS = 200


def solve_power_exponent(probabilities: npt.ArrayLike, target_sum: float) -> float:
    """Find k such that sum(p**k) == target_sum, by bisection.

    Raises:
        ValueError: If target_sum is below 1 or at/above the number of
            outcomes, where no exponent exists.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("every probability must lie strictly in (0, 1)")
    if target_sum < 1.0:
        raise ValueError(f"target book sum must be at least 1, got {target_sum}")
    if target_sum >= p.size:
        raise ValueError(
            f"target book sum {target_sum} is unreachable: sum(p**k) approaches "
            f"{p.size} as k approaches 0"
        )

    # sum(p**k) is strictly decreasing in k: it is p.size at k=0 and sum(p) at k=1.
    lo, hi = 0.0, 1.0
    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        value = float(np.sum(p**mid))
        if abs(value - target_sum) < _TOLERANCE:
            return mid
        if value > target_sum:
            lo = mid
        else:
            hi = mid
    raise ValueError(f"bisection failed to reach book sum {target_sum} in {_MAX_ITERATIONS} steps")


def apply_overround(probabilities: npt.ArrayLike, target_sum: float) -> npt.NDArray[np.float64]:
    """Offered decimal odds whose implied probabilities sum to target_sum."""
    p = np.asarray(probabilities, dtype=np.float64)
    k = solve_power_exponent(p, target_sum)
    odds: npt.NDArray[np.float64] = p ** (-k)

    achieved = float(np.sum(1.0 / odds))
    if abs(achieved - target_sum) > 1e-9:
        raise ValueError(f"postcondition failed: book sum {achieved}, target {target_sum}")
    return odds


def margin_double_chance(m: Matrix, target_sum: float) -> npt.NDArray[np.float64]:
    """Double-chance odds, margined independently against 2 * target_sum.

    MODEL.md §8.3. A fair double-chance book already sums to 2, since each
    selection covers two of three outcomes.

    Summing the already-margined 1X2 legs looks tidier and is wrong: for a
    heavy favourite it yields decimal odds below 1, which is not a payable
    price. Margining independently is structurally safe because p**k < 1 for
    any p < 1 and k > 0.
    """
    return apply_overround(np.array(double_chance(m)) / 2.0, target_sum) / 2.0
```

Note on the final line: double-chance probabilities sum to 2, so they are halved into a simplex, margined, and the resulting odds halved back.

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_overround.py -v`
Expected: 8 passed. The favourite–longshot and double-chance assertions are Layer-1 targets; a failure there is a port defect, not a tolerance problem.

- [ ] **Step 5: Commit**

```bash
git add src/footy/market/overround.py tests/test_overround.py
git commit -m "feat(market): power-method overround with double-chance validity

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Shin's method — the inverse

**Files:**
- Create: `src/footy/market/demargin.py`, `tests/test_demargin.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `shin(implied: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], float]` — (true probabilities, z)

Plan 3 adds `proportional`, `power` and `odds_ratio` to this module and the dispatch table that compares them. Only Shin is ported here, because only Shin is specified in `MODEL.md`.

- [ ] **Step 1: Write the failing test**

`tests/test_demargin.py`:

```python
"""MODEL.md §9 — recovering true probabilities from offered odds."""

from typing import Any

import numpy as np
import pytest

from footy.market.demargin import shin


def test_reproduces_the_model_md_round_trip(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §9."""
    spec = model_md["shin_round_trip"]
    recovered, z = shin(spec["implied"])
    assert recovered == pytest.approx(spec["recovered"], abs=spec["tolerance"])
    assert z == pytest.approx(spec["z"], abs=spec["tolerance"])


def test_recovered_probabilities_sum_to_one() -> None:
    recovered, _ = shin([0.5, 0.35, 0.25])
    assert float(recovered.sum()) == pytest.approx(1.0, abs=1e-12)


def test_a_fair_book_is_returned_unchanged() -> None:
    """With no margin there is no insider mass to remove."""
    fair = np.array([0.5, 0.3, 0.2])
    recovered, z = shin(fair)
    assert recovered == pytest.approx(fair, abs=1e-9)
    assert z == pytest.approx(0.0, abs=1e-6)


def test_shortens_the_longshot_more_than_proportional_normalisation() -> None:
    """Shin attributes margin to insider trading, which concentrates on longshots."""
    implied = np.array([0.5, 0.35, 0.25])
    recovered, _ = shin(implied)
    proportional = implied / implied.sum()
    longshot = int(np.argmin(implied))
    assert recovered[longshot] < proportional[longshot]


def test_rejects_a_book_summing_below_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        shin([0.3, 0.3, 0.3])
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_demargin.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/market/demargin.py`:

```python
"""Recovering true probabilities from bookmaker odds.

MODEL.md §9. Shin (1993) models a bookmaker's prices as containing a
proportion z of insider money, and inverts:

    pi_i = [ sqrt(z**2 + 4(1-z) * p_i**2 / B) - z ] / (2(1-z))

with z solved by bisection so that sum(pi) == 1.

This runs in the opposite direction to overround.py. Applying margin and
removing it are different operations, and conflating them is an easy mistake:
an inverse method must never be used to price a market.

Plan 3 adds the proportional, power and odds-ratio transforms alongside this
one. They are the instrument of the study.
"""

import numpy as np
import numpy.typing as npt

_TOLERANCE = 1e-12
_MAX_ITERATIONS = 200


def _shin_probabilities(p: npt.NDArray[np.float64], book_sum: float, z: float) -> npt.NDArray[np.float64]:
    if z <= 0.0:
        return p / book_sum
    inner = z**2 + 4.0 * (1.0 - z) * p**2 / book_sum
    result: npt.NDArray[np.float64] = (np.sqrt(inner) - z) / (2.0 * (1.0 - z))
    return result


def shin(implied: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], float]:
    """Recover true probabilities and the insider proportion z.

    Args:
        implied: Raw implied probabilities 1/odds, summing to at least 1.

    Returns:
        (true probabilities summing to 1, the fitted z).
    """
    p = np.asarray(implied, dtype=np.float64)
    if np.any(p <= 0.0):
        raise ValueError("every implied probability must be strictly positive")
    book_sum = float(p.sum())
    if book_sum < 1.0 - 1e-12:
        raise ValueError(f"book sum must be at least 1, got {book_sum}")

    if abs(book_sum - 1.0) < 1e-12:
        return p.copy(), 0.0

    # sum(pi) is decreasing in z over [0, 1); at z = 0 it is book_sum > 1.
    lo, hi = 0.0, 1.0 - 1e-12
    z = 0.0
    for _ in range(_MAX_ITERATIONS):
        z = (lo + hi) / 2.0
        total = float(_shin_probabilities(p, book_sum, z).sum())
        if abs(total - 1.0) < _TOLERANCE:
            break
        if total > 1.0:
            lo = z
        else:
            hi = z

    recovered = _shin_probabilities(p, book_sum, z)
    return recovered / recovered.sum(), z
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_demargin.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/market/demargin.py tests/test_demargin.py
git commit -m "feat(market): Shin de-margining, reproducing the MODEL.md round trip

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Fractional Kelly

**Files:**
- Create: `src/footy/market/kelly.py`, `tests/test_kelly.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `kelly_fraction(probability: float, decimal_odds: float, fraction: float = 0.25) -> float`
  - `expected_value(probability: float, decimal_odds: float) -> float`

- [ ] **Step 1: Write the failing test**

`tests/test_kelly.py`:

```python
"""MODEL.md §10 — staking."""

import pytest

from footy.market.kelly import expected_value, kelly_fraction


def test_matches_the_closed_form() -> None:
    p, d = 0.55, 2.10
    b, q = d - 1.0, 1.0 - p
    assert kelly_fraction(p, d, fraction=1.0) == pytest.approx((b * p - q) / b, abs=1e-12)


def test_kelly_and_expected_value_never_disagree() -> None:
    """MODEL.md §10: b*p - q == d*p - 1, so edge, EV and Kelly agree by identity."""
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        for d in (1.2, 1.8, 2.5, 4.0, 11.0):
            assert (kelly_fraction(p, d, fraction=1.0) > 0.0) == (expected_value(p, d) > 0.0)


def test_negative_edge_returns_zero_not_a_reverse_bet() -> None:
    """MODEL.md §10: a negative f* means do not bet, not bet the other side."""
    assert kelly_fraction(0.2, 2.0) == 0.0


def test_quarter_kelly_is_the_default() -> None:
    p, d = 0.55, 2.10
    assert kelly_fraction(p, d) == pytest.approx(kelly_fraction(p, d, fraction=1.0) / 4.0, abs=1e-12)


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="probability"):
        kelly_fraction(1.5, 2.0)
    with pytest.raises(ValueError, match="greater than 1"):
        kelly_fraction(0.5, 1.0)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_kelly.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/market/kelly.py`:

```python
"""Bankroll fraction maximising long-run logarithmic growth.

MODEL.md §10:

    f* = (b*p - q) / b,   b = d - 1,  q = 1 - p

Note b*p - q == d*p - 1, so Kelly, edge and expected value can never disagree
about whether a bet is worth taking.

A negative f* means do not bet -- not bet the other side, which has its own
price and its own assessment.

The default is quarter Kelly. Full Kelly is intolerably volatile once the
probability estimate itself carries error.
"""

DEFAULT_FRACTION = 0.25


def expected_value(probability: float, decimal_odds: float) -> float:
    """Expected profit per unit staked: d*p - 1."""
    _validate(probability, decimal_odds)
    return decimal_odds * probability - 1.0


def kelly_fraction(
    probability: float, decimal_odds: float, fraction: float = DEFAULT_FRACTION
) -> float:
    """Fraction of bankroll to stake, floored at zero."""
    _validate(probability, decimal_odds)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Kelly fraction must lie in (0, 1], got {fraction}")
    b = decimal_odds - 1.0
    full = (b * probability - (1.0 - probability)) / b
    return max(0.0, full * fraction)


def _validate(probability: float, decimal_odds: float) -> None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    if not decimal_odds > 1.0:
        raise ValueError(f"decimal odds must be greater than 1, got {decimal_odds}")
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_kelly.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/market/kelly.py tests/test_kelly.py
git commit -m "feat(market): fractional Kelly staking

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Scoring rules

**Files:**
- Create: `src/footy/eval/scoring.py`, `tests/test_scoring.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ranked_probability_score(forecast: npt.ArrayLike, outcome: int) -> float`
  - `brier_score(forecast: npt.ArrayLike, outcome: int) -> float`
  - `log_loss(forecast: npt.ArrayLike, outcome: int) -> float`

Outcome is an index into the forecast vector: for 1X2, `0 = home, 1 = draw, 2 = away`.

- [ ] **Step 1: Write the failing test**

`tests/test_scoring.py`:

```python
"""MODEL.md §11 — evaluating a forecast."""

import math

import pytest

from footy.eval.scoring import brier_score, log_loss, ranked_probability_score


def test_perfect_forecast_scores_zero() -> None:
    assert ranked_probability_score([1.0, 0.0, 0.0], 0) == pytest.approx(0.0, abs=1e-12)
    assert brier_score([1.0, 0.0, 0.0], 0) == pytest.approx(0.0, abs=1e-12)


def test_rps_matches_the_closed_form() -> None:
    """MODEL.md §11: RPS = 1/(r-1) * sum_i (cumsum(p) - cumsum(o))**2 over i < r."""
    p, outcome = [0.5, 0.3, 0.2], 1
    o = [0.0, 1.0, 0.0]
    cp = cq = 0.0
    total = 0.0
    for i in range(len(p) - 1):
        cp += p[i]
        cq += o[i]
        total += (cp - cq) ** 2
    assert ranked_probability_score(p, outcome) == pytest.approx(total / (len(p) - 1), abs=1e-12)


def test_rps_is_distance_sensitive_where_brier_is_not() -> None:
    """MODEL.md §11: forecasting a home win scores the same under Brier whether
    the match was drawn or lost. RPS must distinguish them."""
    forecast = [0.7, 0.2, 0.1]
    assert brier_score(forecast, 1) == pytest.approx(brier_score(forecast, 2), abs=1e-12)
    assert ranked_probability_score(forecast, 1) < ranked_probability_score(forecast, 2)


def test_brier_sums_over_categories() -> None:
    p, outcome = [0.5, 0.3, 0.2], 1
    o = [0.0, 1.0, 0.0]
    assert brier_score(p, outcome) == pytest.approx(
        sum((a - b) ** 2 for a, b in zip(p, o, strict=True)), abs=1e-12
    )


def test_log_loss_matches_the_closed_form() -> None:
    assert log_loss([0.5, 0.3, 0.2], 1) == pytest.approx(-math.log(0.3), abs=1e-12)


def test_log_loss_is_finite_for_a_zero_probability_outcome() -> None:
    assert math.isfinite(log_loss([1.0, 0.0, 0.0], 1))


def test_rejects_a_forecast_that_does_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ranked_probability_score([0.5, 0.3, 0.3], 0)


def test_rejects_an_out_of_range_outcome() -> None:
    with pytest.raises(IndexError):
        brier_score([0.5, 0.3, 0.2], 3)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/eval/scoring.py`:

```python
"""Scoring rules for probabilistic forecasts.

MODEL.md §11. Football outcomes are *ordered* -- home, draw, away is not an
arbitrary set of three labels -- and the choice of metric turns on that.

Brier is blind to ordering: forecasting a home win scores the same whether the
match was drawn or lost. RPS (Epstein, 1969) is distance-sensitive and
punishes a near miss less than a distant one.

This is contested. Wheatcroft (2021) argues distance sensitivity is not in
fact desirable here. Both are implemented so the choice stays explicit.
"""

import numpy as np
import numpy.typing as npt

_EPSILON = 1e-15
"""Floor for log loss, so a zero-probability realised outcome is finite."""


def _prepare(forecast: npt.ArrayLike, outcome: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    p = np.asarray(forecast, dtype=np.float64)
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError(f"forecast must sum to 1, got {float(p.sum())}")
    if not 0 <= outcome < p.size:
        raise IndexError(f"outcome {outcome} outside the {p.size}-category forecast")
    observed = np.zeros_like(p)
    observed[outcome] = 1.0
    return p, observed


def ranked_probability_score(forecast: npt.ArrayLike, outcome: int) -> float:
    """Distance-sensitive score for ordered categories. Lower is better."""
    p, observed = _prepare(forecast, outcome)
    cumulative = np.cumsum(p) - np.cumsum(observed)
    return float(np.sum(cumulative[:-1] ** 2) / (p.size - 1))


def brier_score(forecast: npt.ArrayLike, outcome: int) -> float:
    """Squared error summed over categories. Lower is better. Ignores ordering."""
    p, observed = _prepare(forecast, outcome)
    return float(np.sum((p - observed) ** 2))


def log_loss(forecast: npt.ArrayLike, outcome: int) -> float:
    """Negative log probability of the realised outcome. Lower is better."""
    p, _ = _prepare(forecast, outcome)
    return float(-np.log(max(float(p[outcome]), _EPSILON)))
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/footy/eval/scoring.py tests/test_scoring.py
git commit -m "feat(eval): RPS, Brier and log loss

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: Murphy decomposition with the within-bin variance

The subtle one. The classical three-way identity is exact only when each bin holds a single distinct forecast; binning continuous forecasts leaves a residual equal to the within-bin variance. `MODEL.md` §11.1 is explicit that the engine got this wrong the first time.

**Files:**
- Create: `src/footy/eval/murphy.py`, `tests/test_murphy.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MurphyTerms` — a `NamedTuple` with `reliability`, `resolution`, `uncertainty`, `within_bin_variance`, `brier`
  - `murphy_decomposition(forecasts: npt.ArrayLike, outcomes: npt.ArrayLike, n_bins: int = 10) -> MurphyTerms`

Binary forecasts only: `forecasts[i]` is P(event), `outcomes[i]` is 0 or 1.

- [ ] **Step 1: Write the failing test**

`tests/test_murphy.py`:

```python
"""MODEL.md §11.1 — the identity is not exact for continuous forecasts."""

from typing import Any

import numpy as np
import pytest

from footy.eval.murphy import murphy_decomposition


def test_reproduces_the_model_md_table(model_md: dict[str, Any]) -> None:
    """Layer 1. MODEL.md §11.1.

    The outcome vector is derived rather than transcribed; see the fixture's
    _outcomes_provenance note.
    """
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    tol = spec["tolerance"]

    assert terms.brier == pytest.approx(spec["brier"], abs=tol)
    assert terms.uncertainty == pytest.approx(spec["uncertainty"], abs=tol)
    assert terms.within_bin_variance == pytest.approx(spec["wbv"], abs=tol)

    three_way = terms.reliability - terms.resolution + terms.uncertainty
    assert three_way == pytest.approx(spec["three_way_sum"], abs=tol)


def test_the_four_term_identity_is_exact(model_md: dict[str, Any]) -> None:
    """MODEL.md §11.1: BS = REL - RES + UNC + WBV, to 1e-12."""
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    assert (
        terms.reliability - terms.resolution + terms.uncertainty + terms.within_bin_variance
    ) == pytest.approx(terms.brier, abs=1e-12)


def test_the_residual_is_the_within_bin_variance(model_md: dict[str, Any]) -> None:
    """Guards against anyone quietly simplifying back to three terms."""
    spec = model_md["murphy_decomposition"]
    terms = murphy_decomposition(spec["forecasts"], spec["outcomes"], n_bins=spec["n_bins"])
    residual = terms.brier - (terms.reliability - terms.resolution + terms.uncertainty)
    assert residual == pytest.approx(terms.within_bin_variance, abs=1e-12)


def test_wbv_vanishes_when_every_bin_holds_one_distinct_forecast() -> None:
    """The discrete case Murphy was writing about: the three-way identity is exact."""
    forecasts = [0.05, 0.05, 0.45, 0.45, 0.75, 0.75, 0.95, 0.95]
    outcomes = [0, 0, 1, 0, 1, 1, 1, 1]
    terms = murphy_decomposition(forecasts, outcomes, n_bins=10)
    assert terms.within_bin_variance == pytest.approx(0.0, abs=1e-12)
    assert (terms.reliability - terms.resolution + terms.uncertainty) == pytest.approx(
        terms.brier, abs=1e-12
    )


def test_a_perfectly_calibrated_forecast_has_zero_reliability() -> None:
    forecasts = [0.5] * 100
    outcomes = [i % 2 for i in range(100)]
    terms = murphy_decomposition(forecasts, outcomes, n_bins=10)
    assert terms.reliability == pytest.approx(0.0, abs=1e-12)


def test_a_base_rate_forecast_has_zero_resolution() -> None:
    """Saying the base rate every time means saying nothing."""
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    terms = murphy_decomposition([0.3] * 10, outcomes, n_bins=10)
    assert terms.resolution == pytest.approx(0.0, abs=1e-12)


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        murphy_decomposition([0.5, 0.5], [1], n_bins=10)


def test_rejects_non_binary_outcomes() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        murphy_decomposition([0.5, 0.5], [0, 2], n_bins=10)
```

- [ ] **Step 2: Run and confirm it fails**

Run: `uv run pytest tests/test_murphy.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/footy/eval/murphy.py`:

```python
"""Murphy's vector partition of the Brier score.

MODEL.md §11. Turns "the model scored 0.58" into a statement about why:

    BS = REL - RES + UNC + WBV

REL (reliability) asks whether stated probabilities match observed
frequencies; zero is perfect calibration. RES (resolution) asks whether the
model says anything beyond the base rate; zero means it does not. UNC belongs
to the sport, not the model.

WBV is the part that is easy to get wrong, and this engine got it wrong first
time. The classical three-way identity is exact only when each bin holds a
single distinct forecast -- the discrete case Murphy (1973) was writing about.
Bin *continuous* forecasts and a residual appears, because REL compares each
bin's mean forecast against its observed frequency while BS uses each
individual forecast. That residual is exactly the within-bin variance, so the
identity that holds for arbitrary forecasts carries four terms, not three.

All four are reported. Reducing this to three terms is a bug, not a
simplification.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class MurphyTerms(NamedTuple):
    """BS == reliability - resolution + uncertainty + within_bin_variance."""

    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float
    brier: float


def murphy_decomposition(
    forecasts: npt.ArrayLike, outcomes: npt.ArrayLike, n_bins: int = 10
) -> MurphyTerms:
    """Decompose the binary Brier score into its four exact components.

    Args:
        forecasts: Probability of the event, one per observation, in [0, 1].
        outcomes: Realised outcomes, each 0 or 1.
        n_bins: Equal-width bins over [0, 1].
    """
    p = np.asarray(forecasts, dtype=np.float64)
    o = np.asarray(outcomes, dtype=np.float64)

    if p.shape != o.shape:
        raise ValueError(f"forecasts and outcomes must be the same length, got {p.shape} and {o.shape}")
    if p.size == 0:
        raise ValueError("need at least one observation")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("forecasts must lie in [0, 1]")
    if not np.all(np.isin(o, (0.0, 1.0))):
        raise ValueError("outcomes must each be 0 or 1")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    n = p.size
    base_rate = float(o.mean())
    brier = float(np.mean((p - o) ** 2))

    # Equal-width bins; a forecast of exactly 1.0 belongs in the top bin.
    bin_index = np.clip((p * n_bins).astype(int), 0, n_bins - 1)

    reliability = resolution = within_bin_variance = 0.0
    for k in range(n_bins):
        members = bin_index == k
        count = int(members.sum())
        if count == 0:
            continue
        mean_forecast = float(p[members].mean())
        observed_rate = float(o[members].mean())

        reliability += count * (mean_forecast - observed_rate) ** 2
        resolution += count * (observed_rate - base_rate) ** 2
        within_bin_variance += float(np.sum((p[members] - mean_forecast) ** 2))

    return MurphyTerms(
        reliability=reliability / n,
        resolution=resolution / n,
        uncertainty=base_rate * (1.0 - base_rate),
        within_bin_variance=within_bin_variance / n,
        brier=brier,
    )
```

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest tests/test_murphy.py -v`
Expected: 8 passed.

If `test_reproduces_the_model_md_table` fails on `brier` while the others pass, the derived outcome vector in the fixture is wrong — re-run the Task 3 Step 2 check before touching this module.

- [ ] **Step 5: Commit**

```bash
git add src/footy/eval/murphy.py tests/test_murphy.py
git commit -m "feat(eval): Murphy decomposition with the within-bin variance term

The three-way identity is exact only for discrete forecasts. Binning
continuous forecasts leaves a residual equal to the within-bin variance, so
all four terms are reported and the exact identity is asserted.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 15: Purity enforcement and Layer-2 property tests

**Files:**
- Create: `tests/test_purity.py`, `tests/test_properties.py`

**Interfaces:**
- Consumes: every module built so far
- Produces: the two guarantees the spec's §7.2 and §8 Layer 2 require

- [ ] **Step 1: Write the purity test**

`tests/test_purity.py`:

```python
"""Spec §7.2 — core/ performs no I/O, reads no clock, and has no randomness.

Carried forward from the TypeScript engine, where the implementation being
replaced had Math.random() inside the pricing path: odds changed on every
refresh, so a bettor could re-roll a price until it suited them.
"""

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).parent.parent / "src" / "footy" / "core"
BANNED_MODULES = {"random", "datetime", "time", "os", "pathlib", "io", "requests", "urllib"}
BANNED_CALLS = {"open", "input", "print"}

CORE_FILES = sorted(CORE.glob("*.py"))


def test_core_directory_is_not_empty() -> None:
    """Guards against the scan silently passing because it found nothing."""
    assert len(CORE_FILES) >= 5


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_no_banned_imports(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in BANNED_MODULES, f"{path.name} imports {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in BANNED_MODULES, f"{path.name} imports from {node.module}"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_no_io_calls(path: Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in BANNED_CALLS, f"{path.name} calls {node.func.id}()"


def test_pricing_is_deterministic() -> None:
    """Same inputs, byte-identical output, always."""
    from footy.core.matrix import scoreline_matrix

    first = scoreline_matrix(1.6, 1.1, -0.10)
    second = scoreline_matrix(1.6, 1.1, -0.10)
    assert first.tobytes() == second.tobytes()
```

- [ ] **Step 2: Run the purity test**

Run: `uv run pytest tests/test_purity.py -v`
Expected: all pass. If a banned import is found, remove it — do not weaken the test.

- [ ] **Step 3: Write the Layer-2 property tests**

`tests/test_properties.py`:

```python
"""Spec §8 Layer 2 — invariants that must hold for any admissible parameters.

Hypothesis searches the admissible region rather than trusting hand-picked
fixtures, which is what catches the boundary cases a fixed example misses.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from footy.core.dixon_coles import rho_bounds
from footy.core.markets import (
    asian_fair_odds,
    asian_handicap,
    both_teams_to_score,
    correct_score,
    double_chance,
    match_odds,
    totals,
)
from footy.core.matrix import scoreline_matrix
from footy.market.demargin import shin
from footy.market.overround import apply_overround

rates = st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False)


@st.composite
def admissible_parameters(draw: st.DrawFn) -> tuple[float, float, float]:
    """(lam, mu, rho) with rho strictly inside its admissible region."""
    lam = draw(rates)
    mu = draw(rates)
    lo, hi = rho_bounds(lam, mu)
    inset = (hi - lo) * 0.02
    rho = draw(st.floats(min_value=lo + inset, max_value=hi - inset, allow_nan=False))
    return lam, mu, rho


@given(admissible_parameters())
@settings(max_examples=200, deadline=None)
def test_matrix_is_a_probability_distribution(params: tuple[float, float, float]) -> None:
    m = scoreline_matrix(*params)
    assert np.all(m >= 0.0)
    assert m.sum() == pytest.approx(1.0, abs=1e-9)


@given(admissible_parameters())
@settings(max_examples=200, deadline=None)
def test_every_market_is_a_valid_probability(params: tuple[float, float, float]) -> None:
    m = scoreline_matrix(*params)
    values = [*match_odds(m), *totals(m, 2.5), *both_teams_to_score(m), correct_score(m, 1, 1)]
    for value in values:
        assert 0.0 <= value <= 1.0 + 1e-12


@given(admissible_parameters())
@settings(max_examples=200, deadline=None)
def test_markets_cannot_contradict_one_another(params: tuple[float, float, float]) -> None:
    """Spec §8 Layer 2: they are marginals of one distribution."""
    m = scoreline_matrix(*params)
    home, draw, away = match_odds(m)
    dc_1x, _, dc_x2 = double_chance(m)

    assert home + draw + away == pytest.approx(1.0, abs=1e-9)
    assert dc_1x == pytest.approx(home + draw, abs=1e-9)
    assert dc_x2 == pytest.approx(draw + away, abs=1e-9)

    over, under = totals(m, 2.5)
    assert over + under == pytest.approx(1.0, abs=1e-9)

    triangle = sum(
        correct_score(m, x, y) for x in range(m.shape[0]) for y in range(m.shape[1]) if x > y
    )
    assert triangle == pytest.approx(home, abs=1e-9)


@given(admissible_parameters())
@settings(max_examples=100, deadline=None)
def test_level_ball_handicap_equals_draw_no_bet(params: tuple[float, float, float]) -> None:
    m = scoreline_matrix(*params)
    home, _, away = match_odds(m)
    side, _ = asian_handicap(m, 0.0)
    assert asian_fair_odds(side) == pytest.approx((home + away) / home, abs=1e-9)


@given(
    admissible_parameters(),
    st.floats(min_value=1.001, max_value=1.30, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_overround_hits_its_target_and_shortens_prices(
    params: tuple[float, float, float], target: float
) -> None:
    fair = np.array(match_odds(scoreline_matrix(*params)))
    if np.any(fair <= 1e-6):
        return  # a degenerate simplex is not a book
    odds = apply_overround(fair, target)
    assert float(np.sum(1.0 / odds)) == pytest.approx(target, abs=1e-9)
    assert np.all(odds < 1.0 / fair)


@given(
    st.lists(st.floats(min_value=0.02, max_value=0.9), min_size=2, max_size=4),
    st.floats(min_value=1.001, max_value=1.20, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_shin_recovers_a_simplex(raw: list[float], margin: float) -> None:
    implied = np.array(raw) / sum(raw) * margin
    recovered, z = shin(implied)
    assert recovered.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(recovered > 0.0)
    assert 0.0 <= z < 1.0


@given(st.lists(st.floats(min_value=0.05, max_value=0.9), min_size=2, max_size=4))
@settings(max_examples=100, deadline=None)
def test_shin_approaches_the_identity_as_margin_vanishes(raw: list[float]) -> None:
    """Spec §4: all transforms must converge to p as B -> 1."""
    fair = np.array(raw) / sum(raw)
    recovered, _ = shin(fair * 1.0000001)
    assert recovered == pytest.approx(fair, abs=1e-5)
```

- [ ] **Step 4: Run the property tests**

Run: `uv run pytest tests/test_properties.py -v`
Expected: all pass. A Hypothesis counterexample here is a genuine bug in the port — record the failing parameters before fixing.

- [ ] **Step 5: Run the whole suite and the full verification**

Run: `make verify`
Expected: ruff clean, mypy clean, every test passing.

- [ ] **Step 6: Commit**

```bash
git add tests/test_purity.py tests/test_properties.py
git commit -m "test: purity enforcement and Layer-2 property invariants

core/ is scanned for I/O, clock and randomness. Hypothesis searches the
admissible parameter region for violations of the matrix, market-consistency,
overround and Shin invariants.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: Rewrite the README and push

**Files:**
- Modify: `README.md` (full replacement)

**Interfaces:**
- Consumes: everything above
- Produces: a repository whose front page describes the research, not the app

- [ ] **Step 1: Replace README.md**

The current README describes a betting platform. Replace it entirely with a research front page covering, in this order:

1. **The question** — one paragraph. Papers claim to beat the bookmaker; that claim requires de-margining offered odds into a benchmark; the literature uses four different transforms interchangeably; this repository tests whether the choice changes the conclusion.
2. **The three pre-registered predictions** — the P1/P2/P3 table copied from the spec §2.
3. **Status** — a checklist of the five plans, with plan 1 marked done.
4. **Reproducing** — `make install`, `make verify`, and a note that `make data` and `make study` arrive in Plans 3 and 4.
5. **The model** — two sentences plus a link to `packages/quant-engine/docs/MODEL.md`, noting it is the specification the Python was written from.
6. **Layout** — the `src/footy/` table from this plan's File Structure section.
7. **Licence.**

Do not carry over any claim about the model's performance. Nothing has been measured yet, and the previous README's unbacked claims are exactly what this rebuild exists to correct.

- [ ] **Step 2: Verify no application language survives**

Run: `grep -rniE "bet(ting)?[ -]app|wallet|frontend|localhost:[0-9]|npm run dev" README.md`
Expected: no output.

- [ ] **Step 3: Full verification before pushing**

Run: `make verify`
Expected: everything green.

- [ ] **Step 4: Commit and push the branch and tag**

```bash
git add README.md
git commit -m "docs: rewrite the README around the research question

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git push -u origin research/demargining-study
git push origin v1-betting-app
```

- [ ] **Step 5: Report what remains**

Plan 1 is complete when `make verify` is green and every Layer-1 value in `tests/fixtures/model_md_values.json` is asserted by a passing test. Note for Plan 2: `MODEL.md` §4 specifies the likelihood but was never implemented in TypeScript, so the fitting layer has no oracle and no Layer-3 corroboration is possible for it. Its correctness rests entirely on the `check_grad` verification specified in the spec's §8.

---

## Self-Review

**Spec coverage.** §7.1 layout — Tasks 2, 4–14, with `fit/`, `data/` and `study/` deferred to Plans 2–4 as the spec's build order requires. §7.2 purity — Task 15. §8 Layer 1 — Task 3 defines the targets; Tasks 6, 10, 11 and 14 assert them. §8 Layer 2 — Task 15. §8 Layer 3 — deferred to Plan 4 by design, since it must run after the clean-room work is complete. §10.1 wave-1 deletion — Task 1. §4 transforms — only Shin is here; proportional, power and odds-ratio belong to Plan 3, and the `B → 1` convergence property is asserted for Shin in Task 15.

**Two spec items intentionally deferred beyond this plan.** The `paper/` tree is created in Plan 5, not scaffolded empty here. The spec's §11 names two skills to author; the Python-numerics skill is genuinely useful before Plan 2's gradient work, but writing it is not a prerequisite for any task in this plan, and research-writing is not needed until Plan 5. Both are better authored immediately before the plan that needs them.

**Placeholder scan.** No TBDs. Every code step carries runnable code. Task 16's README is described section by section rather than written out, because it summarises results that do not exist yet — the structure is fully specified and the content is mechanical from the spec.

**Type consistency.** `Matrix` is defined in `core/markets.py` and imported by `market/overround.py`. `AsianOutcome` is produced by `asian_handicap` and consumed by `asian_fair_odds`. `MurphyTerms` field names match their assertions in `tests/test_murphy.py`. `GRID` is defined once in `core/dixon_coles.py` and imported by `core/matrix.py`. `model_md` is the session fixture from `tests/conftest.py`, created in Task 4 and used by Tasks 6, 10, 11 and 14 — Task 4 must therefore run before them, which the numbering enforces.
