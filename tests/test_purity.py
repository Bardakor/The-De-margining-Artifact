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

# Banning import roots is not enough. core/ legitimately imports numpy and scipy,
# so `np.random.random()` would sail past a root check — it is an attribute call on
# an already-permitted module. Since randomness in the pricing path is the exact bug
# this scanner exists to prevent, the dotted path of every attribute is checked too.
BANNED_SEGMENTS = {"random", "rand", "stats", "seed", "shuffle", "permutation"}

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


def _dotted_path(node: ast.expr) -> str:
    """Reconstruct `np.random.default_rng` from a nested Attribute chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_no_randomness_through_a_permitted_module(path: Path) -> None:
    """numpy and scipy are permitted, so randomness must be caught by dotted path.

    `np.random.random()` is an attribute call on an already-imported module, which
    the import-root and bare-call scans both miss. This closes that hole — the
    reintroduction vector for the exact defect the purity boundary exists to stop.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted_path(node)
        offending = set(dotted.split(".")) & BANNED_SEGMENTS
        assert not offending, f"{path.name} reaches {dotted} via {sorted(offending)}"


@pytest.mark.parametrize("path", CORE_FILES, ids=lambda p: p.name)
def test_no_randomness_imported_by_name(path: Path) -> None:
    """Catches `from numpy import random`, which names no banned module root."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name not in BANNED_SEGMENTS, (
                    f"{path.name} imports {alias.name} from {node.module}"
                )


def test_the_scanner_actually_catches_a_violation() -> None:
    """The scanner must be able to fail. A green scan that cannot fail is worthless."""
    smuggled = ast.parse("import numpy as np\ndef price(): return np.random.random()")
    found = [
        _dotted_path(n)
        for n in ast.walk(smuggled)
        if isinstance(n, ast.Attribute) and set(_dotted_path(n).split(".")) & BANNED_SEGMENTS
    ]
    assert found, "scanner failed to detect np.random.random() — the net has a hole"


def test_pricing_is_deterministic() -> None:
    """Same inputs, byte-identical output, always."""
    from footy.core.matrix import scoreline_matrix

    first = scoreline_matrix(1.6, 1.1, -0.10)
    second = scoreline_matrix(1.6, 1.1, -0.10)
    assert first.tobytes() == second.tobytes()
