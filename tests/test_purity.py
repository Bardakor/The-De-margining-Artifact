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
