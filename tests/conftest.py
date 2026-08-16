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
