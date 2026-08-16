.PHONY: install compile test lint typecheck verify clean

install:
	uv sync --extra dev

compile:
	uv run python -m compileall -q src tests

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

verify: compile lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
