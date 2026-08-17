.PHONY: install compile test lint typecheck verify clean tables paper

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

# Tables are written out of results/cells.csv, never typed into the paper.
tables:
	uv run python scripts/make_tables.py

# Compiles at any stage: result slots render as "[pending]" until `tables` runs.
# The finished PDF is copied to the repository root so it is visible on GitHub.
paper:
	latexmk -pdf -outdir=build -cd paper/main.tex
	cp paper/build/main.pdf The-Demargining-Artifact.pdf

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache paper/build
	find . -name __pycache__ -type d -exec rm -rf {} +
