.PHONY: install compile test lint typecheck verify clean tables figures paper \
	players-data players-results paper2

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

# Figures are written out of results/cells.csv, results/book_margin.csv and
# results/calibration.json, never typed into the paper.
figures:
	uv run python scripts/make_figures.py

# Compiles at any stage: result slots render as "[pending]" until `tables` runs,
# and figure slots render as "[pending]" until `figures` runs.
# The finished PDF is copied to the repository root so it is visible on GitHub.
paper: figures
	latexmk -pdf -outdir=build -cd paper/main.tex
	cp paper/build/main.pdf The-Demargining-Artifact.pdf

# Paper 2, The Unmeasured Remainder. Data: downloads (cached) and processed tables.
players-data:
	uv run python scripts/players/acquire_transfermarkt.py
	uv run python scripts/players/acquire_statsbomb.py
	uv run python scripts/players/build_panel.py

# Every analysis, master seed 20261114 (pre-registration A12). Hours, not minutes.
players-results:
	uv run python scripts/players/run_descriptive.py
	uv run python scripts/players/run_impact.py
	uv run python scripts/players/run_tournaments.py
	uv run python scripts/players/run_ranking.py

# Numbers, tables and figures come from results/players/, never typed into the paper.
paper2:
	uv run python scripts/players/make_paper2.py
	latexmk -pdf -outdir=build -cd paper-unmeasured/main.tex
	cp paper-unmeasured/build/main.pdf The-Unmeasured-Remainder.pdf

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache paper/build paper-unmeasured/build
	find . -name __pycache__ -type d -exec rm -rf {} +
