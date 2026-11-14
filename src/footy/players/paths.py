"""Where paper 2 keeps its files, so every module agrees (design §4)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "players" / "raw"
PROCESSED = ROOT / "data" / "players" / "processed"
RESULTS = ROOT / "results" / "players"
PAPER = ROOT / "paper-unmeasured"
GENERATED = PAPER / "generated"
