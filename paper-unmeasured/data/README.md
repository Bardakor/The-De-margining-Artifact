# Curated inputs for paper 2

- `v1_ranking.csv` — the "V1" ranking Liam supplied on 2026-11-14 (output of a
  model built elsewhere, scores not recomputable). Used only to define part of
  the player pool and as a comparison point; never as an input to an estimate.
- `ballon_dor.csv` — every ranked player of the men's Ballon d'Or 2019 and 2021–2026
  (2020 cancelled; 2019 is used only as the previous edition for 2021, not for the pool), cross-checked between independent outlets; per-row source
  URLs and every disagreement are in `ballon_dor_sources.md`. Each edition's
  points sum to voters × points per ballot exactly. Join on `player`, not
  `club`.
