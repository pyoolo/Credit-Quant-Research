# Architecture

This document complements the README with the reasoning behind the package
boundaries. The guiding principle is **compartmentalisation**: the three tracks
share a data layer and nothing else, so each can evolve, be tested and be
reasoned about in isolation.

## Layers

### `crv.config` — single source of truth
The original notebooks pasted an identical `paths` block into the first cell of
every file. That block is now one place (`paths.py`), and every magic number
lives in `settings.py`. The project root resolves from `$CRV_ROOT` so the same
code runs locally, in CI and in production without edits.

### `crv.utils` — track-agnostic primitives
`bqlio` is the only module that imports `bql`. This is deliberate: it means the
entire package imports and tests without a Bloomberg terminal, and the tracks
operate on plain DataFrames. `logging` replaces every `print`; `io` stamps each
track cache with provenance metadata; `dates` owns the Friday-anchored weekly
grid.

### `crv.data` — the shared spine
One fetch, one sanity report, one router. The router is the **only** code that
decides track membership, which is what guarantees the three panels are
disjoint. Everything above this layer is track-private.

### `crv.tracks.*` — three independent pipelines
Each track is a package with the same shape:

```
<track>/
  (config.py)     track-specific params + cache paths   [tracks 2 & 3]
  fetch.py        track-specific Bloomberg pulls         [tracks 2 & 3]
  <models>.py     the track's modelling steps
  artifacts.py    persistence of the fitted outputs
  pipeline.py     orchestration: load -> model -> persist
```

Track 1 has the richest model surface (fundamentals → features → Lasso → signal
→ backtest); Tracks 2 and 3 add their own fetch because they need longer history
and instrument-specific fields (call schedules, discount margins, distressed
price series) the shared fetch does not pull.

### `crv.pricing` — artefact consumer
The pricer never re-derives anything; it loads the persisted artefacts
(`model_t1.pkl`, `track2_peers.json`, `track3_params.json`) and dispatches a
spec through the same routing logic the live panel uses. The Track-1 pricer
reconstructs the two fundamental scores from **raw** ratios using the
standardisation params saved at fit time, so a user supplies `net_lev` / `icr`
rather than pre-computed z-scores.

### `crv.app` — presentation
Streamlit, split into a UI-agnostic `data` loader and two views (monitor,
pricer). The loader normalises the three heterogeneous `rv_track*.csv` schemas
into one shape the UI renders uniformly.

## Testing strategy

Pure-logic components (router, solver, ratings, pricing router, fund-score
reconstruction) are unit-tested directly. The Track-1 model is exercised
end-to-end on a synthetic panel built with a known rating effect, so the test
can assert the fit recovers it (`rating_notch` leads the contribution table).
Bloomberg-dependent code is structured so it imports cleanly and is covered by
import tests, but live fetches are out of scope for CI.

## Extending

- **A new feature in Track 1**: add it to `settings.SEL_FEATS` (and
  `SET_FEATS` if it should appear in the selection-path diagnostic), then to the
  relevant builder in `features.py`. The model, signal and pricer pick it up via
  `feats`/`xcols`.
- **A new track**: add a package under `crv.tracks`, extend the router with its
  membership predicate, and register its pipeline in the CLI.
