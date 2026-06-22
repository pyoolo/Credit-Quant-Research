# CRV — HY EUR Credit Relative-Value Engine

A compartmentalised, three-track relative-value system for European high-yield
corporate bonds. A single Bloomberg fetch routes the universe into three
disjoint panels; each track owns its model, artefacts and relative-value output;
a pricing layer reuses the fitted artefacts to price new issues; and a Streamlit
app exposes the monitor and pricer.

This repository is the de-notebooked, packaged form of an eight-notebook
research pipeline. The logic is preserved verbatim; the structure is
production-shaped: one source of truth for configuration, proper logging,
isolated Bloomberg access, per-track packages, a CLI and a test suite.

---

## The three tracks

A bond is routed to exactly one track based on its coupon type, rating and where
its Z-spread sits relative to two bands (`SPREAD_FLOOR`, `DISTRESS_CUTOFF`):

| Track | Module | Universe | Method | Signal |
|------|--------|----------|--------|--------|
| **1 — Cross-section** | `crv.tracks.cross_section` | Fixed-coupon HY corporates | Lasso on `log(z_spread)` vs notch rating, structure & two-sided fundamentals, with sector + week fixed-effects | Residual z-scored within (date × rating bucket) → CHEAP / FAIR / RICH |
| **2 — Self-ref** | `crv.tracks.self_ref` | FRN / perpetual / AT1 | Structural spread (FRN discount margin; PERP spread-to-worst vs EUR swap curve) + rolling hybrid (Huber/GBM) cross-section fair value | Robust MAD-based RV score, self-referential z-score fallback |
| **3 — Distressed** | `crv.tracks.distressed` | Z-spread above the distress cutoff | Constant-hazard PD/recovery frontier, capital-structure coherence, price-trajectory regime, composite | Judgement score in [-1, +1] with an uncertainty band |

The router (`crv.data.router`) is the single arbiter and guarantees the three
panels are mutually exclusive.

---

## Architecture

```
src/crv/
├── config/                 # single source of truth (replaces the copied `paths` cell)
│   ├── paths.py            #   every file location; root from $CRV_ROOT (default ./project)
│   └── settings.py         #   all model / pipeline parameters
├── utils/                  # logging, dates, io, bqlio (the ONLY module importing bql)
├── data/                   # shared layer: one fetch -> router -> 3 panels
│   ├── universe.py  fetch.py  sanity.py  router.py  ratings.py  pipeline.py
└── tracks/
    ├── cross_section/      # Track 1
    │   ├── fundamentals.py features.py model.py signal.py backtest.py
    │   └── artifacts.py    pipeline.py
    ├── self_ref/           # Track 2
    │   ├── config.py fetch.py pricing.py model.py artifacts.py pipeline.py
    └── distressed/         # Track 3
        ├── config.py fetch.py solver.py levels.py signal.py artifacts.py pipeline.py
├── pricing/                # new-issue pricer (engine + batch/scenario/sensitivity)
├── app/                    # Streamlit monitor + pricer
└── cli.py                  # `crv` entry point
```

### Data flow

```
                    ┌─────────────┐
   Bloomberg  ─────▶│  crv.data   │  fetch → sanity → router
                    └──────┬──────┘
            ┌──────────────┼──────────────┐
            ▼              ▼               ▼
       panel_xs       panel_ts        panel_ds
            │              │               │
       ┌────▼────┐    ┌────▼────┐     ┌────▼────┐
       │ Track 1 │    │ Track 2 │     │ Track 3 │
       └────┬────┘    └────┬────┘     └────┬────┘
       rv_track1.csv  rv_track2.csv   rv_track3.csv
       model_t1.pkl   track2_peers    track3_params
            └──────────────┼───────────────┘
                    ┌───────▼────────┐
                    │ crv.pricing /  │  monitor + new-issue pricer
                    │ crv.app        │
                    └────────────────┘
```

The project data tree (`data/`, `outputs/`) is created automatically under
`$CRV_ROOT` (default `./project`) and is git-ignored.

---

## Installation

```bash
pip install -e .            # core engine
pip install -e ".[app]"     # + Streamlit app
pip install -e ".[dev]"     # + pytest / ruff / mypy
```

Requires Python ≥ 3.10. Bloomberg access (`bql`) is provided by the terminal and
is **not** on PyPI — only the fetch steps need it; everything downstream runs
without a terminal.

---

## Usage

### CLI

```bash
crv fetch                       # shared fetch + 3-track router  (needs Bloomberg)
crv track1                      # cross-section pipeline
crv track2                      # self-ref pipeline              (needs Bloomberg)
crv track3                      # distressed pipeline            (needs Bloomberg)
crv all                         # fetch + all three tracks, in order
crv price examples/new_issue_specs.json    # price new issues
crv app                         # launch the Streamlit app
```

Set the project root if you don't want it under the working directory:

```bash
export CRV_ROOT=/data/crv
```

Drop `Financials.xlsx` (9Fin + Back-Up sheets) into `$CRV_ROOT/data/inputs/`
before running Track 1.

### Library

```python
from crv.tracks.cross_section import pipeline as t1
result = t1.run()                       # fit, score, persist, backtest
print(result["snapshot"].head())        # latest CHEAP/RICH table
print(result["backtest"]["ic_mean"])    # forward rank IC

from crv.pricing import engine, analytics
art = engine.load_artefacts()
analytics.batch_price(my_specs, art)    # price a list of new-issue dicts
```

---

## Testing

```bash
pytest                  # full suite
pytest --cov=crv        # with coverage
ruff check src tests    # lint
```

The suite runs entirely on synthetic, schema-faithful data — no Bloomberg
terminal required. It covers the router's disjointness, the distressed solver's
price↔hazard round-trip, rating/notch derivation, the pricing router and
fund-score reconstruction, and an end-to-end Track-1 fit/score.

---

## Notes on the model

- **Why the residual is z-scored within rating, not sector.** Sector is already
  a fair-value driver in the model, so within-sector residual dispersion is
  near-zero; rating buckets retain comparable dispersion, so a BB is read
  against BBs and a CCC against CCCs.
- **Why alpha is selected on a held-out issuer split.** The internal CV optimum
  consistently preferred a higher penalty than what generalises to unseen
  issuers; selecting on a 30 % issuer hold-out removes that divergence.
- **The distressed score is a judgement, not a fair value.** Track 3 reports a
  lo–hi band that widens where data is missing or the trajectory is uncertain.

See module docstrings for the full methodology; each design choice is documented
where it lives.
