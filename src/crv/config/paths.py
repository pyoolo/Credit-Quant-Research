"""Single source of truth for every file location in the CRV pipeline.

The directory tree mirrors the data flow:

    fetch  ->  panels/   ->  tracks  ->  rv_output/  ->  apps
                                    \\->  cache/<track>/  (fitted artefacts)

All folders are created on import (idempotent). Override the root by setting the
``CRV_ROOT`` environment variable before importing anything from ``crv``.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Root                                                                        #
# --------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("CRV_ROOT", Path.cwd() / "project")).resolve()

# --------------------------------------------------------------------------- #
# Top-level folders                                                           #
# --------------------------------------------------------------------------- #
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

# Data sub-folders, one per pipeline stage
CACHE_DIR = DATA_DIR / "cache"        # .pkl caches (raw + 3 tracks)
PANELS_DIR = DATA_DIR / "panels"      # fetch output  -> track input
RVOUT_DIR = DATA_DIR / "rv_output"    # track output  -> app input
INPUTS_DIR = DATA_DIR / "inputs"      # external inputs (Financials.xlsx)

# Per-track artefact folders
CACHE_T1_DIR = CACHE_DIR / "track1"   # Track-1 fitted-model artefacts

_ALL_DIRS = (
    CACHE_DIR,
    PANELS_DIR,
    RVOUT_DIR,
    INPUTS_DIR,
    OUTPUTS_DIR,
    CACHE_T1_DIR,
)


def ensure_dirs() -> None:
    """Create every project folder (idempotent)."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

# --------------------------------------------------------------------------- #
# Caches (.pkl)                                                               #
# --------------------------------------------------------------------------- #
CACHE_RAW = CACHE_DIR / "hy_eur_raw_cache.pkl"   # shared raw dynamic fetch
CACHE_XS = CACHE_DIR / "hy_eur_xs_cache.pkl"     # Track 1 cross-section
CACHE_TS = CACHE_DIR / "hy_eur_ts_cache.pkl"     # Track 2 self-ref
CACHE_DS = CACHE_DIR / "hy_eur_ds_cache.pkl"     # Track 3 distressed

# --------------------------------------------------------------------------- #
# Panels (fetch output, track input)                                          #
# --------------------------------------------------------------------------- #
CSV_XS = PANELS_DIR / "panel_xs.csv"
CSV_TS = PANELS_DIR / "panel_ts.csv"
CSV_DS = PANELS_DIR / "panel_ds.csv"

# --------------------------------------------------------------------------- #
# Relative-value output (track output, app input)                             #
# --------------------------------------------------------------------------- #
RV_T1 = RVOUT_DIR / "rv_track1.csv"
RV_T2 = RVOUT_DIR / "rv_track2.csv"
RV_T3 = RVOUT_DIR / "rv_track3.csv"

T2_PEERS = RVOUT_DIR / "track2_peers.json"       # per-family peer distributions
T3_PARAMS = RVOUT_DIR / "track3_params.json"
T3_TRAJ = RVOUT_DIR / "track3_traj.csv"

# --------------------------------------------------------------------------- #
# Per-track fitted artefacts                                                  #
# --------------------------------------------------------------------------- #
MODEL_T1 = CACHE_T1_DIR / "model_t1.pkl"             # Lasso + scaler + resid quantiles
COMPANY_SNAP = CACHE_T1_DIR / "company_snapshot.csv"  # per-ISIN raw-ratio snapshot

# --------------------------------------------------------------------------- #
# External inputs                                                             #
# --------------------------------------------------------------------------- #
FINANCIALS = INPUTS_DIR / "Financials.xlsx"

# --------------------------------------------------------------------------- #
# Final outputs                                                               #
# --------------------------------------------------------------------------- #
DASHBOARD_HTML = OUTPUTS_DIR / "crv_dashboard.html"


def summary() -> str:
    """Human-readable summary of the resolved layout."""
    return f"paths OK | root: {ROOT}"
