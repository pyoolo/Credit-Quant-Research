"""Track-3-specific parameters and cache paths."""

from __future__ import annotations

import numpy as np

from crv.config import paths, settings

DS_PANEL_WEEKS = 104
RISK_FREE = settings.RISK_FREE          # reference EUR risk-free (simple)
R_SCENARIOS = [0.20, 0.40, 0.60]        # recovery scenarios for the PD frontier
R_GRID = np.linspace(0.05, 0.90, 24)    # recovery grid for the full saved frontier

# Payment rank -> seniority number (lower = more senior, higher recovery)
RANK_ORDER = {
    "Secured": 0, "1st lien": 0, "Sr Secured": 0,
    "Sr Unsecured": 1, "SNDB": 1, "Unsecured": 1,
    "2nd lien": 2, "SBOD": 2, "Subordinated": 3,
    "Jr Subordinated": 4, "JUND": 4,
}

# --- track-3 cache sub-folder ---
CACHE_DS_DIR = paths.CACHE_DIR / "track3"
CACHE_DS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DS_LONG = CACHE_DS_DIR / "ds_long_history.pkl"
