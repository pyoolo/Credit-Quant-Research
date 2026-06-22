"""Track-2-specific parameters and cache paths (kept separate from the global config)."""

from __future__ import annotations

from crv.config import paths

# --- fetch ---
TS_PANEL_WEEKS = 104                          # long self-ref history
EUR_TENORS = [1, 2, 3, 5, 7, 10, 15, 20, 30]  # EUR swap-curve tenors

# --- structural pricing ---
RC_MIN_OBS = 8       # min observations for the self-ref z-score
MAX_CALLS = 10       # worst-of over the first N future calls only

# --- hybrid fair-value model ---
POOL_WEEKS = 8        # rolling pooling window
MIN_POOL_OBS = 40     # min pool obs to fit the cross-section
MIN_FAMILY_DATE = 6   # min family bonds at a date to score
MAD_FLOOR = 1.0       # floor for the MAD z-score (avoid /0)
CV_SPLITS = 4         # GroupKFold folds for estimator choice
HUBER_EPS = 1.35
NUM_BASE = ["log_dur", "months_to_call", "px_minus_callpx", "cpn_freq_f"]

# --- track-2 cache sub-folder ---
CACHE_TS_DIR = paths.CACHE_DIR / "track2"
CACHE_TS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TS_LONG = CACHE_TS_DIR / "ts_long_history.pkl"
CACHE_CURVE = CACHE_TS_DIR / "eur_curve.pkl"
CACHE_CALLSCH = CACHE_TS_DIR / "call_schedule.pkl"
