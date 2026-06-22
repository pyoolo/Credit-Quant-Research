"""Persist Track-2 outputs: rv_track2.csv + per-family peer distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import paths
from crv.utils.io import save_json
from crv.utils.logging import get_logger

log = get_logger(__name__)


def save(rv_ts: pd.DataFrame, panel_ts_priced: pd.DataFrame) -> None:
    """Write the latest-date RV snapshot and the per-family peer distributions."""
    rv = rv_ts.copy()
    rv["date"] = pd.to_datetime(rv["date"], errors="coerce")
    last = rv[rv["date"] == rv["date"].max()].copy()

    pt = panel_ts_priced.copy()
    pt["date"] = pd.to_datetime(pt["date"], errors="coerce")
    want = ["isin", "name", "industry", "rating_bucket", "duration", "family",
            "struct_spread", "spread_method", "rc_zscore"]
    have = [c for c in want if c in pt.columns]
    extra = pt.sort_values("date").groupby("isin", as_index=False).tail(1)[have]
    out = last.merge(extra, on="isin", how="left", suffixes=("", "_x"))

    # map to the dashboard's expected column names (with fallbacks)
    out["z_spread"] = out["struct_spread"] if "struct_spread" in out.columns else np.nan
    out["fair_zs"] = out["z_fair"] if "z_fair" in out.columns else np.nan
    out["resid_bps"] = out["alpha"] if "alpha" in out.columns else np.nan
    out["rv_z"] = out["rv_score"] if "rv_score" in out.columns else np.nan

    cols = ["isin", "name", "industry", "rating_bucket", "duration", "family",
            "z_spread", "fair_zs", "resid_bps", "rv_z", "spread_method"]
    out[[c for c in cols if c in out.columns]].to_csv(paths.RV_T2, index=False)
    log.info("rv_track2.csv saved (%d bonds)", len(out))

    # per-family peer distributions (struct_spread at the latest date)
    dist: dict = {}
    for fam, g in pt.groupby("family"):
        lg = g[g["date"] == g["date"].max()]
        vals = pd.to_numeric(lg["struct_spread"], errors="coerce").dropna()
        if len(vals):
            dist[str(fam)] = {"values": [float(v) for v in vals],
                              "p25": float(vals.quantile(0.25)),
                              "p50": float(vals.quantile(0.50)),
                              "p75": float(vals.quantile(0.75)),
                              "n": int(len(vals))}
    save_json(dist, paths.T2_PEERS)
    log.info("track2_peers.json saved (families: %s)", list(dist))
