"""Persist Track-3 outputs: rv_track3.csv, pricer params, price trajectory."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from crv.config import paths
from crv.tracks.distressed import config as ds_cfg
from crv.tracks.distressed.solver import frontier
from crv.utils.logging import get_logger

log = get_logger(__name__)


def save(ds_signals: pd.DataFrame, ds_static: pd.DataFrame, panel_ds_long: pd.DataFrame) -> None:
    """Write the distressed RV table (with full frontier), params and trajectory."""
    out = ds_signals.copy()
    if "issuer" in out.columns:
        out["name"] = out["issuer"]
    if "rank_txt" in out.columns:
        out["rating_bucket"] = out["rank_txt"]
    out["industry"] = ""
    out["z_spread"] = np.nan
    out["fair_zs"] = np.nan
    out["resid_bps"] = np.nan
    if "ds_score" in out.columns:
        out["rv_z"] = out["ds_score"]

    st = ds_static.set_index("isin")

    def _fr(r):
        isin, px = r["isin"], r.get("px")
        cpn = st["coupon"].get(isin, np.nan) if "coupon" in st.columns else np.nan
        mat = st["workout_dt"].get(isin) if "workout_dt" in st.columns else None
        if mat is None or pd.isna(mat):
            mat = st["maturity"].get(isin) if "maturity" in st.columns else None
        yrs = (max(0.25, (pd.to_datetime(mat) - pd.Timestamp.today()).days / 365.25)
               if (mat is not None and pd.notna(mat)) else np.nan)
        return json.dumps(frontier(px, cpn, yrs))

    out["frontier_json"] = out.apply(_fr, axis=1)

    cols = ["isin", "name", "industry", "rating_bucket", "px", "regime", "pd_40",
            "max_impl_recovery", "cap_struct_flag", "ds_score", "ds_conf",
            "ds_score_lo", "ds_score_hi", "rv_z", "z_spread", "fair_zs",
            "resid_bps", "frontier_json"]
    out[[c for c in cols if c in out.columns]].to_csv(paths.RV_T3, index=False)

    params = {"risk_free": ds_cfg.RISK_FREE, "r_scenarios": ds_cfg.R_SCENARIOS,
              "rank_order": ds_cfg.RANK_ORDER}
    with open(paths.T3_PARAMS, "w") as f:
        json.dump(params, f)

    if {"isin", "date", "px"}.issubset(panel_ds_long.columns):
        panel_ds_long[["isin", "date", "px"]].dropna(subset=["px"]).to_csv(paths.T3_TRAJ, index=False)

    nf = int((out["frontier_json"] != "[]").sum())
    log.info("rv_track3.csv saved (%d bonds, frontier on %d) + params + trajectory", len(out), nf)
