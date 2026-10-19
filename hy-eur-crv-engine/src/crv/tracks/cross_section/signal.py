"""Track-1 relative-value signal: fair-spread scoring -> rv_z -> CHEAP/FAIR/RICH.

The fitted model prices every live bond's fair Z-spread. The residual
(observed minus fair) is z-scored within (date x rating bucket) — which removes
the per-bucket bias and asks "is this bond rich or cheap *relative to its rating
peers*". A 4-week trailing mean of that z-score (``rv_z_4w``) drives the
CHEAP/FAIR/RICH label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import settings
from crv.tracks.cross_section.model import FittedModel
from crv.utils.logging import get_logger

log = get_logger(__name__)


def _score_cross_section(live: pd.DataFrame, fitted: FittedModel) -> pd.DataFrame:
    """Apply the fitted model to the live panel -> fair_zs, resid_bps, rv_z."""
    live = live.copy()
    live["wk"] = live["date"].astype(str)
    sec_n = live.groupby("industry")["isin"].transform("nunique")
    live["sector"] = np.where(sec_n >= settings.MIN_SEC, live["industry"], "OTHER")

    Xnum = live[fitted.feats].copy().fillna(live[fitted.feats].median())
    Xnum_s = pd.DataFrame(fitted.scaler.transform(Xnum), columns=fitted.feats, index=Xnum.index)
    Xcat = pd.get_dummies(live[["sector", "wk"]], drop_first=True)
    X = pd.concat([Xnum_s, Xcat], axis=1)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    X = X.reindex(columns=fitted.xcols, fill_value=0)

    live["fair_zs"] = np.exp(fitted.model.predict(X))
    live["resid_bps"] = live["z_spread"] - live["fair_zs"]
    live["rv_z"] = live.groupby(["date", "rating_bucket"])["resid_bps"].transform(
        lambda s: (s - s.mean()) / s.std() if len(s) >= 5 and s.std() else np.nan)
    return live


def _latest_snapshot(df: pd.DataFrame, col: str = "rv_z") -> pd.DataFrame:
    """Latest-date snapshot with a 4-week trailing mean of the signal column."""
    last = df["date"].max()
    rec = df[df["date"] >= last - pd.Timedelta(weeks=4)]
    agg = rec.groupby("isin")[col].mean().rename(f"{col}_4w").reset_index()
    return df[df["date"] == last].merge(agg, on="isin", how="left")


def score(panel_xs: pd.DataFrame, fitted: FittedModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the panel. Returns ``(live_scored_history, latest_snapshot_with_signal)``."""
    panel_xs = panel_xs.copy()
    panel_xs["log_zs"] = np.log(panel_xs["z_spread"].clip(lower=1))
    live = panel_xs[panel_xs["live"]].dropna(
        subset=["log_zs", "duration", "rating_notch"]).copy()
    live = _score_cross_section(live, fitted)

    out = _latest_snapshot(live)
    out["signal"] = np.where(out["rv_z_4w"] > 1.0, "CHEAP",
                             np.where(out["rv_z_4w"] < -1.0, "RICH", "FAIR"))
    cols = ["isin", "name", "sector", "rating_bucket", "z_spread", "fair_zs",
            "resid_bps", "rv_z_4w", "signal", "fund_quality", "fund_risk",
            "net_lev", "icr", "coupon"]
    out = out[[c for c in cols if c in out.columns]].sort_values("rv_z_4w", ascending=False)
    log.info("scored %d bonds | %d CHEAP, %d RICH",
             len(out), int((out["signal"] == "CHEAP").sum()), int((out["signal"] == "RICH").sum()))
    return live, out
