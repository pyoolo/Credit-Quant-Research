"""Persist Track-1 fitted artefacts for the pricer and the company-prefill form.

Two files are written:

* ``model_t1.pkl`` — the fitted Lasso, scaler, residual quantiles (per rating
  bucket, for the pricer's confidence band), peer-spread snapshot, fundamental
  standardisation params and the rebuild map.
* ``company_snapshot.csv`` — the latest raw-ratio + metadata observation per
  ISIN, so the app can pre-fill the pricer form from a company selection.
"""

from __future__ import annotations

import pandas as pd

from crv.config import paths
from crv.tracks.cross_section.features import FundScoreParams
from crv.tracks.cross_section.model import FittedModel
from crv.utils.io import save_pickle
from crv.utils.logging import get_logger

log = get_logger(__name__)


def save_model(fitted: FittedModel, train: pd.DataFrame, live: pd.DataFrame,
               fund_params: FundScoreParams, path=paths.MODEL_T1) -> dict:
    """Build and persist the fitted-model artefact consumed by the pricer."""
    # per-rating-bucket residual quantiles for the pricer confidence band
    X = _design(train, fitted)
    train = train.copy()
    train["log_resid"] = train["log_zs"] - fitted.model.predict(X)
    resid_q: dict = {}
    for b, g in train.groupby("rating_bucket"):
        if len(g) >= 20:
            resid_q[b] = {"q10": float(g["log_resid"].quantile(0.1)),
                          "q90": float(g["log_resid"].quantile(0.9)),
                          "std": float(g["log_resid"].std()), "n": int(len(g))}

    # peer Z-spread snapshot per bucket at the latest date (pricer histogram)
    last = live[live["date"] == live["date"].max()]
    peer_snap = {b: s.dropna().tolist()
                 for b, s in last.groupby("rating_bucket")["z_spread"]}

    artefact = {
        "model": fitted.model, "scaler": fitted.scaler,
        "xcols": list(fitted.xcols), "feats": fitted.feats,
        "resid_q": resid_q,
        "latest_wk": live["date"].max().strftime("%Y-%m-%d"),
        "peer_snap": peer_snap,
        "feat_median": train[fitted.feats].median().to_dict(),
        "fund_std_params": fund_params.std_params,
        "fund_score_map": fund_params.score_map,
        "bucket_notch": train.groupby("rating_bucket")["rating_notch"].median().to_dict(),
        "sectors": sorted({c.replace("sector_", "")
                           for c in fitted.xcols if c.startswith("sector_")}),
        "alpha": fitted.alpha, "r2_oof": fitted.r2_oof, "r2_holdout": fitted.r2_holdout,
    }
    save_pickle(artefact, path)
    log.info("model_t1.pkl saved (%d buckets, %d peer groups)", len(resid_q), len(peer_snap))
    return artefact


def save_company_snapshot(panel_xs: pd.DataFrame, path=paths.COMPANY_SNAP) -> pd.DataFrame:
    """Latest raw-ratio + metadata observation per ISIN, for form pre-fill."""
    cols = ["isin", "name", "rating_bucket", "sector", "duration", "coupon",
            "yrs_to_call", "is_callable", "age", "net_lev", "icr",
            "ebitda_margin", "ebitda_growth", "capex_pct"]
    have = [c for c in cols if c in panel_xs.columns]
    snap = (panel_xs.sort_values("date").groupby("isin").tail(1)[have]
            .reset_index(drop=True))
    snap.to_csv(path, index=False)
    log.info("company_snapshot.csv saved (%d bonds, %d cols)", len(snap), len(have))
    return snap


def _design(train: pd.DataFrame, fitted: FittedModel) -> pd.DataFrame:
    """Rebuild the training design matrix in the fitted column order."""
    Xnum = train[fitted.feats].copy().fillna(train[fitted.feats].median())
    Xnum_s = pd.DataFrame(fitted.scaler.transform(Xnum), columns=fitted.feats, index=Xnum.index)
    Xcat = pd.get_dummies(train[["sector", "wk"]], drop_first=True)
    X = pd.concat([Xnum_s, Xcat], axis=1)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    return X.reindex(columns=fitted.xcols, fill_value=0)
