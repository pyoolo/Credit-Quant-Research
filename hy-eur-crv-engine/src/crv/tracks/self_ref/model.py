"""Track-2 hybrid fair-value model: rolling cross-section -> alpha -> RV score.

For each family (FRN / PERP) we pick the better of a robust Huber regression and
a gradient-boosting model by grouped CV MAE, then roll an 8-week pool: pricing
each date's cross-section against a temporally de-meaned fair value. The residual
(``alpha``) is turned into a robust MAD-based z-score (``rv_score``). Where the
cross-section can't score a bond, the self-referential z-score is the fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from crv.tracks.self_ref import config as ts_cfg
from crv.utils.logging import get_logger

log = get_logger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """log_dur + months_to_call + px_minus_callpx + cpn_freq + rating/industry dummies."""
    d = df.copy()
    d["log_dur"] = np.log(d["duration"].clip(lower=0.05))
    nc = pd.to_datetime(d.get("next_call_dt"), errors="coerce")
    d["months_to_call"] = ((nc - d["date"]).dt.days / 30.44).clip(lower=0)
    d["months_to_call"] = d["months_to_call"].fillna(d["months_to_call"].median())
    d["px_minus_callpx"] = (d.get("px") - d.get("next_call_px")).fillna(0.0)
    d["cpn_freq_f"] = pd.to_numeric(d.get("cpn_freq"), errors="coerce").fillna(1.0)
    rb = pd.get_dummies(d["rating_bucket"].astype(str), prefix="rt", drop_first=True)
    ind = pd.get_dummies(d["industry"].astype(str), prefix="ind", drop_first=True)
    num = d[["log_dur", "months_to_call", "px_minus_callpx", "cpn_freq_f"]]
    return pd.concat([num, rb, ind], axis=1)


def _choose_estimator(X, y, groups):
    """Return ``(name, MAE_huber, MAE_gbm)`` picking the lower CV MAE."""
    m = y.notna() & X.notna().all(axis=1)
    X, y, groups = X[m], y[m], groups[m]
    if len(y) < max(ts_cfg.MIN_POOL_OBS, ts_cfg.CV_SPLITS * 3) or groups.nunique() < ts_cfg.CV_SPLITS:
        return "huber", np.nan, np.nan
    X = X[X.columns[X.nunique() > 1]]
    base_now = [c for c in ts_cfg.NUM_BASE if c in X.columns]
    gkf = GroupKFold(n_splits=ts_cfg.CV_SPLITS)
    mae_h, mae_g = [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        try:
            Xh_tr, Xh_te = Xtr.copy(), Xte.copy()
            if base_now:
                sc = StandardScaler().fit(Xtr[base_now])
                Xh_tr[base_now] = sc.transform(Xtr[base_now])
                Xh_te[base_now] = sc.transform(Xte[base_now])
            hub = HuberRegressor(epsilon=ts_cfg.HUBER_EPS, alpha=0.0, max_iter=500).fit(Xh_tr, ytr)
            mae_h.append(mean_absolute_error(yte, hub.predict(Xh_te)))
        except Exception:  # noqa: BLE001
            mae_h.append(np.inf)
        try:
            gbm = HistGradientBoostingRegressor(
                max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0).fit(Xtr, ytr)
            mae_g.append(mean_absolute_error(yte, gbm.predict(Xte)))
        except Exception:  # noqa: BLE001
            mae_g.append(np.inf)
    MH, MG = np.nanmean(mae_h), np.nanmean(mae_g)
    return ("gbm" if MG < MH else "huber"), MH, MG


def _fit_predict(X_train, y_train, X_score, kind):
    nonconst = X_train.columns[X_train.nunique() > 1]
    Xtr, Xsc = X_train[nonconst], X_score[nonconst]
    if kind == "gbm":
        mdl = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0).fit(Xtr, y_train)
        return pd.Series(mdl.predict(Xsc), index=X_score.index)
    base = [c for c in ts_cfg.NUM_BASE if c in nonconst]
    sc = StandardScaler().fit(Xtr[base]) if base else None
    Xtr2, Xsc2 = Xtr.copy(), Xsc.copy()
    if sc is not None:
        Xtr2[base] = sc.transform(Xtr[base])
        Xsc2[base] = sc.transform(Xsc[base])
    mdl = HuberRegressor(epsilon=ts_cfg.HUBER_EPS, alpha=0.0, max_iter=500).fit(Xtr2, y_train)
    return pd.Series(mdl.predict(Xsc2), index=X_score.index)


def _fair_value_family(fam_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    """Per date: pool last POOL_WEEKS, temporal de-mean, fit fair value, residual -> alpha."""
    dates = np.sort(fam_df["date"].unique())
    group_col = "issuer_id" if "issuer_id" in fam_df else "isin"
    kind, mae_h, mae_g = _choose_estimator(
        build_features(fam_df), fam_df["struct_spread"], fam_df[group_col])
    log.info("[%s] estimator: %s (MAE Huber=%.0f GBM=%.0f)", family_name, kind.upper(), mae_h, mae_g)

    out = []
    for i, dt in enumerate(dates):
        win = dates[max(0, i - ts_cfg.POOL_WEEKS + 1): i + 1]
        pool = fam_df[fam_df["date"].isin(win)].copy()
        cur = fam_df[fam_df["date"] == dt].copy()
        if cur["isin"].nunique() < ts_cfg.MIN_FAMILY_DATE:
            continue
        pool["y_dm"] = pool["struct_spread"] - pool.groupby("date")["struct_spread"].transform("median")
        cur_med = pool.loc[pool["date"] == dt, "struct_spread"].median()
        Xp = build_features(pool)
        yp = pool["y_dm"]
        valid = yp.notna() & Xp.notna().all(axis=1)
        if valid.sum() < ts_cfg.MIN_POOL_OBS:
            continue
        Xc = build_features(cur).reindex(columns=Xp.columns, fill_value=0)
        yhat_dm = _fit_predict(Xp[valid], yp[valid], Xc, kind)
        z_fair = yhat_dm + cur_med
        alpha = cur["struct_spread"].values - z_fair.values
        tmp = cur[["isin", "date", "family", "struct_spread", "rating_bucket",
                   "industry", "duration"]].copy()
        tmp["z_fair"] = z_fair.values
        tmp["alpha"] = alpha
        med = np.nanmedian(alpha)
        mad = np.nanmedian(np.abs(alpha - med)) * 1.4826
        tmp["rv_score"] = (alpha - med) / max(mad, ts_cfg.MAD_FLOOR)
        tmp["method"] = f"cs_pool_{kind}"
        out.append(tmp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(panel_ts_priced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the hybrid model across families with self-ref fallback.

    Returns ``(rv_history, rv_last_date)``.
    """
    src = panel_ts_priced.copy()
    src["date"] = pd.to_datetime(src["date"])
    src["family"] = np.where(src["is_perp"], "PERP", "FRN")

    parts = []
    for fam in ("FRN", "PERP"):
        fam_df = src[src["family"] == fam].copy()
        if fam_df.empty:
            continue
        r = _fair_value_family(fam_df, fam)
        if not r.empty:
            parts.append(r)
    rv = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # self-ref fallback where the cross-section produced no alpha
    scored = set(zip(rv["isin"], rv["date"], strict=False)) if not rv.empty else set()
    fb = src[~src.apply(lambda r: (r["isin"], r["date"]) in scored, axis=1)].copy()
    fb = fb[fb["rc_zscore"].notna()]
    if len(fb):
        fbr = fb[["isin", "date", "family", "struct_spread", "rating_bucket",
                  "industry", "duration"]].copy()
        fbr["z_fair"] = fb["rc_med"].values
        fbr["alpha"] = fb["struct_spread"].values - fb["rc_med"].values
        fbr["rv_score"] = fb["rc_zscore"].values
        fbr["method"] = "self_ref_fallback"
        rv = pd.concat([rv, fbr], ignore_index=True)

    rv = rv.sort_values(["date", "family", "rv_score"]).reset_index(drop=True)
    last = rv[rv["date"] == rv["date"].max()].copy()
    log.info("Track 2 model | %d bonds | %d rows | %d dates",
             rv["isin"].nunique(), len(rv), rv["date"].nunique())
    return rv, last
