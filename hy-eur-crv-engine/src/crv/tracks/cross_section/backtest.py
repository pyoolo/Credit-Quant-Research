"""Track-1 mini-backtest and signal-persistence diagnostics.

Two questions:

1. **Does the signal work?** Do CHEAP bonds (resid > 0) tighten versus their
   rating peers over the next ``HORIZON_W`` weeks? Measured by weekly Spearman
   rank IC (we expect *negative* IC: high rv_z -> negative forward spread change)
   and by the CHEAP-minus-RICH forward peer-adjusted spread change.

2. **Does it persist?** rv_z must mean-revert to be tradable. We measure
   within-bond autocorrelation of rv_z at 1/2/4/8 weeks (raw and bond-demeaned,
   which bracket the truth) and fit a decay to get an implied half-life.

Caveat: fundamentals are a today-snapshot reused at past dates, so there is mild
look-ahead on ``net_lev`` only (small model weight). Week fixed-effects are not
identifiable out-of-sample, so the per-week macro level is absorbed by demeaning
the residual within each date instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import settings
from crv.tracks.cross_section.model import FittedModel
from crv.utils.logging import get_logger

log = get_logger(__name__)


def _fair_zs_at(df_slice: pd.DataFrame, fitted: FittedModel, feat_med: pd.Series) -> pd.DataFrame:
    """Apply the frozen model to one weekly cross-section, week-FE excluded."""
    d = df_slice.copy()
    sec_n = d.groupby("industry")["isin"].transform("nunique")
    d["sector"] = np.where(sec_n >= settings.MIN_SEC, d["industry"], "OTHER")

    Xn = d[fitted.feats].copy().fillna(feat_med).fillna(0)
    Xn_s = pd.DataFrame(fitted.scaler.transform(Xn), columns=fitted.feats, index=Xn.index)
    Xc = pd.get_dummies(d[["sector"]], drop_first=True)
    X = pd.concat([Xn_s, Xc], axis=1)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]

    # drop week dummies (not identifiable on out-of-sample weeks)
    cols = [c for c in fitted.xcols if not c.startswith("wk_")]
    X = X.reindex(columns=cols, fill_value=0)
    wk_pos = [i for i, c in enumerate(fitted.xcols) if not c.startswith("wk_")]
    coef = fitted.model.coef_[wk_pos]

    d["fair_zs"] = np.exp(fitted.model.intercept_ + X.values @ coef)
    d["resid_bps"] = d["z_spread"] - d["fair_zs"]
    # absorb per-week macro level (the dropped week-FE would have done this)
    d["resid_bps"] = d["resid_bps"] - d.groupby("date")["resid_bps"].transform("mean")
    d["rv_z"] = d.groupby(["date", "rating_bucket"])["resid_bps"].transform(
        lambda s: (s - s.mean()) / s.std() if len(s) >= 5 and s.std() else np.nan)
    return d[["isin", "date", "rating_bucket", "z_spread", "rv_z"]]


def run(panel_xs: pd.DataFrame, fitted: FittedModel, train: pd.DataFrame) -> dict:
    """Run the forward backtest. Returns a report dict (IC, bucket means, edge)."""
    feat_med = train[fitted.feats].median()
    base = panel_xs[panel_xs["live"]].dropna(
        subset=["log_zs", "duration", "rating_notch"]).copy()
    base["date"] = pd.to_datetime(base["date"])
    dts = sorted(base["date"].unique())
    scored = pd.concat([_fair_zs_at(base[base["date"] == dt], fitted, feat_med) for dt in dts],
                       ignore_index=True)
    zs_wide = scored.pivot_table(index="isin", columns="date", values="z_spread")

    rows = []
    for i, dt in enumerate(dts):
        fut_i = i + settings.HORIZON_W
        if fut_i >= len(dts):
            break
        dt_f = dts[fut_i]
        snap = scored[scored["date"] == dt].dropna(subset=["rv_z"]).copy()
        snap["d_zs"] = snap["isin"].map(zs_wide[dt_f]) - snap["isin"].map(zs_wide[dt])
        snap = snap.dropna(subset=["d_zs"])
        snap["d_zs_adj"] = snap["d_zs"] - snap.groupby("rating_bucket")["d_zs"].transform("median")
        snap["bucket_sig"] = np.where(snap["rv_z"] > settings.Z_CUT, "CHEAP",
                                      np.where(snap["rv_z"] < -settings.Z_CUT, "RICH", "FAIR"))
        rows.append(snap[["date", "isin", "rv_z", "d_zs_adj", "bucket_sig"]])
    bt = pd.concat(rows, ignore_index=True)

    ic = bt.groupby("date").apply(
        lambda g: g["rv_z"].corr(g["d_zs_adj"], method="spearman") if len(g) >= 20 else np.nan
    ).dropna()
    grp = bt.groupby("bucket_sig")["d_zs_adj"].agg(["mean", "median", "count"]).round(1)
    edge = (grp.loc["CHEAP", "mean"] - grp.loc["RICH", "mean"]
            if {"CHEAP", "RICH"}.issubset(grp.index) else np.nan)

    report = {
        "horizon_w": settings.HORIZON_W,
        "usable_weeks": int(len(ic)),
        "ic_mean": float(ic.mean()), "ic_median": float(ic.median()),
        "ic_hit_rate": float((ic < 0).mean()),
        "bucket_means": grp.to_dict(),
        "cheap_minus_rich_bps": float(edge) if pd.notna(edge) else None,
        "half_life": half_life(bt),
    }
    log.info("backtest | %d weeks | IC mean %+.3f | CHEAP-RICH %+.1f bps",
             report["usable_weeks"], report["ic_mean"],
             report["cheap_minus_rich_bps"] or float("nan"))
    return report


def half_life(scored: pd.DataFrame) -> dict:
    """Within-bond rv_z autocorrelation at 1/2/4/8w and implied decay half-life."""
    d = scored.dropna(subset=["rv_z", "isin", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["isin", "date"])

    def _autocorr(lag_weeks: int):
        pairs = []
        for isin, g in d.groupby("isin"):
            s = g.sort_values("date")[["date", "rv_z"]].copy()
            s["date_l"] = s["date"] + pd.Timedelta(weeks=lag_weeks)
            m = s.merge(s[["date", "rv_z"]].rename(columns={"date": "date_l", "rv_z": "rv_z_next"}),
                        on="date_l", how="inner")
            if len(m):
                pairs.append(m[["rv_z", "rv_z_next"]].assign(isin=isin))
        if not pairs:
            return np.nan, np.nan, 0
        P = pd.concat(pairs, ignore_index=True)
        raw = P["rv_z"].corr(P["rv_z_next"])
        mu = P.groupby("isin")["rv_z"].transform("mean")
        mu2 = P.groupby("isin")["rv_z_next"].transform("mean")
        dem = (P["rv_z"] - mu).corr(P["rv_z_next"] - mu2)
        return raw, dem, len(P)

    lags = [1, 2, 4, 8]
    rows = [(lag, *_autocorr(lag)) for lag in lags]

    def _hl(idx: int):
        x = np.array([r[0] for r in rows], dtype=float)
        ac = np.array([r[idx] for r in rows], dtype=float)
        ok = ac > 0.02
        if ok.sum() < 2:
            return float("nan")
        slope = np.polyfit(x[ok], np.log(ac[ok]), 1)[0]
        return float(np.log(0.5) / slope) if slope < 0 else float("inf")

    return {
        "autocorr": {lag: {"raw": r[1], "demeaned": r[2], "pairs": r[3]}
                     for lag, r in zip(lags, rows, strict=False)},
        "half_life_raw_weeks": _hl(1),
        "half_life_demeaned_weeks": _hl(2),
    }
