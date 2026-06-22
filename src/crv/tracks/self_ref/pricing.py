"""Structural spread for Track-2 bonds + self-referential rich/cheap z-score.

FRN bonds take the Bloomberg discount margin (falling back to z-spread); perps
take the spread-to-worst over their future call schedule, computing each
yield-to-call via a closed-form annuity PV solved with Brent's method and
comparing to the interpolated EUR swap rate at the call horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from crv.tracks.self_ref import config as ts_cfg
from crv.utils.logging import get_logger

log = get_logger(__name__)


def is_perp(row) -> bool:
    """A bond is a perp if its maturity is missing or literally 'PERP'."""
    m = pd.to_datetime(row.get("maturity"), errors="coerce")
    txt = str(row.get("maturity", "")).upper()
    return pd.isna(m) or "PERP" in txt


def split_families(panel_ts: pd.DataFrame) -> pd.DataFrame:
    """Tag each bond ``is_perp`` (PERP/AT1 family) vs FRN."""
    df = panel_ts.copy()
    df["is_perp"] = df.apply(is_perp, axis=1)
    log.info("FRN family: %d | PERP/AT1 family: %d",
             df.loc[~df["is_perp"], "isin"].nunique(),
             df.loc[df["is_perp"], "isin"].nunique())
    return df


def ytc(price: float, coupon_pct: float, freq: float,
        years_to_call: float, call_px: float) -> float:
    """Yield-to-call via closed-form annuity PV + Brent root-find."""
    if price <= 0 or years_to_call <= 0 or freq <= 0:
        return np.nan
    n = max(1, int(round(years_to_call * freq)))
    cpn = coupon_pct / freq

    def pv(y):
        r = y / 100.0 / freq
        if abs(r) < 1e-12:
            disc = cpn * n + call_px
        else:
            af = (1 - (1 + r) ** (-n)) / r
            disc = cpn * af + call_px / (1 + r) ** n
        return disc - price

    try:
        return brentq(pv, -50, 200, maxiter=80, xtol=1e-4)
    except Exception:  # noqa: BLE001 - non-convergence -> NaN
        return np.nan


def build_curve_interps(eur_curve: pd.DataFrame) -> dict:
    """Pre-build one ``(tenors, rates)`` pair per date, reused across bonds."""
    interps: dict = {}
    cur = eur_curve.copy()
    cur["date"] = pd.to_datetime(cur["date"]).dt.strftime("%Y-%m-%d")
    cur["rate"] = pd.to_numeric(cur["rate"], errors="coerce")
    for ds, g in cur.groupby("date"):
        gg = g.dropna(subset=["rate"]).sort_values("tenor")
        if gg.empty:
            continue
        interps[ds] = (gg["tenor"].to_numpy(float), gg["rate"].to_numpy(float))
    return interps


def _interp(interps: dict, ds: str, tau: float) -> float:
    pair = interps.get(ds)
    if pair is None or pd.isna(tau):
        return np.nan
    xs, ys = pair
    return float(np.interp(tau, xs, ys))


def price_perp(perp: pd.DataFrame, calls_by_isin: dict, interps: dict) -> np.ndarray:
    """Spread-to-worst per perp row: min over future calls of (YTC - curve rate)."""
    out = np.full(len(perp), np.nan)
    cols = {c: i for i, c in enumerate(perp.columns)}
    for ridx, row in enumerate(perp.itertuples(index=False, name=None)):
        isin = row[cols["isin"]]
        ds = row[cols["date"]]
        price = row[cols["px"]]
        cpn = row[cols["coupon"]]
        freq = row[cols["cpn_freq"]]
        callpx = row[cols["next_call_px"]]
        nxt = row[cols["next_call_dt"]]
        if pd.isna(price) or pd.isna(cpn):
            continue
        if pd.isna(callpx):
            callpx = 100.0
        if pd.isna(freq) or freq <= 0:
            freq = 1.0
        dt0 = pd.to_datetime(ds)
        calls = calls_by_isin.get(isin)
        if calls is None or len(calls) == 0:
            calls = (np.array([nxt], dtype="datetime64[ns]")
                     if pd.notna(nxt) else np.array([], dtype="datetime64[ns]"))
        fut = calls[calls > np.datetime64(dt0)]
        if fut.size == 0:
            continue
        fut = np.sort(fut)[:ts_cfg.MAX_CALLS]
        worst = np.nan
        for cd in fut:
            tau = (pd.Timestamp(cd) - dt0).days / 365.25
            if tau <= 0:
                continue
            y = ytc(price, cpn, freq, tau, callpx)
            if np.isnan(y):
                continue
            base = _interp(interps, ds, tau)
            if np.isnan(base):
                continue
            s = (y - base) * 100.0
            worst = s if np.isnan(worst) else min(worst, s)
        out[ridx] = worst
    return out


def add_rich_cheap(df: pd.DataFrame, col: str = "struct_spread") -> pd.DataFrame:
    """Self-referential z-score of ``col`` within each bond's own history."""
    df = df.copy()
    g = df.groupby("isin")[col]
    df["rc_n"] = g.transform("count")
    df["rc_med"] = g.transform("median")
    df["rc_std"] = g.transform("std")
    df["rc_zscore"] = np.where(
        (df["rc_n"] >= ts_cfg.RC_MIN_OBS) & (df["rc_std"] > 0),
        (df[col] - df["rc_med"]) / df["rc_std"], np.nan)
    df["rich_cheap"] = np.where(
        df["rc_zscore"].isna(), "n/a",
        np.where(df["rc_zscore"] < 0, "rich", "cheap"))
    return df


def price(panel_ts: pd.DataFrame, frn_dm: pd.DataFrame, eur_curve: pd.DataFrame,
          call_sched: pd.DataFrame, perp_freq: pd.Series) -> pd.DataFrame:
    """Compute structural spreads for the whole Track-2 panel."""
    base = panel_ts.copy()
    base["date"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    if "px" not in base.columns and {"px_bid", "px_ask"}.issubset(base.columns):
        base["px"] = base[["px_bid", "px_ask"]].mean(axis=1)

    # FRN: discount margin
    frn = base[~base["is_perp"]].copy()
    if len(frn) and len(frn_dm):
        dm = frn_dm.copy()
        dm["date"] = pd.to_datetime(dm["date"]).dt.strftime("%Y-%m-%d")
        frn = frn.merge(dm[["isin", "date", "discount_margin"]], on=["isin", "date"], how="left")
        frn["struct_spread"] = frn["discount_margin"]
        frn["spread_method"] = np.where(frn["struct_spread"].notna(), "DM", "fallback_zspread")
        frn.loc[frn["struct_spread"].isna(), "struct_spread"] = frn["z_spread"]
    elif len(frn):
        frn["struct_spread"] = frn["z_spread"]
        frn["spread_method"] = "fallback_zspread"

    # PERP: YTC worst-of
    perp = base[base["is_perp"]].copy()
    if len(perp):
        if "cpn_freq" not in perp.columns:
            perp["cpn_freq"] = np.nan
        perp["cpn_freq"] = perp["cpn_freq"].fillna(perp["isin"].map(perp_freq))
        calls_by_isin: dict = {}
        if len(call_sched):
            for i, g in call_sched.groupby("isin"):
                calls_by_isin[i] = np.sort(pd.to_datetime(g["call_dt"]).to_numpy())
        interps = build_curve_interps(eur_curve)
        for c in ("px", "coupon", "cpn_freq", "next_call_px", "next_call_dt", "date", "isin"):
            if c not in perp.columns:
                perp[c] = np.nan
        perp["struct_spread"] = price_perp(perp, calls_by_isin, interps)
        perp["spread_method"] = np.where(perp["struct_spread"].notna(), "YTC_worst", "fallback_zspread")
        perp.loc[perp["struct_spread"].isna(), "struct_spread"] = perp["z_spread"]

    out = pd.concat([frn, perp], ignore_index=True)
    out = add_rich_cheap(out, "struct_spread")
    log.info("Track 2 priced | %d bonds, %d rows", out["isin"].nunique(), len(out))
    return out
