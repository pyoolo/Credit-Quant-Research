"""Track-2 Bloomberg fetch: long history, EUR curve, call schedule, FRN DM.

Each fetch is incremental and cached separately under ``data/cache/track2``.
This module imports :mod:`bql` lazily via :mod:`crv.utils.bqlio` so the package
imports cleanly without a terminal.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from crv.config import settings
from crv.tracks.self_ref import config as ts_cfg
from crv.utils import bqlio, dates
from crv.utils.io import load_pickle, save_pickle
from crv.utils.logging import get_logger

log = get_logger(__name__)


def _grid() -> list[str]:
    return dates.weekly_grid(ts_cfg.TS_PANEL_WEEKS)


def _save(obj, path, **meta):
    save_pickle({"data": obj, "saved_at": datetime.now().isoformat(), **meta}, path)


def _load(path):
    d = load_pickle(path)
    if d is None:
        return None
    log.info("cache %s: %s", path.name, d["saved_at"][:16])
    return d["data"]


# --------------------------------------------------------------------------- #
# A) long bond history                                                        #
# --------------------------------------------------------------------------- #
def _dyn_fields():
    import bql  # noqa: F401
    bq = bqlio.service()
    specs = [
        ("duration", bq.data.mty_dur_mid), ("z_spread", bq.data.spread),
        ("ytm", bq.data.yield_), ("px_bid", bq.data.px_bid), ("px_ask", bq.data.px_ask),
        ("next_call_dt", bq.data.nxt_call_dt), ("next_call_px", bq.data.nxt_call_px),
        ("oas", bq.data.risk_oas_mid), ("workout_dt", bq.data.workout_date_mid_to_worst),
    ]
    out = []
    for name, fac in specs:
        try:
            out.append(fac())
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", name, str(exc)[:18])
    return out


def _norm_dyn(d: pd.DataFrame) -> pd.DataFrame:
    d = d.rename(columns={"isin_full": "isin"}).rename(columns=settings.COL_MAP)
    d["isin"] = d["isin"].astype(str).str.replace(" Corp", "").str.strip()
    d = d[["isin", "date"] + [c for c in settings.DYNAMIC_COLS if c in d.columns]]
    for c in ("z_spread", "duration", "ytm", "px_bid", "px_ask", "next_call_px", "oas"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("next_call_dt", "workout_dt"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    return d


def _fetch_dyn_date(sub_universe, date_str):
    df = bqlio.fetch_dated(sub_universe, _dyn_fields(), date_str)
    df["date"] = date_str
    return df


def fetch_bond_history(ids, panel_ts: pd.DataFrame) -> pd.DataFrame:
    """Long weekly dynamic history, seeded from the short panel, incremental."""
    grid = _grid()
    have = _load(ts_cfg.CACHE_TS_LONG)
    if have is None:
        cols = ["isin", "date"] + [c for c in settings.DYNAMIC_COLS if c in panel_ts.columns]
        have = panel_ts[cols].copy()
    have["date"] = pd.to_datetime(have["date"]).dt.strftime("%Y-%m-%d")
    have_weeks = set(pd.to_datetime(have["date"]).dt.strftime("%G-%V"))
    missing = sorted([d for d in grid if dates.iso_week(d) not in have_weeks], reverse=True)
    log.info("long history: %d weeks to fetch", len(missing))

    new = []
    for ds in missing:
        try:
            d = bqlio.fetch_chunked(_fetch_dyn_date, ids, ds, currency="TSL")
            if d is None or not len(d):
                raise ValueError("empty")
            new.append(_norm_dyn(d))
            log.info("%s: %d bonds", ds, new[-1]["isin"].nunique())
        except Exception as exc:  # noqa: BLE001
            log.warning("%s skip (%s)", ds, str(exc)[:28])

    have = have[have["isin"].isin(ids)]
    long = pd.concat([have] + new, ignore_index=True).drop_duplicates(["isin", "date"], keep="last")
    _save(long, ts_cfg.CACHE_TS_LONG, weeks=ts_cfg.TS_PANEL_WEEKS)
    return long


# --------------------------------------------------------------------------- #
# B) EUR swap curve                                                           #
# --------------------------------------------------------------------------- #
def fetch_eur_curve() -> pd.DataFrame:
    bq = bqlio.service()
    grid = _grid()
    tickers = [f"EUSA{t} Curncy" for t in ts_cfg.EUR_TENORS]
    have = _load(ts_cfg.CACHE_CURVE)
    if have is None:
        have = pd.DataFrame(columns=["date", "tenor", "rate"])
    have_weeks = (set(pd.to_datetime(have["date"]).dt.strftime("%G-%V")) if len(have) else set())
    missing = sorted([d for d in grid if dates.iso_week(d) not in have_weeks], reverse=True)

    rows = []
    for ds in missing:
        try:
            d = bqlio.fetch_dated(tickers, [bq.data.px_last()], ds)
            d = d.rename(columns={"isin_full": "ticker", "PX_LAST": "rate"})
            d["tenor"] = d["ticker"].str.extract(r"EUSA(\d+)").astype(float)
            d["date"] = ds
            rows.append(d[["date", "tenor", "rate"]])
        except Exception as exc:  # noqa: BLE001
            log.warning("%s curve skip (%s)", ds, str(exc)[:22])
    curve = pd.concat([have] + rows, ignore_index=True) if rows else have
    curve["rate"] = pd.to_numeric(curve["rate"], errors="coerce")
    curve = curve.drop_duplicates(["date", "tenor"], keep="last")
    _save(curve, ts_cfg.CACHE_CURVE, tenors=ts_cfg.EUR_TENORS)
    log.info("curve: %d dates x %d tenors", curve["date"].nunique(), curve["tenor"].nunique())
    return curve


# --------------------------------------------------------------------------- #
# C) call schedule + coupon frequency (perps)                                 #
# --------------------------------------------------------------------------- #
def fetch_call_schedule(perp_ids):
    import bql
    bq = bqlio.service()
    if not len(perp_ids):
        return pd.DataFrame(columns=["isin", "call_dt"]), pd.Series(dtype=float)
    cached = _load(ts_cfg.CACHE_CALLSCH)
    if cached is not None:
        sched, freq = cached["sched"], cached["freq"]
        new_ids = set(perp_ids) - set(sched["isin"].unique())
        if not new_ids:
            return sched, freq
        perp_ids = list(new_ids)
    univ = bqlio.as_corp_universe(perp_ids)

    def _parse_schedule(res):
        parts = [item.df().reset_index() for item in res]
        raw = pd.concat(parts, axis=0, ignore_index=True)
        isin_col = next((c for c in raw.columns
                         if c.upper() in ("ISIN_FULL", "ID", "SECURITY", "ISIN")
                         or raw.columns.get_loc(c) == 0), raw.columns[0])
        date_col = next((c for c in raw.columns
                         if "DATE" in c.upper() or "CALL_SCHEDULE" in c.upper()), None)
        out = pd.DataFrame({
            "isin": raw[isin_col].astype(str).str.replace(" Corp", "").str.strip(),
            "call_dt": pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT,
        })
        return out.dropna(subset=["call_dt"]).drop_duplicates(["isin", "call_dt"])

    sched_new = _parse_schedule(bq.execute(bql.Request(univ, [bq.data.call_schedule()])))
    fq = bqlio.parse_response(bq.execute(bql.Request(univ, [bq.data.cpn_freq()])))
    fq = fq.rename(columns={"isin_full": "isin"})
    fq["isin"] = fq["isin"].astype(str).str.replace(" Corp", "").str.strip()
    fcol = next((c for c in fq.columns if "CPN_FREQ" in c.upper()), None)
    freq_new = (pd.Series(pd.to_numeric(fq[fcol], errors="coerce").values, index=fq["isin"].values)
                .dropna().groupby(level=0).first()) if fcol else pd.Series(dtype=float)

    if cached is not None:
        sched = pd.concat([cached["sched"], sched_new], ignore_index=True) \
            .drop_duplicates(["isin", "call_dt"])
        freq = pd.concat([cached["freq"], freq_new])
        freq = freq[~freq.index.duplicated(keep="first")]
    else:
        sched, freq = sched_new, freq_new
    _save({"sched": sched, "freq": freq}, ts_cfg.CACHE_CALLSCH)
    log.info("call schedule: %d perp | %d rows", sched["isin"].nunique(), len(sched))
    return sched, freq


# --------------------------------------------------------------------------- #
# D) FRN discount margin                                                      #
# --------------------------------------------------------------------------- #
def fetch_frn_dm(frn_ids) -> pd.DataFrame:
    bq = bqlio.service()
    if not len(frn_ids):
        return pd.DataFrame(columns=["isin", "date", "discount_margin", "cpn_freq"])
    frames = []
    for ds in _grid():
        try:
            d = bqlio.fetch_dated(bqlio.as_corp_universe(frn_ids),
                                  [bq.data.discount_margin(), bq.data.cpn_freq()], ds)
            d = d.rename(columns={"isin_full": "isin", "DISCOUNT_MARGIN": "discount_margin",
                                  "CPN_FREQ": "cpn_freq"})
            d["isin"] = d["isin"].astype(str).str.replace(" Corp", "").str.strip()
            d["date"] = ds
            frames.append(d[["isin", "date", "discount_margin", "cpn_freq"]])
        except Exception as exc:  # noqa: BLE001
            log.warning("%s DM skip (%s)", ds, str(exc)[:22])
    dm = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(dm):
        dm["discount_margin"] = pd.to_numeric(dm["discount_margin"], errors="coerce")
        dm["cpn_freq"] = pd.to_numeric(dm["cpn_freq"], errors="coerce")
    return dm


def fetch_all(panel_ts: pd.DataFrame):
    """Fetch every Track-2 input and reattach static columns to the long history."""
    ids = panel_ts["isin"].dropna().unique()
    perp_ids = panel_ts.loc[panel_ts["is_perp"], "isin"].dropna().unique()
    frn_ids = panel_ts.loc[~panel_ts["is_perp"], "isin"].dropna().unique()
    log.info("Track-2 universe: %d bonds (%d FRN, %d perp)", len(ids), len(frn_ids), len(perp_ids))

    long_bonds = fetch_bond_history(ids, panel_ts)
    eur_curve = fetch_eur_curve()
    call_sched, perp_freq = fetch_call_schedule(perp_ids)
    frn_dm = fetch_frn_dm(frn_ids)

    static_cols = [c for c in panel_ts.columns if c not in settings.DYNAMIC_COLS and c != "date"]
    static_snap = panel_ts[static_cols].drop_duplicates("isin")
    long_bonds = long_bonds[long_bonds["isin"].isin(ids)].merge(static_snap, on="isin", how="left")
    if "px" not in long_bonds.columns and {"px_bid", "px_ask"}.issubset(long_bonds.columns):
        long_bonds["px"] = long_bonds[["px_bid", "px_ask"]].mean(axis=1)
    if "cpn_freq" not in long_bonds.columns:
        long_bonds["cpn_freq"] = np.nan
    long_bonds["cpn_freq"] = long_bonds["cpn_freq"].fillna(long_bonds["isin"].map(perp_freq))

    log.info("Track-2 fetch complete | %d bonds", long_bonds["isin"].nunique())
    return long_bonds, eur_curve, call_sched, perp_freq, frn_dm
