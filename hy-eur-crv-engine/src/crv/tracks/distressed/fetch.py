"""Track-3 Bloomberg fetch: distressed static descriptors + long price history.

Distressed names need their value column read carefully (Bloomberg returns
date/currency/multiplier alongside the value), so :func:`_parse_value` extracts
the right column per field.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from crv.tracks.distressed import config as ds_cfg
from crv.utils import bqlio, dates
from crv.utils.io import load_pickle, save_pickle
from crv.utils.logging import get_logger

log = get_logger(__name__)

_ANCILLARY = {"ID", "DATE", "CURRENCY", "CURRENCY_OF_ISSUE", "MULTIPLIER",
              "CPN_TYP", "ISIN_FULL", "INDEX"}


def _parse_value(res, value_hint: str) -> pd.DataFrame:
    """Extract ID + the correct VALUE column (not DATE/CURRENCY/MULTIPLIER/...)."""
    raw = res[0].df().reset_index()
    cand = [c for c in raw.columns if value_hint.upper() in c.upper()]
    if not cand:
        cand = [c for c in raw.columns if c.upper() not in _ANCILLARY]
    col = cand[-1] if cand else raw.columns[-1]
    idc = "ID" if "ID" in raw.columns else raw.columns[0]
    return pd.DataFrame({
        "isin": raw[idc].astype(str).str.replace(" Corp", "").str.strip(),
        value_hint: raw[col].values,
    })


def _grid() -> list[str]:
    return dates.weekly_grid(ds_cfg.DS_PANEL_WEEKS)


def _save(obj, path, **m):
    save_pickle({"data": obj, "saved_at": datetime.now().isoformat(), **m}, path)


def _load(path):
    d = load_pickle(path)
    if d is None:
        return None
    log.info("cache %s: %s", path.name, d["saved_at"][:16])
    return d["data"]


def fetch_static(ids) -> pd.DataFrame:
    """Descriptive fields per distressed bond, read from the correct value column."""
    import bql
    bq = bqlio.service()
    univ = bqlio.as_corp_universe(ids)
    specs = [
        ("coupon", bq.data.cpn), ("maturity", bq.data.maturity),
        ("workout_dt", bq.data.workout_date_mid_to_worst),
        ("amount", bq.data.amt_outstanding), ("rank_txt", bq.data.payment_rank),
        ("pay_rank", bq.data.mifid_seniority_of_bond), ("cpn_typ", bq.data.cpn_typ),
        ("issuer_id", bq.data.id_bb_company), ("issuer", bq.data.issuer),
    ]
    parts = []
    for name, fac in specs:
        try:
            df = _parse_value(bq.execute(bql.Request(univ, [fac()])), name)
            parts.append(df.set_index("isin")[name])
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", name, str(exc)[:25])
    st = pd.concat(parts, axis=1).reset_index()
    for c in ("coupon", "amount"):
        if c in st.columns:
            st[c] = pd.to_numeric(st[c], errors="coerce")
    for c in ("maturity", "workout_dt"):
        if c in st.columns:
            st[c] = pd.to_datetime(st[c], errors="coerce")
    log.info("static: %d bonds | coupon coverage %.0f%%",
             st["isin"].nunique(), 100 * st["coupon"].notna().mean())
    return st


def _retry_px(ids, date_str: str) -> pd.DataFrame:
    """Fetch price + spread for one date, with retry on transient errors."""
    import bql
    bq = bqlio.service()
    univ = bqlio.as_corp_universe(ids)
    from crv.config import settings
    for att in range(1, settings.N_RETRY + 1):
        try:
            req = bql.Request(univ, [bq.data.px_last(), bq.data.spread()])
            q = req.to_string().replace(
                "get(", f"with(dates=range('{date_str}','{date_str}')) get(", 1)
            res = bq.execute(q)
            px = _parse_value(res, "PX_LAST").rename(columns={"PX_LAST": "px"})
            try:
                sp = _parse_value([res[1]], "SPREAD").rename(columns={"SPREAD": "z_spread"})
                out = px.merge(sp, on="isin", how="left")
            except Exception:  # noqa: BLE001
                out = px
                out["z_spread"] = np.nan
            out["date"] = date_str
            return out
        except Exception as exc:  # noqa: BLE001
            if att < settings.N_RETRY and bqlio.is_transient(exc):
                time.sleep(settings.BASE_WAIT * att)
                continue
            raise


def fetch_history(ids, panel_ds: pd.DataFrame) -> pd.DataFrame:
    """Long weekly price/spread history for the distressed names (incremental)."""
    have = _load(ds_cfg.CACHE_DS_LONG)
    if have is None:
        cols = [c for c in ("isin", "date", "px", "z_spread") if c in panel_ds.columns]
        have = panel_ds[cols].copy() if cols else pd.DataFrame(columns=["isin", "date", "px", "z_spread"])
    if len(have):
        have["date"] = pd.to_datetime(have["date"]).dt.strftime("%Y-%m-%d")
    have_weeks = set(pd.to_datetime(have["date"]).dt.strftime("%G-%V")) if len(have) else set()
    missing = sorted([d for d in _grid() if dates.iso_week(d) not in have_weeks], reverse=True)
    log.info("distressed history: %d weeks to fetch", len(missing))

    new = []
    for ds in missing:
        try:
            d = _retry_px(ids, ds)
            d["px"] = pd.to_numeric(d["px"], errors="coerce")
            d["z_spread"] = pd.to_numeric(d["z_spread"], errors="coerce")
            new.append(d[["isin", "date", "px", "z_spread"]])
        except Exception as exc:  # noqa: BLE001
            log.warning("%s skip (%s)", ds, str(exc)[:24])
    long = pd.concat([have] + new, ignore_index=True).drop_duplicates(["isin", "date"], keep="last")
    long = long[long["isin"].isin(ids)]
    _save(long, ds_cfg.CACHE_DS_LONG, weeks=ds_cfg.DS_PANEL_WEEKS)
    log.info("history: %d bonds | %d rows", long["isin"].nunique(), len(long))
    return long


def fetch_all(panel_ds: pd.DataFrame):
    """Fetch static + long history and join. Returns ``(panel_ds_long, ds_static)``."""
    ids = panel_ds["isin"].dropna().unique()
    log.info("Track-3 universe: %d distressed bonds", len(ids))
    ds_static = fetch_static(ids)
    panel_ds_long = fetch_history(ids, panel_ds).merge(ds_static, on="isin", how="left")
    return panel_ds_long, ds_static
