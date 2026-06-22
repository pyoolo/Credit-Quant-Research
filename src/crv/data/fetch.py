"""Assemble the raw weekly panel, fetching only the missing weeks from Bloomberg.

The raw cache stores the un-routed panel plus its static snapshot. On each run
we compute the set of needed ISO weeks, subtract what the cache already holds,
and fetch only the gap (checkpointing after every date so progress survives a
crash).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from crv.config import paths, settings
from crv.data import ratings, universe
from crv.utils import bqlio, dates
from crv.utils.io import load_pickle, save_pickle
from crv.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Raw cache                                                                   #
# --------------------------------------------------------------------------- #
def save_raw_cache(df_panel: pd.DataFrame, static_snap: pd.DataFrame,
                   path=paths.CACHE_RAW, panel_weeks=settings.PANEL_WEEKS,
                   latest: str | None = None) -> None:
    save_pickle({
        "df_panel": df_panel, "static_snap": static_snap,
        "saved_at": datetime.now().isoformat(),
        "panel_weeks": panel_weeks, "latest": latest,
    }, path)


def load_raw_cache(path=paths.CACHE_RAW):
    """Return ``(df_panel, static_snap, saved_at)`` or ``(None, None, None)``."""
    d = load_pickle(path)
    if d is None:
        return None, None, None
    age_h = (datetime.now() - datetime.fromisoformat(d["saved_at"])).total_seconds() / 3600
    log.info("raw cache: %s (%.1fh) | %sw", d["saved_at"][:16], age_h, d.get("panel_weeks", "?"))
    return d["df_panel"], d["static_snap"], datetime.fromisoformat(d["saved_at"])


# --------------------------------------------------------------------------- #
# Panel assembly helpers                                                      #
# --------------------------------------------------------------------------- #
def _kept_from_cache(df_cached, needed_weeks, missing_weeks):
    if df_cached is None:
        return None
    keep = needed_weeks - missing_weeks
    wk = pd.to_datetime(df_cached["date"]).dt.strftime("%G-%V")
    cols = ["bb_id", "date"] + [c for c in settings.DYNAMIC_COLS if c in df_cached.columns]
    return df_cached.loc[wk.isin(keep), cols].copy()


def _merge_panel(df_dyn, static_snap):
    """Join per-week dynamic data onto per-bond static data; coerce dtypes."""
    df = df_dyn.merge(static_snap, on="bb_id", how="left")
    for c in ("z_spread", "duration", "coupon", "amount", "ytm",
              "px_bid", "px_ask", "next_call_px", "oas"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("next_call_dt", "workout_dt"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "coupon" in df.columns:
        df["coupon"] = df["coupon"].fillna(0)
    return df


def _rebuild_dynamic(df_cached, new_frames, needed_weeks, missing_weeks):
    frames = []
    kept = _kept_from_cache(df_cached, needed_weeks, missing_weeks)
    if kept is not None:
        frames.append(kept)
    if new_frames:
        frames.append(pd.concat(new_frames, ignore_index=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save_partial(df_cached, new_frames, needed_weeks, missing_weeks, static_snap, latest):
    """Checkpoint the cache after each new date so progress survives a crash."""
    try:
        accum = []
        kept = _kept_from_cache(df_cached, needed_weeks, missing_weeks)
        if kept is not None:
            accum.append(kept)
        accum.append(pd.concat(new_frames, ignore_index=True))
        df_dyn = pd.concat(accum, ignore_index=True)
        save_raw_cache(_merge_panel(df_dyn, static_snap), static_snap, latest=latest)
        log.info("checkpoint raw cache (%d dates)", df_dyn["date"].nunique())
    except Exception as exc:  # noqa: BLE001 - best-effort checkpoint
        log.warning("partial save failed: %s", str(exc)[:40])


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def fetch_panel(panel_weeks: int = settings.PANEL_WEEKS):
    """Assemble the raw weekly panel, fetching only missing weeks from Bloomberg.

    Returns ``(df_panel, static_snap)`` where ``df_panel`` carries one row per
    (bond, week) and ``static_snap`` one row per bond with the rating bucket.
    """
    date_strs = dates.weekly_grid(panel_weeks)
    latest = date_strs[0]
    log.info("HY %s fetch | %s -> %s", settings.CURRENCY, date_strs[-1], latest)

    df_cached, static_cached, saved_at = load_raw_cache()
    cached_weeks: set[str] = set()
    if df_cached is not None:
        df_cached["date"] = pd.to_datetime(df_cached["date"]).dt.strftime("%Y-%m-%d")
        cached_weeks = set(pd.to_datetime(df_cached["date"]).dt.strftime("%G-%V"))

    needed = {dates.iso_week(d) for d in date_strs}
    missing = needed - cached_weeks
    missing_dates = sorted([d for d in date_strs if dates.iso_week(d) in missing], reverse=True)
    log.info("cached weeks: %d | missing: %d", len(cached_weeks), len(missing))

    # Cache is "fresh" if saved today AND it already contains the latest week.
    fresh = (saved_at is not None and saved_at.date() == datetime.today().date()
             and dates.iso_week(latest) in cached_weeks)
    if fresh:
        log.info("raw cache up-to-date")
        ss = ratings.add_rating_bucket(static_cached)
        cols = ["bb_id", "date"] + [c for c in settings.DYNAMIC_COLS if c in df_cached.columns]
        return _merge_panel(df_cached[cols], ss), ss

    # --- Static fetch (one value per bond) ---
    log.info("static fetch")
    univ = universe.hy_eur_universe()
    static_raw = universe.fetch_static(univ).rename(columns=settings.COL_MAP)
    static_raw = static_raw.rename(columns={"isin_full": "bb_id"})
    static_raw["bb_id"] = static_raw["bb_id"].astype(str).str.replace(" Corp", "").str.strip()
    static_raw["isin"] = static_raw["isin"].astype(str).str.strip()
    keep = ["bb_id"] + settings.STATIC_COLS
    static_snap = static_raw[[c for c in keep if c in static_raw.columns]].copy()
    log.info("%d HY %s bonds", len(static_snap), settings.CURRENCY)

    # --- Dynamic fetch (one value per bond per missing week) ---
    peer = static_snap["bb_id"].dropna().unique()
    new_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    if missing_dates:
        log.info("dynamic fetch (%d dates)", len(missing_dates))
        fetcher = universe.dynamic_fetcher(univ)
        for ds in missing_dates:
            try:
                d = bqlio.fetch_chunked(fetcher, peer, ds)
                if d is None or not len(d):
                    raise ValueError("empty")
                d = d.rename(columns=settings.COL_MAP).rename(columns={"isin_full": "bb_id"})
                d["bb_id"] = d["bb_id"].astype(str).str.replace(" Corp", "").str.strip()
                d = d[["bb_id", "date"] + [c for c in settings.DYNAMIC_COLS if c in d.columns]]
                new_frames.append(d)
                log.info("%s: %d bonds", ds, d["bb_id"].nunique())
                if settings.SAVE_EVERY_DATE:
                    _save_partial(df_cached, new_frames, needed, missing, static_snap, latest)
            except Exception as exc:  # noqa: BLE001 - skip bad dates, keep going
                failed.append(ds)
                log.warning("%s skipped (%s)", ds, str(exc)[:40])
        if failed:
            log.warning("%d dates not completed", len(failed))
    else:
        log.info("no new dates")

    df_dyn = _rebuild_dynamic(df_cached, new_frames, needed, missing)
    df_panel = _merge_panel(df_dyn, static_snap)
    save_raw_cache(df_panel, static_snap, latest=latest)
    static_snap = ratings.add_rating_bucket(static_snap)
    df_panel = df_panel.merge(static_snap[["isin", "rating_bucket"]], on="isin", how="left")
    log.info("panel: %d dates | %d bonds",
             df_panel["date"].nunique(), df_panel["isin"].nunique())
    return df_panel, static_snap
