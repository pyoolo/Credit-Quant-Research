"""Split the corporate panel into the three disjoint tracks.

The routing logic is the single arbiter of which bond belongs to which track,
guaranteeing the three panels are mutually exclusive:

    broken      = z NaN or z < SPREAD_FLOOR                      -> DROPPED
    distressed  = (not broken) and z > DISTRESS_CUTOFF           -> Track 3 (ds)
    cross_sect  = clean, fixed-coupon, HY bucket                 -> Track 1 (xs)
    self_ref    = clean, not fixed-coupon (FRN / perp / etc.)    -> Track 2 (ts)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crv.config import settings
from crv.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RoutedPanels:
    """The three disjoint track panels produced by :func:`route`."""
    cross_section: pd.DataFrame   # Track 1 (xs)
    self_ref: pd.DataFrame        # Track 2 (ts)
    distressed: pd.DataFrame      # Track 3 (ds)


def mid_px(df: pd.DataFrame) -> pd.Series:
    """Mid price from bid/ask, NaN if either side is missing."""
    if "px_bid" in df.columns and "px_ask" in df.columns:
        return df[["px_bid", "px_ask"]].mean(axis=1)
    return pd.Series(np.nan, index=df.index)


def route(df: pd.DataFrame) -> RoutedPanels:
    """Route the raw panel into the three track panels."""
    df = df.copy()
    pre = df["isin"].nunique()
    df = df[~df["industry"].isin(settings.NON_CORP)].copy()
    log.info("drop non-corporate: %d issuers removed -> %d corporate",
             pre - df["isin"].nunique(), df["isin"].nunique())

    df["px"] = mid_px(df)
    df["coupon_type"] = df["coupon_type"].astype(str).str.upper().str.strip()

    z = df["z_spread"]
    broken = z.isna() | (z < settings.SPREAD_FLOOR)
    distressed = (~broken) & (z > settings.DISTRESS_CUTOFF)
    is_fixed = df["coupon_type"].isin({c.upper() for c in settings.TRACK1_CPN})
    is_hy_t1 = df["rating_bucket"].isin(settings.HY_BUCKETS_T1)

    t1 = (~broken) & (~distressed) & is_fixed & is_hy_t1
    t2 = (~broken) & (~distressed) & (~is_fixed)

    panel_xs = df[t1].copy()
    panel_ts = df[t2].copy()
    panel_ds = df[distressed].copy()

    # Track-1-only flags: usable row, and date with a wide-enough cross-section
    date_ok = panel_xs.groupby("date")["isin"].transform("nunique") >= settings.MIN_XS_PER_DATE
    panel_xs["live"] = panel_xs["z_spread"].notna() & panel_xs["duration"].notna()
    panel_xs["in_train"] = panel_xs["live"] & date_ok.reindex(panel_xs.index, fill_value=False)

    log.info("DROPPED (z<%d or NaN): %d rows", settings.SPREAD_FLOOR, int(broken.sum()))
    log.info("Track 1 cross-section: %d bonds | in_train %d rows",
             panel_xs["isin"].nunique(), int(panel_xs["in_train"].sum()))
    log.info("Track 2 self-ref:      %d bonds", panel_ts["isin"].nunique())
    log.info("Track 3 distressed:    %d bonds (z>%d)",
             panel_ds["isin"].nunique(), settings.DISTRESS_CUTOFF)
    return RoutedPanels(panel_xs, panel_ts, panel_ds)
