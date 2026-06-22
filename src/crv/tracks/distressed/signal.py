"""Compose the four distressed levels into a ranked signal table."""

from __future__ import annotations

import pandas as pd

from crv.tracks.distressed import levels
from crv.utils.logging import get_logger

log = get_logger(__name__)


def run(panel_ds_long: pd.DataFrame, ds_static: pd.DataFrame):
    """Run L1-L4 on the latest snapshot. Returns ``(ds_signals, l3_traj, static_px)``."""
    last_date = pd.to_datetime(panel_ds_long["date"]).max()
    snap = panel_ds_long[pd.to_datetime(panel_ds_long["date"]) == last_date].copy()
    static = ds_static.merge(snap[["isin", "px"]], on="isin", how="left")

    l1 = levels.level1_pd_frontier(static)
    log.info("L1 PD-frontier: applicable on %d/%d bonds", int(l1["l1_applicable"].sum()), len(l1))
    l2 = levels.level2_capstructure(static, l1)
    nviol = int(l2["seniority_violation"].sum()) if len(l2) else 0
    log.info("L2 cap-structure: %d multi-bond issuers | %d seniority violations", len(l2), nviol)
    l3 = levels.level3_trajectory(panel_ds_long)
    log.info("L3 trajectory: %s", l3["regime"].value_counts().to_dict())
    l4 = levels.level4_composite(l1, l2, l3, static)

    ds_signals = l4.sort_values("ds_score", ascending=False).reset_index(drop=True)
    log.info("Track 3 signals ready (%d bonds, last date %s)", len(ds_signals), last_date.date())
    return ds_signals, l3, static
