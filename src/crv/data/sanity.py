"""Read-only data-quality diagnostics on the raw panel.

Emits a structured report (returned as a dict and logged) rather than bare
prints, so it can be asserted on in tests or surfaced in a dashboard.
"""

from __future__ import annotations

import pandas as pd

from crv.utils.logging import get_logger

log = get_logger(__name__)

_COVERAGE_COLS = [
    "z_spread", "duration", "coupon", "rating_bucket", "industry", "payment_rank",
    "next_call_dt", "amount", "issue_date", "px_bid", "px_ask",
]


def sanity_check(df: pd.DataFrame) -> dict:
    """Compute and log diagnostics; return them as a dict."""
    z = df["z_spread"]
    report: dict[str, object] = {
        "n_rows": len(df),
        "n_bonds": int(df["isin"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "coverage": {c: round(100 * df[c].notna().mean(), 1)
                     for c in _COVERAGE_COLS if c in df.columns},
        "z_spread": {
            "min": float(z.min()), "p1": float(z.quantile(0.01)),
            "median": float(z.median()), "p99": float(z.quantile(0.99)),
            "max": float(z.max()),
            "negative": int((z < 0).sum()),
            "near_zero": int((z < 10).sum()),
            "huge": int((z > 5000).sum()),
        },
        "duplicates": int(df.duplicated(["isin", "date"]).sum()),
    }
    wk = df.groupby("isin")["date"].nunique()
    report["short_history"] = int((wk < 13).sum())

    if {"px_bid", "px_ask"}.issubset(df.columns):
        spr = df["px_ask"] - df["px_bid"]
        report["bid_ask"] = {"negative": int((spr < 0).sum()), "wide_gt_5pt": int((spr > 5).sum())}

    log.info("sanity: %d rows | %d bonds | %d dates | %d duplicates",
             report["n_rows"], report["n_bonds"], report["n_dates"], report["duplicates"])
    log.info("z-spread bps: min %.0f median %.0f max %.0f | %d negative",
             report["z_spread"]["min"], report["z_spread"]["median"],
             report["z_spread"]["max"], report["z_spread"]["negative"])
    return report
