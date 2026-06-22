"""Shared fixtures: a synthetic routed Track-1 panel that needs no Bloomberg."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_xs_panel() -> pd.DataFrame:
    """A small but schema-faithful cross-section panel with a built-in rating effect."""
    rng = np.random.default_rng(0)
    n_bonds, n_weeks = 120, 8
    isins = [f"ISIN{i:04d}" for i in range(n_bonds)]
    dates = [(datetime(2025, 1, 3) + timedelta(weeks=w)).strftime("%Y-%m-%d") for w in range(n_weeks)]
    buckets = rng.choice(["HY_BB", "HY_B", "HY_CCC"], n_bonds, p=[0.5, 0.35, 0.15])
    sectors = rng.choice(["Retail", "Auto", "Telecom", "Energy", "OTHER"], n_bonds)

    rows = []
    for i, isin in enumerate(isins):
        notch = {"HY_BB": 12, "HY_B": 15, "HY_CCC": 18}[buckets[i]] + rng.standard_normal()
        dur = rng.uniform(2, 7)
        cpn = rng.uniform(3, 9)
        base_zs = np.exp(4.0 + 0.12 * notch + 0.05 * dur + rng.standard_normal() * 0.2)
        for d in dates:
            rows.append(dict(
                isin=isin, date=d, name=f"Bond {i}", rating_bucket=buckets[i],
                industry=sectors[i],
                rating_moody="Ba2" if buckets[i] == "HY_BB" else "B2",
                rating_sp="BB" if buckets[i] == "HY_BB" else "B", rating_fitch="",
                z_spread=base_zs * np.exp(rng.standard_normal() * 0.05),
                duration=dur, coupon=cpn, coupon_type="FIXED", amount=5e8,
                maturity="2030-01-01", issue_date="2021-01-01",
                next_call_dt="2027-01-01", next_call_px=102.0,
                px_bid=99, px_ask=101, px=100, issuer=f"Issuer{i}",
                issuer_id=f"C{i}", workout_dt=None,
            ))
    panel = pd.DataFrame(rows)
    panel["live"] = True
    date_ok = panel.groupby("date")["isin"].transform("nunique") >= 50
    panel["in_train"] = panel["live"] & date_ok
    return panel


@pytest.fixture
def synthetic_fundamentals() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 120
    return pd.DataFrame({
        "isin": [f"ISIN{i:04d}" for i in range(n)],
        "issuer_key": [f"Issuer{i}" for i in range(n)],
        "net_lev": rng.uniform(2, 6, n), "icr": rng.uniform(1, 5, n),
        "ebitda_margin": rng.uniform(0.1, 0.4, n), "capex_pct": rng.uniform(0.02, 0.1, n),
        "ebitda_growth": rng.uniform(-0.1, 0.2, n),
    })
