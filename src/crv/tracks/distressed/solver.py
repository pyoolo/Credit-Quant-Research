"""Constant-hazard credit solver: PV, implied default intensity, PD frontier.

A defaultable bond's price is the present value of coupons paid while alive plus
recovery on default plus principal if it survives, under a constant hazard rate
``lambda``. Given a price and a recovery assumption ``R`` we solve for the
``lambda`` that reprices the bond (Brent), then read off the cumulative default
probability. This module is shared by L1 and the saved full frontier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from crv.tracks.distressed import config as ds_cfg


def px_given_lambda(lam: float, coupon: float, freq: float, years: float,
                    R: float, r: float, face: float = 100.0) -> float:
    """Present value of a defaultable bond under constant hazard ``lam``."""
    n = max(1, int(round(years * freq)))
    cpn = coupon / freq
    pv = 0.0
    for k in range(1, n + 1):
        t = k / freq
        S = np.exp(-lam * t)
        S0 = np.exp(-lam * (t - 1 / freq))
        DF = np.exp(-r * t)
        pv += cpn * S * DF + R * face * (S0 - S) * DF
    pv += face * np.exp(-(lam + r) * (n / freq))
    return pv


def implied_lambda(price: float, coupon: float, freq: float, years: float,
                   R: float, r: float = ds_cfg.RISK_FREE) -> float:
    """Solve for the constant hazard that reprices the bond (NaN if no root)."""
    def f(lam):
        return px_given_lambda(lam, coupon, freq, years, R, r) - price
    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0, maxiter=100, xtol=1e-6)
    except Exception:  # noqa: BLE001
        return np.nan


def max_implied_recovery(price: float, coupon: float, freq: float, years: float,
                         r: float = ds_cfg.RISK_FREE) -> float:
    """Highest recovery under which the price becomes explainable (lambda solvable)."""
    best = np.nan
    for R in np.linspace(0.01, 0.95, 40):
        if not np.isnan(implied_lambda(price, coupon, freq, years, R, r)):
            best = R
    return best


def frontier(px: float, coupon: float, years: float, freq: float = 2.0) -> list[list[float]]:
    """Full PD-recovery frontier as ``[[R, cum_pd], ...]`` over the recovery grid."""
    if any(pd.isna(x) for x in (px, coupon, years)) or px <= 0 or years <= 0:
        return []
    out = []
    for R in ds_cfg.R_GRID:
        lam = implied_lambda(px, coupon, freq, years, R)
        if not np.isnan(lam):
            out.append([round(float(R), 3), round(float(1 - np.exp(-lam * years)), 4)])
    return out
