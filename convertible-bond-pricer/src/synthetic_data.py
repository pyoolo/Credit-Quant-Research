"""
synthetic_data.py
-----------------
Generates synthetic market scenarios and CB specifications
for testing, benchmarking, and plotting.

No real market data is required — everything is reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict

from models import ConvertibleBond, MarketParams


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Pre-built representative bonds
# ---------------------------------------------------------------------------

def make_vanilla_cb() -> tuple[ConvertibleBond, MarketParams]:
    """
    Plain-vanilla CB: no call/put, semi-annual coupons.
    Good reference case — closed-form limits exist.
    """
    cb = ConvertibleBond(
        face_value=1000.0,
        maturity=3.0,
        coupon_rate=0.04,
        coupon_freq=2,
        conversion_ratio=20.0,
    )
    mkt = MarketParams(
        S0=50.0,
        r=0.03,
        sigma=0.30,
        credit_spread=0.05,
        recovery=0.40,
    )
    return cb, mkt


def make_callable_cb() -> tuple[ConvertibleBond, MarketParams]:
    """
    CB with issuer call at year 2 at 110% of face.
    Tests that TF correctly clips to max(conv, call_price).
    """
    cb = ConvertibleBond(
        face_value=1000.0,
        maturity=5.0,
        coupon_rate=0.03,
        coupon_freq=2,
        conversion_ratio=18.0,
        call_schedule=[(2.0, 1050.0), (3.0, 1030.0), (4.0, 1010.0)],
    )
    mkt = MarketParams(
        S0=55.0,
        r=0.03,
        sigma=0.25,
        credit_spread=0.04,
        recovery=0.40,
    )
    return cb, mkt


def make_putable_cb() -> tuple[ConvertibleBond, MarketParams]:
    """
    CB with holder put at year 2 at par.
    Tests put floor — price should never fall below put value.
    """
    cb = ConvertibleBond(
        face_value=1000.0,
        maturity=5.0,
        coupon_rate=0.025,
        coupon_freq=2,
        conversion_ratio=15.0,
        put_schedule=[(2.0, 1000.0), (3.0, 1000.0)],
    )
    mkt = MarketParams(
        S0=40.0,
        r=0.03,
        sigma=0.35,
        credit_spread=0.08,
        recovery=0.35,
    )
    return cb, mkt


def make_distressed_cb() -> tuple[ConvertibleBond, MarketParams]:
    """
    'Busted convert': wide spread, low stock price.
    CB should behave like a risky bond (delta ~ 0).
    """
    cb = ConvertibleBond(
        face_value=1000.0,
        maturity=3.0,
        coupon_rate=0.06,
        coupon_freq=2,
        conversion_ratio=20.0,
    )
    mkt = MarketParams(
        S0=25.0,       # far out-of-the-money
        r=0.03,
        sigma=0.55,    # high vol reflects distress
        credit_spread=0.15,
        recovery=0.30,
    )
    return cb, mkt


def make_equity_sensitive_cb() -> tuple[ConvertibleBond, MarketParams]:
    """
    'Equity-like' CB: in-the-money, tight spread.
    CB should behave like stock (delta ~ conversion_ratio).
    """
    cb = ConvertibleBond(
        face_value=1000.0,
        maturity=3.0,
        coupon_rate=0.02,
        coupon_freq=2,
        conversion_ratio=20.0,
    )
    mkt = MarketParams(
        S0=80.0,       # deep in-the-money (conv value = 1600)
        r=0.03,
        sigma=0.20,
        credit_spread=0.01,
        recovery=0.40,
    )
    return cb, mkt


# ---------------------------------------------------------------------------
# Scenario grid for model comparison
# ---------------------------------------------------------------------------

def make_scenario_grid(n_scenarios: int = 200) -> pd.DataFrame:
    """
    Generate a grid of random CB scenarios.
    Each row is one (bond, market) pair.
    Returns a DataFrame with columns for all inputs + computed outputs.

    Used for:
    - TF vs BS divergence analysis
    - Sensitivity heatmaps
    - Credit-equity regime classification
    """
    S0            = RNG.uniform(30, 120, n_scenarios)
    sigma         = RNG.uniform(0.15, 0.60, n_scenarios)
    credit_spread = RNG.uniform(0.01, 0.20, n_scenarios)
    r             = RNG.uniform(0.01, 0.06, n_scenarios)
    maturity      = RNG.choice([2.0, 3.0, 5.0], n_scenarios)
    coupon_rate   = RNG.uniform(0.02, 0.07, n_scenarios)
    face_value    = 1000.0
    conv_ratio    = 20.0

    records = []
    for i in range(n_scenarios):
        cb  = ConvertibleBond(
            face_value=face_value,
            maturity=float(maturity[i]),
            coupon_rate=float(coupon_rate[i]),
            coupon_freq=2,
            conversion_ratio=conv_ratio,
        )
        mkt = MarketParams(
            S0=float(S0[i]),
            r=float(r[i]),
            sigma=float(sigma[i]),
            credit_spread=float(credit_spread[i]),
            recovery=0.40,
        )

        from models import price_tsiveriotis_fernandes, price_bs_benchmark
        tf = price_tsiveriotis_fernandes(cb, mkt, N=150)
        bs = price_bs_benchmark(cb, mkt)

        conversion_value = conv_ratio * S0[i]
        regime = _classify_regime(tf["price"], conversion_value, face_value, credit_spread[i])

        records.append({
            "S0"            : S0[i],
            "sigma"         : sigma[i],
            "credit_spread" : credit_spread[i],
            "r"             : r[i],
            "maturity"      : maturity[i],
            "coupon_rate"   : coupon_rate[i],
            "conversion_value": conversion_value,
            "tf_price"      : tf["price"],
            "bs_price"      : bs["price"],
            "tf_delta"      : tf["delta"],
            "bs_delta"      : bs["delta"],
            "tf_premium"    : tf["premium"],
            "bs_premium"    : bs["premium"],
            "cash_fraction" : tf["bond_floor_proxy"],
            "tf_minus_bs"   : tf["price"] - bs["price"],
            "regime"        : regime,
        })

    return pd.DataFrame(records)


def _classify_regime(price: float, conv_value: float, face: float, spread: float) -> str:
    """
    Classify each CB into one of three regimes:
    - 'equity'    : conversion value > 110% of face, tight spread
    - 'distressed': spread > 10%, conv value < 80% of face
    - 'balanced'  : everything else
    """
    if conv_value > 1.10 * face and spread < 0.05:
        return "equity"
    elif spread > 0.10 and conv_value < 0.80 * face:
        return "distressed"
    else:
        return "balanced"


# ---------------------------------------------------------------------------
# Convergence data: price vs N (tree steps)
# ---------------------------------------------------------------------------

def make_convergence_data(
    cb: ConvertibleBond,
    mkt: MarketParams,
    steps: list[int] | None = None,
) -> pd.DataFrame:
    """
    Price the same CB with increasing N to show tree convergence.
    """
    if steps is None:
        steps = [10, 25, 50, 100, 200, 300, 400, 500]

    from models import price_tsiveriotis_fernandes
    records = []
    for N in steps:
        r = price_tsiveriotis_fernandes(cb, mkt, N=N)
        records.append({"N": N, "price": r["price"], "delta": r["delta"]})

    return pd.DataFrame(records)
