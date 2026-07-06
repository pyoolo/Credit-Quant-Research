"""
Data layer.

Bloomberg/vendor prices for defaulted Venezuelan debt cannot be
redistributed, so the repository ships a *synthetic* generator that
reproduces the stylised facts of the post-default regime (paper, Sec. 5):

  1. pre-default: conventional maturity-differentiated pricing, macro beta;
  2. at default: discrete collapse of the level, curve compression;
  3. post-default: long flat spells + jump repricing at political events,
     a dominant common level factor, negligible macro sensitivity.

Real data drop-in: place a CSV in ``data/`` with a ``date`` column and one
column per bond, and use :func:`load_price_panel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_BONDS: dict[str, float] = {
    # bond label -> contractual maturity (year fraction)
    "VENZ 9.25 09/15/27": 2027.7,
    "VENZ 11.95 08/05/31": 2031.6,
    "VENZ 9.375 01/13/34": 2034.0,
    "VENZ 7 03/31/38": 2038.2,
    "PDVSA 5.375 04/12/27": 2027.3,
    "PDVSA 5.5 04/12/37": 2037.3,
}

MACRO_COLS = ["DXY", "Brent", "WTI", "EMBI", "EM_HY", "US10Y", "VIX", "MOVE"]


@dataclass
class SimulatedPanel:
    prices: pd.DataFrame          # dates x bonds, cents on the dollar
    macro: pd.DataFrame           # dates x macro variables (returns / diffs)
    maturities: pd.Series         # bond -> maturity (year fraction)
    default_date: pd.Timestamp
    event_dates: list[pd.Timestamp] = field(default_factory=list)
    latent_prob: pd.Series | None = None  # true normalization probability path


def simulate_default_panel(
    start: str = "2015-01-01",
    end: str = "2026-01-31",
    default_date: str = "2017-11-15",
    r_low: float = 5.0,
    r_high: float = 50.0,
    n_events: int = 6,
    seed: int = 42,
) -> SimulatedPanel:
    """Simulate a sovereign/PDVSA panel with a political-option data-generating
    process.

    Post-default prices are generated *from the model*:
        P_i(t) = R_low + (R_high - R_low) * p(t) * m_i + noise,
    where p(t) is a latent normalization probability following a jump
    process (political events) plus small diffusion, and m_i > is a mild
    maturity tilt making longer bonds slightly more exposed (Sec. 7.2).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    default_ts = pd.Timestamp(default_date)
    bonds = list(DEFAULT_BONDS)
    maturities = pd.Series(DEFAULT_BONDS, name="maturity")

    # ---- macro variables: iid-ish returns/diffs -------------------------- #
    macro = pd.DataFrame(
        rng.normal(0.0, 0.008, size=(len(dates), len(MACRO_COLS))),
        index=dates,
        columns=MACRO_COLS,
    )
    macro["US10Y"] = rng.normal(0.0, 3.0, len(dates))   # delta bps
    macro["VIX"] = rng.normal(0.0, 0.05, len(dates))    # delta log
    macro["MOVE"] = rng.normal(0.0, 0.03, len(dates))   # delta log

    pre_mask = dates < default_ts
    post_mask = ~pre_mask

    # ---- latent normalization probability (post-default) ----------------- #
    n_post = int(post_mask.sum())
    event_pos = np.sort(rng.choice(np.arange(60, n_post - 30), size=n_events, replace=False))
    jumps = np.zeros(n_post)
    jumps[event_pos] = rng.normal(0.10, 0.05, n_events)  # mostly positive news
    jumps[event_pos[-1]] = 0.25                          # one salient Jan-2026-style shock
    drift = rng.normal(0.0, 0.004, n_post)               # small diffusion
    p_latent = 0.35 + np.cumsum(jumps + drift)
    p_latent = np.clip(p_latent - np.linspace(0.0, 0.25, n_post), 0.02, 0.95)  # slow decay then jumps

    # ---- prices ----------------------------------------------------------- #
    prices = pd.DataFrame(index=dates, columns=bonds, dtype=float)

    # pre-default: conventional pricing with maturity profile and macro beta
    mat_rank = maturities.rank()
    base_pre = 105.0 - 4.5 * mat_rank  # downward-sloping distressed-ish curve
    macro_beta = np.array([60.0, 25.0, 0, 15.0, 0, 0, 0, 0])  # loads on DXY/Brent/EMBI
    common_pre = (macro[MACRO_COLS].to_numpy() @ macro_beta)[pre_mask].cumsum()
    for j, b in enumerate(bonds):
        idio = rng.normal(0, 0.4, pre_mask.sum()).cumsum()
        prices.loc[pre_mask, b] = base_pre.iloc[j] + common_pre + idio
        prices.loc[pre_mask, b] = prices.loc[pre_mask, b].clip(lower=40.0)

    # post-default: political-option DGP, no macro exposure
    tilt = 0.92 + 0.03 * (mat_rank - mat_rank.mean())  # longer maturity => higher p exposure
    for j, b in enumerate(bonds):
        noise = rng.normal(0, 0.15, n_post)
        p_i = np.clip(p_latent * tilt.iloc[j], 0.0, 1.0)
        prices.loc[post_mask, b] = r_low + (r_high - r_low) * p_i + noise
    prices = prices.clip(lower=1.0)

    event_dates = list(pd.DatetimeIndex(dates[post_mask])[event_pos])
    latent = pd.Series(p_latent, index=dates[post_mask], name="p_latent")
    return SimulatedPanel(
        prices=prices,
        macro=macro,
        maturities=maturities,
        default_date=default_ts,
        event_dates=event_dates,
        latent_prob=latent,
    )


def load_price_panel(path: str, date_col: str = "date") -> pd.DataFrame:
    """Load a real (dates x bonds) price panel from CSV."""
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    return df.apply(pd.to_numeric, errors="coerce")
