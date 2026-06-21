"""Synthetic corporate-credit panel with *correlated* signals.

Each issuer has a rating, a latent deterioration factor z, and a realized
1-year default outcome. Three signals -- downgrade, earnings_miss,
spread_widening -- all load on the same z, so they are correlated with one
another and with default. This is exactly the regime in which naive
independent LR multiplication breaks, which makes the dataset useful for
demonstrating the correlation-aware methods.

The observed CDS spread is generated from the *rating* base PD (plus noise),
i.e. it reflects the stale prior the market quotes from the rating, while the
signals carry the new, not-yet-priced information.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .priors import spread_from_pd

__all__ = ["RATINGS", "BASE_LOGODDS", "generate"]

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
# Rough monotone 1y default log-odds by rating.
BASE_LOGODDS = {"AAA": -7.0, "AA": -6.0, "A": -5.0, "BBB": -3.9,
                "BB": -2.8, "B": -1.9, "CCC": -0.9}
_RATING_P = np.array([0.03, 0.10, 0.22, 0.30, 0.20, 0.12, 0.03])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 4000, seed: int = 0, gamma: float = 1.4,
             recovery: float = 0.4, T: float = 1.0,
             signal_load=(2.0, 1.8, 2.2),
             signal_base=(-1.3, -1.1, -1.0),
             spread_noise: float = 0.25) -> pd.DataFrame:
    """Generate a synthetic issuer panel.

    Parameters mirror the generative model in the paper. `gamma` controls how
    strongly the latent deterioration drives default; `signal_load` how strongly
    it drives each signal (larger => more correlated signals).
    """
    rng = np.random.default_rng(seed)
    rating = rng.choice(RATINGS, size=n, p=_RATING_P)
    a = np.array([BASE_LOGODDS[r] for r in rating])

    z = rng.standard_normal(n)                     # latent deterioration
    p_true = _sigmoid(a + gamma * z)
    default = rng.binomial(1, p_true)

    dl, ml, sl = signal_load
    db, mb, sb = signal_base
    downgrade = rng.binomial(1, _sigmoid(db + dl * z))
    earnings_miss = rng.binomial(1, _sigmoid(mb + ml * z))
    spread_widening = rng.binomial(1, _sigmoid(sb + sl * z))

    base_pd = _sigmoid(a)                           # rating-implied prior PD
    spread = spread_from_pd(base_pd, recovery, T) * np.exp(
        spread_noise * rng.standard_normal(n))

    return pd.DataFrame({
        "rating": rating,
        "cds_spread": spread,
        "downgrade": downgrade,
        "earnings_miss": earnings_miss,
        "spread_widening": spread_widening,
        "default": default,
        "_z": z,
        "_p_true": p_true,
    })
