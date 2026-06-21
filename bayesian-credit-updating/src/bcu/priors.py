"""Map credit spreads to prior default probabilities (the 'credit triangle')
and convert freely between probability and odds.

The market spread of a bond or CDS encodes a *prior* default probability.
Under a constant-hazard (reduced-form) approximation, the credit triangle gives

    hazard  lambda = spread / (1 - recovery)
    survival S(T)   = exp(-lambda * T)
    default  PD(T)  = 1 - S(T).

This module is deliberately small and dependency-light (numpy only): it is the
'prior' half of the prior x likelihood updating used throughout the package.
"""
from __future__ import annotations
import numpy as np

__all__ = [
    "hazard_from_spread", "survival_prob", "pd_from_hazard",
    "pd_from_spread", "spread_from_pd", "prob_to_odds", "odds_to_prob",
]


def hazard_from_spread(spread, recovery: float = 0.4):
    """Constant hazard rate implied by a credit spread (decimal, e.g. 0.012)."""
    spread = np.asarray(spread, dtype=float)
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must be in [0, 1)")
    return spread / (1.0 - recovery)


def survival_prob(hazard, T: float = 1.0):
    """Survival probability over horizon T under a constant hazard."""
    return np.exp(-np.asarray(hazard, dtype=float) * T)


def pd_from_hazard(hazard, T: float = 1.0):
    """Default probability over horizon T under a constant hazard."""
    return 1.0 - survival_prob(hazard, T)


def pd_from_spread(spread, recovery: float = 0.4, T: float = 1.0):
    """Prior default probability implied by a spread via the credit triangle."""
    return pd_from_hazard(hazard_from_spread(spread, recovery), T)


def spread_from_pd(pd, recovery: float = 0.4, T: float = 1.0):
    """Inverse of :func:`pd_from_spread`: spread consistent with a given PD."""
    pd = np.clip(np.asarray(pd, dtype=float), 1e-12, 1 - 1e-12)
    hazard = -np.log(1.0 - pd) / T
    return hazard * (1.0 - recovery)


def prob_to_odds(p):
    """Convert probability to odds p / (1 - p)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return p / (1.0 - p)


def odds_to_prob(o):
    """Convert odds to probability o / (1 + o)."""
    o = np.asarray(o, dtype=float)
    return o / (1.0 + o)
