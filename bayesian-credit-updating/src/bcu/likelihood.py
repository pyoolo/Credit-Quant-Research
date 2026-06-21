"""Estimate likelihood ratios for credit signals from historical data.

For a binary signal E (e.g. a rating downgrade) and default indicator D,

    LR(E) = P(E | D) / P(E | not D).

The *joint* likelihood ratio for a set of signals uses the empirically observed
co-occurrence, P(E1 & ... & Ek | D) / P(E1 & ... & Ek | not D); unlike the
product of marginal LRs, it does not assume the signals are independent.
All estimates use Laplace (add-alpha) smoothing so rare cells stay finite.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["signal_rates", "likelihood_ratio", "joint_likelihood_ratio",
           "all_marginal_lrs"]


def _rate(numer_mask, denom_mask, alpha: float) -> float:
    return (numer_mask.sum() + alpha) / (denom_mask.sum() + 2.0 * alpha)


def signal_rates(data: pd.DataFrame, signal: str,
                 default_col: str = "default", alpha: float = 0.5):
    """Return (P(signal=1 | default), P(signal=1 | survive))."""
    d = data[default_col].astype(bool).values
    s = data[signal].astype(bool).values
    p_given_d = _rate(s & d, d, alpha)
    p_given_nd = _rate(s & ~d, ~d, alpha)
    return float(p_given_d), float(p_given_nd)


def likelihood_ratio(data: pd.DataFrame, signal: str,
                     default_col: str = "default", alpha: float = 0.5) -> float:
    """Marginal likelihood ratio of a single signal."""
    p_d, p_nd = signal_rates(data, signal, default_col, alpha)
    return p_d / p_nd


def joint_likelihood_ratio(data: pd.DataFrame, signals,
                           default_col: str = "default",
                           alpha: float = 0.5) -> float:
    """Joint LR for the simultaneous occurrence of all `signals` (all = 1)."""
    d = data[default_col].astype(bool).values
    mask = np.ones(len(data), dtype=bool)
    for s in signals:
        mask &= data[s].astype(bool).values
    p_given_d = _rate(mask & d, d, alpha)
    p_given_nd = _rate(mask & ~d, ~d, alpha)
    return float(p_given_d / p_given_nd)


def all_marginal_lrs(data: pd.DataFrame, signals,
                     default_col: str = "default", alpha: float = 0.5):
    """Dict of {signal: marginal LR}."""
    return {s: likelihood_ratio(data, s, default_col, alpha) for s in signals}
