"""
Cross-maturity monotonicity restriction (paper, Secs. 8.2 and 9.2).

Because normalization is a sovereign-level event, implied probabilities
must be weakly increasing in maturity:  T_i < T_j  =>  p_i(t) <= p_j(t).

The violation mass on date t is

    V(t) = sum_k max{0, p_(k)(t) - p_(k+1)(t)},

with bonds sorted by increasing maturity. V(t) ~ 0 indicates consistency
with a single underlying normalization probability curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def violation_mass(probs_sorted_by_maturity: np.ndarray) -> float:
    """V(t) for a single date; input must already be maturity-sorted."""
    p = np.asarray(probs_sorted_by_maturity, dtype=float)
    p = p[~np.isnan(p)]
    if p.size < 2:
        return 0.0
    diffs = p[:-1] - p[1:]
    return float(np.sum(np.clip(diffs, 0.0, None)))


def violation_series(
    probs: pd.DataFrame,
    maturities: pd.Series,
    smoothing_window: int | None = 60,
) -> pd.DataFrame:
    """Time series of V(t) for a (dates x bonds) probability panel.

    Parameters
    ----------
    probs : DataFrame of implied probabilities (dates x bonds)
    maturities : Series mapping bond -> maturity; used to sort columns
    smoothing_window : optional moving-average window (paper Fig. 8 uses 60d)

    Returns
    -------
    DataFrame with columns ``V`` and (if requested) ``V_ma``.
    """
    order = maturities.reindex(probs.columns).sort_values().index
    sorted_probs = probs[order]
    v = sorted_probs.apply(lambda row: violation_mass(row.to_numpy()), axis=1).rename("V")
    out = v.to_frame()
    if smoothing_window:
        out["V_ma"] = v.rolling(smoothing_window, min_periods=max(5, smoothing_window // 4)).mean()
    return out
