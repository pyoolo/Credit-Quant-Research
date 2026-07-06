"""
Implied hazard rates of political normalization (paper, Sec. 8.3).

Under a constant hazard lambda, survival of the political stalemate is

    S(t, T) = exp(-lambda * (T - t)),

hence, given an implied probability p = P_t(tau <= T),

    lambda = - ln(1 - p) / (T - t).

The piecewise-constant extension bootstraps a hazard *term structure*
from the cross-section of maturities, in the spirit of CDS bootstrapping,
extending the single-lambda summary of the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def implied_hazard(prob: float | np.ndarray, horizon: float | np.ndarray) -> float | np.ndarray:
    """Constant implied hazard rate (Sec. 8.3).

    Parameters
    ----------
    prob : implied normalization probability in [0, 1)
    horizon : T_i - t, in years (must be > 0)
    """
    prob = np.asarray(prob, dtype=float)
    horizon = np.asarray(horizon, dtype=float)
    if np.any(horizon <= 0):
        raise ValueError("Horizon must be strictly positive.")
    p = np.clip(prob, 0.0, 1.0 - _EPS)
    out = -np.log1p(-p) / horizon
    return float(out) if out.ndim == 0 else out


def hazard_term_structure(
    probs: pd.Series,
    horizons: pd.Series,
) -> pd.DataFrame:
    """Piecewise-constant hazard bootstrap across maturities.

    Given implied probabilities p_k for horizons T_1 < ... < T_K at a fixed
    date, survival S_k = 1 - p_k, and the forward hazard on (T_{k-1}, T_k] is

        lambda_k = - (ln S_k - ln S_{k-1}) / (T_k - T_{k-1}),  S_0 = 1, T_0 = 0.

    Negative forward hazards flag monotonicity violations (Sec. 8.2/9.2) and
    are returned as-is so callers can inspect them.

    Returns
    -------
    DataFrame indexed like the sorted input, with columns
    ``horizon``, ``prob``, ``survival``, ``forward_hazard``.
    """
    df = pd.DataFrame({"horizon": horizons, "prob": probs}).dropna().sort_values("horizon")
    if df.empty:
        raise ValueError("No valid (prob, horizon) pairs.")
    if (df["horizon"] <= 0).any():
        raise ValueError("All horizons must be strictly positive.")
    surv = np.clip(1.0 - df["prob"].to_numpy(), _EPS, 1.0)
    log_s = np.log(surv)
    t = df["horizon"].to_numpy()
    prev_log_s = np.concatenate([[0.0], log_s[:-1]])
    prev_t = np.concatenate([[0.0], t[:-1]])
    fwd = -(log_s - prev_log_s) / (t - prev_t)
    df["survival"] = surv
    df["forward_hazard"] = fwd
    return df
