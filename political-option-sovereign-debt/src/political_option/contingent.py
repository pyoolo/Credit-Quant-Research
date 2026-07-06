"""
Contingent-claim pricing of defaulted sovereign bonds (paper, Secs. 6-9).

Core identity (Sec. 8.1):

    P_i(t) = R_low + (R_high,i(t) - R_low) * P_t(tau <= T_i)

which inverts to the *implied normalization probability*

    p_i(t) = (P_i(t) - R_low) / (R_high,i(t) - R_low).

Probabilities are pricing (risk-neutral-like) probabilities: they mix
beliefs and compensation for bearing political event risk (Sec. 8.4).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .monotonicity import violation_mass


# --------------------------------------------------------------------------- #
# Inversion: prices <-> implied probabilities
# --------------------------------------------------------------------------- #
def implied_probability(
    price: float | np.ndarray | pd.Series | pd.DataFrame,
    r_low: float,
    r_high: float | np.ndarray,
    clip: bool = True,
):
    """Implied market probability of political normalization (Sec. 8.1, 9.1).

    Parameters
    ----------
    price : scalar, array, Series or DataFrame
        Observed bond price(s), cents on the dollar.
    r_low : float
        Distressed floor under persistent political stalemate.
    r_high : float or array
        Haircut-adjusted recovery benchmark conditional on normalization.
        May be bond-specific (broadcast against ``price``).
    clip : bool
        Truncate to [0, 1] for numerical stability, as in the paper.

    Returns
    -------
    Same type/shape as ``price``.
    """
    denom = np.asarray(r_high, dtype=float) - float(r_low)
    if np.any(denom <= 0):
        raise ValueError("r_high must exceed r_low for every bond.")
    p = (price - r_low) / denom
    if clip:
        p = np.clip(p, 0.0, 1.0) if not isinstance(p, (pd.Series, pd.DataFrame)) else p.clip(0.0, 1.0)
    return p


def price_from_probability(
    prob: float | np.ndarray,
    r_low: float,
    r_high: float | np.ndarray,
):
    """Forward map: convex combination of regime benchmarks (Sec. 6.1).

    P = R_low * S + R_high * (1 - S),  with S = 1 - prob.
    """
    return r_low + (np.asarray(r_high, dtype=float) - float(r_low)) * prob


def implied_probability_panel(
    prices: pd.DataFrame,
    r_low: float,
    r_high: float | dict[str, float],
    clip: bool = True,
) -> pd.DataFrame:
    """Vectorised inversion for a (dates x bonds) price panel.

    ``r_high`` may be a scalar (baseline implementation, Sec. 9.1:
    R_low = 5, R_high = 50) or a dict {bond: R_high_i} allowing the
    bond-specific recoveries of the general framework (Sec. 6.1).
    """
    if isinstance(r_high, dict):
        missing = set(prices.columns) - set(r_high)
        if missing:
            raise KeyError(f"Missing R_high for bonds: {sorted(missing)}")
        rh = pd.Series(r_high).reindex(prices.columns).astype(float)
        out = (prices - r_low).div(rh - r_low, axis=1)
    else:
        out = (prices - r_low) / (float(r_high) - r_low)
    return out.clip(0.0, 1.0) if clip else out


# --------------------------------------------------------------------------- #
# Benchmark calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationResult:
    r_low: float
    r_high: float
    objective: float
    grid: pd.DataFrame  # full objective surface, for diagnostics


def calibrate_benchmarks(
    prices: pd.DataFrame,
    maturities: pd.Series,
    r_low_grid: np.ndarray | None = None,
    r_high_grid: np.ndarray | None = None,
    truncation_weight: float = 1.0,
    span_weight: float = 0.5,
    quantiles: tuple[float, float] = (0.005, 0.995),
) -> CalibrationResult:
    """Data-driven choice of (R_low, R_high) beyond the paper's fixed (5, 50).

    The paper fixes the benchmarks a priori (Sec. 9.1). Here we select them
    by minimising a three-part loss on the *unclipped* implied probabilities:

        L =   mean_t V(t)              (monotonicity mass, Sec. 8.2)
            + w_trunc * truncation     (probability mass outside [0, 1])
            + w_span  * slack          (unused probability range)

    where ``slack`` is the distance between the benchmarks and the extreme
    price quantiles, normalised by the benchmark span.

    Identification note. Because the price->probability map is affine, the
    monotonicity mass alone is *invariant* to (R_low, R_high): benchmarks
    are only set-identified by the model's internal restrictions. The
    truncation term pins the benchmarks to bracket observed prices, and the
    slack term selects the *tightest* such bracket — i.e., the benchmarks
    are interpreted as the observed distressed floor and the maximal
    haircut-adjusted recovery priced over the sample. Recovery of "true"
    benchmarks therefore requires the latent probability to approach 0 and
    1 somewhere in the sample; otherwise the estimate is a conservative
    inner bracket and results should be read as sensitivity analysis around
    the paper's baseline.

    Parameters
    ----------
    prices : DataFrame (dates x bonds)
    maturities : Series mapping bond -> maturity (in years or dates ordinal)
    """
    lo_grid = np.arange(0.0, 15.0 + 1e-9, 1.0) if r_low_grid is None else np.asarray(r_low_grid)
    hi_grid = np.arange(30.0, 80.0 + 1e-9, 2.5) if r_high_grid is None else np.asarray(r_high_grid)

    order = maturities.reindex(prices.columns).sort_values().index
    px = prices[order]
    vals = px.to_numpy().ravel()
    q_lo, q_hi = np.nanquantile(vals, quantiles[0]), np.nanquantile(vals, quantiles[1])
    records = []
    for r_lo, r_hi in itertools.product(lo_grid, hi_grid):
        if r_hi <= r_lo:
            continue
        raw = (px - r_lo) / (r_hi - r_lo)  # unclipped
        trunc = (raw.clip(upper=0.0).abs() + (raw - 1.0).clip(lower=0.0)).mean().mean()
        v = raw.clip(0.0, 1.0).apply(lambda row: violation_mass(row.to_numpy()), axis=1).mean()
        slack = (max(0.0, q_lo - r_lo) + max(0.0, r_hi - q_hi)) / (r_hi - r_lo)
        records.append((r_lo, r_hi, v + truncation_weight * trunc + span_weight * slack))
    grid = pd.DataFrame(records, columns=["r_low", "r_high", "objective"])
    best = grid.loc[grid["objective"].idxmin()]
    return CalibrationResult(
        r_low=float(best["r_low"]),
        r_high=float(best["r_high"]),
        objective=float(best["objective"]),
        grid=grid,
    )
