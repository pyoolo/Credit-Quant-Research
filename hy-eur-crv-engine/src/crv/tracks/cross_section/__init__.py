"""Track 1 — cross-sectional fair-spread Lasso (the ``xs`` panel).

A panel of fixed-coupon HY corporates is priced with a Lasso regression of
``log(z_spread)`` on a granular notch rating, structural features and two-sided
fundamental scores, plus sector and week fixed-effects. The fitted residual,
z-scored within (date x rating bucket), is the relative-value signal.

Modules:

    ``fundamentals``  — build per-ISIN credit ratios from Financials.xlsx
    ``features``      — structural features + two-sided fundamental scores + notch
    ``model``         — Lasso path, held-out alpha selection, final fit
    ``signal``        — fair-spread scoring -> rv_z -> CHEAP/FAIR/RICH
    ``backtest``      — forward IC, CHEAP-vs-RICH, rv_z half-life
    ``artifacts``     — persist the fitted model + company snapshot for the pricer
    ``pipeline``      — orchestrates the above end to end
"""

from crv.tracks.cross_section import (
    artifacts,
    backtest,
    features,
    fundamentals,
    model,
    signal,
)

__all__ = [
    "artifacts", "backtest", "features", "fundamentals", "model", "signal",
]
