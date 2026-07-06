"""
Model-free cross-maturity summary measures (paper, Secs. 4.1-4.2, 5.2).

The object of analysis is the *price curve*, not yields or spreads:
in prolonged default, contractual cash flows lose enforceability and
yield-based metrics are uninformative.
"""

from __future__ import annotations

import pandas as pd


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize each date's cross-section by its cross-sectional mean (Sec. 4.1).

    Removes common level shifts and isolates the *shape* of the price
    profile across maturities.
    """
    return prices.div(prices.mean(axis=1), axis=0)


def level_factor(prices: pd.DataFrame) -> pd.Series:
    """Cross-sectional mean price: the common 'level' factor (Secs. 5.2-5.4)."""
    return prices.mean(axis=1).rename("level")


def curve_summary(prices: pd.DataFrame, maturities: pd.Series) -> pd.DataFrame:
    """Level, dispersion and slope of the price curve at each date (Sec. 4.2).

    slope = mean price of the longest-maturity half minus the shortest half;
    a flat (compressed) curve — the post-default signature — has slope ~ 0
    and low dispersion.
    """
    order = maturities.reindex(prices.columns).sort_values().index
    px = prices[order]
    n = len(order)
    short_leg = px.iloc[:, : max(1, n // 2)].mean(axis=1)
    long_leg = px.iloc[:, -max(1, n // 2):].mean(axis=1)
    return pd.DataFrame(
        {
            "level": px.mean(axis=1),
            "dispersion": px.std(axis=1),
            "slope": long_leg - short_leg,
        }
    )
