"""
Macro-financial falsification regressions (paper, Sec. 5.4 and App. B).

Reduced-form regressions of the sovereign level factor on global
macro-financial variables (DXY, Brent/WTI, EMBI, EM HY, US 10Y, VIX, MOVE),
with HAC standard errors. Purpose is strictly falsification: in the
post-default regime, coefficients should be economically negligible and
statistically insignificant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class RegressionResult:
    params: pd.Series
    std_errors: pd.Series
    pvalues: pd.Series
    r_squared: float
    nobs: int

    def table(self) -> pd.DataFrame:
        """App.-B-style summary table."""
        return pd.DataFrame(
            {"Coefficient": self.params, "Std. Error": self.std_errors, "p-value": self.pvalues}
        ).round(4)


def hac_regression(y: pd.Series, X: pd.DataFrame, maxlags: int | None = None) -> RegressionResult:
    """OLS with Newey-West (HAC) covariance.

    ``maxlags`` defaults to the Newey-West rule floor(4 * (n/100)^(2/9)).
    """
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("Too few overlapping observations (< 30).")
    yv = df["_y"]
    Xv = sm.add_constant(df.drop(columns="_y"))
    if maxlags is None:
        maxlags = int(np.floor(4 * (len(df) / 100.0) ** (2.0 / 9.0)))
    fit = sm.OLS(yv, Xv).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, maxlags)})
    return RegressionResult(
        params=fit.params,
        std_errors=fit.bse,
        pvalues=fit.pvalues,
        r_squared=float(fit.rsquared),
        nobs=int(fit.nobs),
    )


def macro_falsification(
    level: pd.Series,
    macro: pd.DataFrame,
    freq: str | None = None,
) -> RegressionResult:
    """Baseline falsification regression (Tables 1-2).

    Parameters
    ----------
    level : sovereign level factor (mean price), in levels; first-differenced
        internally.
    macro : DataFrame of macro variables *already transformed* into log
        returns / first differences (App. A.3).
    freq : optional pandas offset alias ('W-FRI' for weekly aggregation,
        Table 2). Levels are resampled last-of-period before differencing;
        macro returns are summed within period.
    """
    y = level.copy()
    X = macro.copy()
    if freq:
        y = y.resample(freq).last()
        X = X.resample(freq).sum(min_count=1)
    dy = y.diff().rename("d_level")
    return hac_regression(dy, X)


def post_default_interactions(
    level: pd.Series,
    macro: pd.DataFrame,
    default_date: pd.Timestamp,
) -> RegressionResult:
    """Regime-shift regression with post-default dummy and interactions (Table 3).

    The dummy enters the *level* equation (capturing the discrete drop at
    default); macro variables and their post-default interactions enter in
    differences. Expected pattern: dummy large and significant, all macro
    terms insignificant — default shifts the level of prices, not their
    macro exposure.
    """
    post = (level.index >= pd.Timestamp(default_date)).astype(float)
    post = pd.Series(post, index=level.index, name="post_default")
    dy = level.diff().rename("d_level")
    d_post = post.diff().rename("post_default_jump")  # spikes at default date
    inter = macro.mul(post, axis=0).add_suffix(" x post")
    X = pd.concat([d_post, macro, inter], axis=1)
    return hac_regression(dy, X)
