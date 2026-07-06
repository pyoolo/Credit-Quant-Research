import numpy as np
import pandas as pd
import pytest

from political_option.curves import curve_summary, level_factor, normalize_prices
from political_option.data import simulate_default_panel
from political_option.events import cumulative_abnormal_returns, event_study
from political_option.regressions import (
    hac_regression,
    macro_falsification,
    post_default_interactions,
)


@pytest.fixture(scope="module")
def panel():
    return simulate_default_panel(seed=7)


# ------------------------------ events ------------------------------------ #
def test_event_study_detects_jump():
    dates = pd.bdate_range("2024-01-01", periods=400)
    px = pd.Series(20.0, index=dates)
    event = dates[300]
    px.loc[event:] = 26.0  # +30% jump on impact
    res = event_study(px, event)
    assert res.jump_on_impact > 0.25
    assert abs(res.pre_event_drift) < 1e-6
    car = cumulative_abnormal_returns(px, event, window=(-10, 4))
    assert car.index.min() == -10 and car.index.max() == 4
    assert car.loc[4] > car.loc[-1]


# ----------------------------- regressions -------------------------------- #
def test_hac_regression_recovers_beta():
    rng = np.random.default_rng(1)
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n)
    x = pd.DataFrame({"x1": rng.normal(0, 1, n)}, index=idx)
    y = pd.Series(2.0 * x["x1"] + rng.normal(0, 0.5, n), index=idx)
    res = hac_regression(y, x)
    assert abs(res.params["x1"] - 2.0) < 0.15
    assert res.pvalues["x1"] < 1e-6
    assert set(res.table().columns) == {"Coefficient", "Std. Error", "p-value"}


def test_macro_falsification_post_default_insignificant(panel):
    post = panel.prices.loc[panel.default_date :]
    lvl = level_factor(post)
    res = macro_falsification(lvl, panel.macro.loc[panel.default_date :])
    slopes = res.pvalues.drop("const")
    # DGP has no macro channel post-default: most coefficients insignificant
    assert (slopes > 0.05).mean() >= 0.75


def test_weekly_aggregation_runs(panel):
    lvl = level_factor(panel.prices.loc[panel.default_date :])
    res = macro_falsification(lvl, panel.macro.loc[panel.default_date :], freq="W-FRI")
    assert res.nobs > 50


def test_post_default_dummy_significant(panel):
    lvl = level_factor(panel.prices)
    res = post_default_interactions(lvl, panel.macro, panel.default_date)
    assert res.params["post_default_jump"] < 0          # price collapse at default
    assert res.pvalues["post_default_jump"] < 0.05


# -------------------------------- data ------------------------------------ #
def test_simulated_panel_stylised_facts(panel):
    pre = panel.prices.loc[: panel.default_date - pd.Timedelta(days=1)]
    post = panel.prices.loc[panel.default_date :]
    # level collapse at default
    assert post.mean().mean() < 0.6 * pre.mean().mean()
    # curve compression: post-default dispersion below pre-default dispersion
    assert post.std(axis=1).mean() < pre.std(axis=1).mean()
    # post-default cross-maturity co-movement: avg pairwise return corr high
    rets = post.pct_change().dropna()
    corr = rets.corr().to_numpy()
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert off_diag.mean() > 0.6


def test_curve_summary_and_normalization(panel):
    summ = curve_summary(panel.prices, panel.maturities)
    assert {"level", "dispersion", "slope"} <= set(summ.columns)
    norm = normalize_prices(panel.prices)
    assert np.allclose(norm.mean(axis=1).dropna(), 1.0)
