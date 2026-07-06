import numpy as np
import pandas as pd
import pytest

from political_option.contingent import (
    calibrate_benchmarks,
    implied_probability,
    implied_probability_panel,
    price_from_probability,
)


def test_round_trip_scalar():
    p = implied_probability(27.5, r_low=5, r_high=50)
    assert np.isclose(price_from_probability(p, 5, 50), 27.5)
    assert np.isclose(p, 0.5)


def test_clipping():
    assert implied_probability(2.0, 5, 50) == 0.0
    assert implied_probability(80.0, 5, 50) == 1.0
    raw = implied_probability(2.0, 5, 50, clip=False)
    assert raw < 0


def test_invalid_benchmarks_raise():
    with pytest.raises(ValueError):
        implied_probability(20.0, r_low=50, r_high=50)


def test_panel_scalar_and_bond_specific_rhigh():
    idx = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0, 20, 30], "B": [15.0, 25, 35]}, index=idx)
    p_scalar = implied_probability_panel(prices, 5, 50)
    assert p_scalar.shape == prices.shape
    assert ((p_scalar >= 0) & (p_scalar <= 1)).all().all()

    p_dict = implied_probability_panel(prices, 5, {"A": 50.0, "B": 65.0})
    # same price maps to lower prob with higher R_high
    assert p_dict.loc[idx[0], "B"] < p_scalar.loc[idx[0], "B"]

    with pytest.raises(KeyError):
        implied_probability_panel(prices, 5, {"A": 50.0})


def test_calibration_recovers_true_benchmarks():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    maturities = pd.Series({"B1": 2027.0, "B2": 2031.0, "B3": 2038.0})
    true_lo, true_hi = 5.0, 50.0
    # latent probability spanning nearly [0, 1]: benchmarks point-identified
    p = np.clip(np.linspace(0.005, 0.995, len(dates)) + rng.normal(0, 0.01, len(dates)), 0.0, 1.0)
    tilt = {"B1": 0.95, "B2": 1.0, "B3": 1.05}
    prices = pd.DataFrame(
        {b: true_lo + (true_hi - true_lo) * np.clip(p * t, 0, 1) for b, t in tilt.items()},
        index=dates,
    )
    res = calibrate_benchmarks(
        prices,
        maturities,
        r_low_grid=np.arange(0, 11, 1.0),
        r_high_grid=np.arange(40, 61, 2.5),
    )
    assert abs(res.r_low - true_lo) <= 2.0
    assert abs(res.r_high - true_hi) <= 5.0
    assert res.objective >= 0
