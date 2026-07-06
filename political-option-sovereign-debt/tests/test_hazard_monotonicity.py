import numpy as np
import pandas as pd
import pytest

from political_option.hazard import hazard_term_structure, implied_hazard
from political_option.monotonicity import violation_mass, violation_series


# ------------------------------ hazard ------------------------------------ #
def test_hazard_matches_closed_form():
    lam = 0.15
    T = 4.0
    p = 1 - np.exp(-lam * T)
    assert np.isclose(implied_hazard(p, T), lam)


def test_hazard_positive_horizon_required():
    with pytest.raises(ValueError):
        implied_hazard(0.3, 0.0)


def test_term_structure_flat_hazard():
    lam = 0.10
    horizons = pd.Series({"B1": 2.0, "B2": 5.0, "B3": 10.0})
    probs = 1 - np.exp(-lam * horizons)
    ts = hazard_term_structure(probs, horizons)
    assert np.allclose(ts["forward_hazard"], lam, atol=1e-10)


def test_term_structure_flags_violations_with_negative_forward():
    horizons = pd.Series({"B1": 2.0, "B2": 5.0})
    probs = pd.Series({"B1": 0.6, "B2": 0.4})  # non-monotone
    ts = hazard_term_structure(probs, horizons)
    assert (ts["forward_hazard"] < 0).any()


# --------------------------- monotonicity --------------------------------- #
def test_violation_mass_zero_when_sorted():
    assert violation_mass(np.array([0.1, 0.2, 0.2, 0.5])) == 0.0


def test_violation_mass_value():
    v = violation_mass(np.array([0.5, 0.3, 0.6, 0.55]))
    assert np.isclose(v, 0.2 + 0.05)


def test_violation_mass_handles_nan_and_short():
    assert violation_mass(np.array([np.nan, 0.4])) == 0.0
    assert violation_mass(np.array([0.7])) == 0.0


def test_violation_series_sorts_by_maturity():
    idx = pd.date_range("2020-01-01", periods=5)
    # columns given out of maturity order on purpose
    probs = pd.DataFrame({"LONG": 0.5, "SHORT": 0.2}, index=idx)
    mats = pd.Series({"LONG": 2038.0, "SHORT": 2027.0})
    v = violation_series(probs, mats, smoothing_window=None)
    assert (v["V"] == 0).all()  # short(0.2) <= long(0.5): no violation
