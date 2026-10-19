"""Tests for the signal layer."""
import numpy as np
import pandas as pd
import pytest

from cds_basis.signal import OUParams, SignalConfig, calibrate_ou, zscore_signal


def _simulate_ou(n, dt, kappa, theta, sigma, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = theta
    exp_kdt = np.exp(-kappa * dt)
    std = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    for t in range(1, n):
        x[t] = theta + (x[t - 1] - theta) * exp_kdt + std * rng.standard_normal()
    return x


def test_ou_calibration_recovers_params():
    dt = 1 / 252
    true = OUParams(kappa=10.0, theta=-0.0008, sigma=0.003)
    path = _simulate_ou(5000, dt, true.kappa, true.theta, true.sigma, seed=1)
    est = calibrate_ou(pd.Series(path), dt=dt)
    assert est.kappa == pytest.approx(true.kappa, rel=0.25)
    assert est.theta == pytest.approx(true.theta, abs=3e-4)
    assert est.sigma == pytest.approx(true.sigma, rel=0.25)


def test_ou_half_life_positive():
    p = OUParams(kappa=12.0, theta=0.0, sigma=0.01)
    assert p.half_life > 0
    assert p.stationary_std > 0


def test_zscore_signal_enters_on_extreme():
    # Construct a series that dips far below its mean then recovers
    base = np.concatenate([
        np.zeros(60),
        np.full(10, -0.01),   # large negative deviation
        np.zeros(60),
    ])
    s = pd.Series(base, index=pd.bdate_range("2022-01-03", periods=len(base)))
    out = zscore_signal(s, SignalConfig(lookback=30, entry_z=2.0, exit_z=0.5))
    # Should take a long-basis position (+1) somewhere during the dip
    assert (out["position"] == 1).any()


def test_signal_has_no_position_during_warmup():
    s = pd.Series(np.random.default_rng(0).standard_normal(100) * 0.001)
    out = zscore_signal(s, SignalConfig(lookback=60))
    assert (out["position"].iloc[:59] == 0).all()
