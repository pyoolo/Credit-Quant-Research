"""
Tests for the ascredit (Avellaneda-Stoikov, credit) package. Run with: pytest -q
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ascredit import (ASParams, optimal_quotes, symmetric_quotes,
                      reservation_spread, optimal_half_spread,
                      SpreadModel, fill_probability, calibrate_k_from_halflife,
                      SimConfig, run_session, monte_carlo)


def _params(gamma=0.05):
    return ASParams(gamma=gamma, sigma=4.0,
                    k=calibrate_k_from_halflife(3.0), A=140.0, T=1.0, dv01=1.0)


def test_reservation_spread_zero_inventory():
    p = _params()
    # with q=0 the reservation spread equals the mid
    assert reservation_spread(120.0, 0.0, 0.5, p) == 120.0


def test_skew_sign_long_credit():
    """A long-credit dealer (q>0) shades the reservation spread UP."""
    p = _params()
    r_long = reservation_spread(120.0, 5.0, 0.5, p)
    r_flat = reservation_spread(120.0, 0.0, 0.5, p)
    r_short = reservation_spread(120.0, -5.0, 0.5, p)
    assert r_long > r_flat > r_short


def test_bid_above_ask_in_spread_space():
    """Spread is inverse to price, so bid_spread > ask_spread."""
    p = _params()
    bid, ask = optimal_quotes(120.0, 0.0, 0.5, p)
    assert bid > ask


def test_half_spread_decreases_to_terminal():
    p = _params()
    early = optimal_half_spread(0.0, p)
    late = optimal_half_spread(0.99, p)
    assert early > late  # inventory-risk term shrinks toward T


def test_fill_probability_monotone_in_distance():
    # quoting further from mid lowers fill probability
    near = fill_probability(1.0, A=140.0, k=0.3, dt=0.001)
    far = fill_probability(8.0, A=140.0, k=0.3, dt=0.001)
    assert 0.0 <= far < near <= 1.0


def test_spread_model_positivity_and_shape():
    rng = np.random.default_rng(0)
    sm = SpreadModel(s0=120.0, sigma=4.0, kappa=1.0, theta=120.0, floor=1.0)
    path = sm.simulate(T=1.0, n_steps=500, rng=rng)
    assert path.shape == (501,)
    assert (path >= 1.0).all()


def test_calibrate_k_halflife():
    k = calibrate_k_from_halflife(3.0)
    # intensity should halve at 3 bp: exp(-k*3) == 0.5
    assert abs(np.exp(-k * 3.0) - 0.5) < 1e-9


def test_session_runs():
    rng = np.random.default_rng(1)
    p = _params()
    sm = SpreadModel(s0=120.0, sigma=4.0, kappa=1.0, theta=120.0)
    res = run_session(p, sm, SimConfig(strategy="as", n_steps=500), rng)
    assert np.isfinite(res.pnl)
    assert res.s_path.shape == (501,)
    assert abs(res.terminal_q) <= 25.0  # respects inventory cap


def test_as_controls_inventory_better_than_symmetric():
    """The headline guard test: AS must reduce terminal-inventory dispersion
    versus the inventory-agnostic symmetric benchmark. If the skew sign is
    wrong, AS would *accumulate* inventory and this fails."""
    p = _params()
    sm = SpreadModel(s0=120.0, sigma=4.0, kappa=1.0, theta=120.0)
    as_res = monte_carlo(p, sm, SimConfig(strategy="as"), n_paths=1500, seed=42)
    sym_res = monte_carlo(p, sm, SimConfig(strategy="symmetric"), n_paths=1500, seed=42)
    assert as_res["std_terminal_q"] < sym_res["std_terminal_q"]


def test_higher_gamma_reduces_pnl_variance():
    sm = SpreadModel(s0=120.0, sigma=4.0, kappa=1.0, theta=120.0)
    lo = monte_carlo(_params(0.01), sm, SimConfig(strategy="as"), 1200, seed=7)
    hi = monte_carlo(_params(0.4), sm, SimConfig(strategy="as"), 1200, seed=7)
    assert hi["std_pnl"] <= lo["std_pnl"]
