"""The constant-hazard solver must round-trip price <-> lambda and behave monotonically."""

from __future__ import annotations

import numpy as np

from crv.tracks.distressed import solver


def test_pv_at_zero_hazard_is_above_distressed_price():
    # with no default risk the bond is worth roughly par-plus-coupon-carry
    pv = solver.px_given_lambda(0.0, 6.0, 2.0, 3.0, 0.40, r=0.03)
    assert pv > 90


def test_implied_lambda_reprices():
    price, cpn, freq, yrs, R, r = 60.0, 6.0, 2.0, 3.0, 0.40, 0.03
    lam = solver.implied_lambda(price, cpn, freq, yrs, R, r)
    assert not np.isnan(lam)
    repriced = solver.px_given_lambda(lam, cpn, freq, yrs, R, r)
    assert abs(repriced - price) < 1e-2


def test_higher_recovery_implies_higher_default():
    # at a fixed (distressed) price, assuming more recovery means more implied default
    lo = solver.implied_lambda(60.0, 6.0, 2.0, 3.0, 0.20, 0.03)
    hi = solver.implied_lambda(60.0, 6.0, 2.0, 3.0, 0.60, 0.03)
    assert hi > lo


def test_frontier_is_increasing_in_recovery():
    fr = solver.frontier(60.0, 6.0, 3.0)
    assert len(fr) > 5
    pds = [pd for _, pd in fr]
    assert pds == sorted(pds)  # cumulative PD rises with assumed recovery
