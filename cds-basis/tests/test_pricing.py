"""Tests for the pricing layer."""
import numpy as np
import pytest

from cds_basis.pricing import CDS, Bond, CashSpread, DiscountCurve, compute_basis


def test_flat_curve_discount_factor():
    curve = DiscountCurve.flat(0.03)
    # DF at t=0 limit and monotonic decreasing
    assert curve.discount_factor(1.0) < 1.0
    assert curve.discount_factor(5.0) < curve.discount_factor(1.0)
    np.testing.assert_allclose(curve.discount_factor(2.0), np.exp(-0.03 * 2.0), rtol=1e-12)


def test_curve_rejects_non_increasing_tenors():
    with pytest.raises(ValueError):
        DiscountCurve(np.array([1.0, 1.0, 2.0]), np.array([0.03, 0.03, 0.03]))


def test_zspread_roundtrip():
    curve = DiscountCurve.flat(0.03)
    bond = Bond(maturity=5.0, coupon=0.05)
    target_spread = 0.012
    price = bond.price_from_curve(curve, target_spread)
    recovered = bond.z_spread(curve, price)
    assert recovered == pytest.approx(target_spread, abs=1e-8)


def test_cds_par_spread_hazard_roundtrip():
    curve = DiscountCurve.flat(0.03)
    cds = CDS(maturity=5.0, recovery=0.4)
    hazard = 0.02
    spread = cds.par_spread(curve, hazard)
    recovered = cds.hazard_from_spread(curve, spread)
    assert recovered == pytest.approx(hazard, rel=1e-6)


def test_credit_triangle_approximation():
    cds = CDS(maturity=5.0, recovery=0.4)
    curve = DiscountCurve.flat(0.03)
    spread = 0.012
    h_exact = cds.hazard_from_spread(curve, spread)
    h_approx = CDS.credit_triangle(spread, 0.4)
    # Triangle is a first-order approximation; should be in the right ballpark
    assert h_approx == pytest.approx(h_exact, rel=0.1)


def test_compute_basis_sign():
    curve = DiscountCurve.flat(0.03)
    bond = Bond(maturity=5.0, coupon=0.05)
    cds = CDS(maturity=5.0, recovery=0.4)
    # Price the bond at a known cash spread, set CDS below it -> negative basis
    cash_spread = 0.012
    price = bond.price_from_curve(curve, cash_spread)
    res = compute_basis(curve, bond, price, cds, cds_spread=0.010,
                        cash_spread_type=CashSpread.Z_SPREAD)
    assert res.basis < 0
    assert res.cash_spread == pytest.approx(cash_spread, abs=1e-6)
