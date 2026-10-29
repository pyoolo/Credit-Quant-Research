import numpy as np
import pandas as pd
import pytest

from at1_coco import CET1Params
from at1_coco.market import (
    DatedAT1, MarketState, ModelSettings, SpreadParams, cash_flow_profile,
    cont_rate, default_randoms, implied_ponv,
)

BOND = DatedAT1(isin="US05602XQR25", coupon=0.06875, first_reset=pd.Timestamp("2033-12-15"),
                reset_margin=0.02853, floor_price_usd=61.1189)
MKT = MarketState(pd.Timestamp("2026-09-25"), 94.394, 0.0516, 0.0516, 0.027, 12.97, 113.0)


def _ms(vol=0.0):
    cet1 = CET1Params(c0=13, kappa=0.32, theta=13, sigma=0.41, jump_intensity=0.12,
                      jump_median=2.2, jump_vol=0.5, ponv_intensity=0.0)
    return ModelSettings(cet1=cet1, spread=SpreadParams(long_run=0.027, vol=vol))


@pytest.fixture(scope="module")
def randoms():
    return default_randoms(n_paths=3000, seed=1)


def test_accrued_matches_capital_iq():
    # CIQ bond detail: T0 accrued 1.910, 100 accrual days (30/360) on 25-Sep-2026
    assert BOND.accrued(pd.Timestamp("2026-09-25")) == pytest.approx(1.910, abs=1e-3)


def test_calls_only_on_reset_dates():
    t, is_reset, after = BOND.schedule(pd.Timestamp("2026-09-25"), 40)
    assert is_reset.sum() == 7                     # 2033, 2038, ..., 2063 before Sep-2066
    first = np.argmax(is_reset)
    assert t[first] == pytest.approx((pd.Timestamp("2033-12-15") - pd.Timestamp("2026-09-25")).days / 365.25)
    assert not after[first] and after[first + 1]   # reset coupon starts after the first reset


def test_price_decreasing_in_ponv(randoms):
    prof = cash_flow_profile(BOND, MKT, _ms(), randoms)
    r = cont_rate(MKT.discount_yield)
    prices = [prof.dirty_price(r, lam) for lam in (0.0, 0.01, 0.02, 0.05)]
    assert all(np.diff(prices) < 0)


def test_implied_ponv_reprices(randoms):
    lam, prof = implied_ponv(BOND, MKT, _ms(0.01), randoms)
    dirty = prof.dirty_price(cont_rate(MKT.discount_yield), lam)
    assert dirty == pytest.approx(MKT.clean_price + BOND.accrued(MKT.val_date), abs=1e-5)
    assert 0 < lam < 0.2


def test_deterministic_spread_call_rule(randoms):
    # margin 285bp below a flat 300bp refi spread: a healthy issuer never calls
    ms = ModelSettings(cet1=_ms().cet1, spread=SpreadParams(long_run=0.03, vol=0.0))
    mkt = MarketState(MKT.val_date, 94.0, 0.05, 0.05, 0.03, 13.0, 113.0)
    prof = cash_flow_profile(BOND, mkt, ms, randoms)
    assert prof.diagnostics["p_called"] == 0.0
    # and always calls at the first reset when refinancing is cheaper
    ms = ModelSettings(cet1=_ms().cet1, spread=SpreadParams(long_run=0.02, vol=0.0))
    mkt = MarketState(MKT.val_date, 94.0, 0.05, 0.05, 0.02, 13.0, 113.0)
    prof = cash_flow_profile(BOND, mkt, ms, randoms)
    assert prof.diagnostics["p_call_first"] > 0.9


def test_conversion_recovery_capped_at_par(randoms):
    mkt = MarketState(MKT.val_date, 94.0, 0.05, 0.05, 0.027, 13.0, 10_000.0)
    prof = cash_flow_profile(BOND, mkt, _ms(), randoms)
    assert prof.diagnostics["recovery"] == 1.0


def test_issuer_call_option_costs_the_investor(randoms):
    # the issuer's option to extend can only lower the investor's value
    from dataclasses import replace
    ms = _ms(0.01)
    lam, _ = implied_ponv(BOND, MKT, ms, randoms)
    r = cont_rate(MKT.discount_yield)
    p_model = cash_flow_profile(BOND, MKT, ms, randoms).dirty_price(r, lam)
    p_first = cash_flow_profile(BOND, MKT, ms, randoms, call_rule="first").dirty_price(r, lam)
    assert p_first >= p_model
    # a capital shock that crosses the MDA threshold costs more than one that does not
    p_small = cash_flow_profile(BOND, replace(MKT, cet1=11.5), ms, randoms).dirty_price(r, lam)
    p_big = cash_flow_profile(BOND, replace(MKT, cet1=9.0), ms, randoms).dirty_price(r, lam)
    assert p_model > p_small > p_big
    assert (p_small - p_big) > (p_model - p_small)


def test_buffer_path_steps():
    from at1_coco.market import buffer_path
    from dataclasses import replace
    ms = replace(_ms(), combined_buffer=4.79, buffer_steps=(("2027-01-01", 0.25), ("2028-01-01", 0.25)))
    t = np.array([0.0, 0.5, 1.5])
    w = buffer_path(ms, pd.Timestamp("2026-09-25"), t)
    assert np.allclose(w, [4.79, 5.04, 5.29])


def _t2_setup(randoms):
    from at1_coco.market import DatedBullet, hazard_state
    bond = DatedBullet(isin="US09659T2D24", coupon=0.05906, call_date=pd.Timestamp("2034-11-19"))
    ms = _ms(vol=0.01)
    return bond, ms, hazard_state(MKT, ms, randoms)


def test_t2_accrued_matches_capital_iq():
    from at1_coco.market import DatedBullet
    bond = DatedBullet(isin="US09659T2D24", coupon=0.05906, call_date=pd.Timestamp("2034-11-19"))
    # CIQ bond detail on 27-Oct-2026: T0 accrued 2.592 (158 days, 30/360)
    assert bond.accrued(pd.Timestamp("2026-10-27")) == pytest.approx(2.592, abs=1e-3)


def test_bullet_without_risk_is_discounted_cash_flows(randoms):
    from at1_coco.market import bullet_profile
    bond, ms, st = _t2_setup(randoms)
    prof = bullet_profile(bond, st, MKT.val_date, ms.dt)
    r = 0.045
    t = bond.schedule(MKT.val_date)
    expected = (bond.coupon * 50 * np.exp(-r * t)).sum() + 100 * np.exp(-r * t[-1])
    assert prof.dirty_price(r, 0.0, 0.2) == pytest.approx(expected, rel=1e-12)


def test_full_recovery_removes_credit_risk(randoms):
    from at1_coco.market import bullet_profile
    bond, ms, st = _t2_setup(randoms)
    prof = bullet_profile(bond, st, MKT.val_date, ms.dt)
    p0 = prof.dirty_price(0.045, 0.0, 1.0)
    # recovery of par paid at default instead of later cash flows: close to, not exactly, riskless
    assert prof.dirty_price(0.045, 0.05, 1.0) == pytest.approx(p0, rel=0.03)


def test_bullet_price_decreasing_in_hazard_and_increasing_in_recovery(randoms):
    from at1_coco.market import bullet_profile
    bond, ms, st = _t2_setup(randoms)
    prof = bullet_profile(bond, st, MKT.val_date, ms.dt)
    p = [prof.dirty_price(0.045, lam, 0.2) for lam in (0.0, 0.01, 0.02, 0.04)]
    assert all(a > b for a, b in zip(p, p[1:]))
    assert prof.dirty_price(0.045, 0.02, 0.4) > prof.dirty_price(0.045, 0.02, 0.0)


def test_implied_resolution_hazard_roundtrip(randoms):
    from at1_coco.market import bullet_profile, implied_resolution_hazard
    bond, ms, st = _t2_setup(randoms)
    prof = bullet_profile(bond, st, MKT.val_date, ms.dt)
    y = 0.05
    dirty = prof.dirty_price(cont_rate(y), 0.015, 0.2)
    clean = dirty - bond.accrued(MKT.val_date)
    lam, _ = implied_resolution_hazard(bond, clean, y, st, MKT.val_date, ms.dt, 0.2)
    assert lam == pytest.approx(0.015, abs=1e-6)
