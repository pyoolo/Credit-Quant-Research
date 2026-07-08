import numpy as np
import pytest

from at1_coco.instrument import AT1Note, mda_fraction
from at1_coco.pricer import MonteCarloPricer
from at1_coco.processes import CET1Params, draw_randoms


def mda_params(**kw):
    d = dict(
        c0=13.0, kappa=0.3, theta=13.0, sigma=1.2,
        jump_intensity=0.15, jump_median=2.0, jump_vol=0.5, ponv_intensity=0.005,
    )
    d.update(kw)
    return CET1Params(**d)


# ---- MDA schedule ------------------------------------------------------
def test_mda_schedule_quartiles():
    thr, buf = 9.0, 4.0  # buffer spans 9..13
    assert mda_fraction(13.5, thr, buf) == 1.0          # above buffer
    assert mda_fraction(8.5, thr, buf) == 0.0           # below floor
    assert mda_fraction(9.5, thr, buf) == 0.0           # bottom quartile
    assert mda_fraction(10.5, thr, buf) == pytest.approx(0.20)
    assert mda_fraction(11.5, thr, buf) == pytest.approx(0.40)
    assert mda_fraction(12.5, thr, buf) == pytest.approx(0.60)


def test_mda_vectorised():
    out = mda_fraction(np.array([8.0, 9.5, 12.5, 20.0]), 9.0, 4.0)
    assert np.allclose(out, [0.0, 0.0, 0.60, 1.0])


# ---- pricer ------------------------------------------------------------
def build(note=None, params=None, **kw):
    note = note or AT1Note()
    params = params or mda_params()
    return MonteCarloPricer(note, params, discount_rate=0.03, horizon=40.0, dt=1 / 12, **kw)


def test_price_positive_and_bounded():
    res = build().price(n_paths=20_000, seed=0)
    # a par-100 AT1 with plausible parameters should trade in a sane band
    assert 40.0 < res.price < 130.0
    assert res.std_error < 1.0


def test_probabilities_partition():
    res = build().price(n_paths=20_000, seed=0)
    total = res.prob_absorption + res.prob_called + res.prob_survived
    assert total == pytest.approx(1.0, abs=1e-9)
    assert res.prob_mechanical + res.prob_ponv == pytest.approx(res.prob_absorption, abs=1e-9)


def test_weaker_bank_prices_lower():
    strong = build(params=mda_params(c0=16.0, theta=16.0)).price(n_paths=20_000, seed=1)
    weak = build(params=mda_params(c0=9.5, theta=9.5, sigma=2.5)).price(n_paths=20_000, seed=1)
    assert weak.price < strong.price
    assert weak.prob_absorption > strong.prob_absorption


def test_recovery_raises_price():
    lo = build(note=AT1Note(trigger_recovery=0.0)).price(n_paths=20_000, seed=2)
    hi = build(note=AT1Note(trigger_recovery=0.6)).price(n_paths=20_000, seed=2)
    assert hi.price >= lo.price


def test_common_random_numbers_are_reused():
    r = draw_randoms(10_000, 480, seed=42)
    a = build().price(randoms=r)
    b = build().price(randoms=r)
    assert a.price == pytest.approx(b.price, abs=1e-12)


def test_higher_ponv_lowers_price():
    lo = build(params=mda_params(ponv_intensity=0.0)).price(n_paths=20_000, seed=4)
    hi = build(params=mda_params(ponv_intensity=0.08)).price(n_paths=20_000, seed=4)
    assert hi.price < lo.price
