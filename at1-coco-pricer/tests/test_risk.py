import numpy as np
import pytest

from at1_coco.instrument import AT1Note
from at1_coco.pricer import MonteCarloPricer
from at1_coco.processes import CET1Params, draw_randoms
from at1_coco.risk import greeks, loss_distribution, trigger_term_structure


def make_pricer(**pk):
    p = CET1Params(
        c0=12.0, kappa=0.3, theta=12.0, sigma=1.5,
        jump_intensity=0.20, jump_median=2.5, jump_vol=0.5, ponv_intensity=0.01,
    )
    return MonteCarloPricer(AT1Note(), p, discount_rate=0.03, horizon=40.0, dt=1 / 12)


def test_greek_signs():
    g = greeks(make_pricer(), n_paths=20_000, seed=0)
    # more capital -> worth more
    assert g["delta_cet1"] > 0
    # more absorption risk -> worth less
    assert g["jump_sensitivity"] < 0
    assert g["ponv_sensitivity"] < 0
    # a lower trigger is safer, so price falls as the trigger level rises
    assert g["trigger_sensitivity"] < 0
    # a wider combined-buffer *requirement* raises the MDA hurdle, so full
    # coupons need more capital -> more cancellation risk -> lower price
    assert g["buffer_sensitivity"] < 0
    # long perpetual cash flows -> negative DV01 (price falls as rates rise)
    assert g["dv01"] < 0


def test_term_structure_monotone():
    tenors, probs = trigger_term_structure(make_pricer(), n_paths=20_000, seed=0)
    assert np.all(np.diff(probs) >= -1e-12)   # cumulative, non-decreasing
    assert probs[0] >= 0 and probs[-1] <= 1


def test_loss_distribution_coherence():
    res = make_pricer().price(n_paths=20_000, seed=0)
    ld = loss_distribution(res, par=100.0, alpha=0.99)
    assert ld["ES_99"] >= ld["VaR_99"]           # ES no smaller than VaR
    assert 0.0 <= ld["prob_loss"] <= 1.0
    assert ld["worst_loss"] <= 100.0             # cannot lose more than par
