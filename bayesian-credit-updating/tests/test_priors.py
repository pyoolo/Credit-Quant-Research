import numpy as np
from bcu import priors


def test_credit_triangle_roundtrip():
    pd0 = 0.02
    s = priors.spread_from_pd(pd0, recovery=0.4, T=1.0)
    pd1 = priors.pd_from_spread(s, recovery=0.4, T=1.0)
    assert abs(pd1 - pd0) < 1e-9


def test_hazard_monotonic_in_spread():
    s = np.array([0.005, 0.01, 0.02, 0.05])
    pds = priors.pd_from_spread(s)
    assert np.all(np.diff(pds) > 0)


def test_odds_inverse():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    assert np.allclose(priors.odds_to_prob(priors.prob_to_odds(p)), p)
