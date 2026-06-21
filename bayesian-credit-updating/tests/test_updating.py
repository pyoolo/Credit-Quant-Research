import numpy as np
from bcu import priors, updating, likelihood, data


def test_naive_update_matches_manual():
    # 2% prior, LRs 6 and 2.5 -> ~23.4% (the worked credit example)
    post = updating.naive_update(0.02, [6.0, 2.5])
    assert abs(post - 0.2343) < 1e-3


def test_single_signal_update():
    post = updating.naive_update(0.02, 6.0)   # downgrade only -> ~10.9%
    assert abs(post - 0.1089) < 1e-3


def test_naive_overstates_under_correlation():
    """With positively correlated signals, multiplying marginal LRs should
    give a higher posterior than the correlation-aware joint update."""
    df = data.generate(n=8000, seed=1)
    signals = ["downgrade", "earnings_miss", "spread_widening"]
    marg = [likelihood.likelihood_ratio(df, s) for s in signals]
    joint = likelihood.joint_likelihood_ratio(df, signals)
    prior = 0.02
    naive = updating.naive_update(prior, marg)
    aware = updating.joint_update(prior, joint)
    assert naive > aware            # the central thesis of the repo
    assert np.prod(marg) > joint    # marginal product exceeds joint LR
