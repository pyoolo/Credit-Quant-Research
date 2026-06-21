"""Bayesian updating in odds form.

    posterior_odds = prior_odds x  prod_j LR_j        (independent signals)

The independence product is exact only when signals are conditionally
independent given default. When they are correlated (the usual case in credit,
where a downgrade and an earnings miss often share the same underlying
deterioration), multiplying marginal LRs double-counts the shared information
and *overstates* the posterior. :func:`joint_update` uses a joint LR instead.
"""
from __future__ import annotations
import numpy as np
from .priors import prob_to_odds, odds_to_prob

__all__ = ["update_odds", "naive_update", "joint_update"]


def update_odds(prior_odds, lrs):
    """Multiply prior odds by one or more likelihood ratios."""
    return np.asarray(prior_odds, float) * np.prod(np.atleast_1d(lrs))


def naive_update(prior_pd, lrs):
    """Posterior PD assuming the signals are independent (product of LRs)."""
    o = prob_to_odds(prior_pd) * np.prod(np.atleast_1d(lrs))
    return odds_to_prob(o)


def joint_update(prior_pd, joint_lr):
    """Posterior PD using a single joint likelihood ratio (correlation-aware)."""
    o = prob_to_odds(prior_pd) * float(joint_lr)
    return odds_to_prob(o)
