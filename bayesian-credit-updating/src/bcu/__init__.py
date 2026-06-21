"""Bayesian Credit Updating (bcu).

A small research package showing that conditional credit pricing is Bayesian
updating: a spread-implied prior default probability is revised by the
likelihood ratios of new signals, with explicit handling of the correlation
between signals (where naive independent multiplication overstates risk).
"""
from . import priors, likelihood, updating, data, backtest

__version__ = "0.1.0"
__all__ = ["priors", "likelihood", "updating", "data", "backtest"]

try:                       # PyMC is an optional (heavy) dependency
    from . import bayesian_model
    __all__.append("bayesian_model")
except Exception:          # pragma: no cover
    bayesian_model = None
