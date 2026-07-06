"""
political_option
================

Reference implementation of the framework in

    "When Sovereign Debt Becomes a Political Option"
    P. Mastrogiacomo, EDHEC Business School (2026)

Defaulted sovereign bonds are modelled as *political contingent claims*:
a baseline distressed value R_low plus an embedded cash-or-nothing digital
payoff (R_high,i(t) - R_low) triggered by a sovereign-level political
normalization event tau (paper, Secs. 6-8).

Modules
-------
data           Synthetic post-default price generator + CSV loaders
curves         Model-free cross-maturity summary measures (Sec. 4.2, 5.2)
contingent     Implied normalization probabilities, benchmark calibration (Sec. 8-9)
hazard         Constant and piecewise-constant implied hazard rates (Sec. 8.3)
monotonicity   Cross-maturity monotonicity restriction and violation mass V(t) (Sec. 9.2)
events         Event-study machinery: abnormal returns and CARs (Sec. 5.3)
regressions    HAC macro-financial falsification regressions (Sec. 5.4, App. B)
plotting       Publication-quality replications of Figures 1-8
"""

from .contingent import (
    implied_probability,
    price_from_probability,
    implied_probability_panel,
    calibrate_benchmarks,
)
from .hazard import implied_hazard, hazard_term_structure
from .monotonicity import violation_mass, violation_series
from .events import abnormal_returns, cumulative_abnormal_returns, event_study
from .curves import normalize_prices, curve_summary, level_factor
from .regressions import hac_regression, macro_falsification, post_default_interactions
from .data import simulate_default_panel, load_price_panel

__version__ = "0.1.0"

__all__ = [
    "implied_probability",
    "price_from_probability",
    "implied_probability_panel",
    "calibrate_benchmarks",
    "implied_hazard",
    "hazard_term_structure",
    "violation_mass",
    "violation_series",
    "abnormal_returns",
    "cumulative_abnormal_returns",
    "event_study",
    "normalize_prices",
    "curve_summary",
    "level_factor",
    "hac_regression",
    "macro_falsification",
    "post_default_interactions",
    "simulate_default_panel",
    "load_price_panel",
]
