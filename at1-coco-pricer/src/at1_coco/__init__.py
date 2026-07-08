"""at1_coco — Monte-Carlo valuation and risk for AT1 contingent-convertible bonds.

The instrument is priced as a contingent claim on a bank's CET1 capital ratio.
A single simulated state variable (the CET1 ratio) drives, jointly:

    * mechanical loss absorption  (CET1 crosses a hard trigger),
    * regulatory loss absorption  (Point of Non-Viability, a hazard),
    * coupon cancellation         (Maximum Distributable Amount buffer),
    * extension / call behaviour  (issuer refinancing economics).

See ``paper/at1_coco_note.tex`` for the model write-up.
"""

from .instrument import AT1Note
from .processes import CET1Params, simulate_cet1, draw_randoms
from .pricer import MonteCarloPricer, PricingResult
from .risk import greeks, trigger_term_structure, loss_distribution

__all__ = [
    "AT1Note",
    "CET1Params",
    "simulate_cet1",
    "draw_randoms",
    "MonteCarloPricer",
    "PricingResult",
    "greeks",
    "trigger_term_structure",
    "loss_distribution",
]

__version__ = "0.1.0"
