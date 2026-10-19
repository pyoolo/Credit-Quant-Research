"""
Avellaneda-Stoikov optimal market-making model, adapted to credit (spread space).

Reference
---------
Avellaneda, M. and Stoikov, S. (2008). "High-frequency trading in a limit
order book." Quantitative Finance, 8(3), 217-224.

Credit adaptation
------------------
The canonical model is written in *price* space for an equity-like asset whose
mid follows arithmetic Brownian motion. For a corporate bond, the natural state
variable is the credit *spread* s_t (e.g. the par CDS spread or the bond's
asset-swap spread, in basis points). We model the spread as the risk driver and
quote bid/ask *spreads* around the mid spread. Because dealer P&L is driven by
the bond *price*, and price is locally linear in spread through (minus) spread
duration DV01, the inventory risk term inherits a duration scaling.

Conventions
-----------
* s_t       : mid credit spread (bp). Risk-neutral-style reference; we do not
              impose drift (delta-hedged book assumption).
* sigma     : volatility of the spread, in bp per sqrt(unit time).
* q         : signed inventory in "DV01 units" (risk-equivalent face). q>0 means
              the dealer is long credit risk (long the bond / short protection).
* gamma     : inventory risk-aversion (CARA coefficient).
* k         : order-flow decay. Fill intensity lambda = A * exp(-k * delta),
              where delta is the quoted half-spread distance from mid (bp).
* A         : baseline arrival intensity at delta=0.
* T         : terminal time (end of trading session); t in [0, T].

All formulas below are the closed-form approximations from AS Section 3, with
sign/scaling adjusted so that increasing credit spread *reduces* the value of a
long-credit inventory (a dealer long the bond loses when spreads widen).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ASParams:
    gamma: float          # risk aversion
    sigma: float          # spread vol (bp / sqrt(time))
    k: float              # order-flow decay (1/bp)
    A: float              # baseline intensity
    T: float = 1.0        # session horizon
    dv01: float = 1.0     # price sensitivity per bp (scales risk; >0)

    def __post_init__(self):
        assert self.gamma > 0, "gamma must be positive"
        assert self.sigma >= 0
        assert self.k > 0 and self.A > 0
        assert self.T > 0 and self.dv01 > 0


def reservation_spread(s: float, q: float, t: float, p: ASParams) -> float:
    """
    Indifference (reservation) spread r(s, q, t).

    AS eq. (with credit/duration scaling):
        r = s + q * gamma * (sigma_eff)^2 * (T - t)

    where sigma_eff = dv01 * sigma converts spread vol into price-risk vol that
    the CARA dealer actually cares about.

    Sign in spread space: spread moves *inversely* to price. A long-credit
    dealer (q>0) wants to *reduce* inventory, i.e. sell the bond. Selling the
    bond happens when a client lifts the dealer's ask (the tighter-spread side).
    Shifting the reservation spread *up* (r > s) widens the bid distance
    (b - mid grows -> fewer fills that lengthen the book) and tightens the ask
    distance (mid - a shrinks -> more fills that flatten the book). That is the
    correct skew: a long dealer attracts sellers-to-them less and buyers-from-
    them more, bleeding inventory back toward zero. Short dealer: mirror image.
    """
    sigma_eff2 = (p.dv01 * p.sigma) ** 2
    return s + q * p.gamma * sigma_eff2 * (p.T - t)


def optimal_half_spread(t: float, p: ASParams) -> float:
    """
    Optimal total quoted spread delta_a + delta_b (AS closed form):

        delta_total = gamma * sigma_eff^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    Returned value is the *total* width; bid/ask half-distances are obtained by
    splitting around the reservation spread.
    """
    sigma_eff2 = (p.dv01 * p.sigma) ** 2
    return p.gamma * sigma_eff2 * (p.T - t) + (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.k)


def optimal_quotes(s: float, q: float, t: float, p: ASParams) -> tuple[float, float]:
    """
    Return (bid_spread, ask_spread) in bp.

    In credit/spread space a *buyer of the bond* (provider of liquidity on the
    dealer's bid) transacts at a spread *wider* than mid (cheaper price), and a
    *seller* lifts the dealer's ask at a *tighter* spread. So:

        ask_quote (dealer buys bond / sells protection) = r - delta/2  (tighter)
        bid_quote (dealer sells bond / buys protection)  = r + delta/2  (wider)

    We return them named by the side of the *bond* market the client hits.
    bid_spread > ask_spread by construction (spread is inverse to price).
    """
    r = reservation_spread(s, q, t, p)
    delta = optimal_half_spread(t, p)
    bid_spread = r + delta / 2.0   # client sells bond to dealer (dealer gets longer)
    ask_spread = r - delta / 2.0   # client buys bond from dealer (dealer gets shorter)
    return bid_spread, ask_spread


def symmetric_quotes(s: float, t: float, p: ASParams) -> tuple[float, float]:
    """Inventory-agnostic benchmark: quote symmetric around the true mid."""
    delta = optimal_half_spread(t, p)
    return s + delta / 2.0, s - delta / 2.0
