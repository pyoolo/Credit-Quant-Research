"""Fixed-rate bond pricing, Z-spread and (par-par) asset-swap spread.

The asset-swap spread is the spread that an asset-swap package pays over
the floating leg so that the package is worth par at inception.  We use the
standard par-par approximation, which is the market convention for
investment-grade names trading near par.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curves import DiscountCurve


@dataclass
class Bond:
    """A fixed-rate coupon bond.

    Parameters
    ----------
    maturity:
        Time to maturity in years.
    coupon:
        Annual coupon rate (e.g. 0.05 for 5%).
    freq:
        Coupon payments per year.
    face:
        Redemption / notional amount.
    """

    maturity: float
    coupon: float
    freq: int = 2
    face: float = 100.0

    def cashflow_times(self) -> np.ndarray:
        n = int(round(self.maturity * self.freq))
        if n < 1:
            raise ValueError("bond must have at least one coupon period")
        return np.array([(i + 1) / self.freq for i in range(n)])

    def cashflows(self) -> np.ndarray:
        times = self.cashflow_times()
        cpn = self.coupon / self.freq * self.face
        cf = np.full(times.shape, cpn)
        cf[-1] += self.face
        return cf

    def price_from_curve(self, curve: DiscountCurve, spread: float = 0.0) -> float:
        """Dirty price discounting at curve zero rates plus a constant ``spread``.

        ``spread`` is the Z-spread (continuously compounded) added to every
        zero rate.
        """
        times = self.cashflow_times()
        cf = self.cashflows()
        z = curve.zero_rate(times) + spread
        df = np.exp(-z * times)
        return float(np.dot(cf, df))

    def z_spread(self, curve: DiscountCurve, market_price: float) -> float:
        """Solve for the Z-spread that reprices the bond to ``market_price``."""

        def obj(s: float) -> float:
            return self.price_from_curve(curve, s) - market_price

        return float(brentq(obj, -0.2, 0.5, xtol=1e-10))

    def asset_swap_spread(self, curve: DiscountCurve, market_price: float) -> float:
        """Par-par asset-swap spread.

        ASW = (coupon-leg PV at par - (market dirty price - par)) relationship,
        reduced to the standard closed form:

            ASW = (PV_fixed - P_market) / annuity

        where PV_fixed prices the fixed leg off the risk-free curve and the
        annuity is the PV of 1bp on the (assumed) floating schedule.
        """
        times = self.cashflow_times()
        cpn = self.coupon / self.freq * self.face
        df = curve.discount_factor(times)
        # PV of fixed coupons + redemption off the risk-free curve
        pv_fixed = cpn * df.sum() + self.face * df[-1]
        # Annuity (PV of 1 unit of spread per year) on the floating schedule
        accruals = np.diff(np.concatenate(([0.0], times)))
        annuity = float(np.dot(accruals, df)) * self.face
        return float((pv_fixed - market_price) / annuity)
