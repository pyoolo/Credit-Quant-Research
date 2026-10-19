"""Reduced-form CDS pricing (JP Morgan / ISDA-style approximation).

A flat hazard rate ``h`` defines survival probabilities ``Q(t) = exp(-h t)``.
The par spread equates the protection leg and the premium leg (including
accrued-on-default).  We also expose the inverse: bootstrapping a hazard
rate from a quoted par spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curves import DiscountCurve


@dataclass
class CDS:
    """A single-name CDS contract priced off a flat hazard rate.

    Parameters
    ----------
    maturity:
        Protection maturity in years.
    recovery:
        Assumed recovery rate (e.g. 0.4 for IG senior unsecured).
    freq:
        Premium payments per year.
    """

    maturity: float
    recovery: float = 0.4
    freq: int = 4

    def _grid(self) -> np.ndarray:
        n = int(round(self.maturity * self.freq))
        if n < 1:
            raise ValueError("CDS must have at least one premium period")
        return np.array([(i + 1) / self.freq for i in range(n)])

    def premium_leg_annuity(self, curve: DiscountCurve, hazard: float) -> float:
        """Risky annuity (PV of 1 unit of spread), with accrued-on-default."""
        times = self._grid()
        dt = np.diff(np.concatenate(([0.0], times)))
        df = curve.discount_factor(times)
        surv = np.exp(-hazard * times)
        surv_prev = np.exp(-hazard * np.concatenate(([0.0], times[:-1])))
        # Full-period survival term
        annuity = float(np.sum(dt * df * surv))
        # Accrued-on-default: half-period accrual on defaults within the period
        accrued = float(np.sum(0.5 * dt * df * (surv_prev - surv)))
        return annuity + accrued

    def protection_leg(self, curve: DiscountCurve, hazard: float) -> float:
        """PV of (1 - R) paid at default, integrated on the premium grid."""
        times = self._grid()
        df = curve.discount_factor(times)
        surv = np.exp(-hazard * times)
        surv_prev = np.exp(-hazard * np.concatenate(([0.0], times[:-1])))
        default_prob = surv_prev - surv
        return float((1.0 - self.recovery) * np.sum(df * default_prob))

    def par_spread(self, curve: DiscountCurve, hazard: float) -> float:
        """Par spread = protection leg / risky annuity."""
        annuity = self.premium_leg_annuity(curve, hazard)
        prot = self.protection_leg(curve, hazard)
        return float(prot / annuity)

    def hazard_from_spread(self, curve: DiscountCurve, spread: float) -> float:
        """Bootstrap the flat hazard rate implied by a quoted par spread."""

        def obj(h: float) -> float:
            return self.par_spread(curve, h) - spread

        return float(brentq(obj, 1e-8, 5.0, xtol=1e-12))

    @staticmethod
    def credit_triangle(spread: float, recovery: float) -> float:
        """Quick hazard approximation h ~= s / (1 - R)."""
        return spread / (1.0 - recovery)
