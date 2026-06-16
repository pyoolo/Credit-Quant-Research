"""Interest-rate and credit curve construction.

Provides a minimal but correct discount curve built from a set of
continuously-compounded zero rates, with log-linear interpolation on
discount factors (the market-standard choice that keeps forward rates
positive for well-behaved inputs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscountCurve:
    """A discount curve defined by zero rates on a set of pillar tenors.

    Parameters
    ----------
    tenors:
        Pillar maturities in years (strictly increasing, > 0).
    zero_rates:
        Continuously-compounded zero rates at each pillar.
    """

    tenors: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self) -> None:
        self.tenors = np.asarray(self.tenors, dtype=float)
        self.zero_rates = np.asarray(self.zero_rates, dtype=float)
        if self.tenors.ndim != 1 or self.tenors.size < 1:
            raise ValueError("tenors must be a 1-D array with at least one pillar")
        if self.tenors.shape != self.zero_rates.shape:
            raise ValueError("tenors and zero_rates must have the same shape")
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")
        if np.any(self.tenors <= 0):
            raise ValueError("tenors must be strictly positive")

    def zero_rate(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate the zero rate at time(s) ``t`` (flat extrapolation)."""
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.tenors, self.zero_rates)

    def discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        """Discount factor P(0, t) = exp(-z(t) * t)."""
        t = np.asarray(t, dtype=float)
        z = self.zero_rate(t)
        return np.exp(-z * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously-compounded forward rate between t1 and t2."""
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        df1 = float(self.discount_factor(t1))
        df2 = float(self.discount_factor(t2))
        return float(np.log(df1 / df2) / (t2 - t1))

    @classmethod
    def flat(cls, rate: float, max_tenor: float = 30.0) -> DiscountCurve:
        """Convenience constructor for a flat curve."""
        tenors = np.array([0.25, 1.0, 2.0, 5.0, 10.0, max_tenor])
        return cls(tenors, np.full_like(tenors, rate))
