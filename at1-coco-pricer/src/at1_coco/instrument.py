"""AT1 term sheet and the CRD-IV Maximum Distributable Amount (MDA) schedule."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mda_fraction(
    cet1: np.ndarray | float,
    mda_threshold: float,
    combined_buffer: float,
) -> np.ndarray:
    """Fraction of the coupon that may be paid under the MDA restriction.

    Under CRD IV, once CET1 falls inside the *combined buffer requirement* the
    Maximum Distributable Amount is capped by quartile:

    ========================  ===========================
    Position in buffer        Max distributable fraction
    ========================  ===========================
    top quartile   (75-100%)  60%
    third quartile (50-75%)   40%
    second quartile(25-50%)   20%
    bottom quartile(0-25%)    0%
    ========================  ===========================

    Above the buffer the coupon is unrestricted (100%); below the buffer floor
    (``mda_threshold``) distributions are fully blocked (0%).  This is the
    endogenous source of AT1 *coupon-cancellation risk*.

    Parameters
    ----------
    cet1:
        CET1 ratio(s) in percentage points.
    mda_threshold:
        Buffer floor = total capital requirement (Pillar 1 + Pillar 2R).
        Below it, MDA fully binds.
    combined_buffer:
        Width of the combined buffer requirement above the floor.
    """
    c = np.asarray(cet1, dtype=float)
    top = mda_threshold + combined_buffer
    q = (c - mda_threshold) / combined_buffer  # position within buffer, 0..1
    within = np.select(
        [q < 0.25, q < 0.50, q < 0.75],
        [0.0, 0.20, 0.40],
        default=0.60,
    )
    frac = np.where(c >= top, 1.0, np.where(c < mda_threshold, 0.0, within))
    return frac


@dataclass
class AT1Note:
    """Term sheet of a perpetual non-cumulative AT1 (contingent convertible).

    Rates are decimals (``0.06`` == 6%); ratios are percentage points
    (``5.125`` == 5.125%).
    """

    notional: float = 100.0
    coupon_rate: float = 0.06          # initial (fixed) coupon
    coupon_freq: int = 2               # coupons per year

    first_call_year: float = 5.0       # first call date
    call_freq_years: float = 1.0       # subsequent call frequency

    # Post-call coupon resets to reference_rate + reset_spread (fixed at issue).
    reference_rate: float = 0.03       # mid-swap / reference at reset
    reset_spread: float = 0.045        # contractual reset spread over reference

    # Issuer refinancing cost widens as capital erodes; call is exercised when
    # keeping the (reset) coupon is dearer than issuing anew.
    base_refi_spread: float = 0.040
    refi_widen_beta: float = 0.010     # extra spread per pp of buffer shortfall

    # Loss absorption
    trigger_level: float = 5.125       # mechanical CET1 trigger
    trigger_recovery: float = 0.0      # fraction of par recovered on absorption

    # MDA / coupon-cancellation geometry
    mda_threshold: float = 9.0         # buffer floor (total requirement)
    combined_buffer: float = 3.5       # combined buffer width above the floor

    # Perpetual tail: yield used to value coupons beyond the horizon if the
    # note is never called and never absorbed.
    tail_yield: float | None = None    # defaults to reference_rate + reset_spread + 1%

    def __post_init__(self) -> None:
        if self.trigger_level >= self.mda_threshold:
            raise ValueError("mechanical trigger should sit below the MDA floor")
        if not 0.0 <= self.trigger_recovery <= 1.0:
            raise ValueError("trigger_recovery must be in [0, 1]")

    @property
    def reset_coupon_rate(self) -> float:
        return self.reference_rate + self.reset_spread

    @property
    def effective_tail_yield(self) -> float:
        if self.tail_yield is not None:
            return self.tail_yield
        return self.reference_rate + self.reset_spread + 0.01

    def coupon_times(self, horizon: float) -> np.ndarray:
        step = 1.0 / self.coupon_freq
        n = int(np.floor(horizon / step + 1e-9))
        return np.arange(1, n + 1) * step

    def call_times(self, horizon: float) -> np.ndarray:
        if self.first_call_year > horizon:
            return np.array([])
        n = int(np.floor((horizon - self.first_call_year) / self.call_freq_years + 1e-9))
        return self.first_call_year + np.arange(n + 1) * self.call_freq_years
