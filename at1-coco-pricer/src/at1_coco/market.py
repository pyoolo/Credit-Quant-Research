"""Pricing real, dated AT1 bonds and calibrating the model to market prices.

The core engine (:mod:`at1_coco.pricer`) values a stylised note from t=0.
Real bonds need four extra things, all handled here:

1. **Dates.**  Valuation happens between coupon dates, so the model value is a
   *dirty* price; market clean prices are converted with 30/360 accrued.
2. **Call only on reset dates.**  BNP's USD AT1s are callable every 5 years,
   on the reset dates, not annually.
3. **Stochastic refinancing spread.**  The issuer calls when its reset margin is
   above what a new AT1 would cost.  That new-issue spread moves with the
   market, not only with the bank's capital, so it is simulated as a
   mean-reverting (OU) process correlated with CET1 shocks, plus the capital
   add-on of the core engine (wider when CET1 is inside the MDA buffer).
4. **Conversion recovery.**  At the 5.125% trigger these notes convert into
   shares at max(current price, floor).  Recovery is
   ``min(1, S_trigger / floor)`` with ``S_trigger`` a stressed share price.

PONV is an independent constant hazard ``lambda``.  Instead of simulating it,
the engine prices it *analytically*: every cash flow at time t is multiplied by
``exp(-lambda t)`` (conditional Monte Carlo).  This removes all Monte-Carlo
noise in the lambda direction, so the price is smooth and monotone in lambda
and the implied hazard can be solved with a root finder on one set of paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .instrument import mda_fraction
from .processes import CET1Params, Randoms, draw_randoms, simulate_cet1


# ---------------------------------------------------------------------------
# instrument
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatedAT1:
    """A perpetual fixed-rate resettable AT1 with real dates."""

    isin: str
    coupon: float                 # initial coupon, decimal
    first_reset: pd.Timestamp     # first call = first reset date
    reset_margin: float           # decimal, over the 5y reference rate
    coupon_freq: int = 2
    reset_freq_years: int = 5     # calls allowed on reset dates only
    trigger_level: float = 5.125
    floor_price_usd: float = 61.0
    notional: float = 100.0

    @classmethod
    def from_row(cls, isin: str, row: pd.Series) -> "DatedAT1":
        return cls(
            isin=isin,
            coupon=float(row["coupon_initial"]),
            first_reset=pd.Timestamp(row["first_reset_date"]),
            reset_margin=float(row["reset_margin"]),
            coupon_freq=int(row["coupon_freq"]),
            reset_freq_years=int(row["reset_freq_years"]),
            trigger_level=float(row["trigger_level_pct"]),
            floor_price_usd=float(row["floor_price_usd"]),
        )

    def coupon_dates(self, until: pd.Timestamp) -> pd.DatetimeIndex:
        """Coupon dates on the first-reset anniversary grid, back and forward."""
        months = 12 // self.coupon_freq
        n_back = 200
        start = self.first_reset - pd.DateOffset(months=months * n_back)
        dates = pd.date_range(start, until, freq=pd.DateOffset(months=months))
        return pd.DatetimeIndex(dates)

    def accrued(self, val_date: pd.Timestamp) -> float:
        """Accrued interest (30/360) per 100 at ``val_date``."""
        dates = self.coupon_dates(val_date + pd.DateOffset(years=1))
        last = dates[dates <= val_date][-1]
        d1, d2 = last, val_date
        days = 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (min(d2.day, 30) - min(d1.day, 30))
        return self.coupon * self.notional * days / 360.0

    def schedule(self, val_date: pd.Timestamp, horizon: float):
        """Year fractions (ACT/365.25) from ``val_date`` of future coupons and calls."""
        end = val_date + pd.DateOffset(days=int(horizon * 365.25))
        dates = self.coupon_dates(end)
        fut = dates[dates > val_date]
        t_cpn = np.asarray((fut - val_date).days / 365.25)
        is_reset = np.array(
            [(d >= self.first_reset) and ((d.year - self.first_reset.year) % self.reset_freq_years == 0)
             and d.month == self.first_reset.month for d in fut]
        )
        after_first_reset = np.asarray(fut > self.first_reset)
        return t_cpn, is_reset, after_first_reset


# ---------------------------------------------------------------------------
# market state and model settings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketState:
    """Everything observable on one valuation date for one bond."""

    val_date: pd.Timestamp
    clean_price: float
    discount_yield: float     # Treasury yield near the first call, s.a. decimal
    reset_reference: float    # proxy for the 5y CMT, decimal
    refi_spread: float        # current new-issue AT1 spread, decimal
    cet1: float               # latest published CET1 ratio (pp)
    share_usd: float          # BNP share price in USD


@dataclass(frozen=True)
class SpreadParams:
    """OU dynamics of the issuer's new-issue AT1 spread (decimals)."""

    kappa: float = 0.5        # mean reversion (per year)
    long_run: float = 0.0275  # long-run level
    vol: float = 0.0          # annual vol; 0 = deterministic (core-engine behaviour)
    rho: float = -0.3         # corr. with CET1 diffusion shocks (capital down -> spread up)
    capital_beta: float = 0.01  # extra spread per pp of CET1 below the buffer top


@dataclass(frozen=True)
class ModelSettings:
    cet1: CET1Params
    spread: SpreadParams
    mda_threshold: float = 5.64      # P1 + P2R (CET1 part)
    combined_buffer: float = 4.87    # CBR
    share_stress: float = 0.20       # share price at trigger as fraction of today's
    ponv_recovery: float = 0.0
    horizon: float = 40.0
    dt: float = 1.0 / 12.0


# ---------------------------------------------------------------------------
# simulation: expected cash-flow profile (lambda-free)
# ---------------------------------------------------------------------------
@dataclass
class CashFlowProfile:
    """Path-averaged cash flows; pricing any lambda is then a dot product."""

    t_cpn: np.ndarray
    e_cpn: np.ndarray           # E[coupon paid at t_cpn]
    t_grid: np.ndarray
    e_call: np.ndarray          # E[redemption at 100] on the grid
    e_trig: np.ndarray          # E[conversion recovery] on the grid
    e_end: np.ndarray           # P(life ends at grid time), for PONV recovery term
    p_survive: float
    tail_coupon: float
    diagnostics: dict

    def dirty_price(self, r_cont: float, lam: float, ponv_recovery: float = 0.0,
                    notional: float = 100.0) -> float:
        k = r_cont + lam
        pv = self.e_cpn @ np.exp(-k * self.t_cpn)
        pv += (self.e_call + self.e_trig) @ np.exp(-k * self.t_grid)
        H = self.t_grid[-1]
        pv += self.p_survive * (self.tail_coupon / k) * np.exp(-k * H)
        if ponv_recovery > 0 and lam > 0:
            # E[ int_0^T_end lam e^{-k s} ds ] * R * N
            cum = lam / k * (1 - np.exp(-k * self.t_grid))
            pv += ponv_recovery * notional * (self.e_end @ cum + self.p_survive * cum[-1])
        return float(pv)


def spread_paths(randoms: Randoms, sp: SpreadParams, s0: float, dt: float, n_steps: int) -> np.ndarray:
    """OU spread driven by a mix of the CET1 normals and an independent block."""
    if sp.vol == 0:
        t = np.arange(n_steps + 1) * dt
        return np.broadcast_to(sp.long_run + (s0 - sp.long_run) * np.exp(-sp.kappa * t),
                               (randoms.n_paths, n_steps + 1))
    rng = np.random.default_rng(12345)
    indep = rng.standard_normal((randoms.n_paths, n_steps))
    z = sp.rho * randoms.normals[:, :n_steps] + np.sqrt(1 - sp.rho**2) * indep
    # exact OU discretisation
    e = np.exp(-sp.kappa * dt)
    sd = sp.vol * np.sqrt((1 - e * e) / (2 * sp.kappa))
    s = np.empty((randoms.n_paths, n_steps + 1))
    s[:, 0] = s0
    for n in range(n_steps):
        s[:, n + 1] = sp.long_run + (s[:, n] - sp.long_run) * e + sd * z[:, n]
    return s


def cash_flow_profile(bond: DatedAT1, mkt: MarketState, ms: ModelSettings,
                      randoms: Randoms) -> CashFlowProfile:
    dt, H = ms.dt, ms.horizon
    n_steps = int(round(H / dt))
    cet1 = replace(ms.cet1, c0=mkt.cet1, ponv_intensity=0.0)   # PONV handled analytically
    sim = simulate_cet1(cet1, bond.trigger_level, H, dt, randoms)
    C = sim.paths
    n_paths = C.shape[0]
    big = np.iinfo(np.int64).max
    trig = np.where(sim.trigger_idx < 0, big, sim.trigger_idx)

    t_cpn, is_reset, after_reset = bond.schedule(mkt.val_date, H)
    cidx = np.minimum(np.round(t_cpn / dt).astype(np.int64), n_steps)

    # --- call decision on reset dates -------------------------------------
    S = spread_paths(randoms, ms.spread, mkt.refi_spread, dt, n_steps)
    top = ms.mda_threshold + ms.combined_buffer
    call_col = np.where(is_reset)[0]
    call_at = np.full(n_paths, big, dtype=np.int64)       # coupon index of the call
    if call_col.size:
        ci = cidx[call_col]
        shortfall = np.maximum(0.0, top - C[:, ci])
        refi = S[:, ci] + ms.spread.capital_beta * shortfall
        # issuer calls when the reset margin costs more than a new AT1, needs
        # the note alive, and (regulatory approval) CET1 above the MDA threshold
        ok = (bond.reset_margin > refi) & (ci[None, :] < trig[:, None]) & (C[:, ci] >= top)
        has = ok.any(axis=1)
        first = np.argmax(ok, axis=1)
        call_at[has] = call_col[first[has]]

    # --- coupons ---------------------------------------------------------
    rate = np.where(after_reset, mkt.reset_reference + bond.reset_margin, bond.coupon)
    cpn_amt = rate / bond.coupon_freq * bond.notional
    k = np.arange(t_cpn.size)
    live = (cidx[None, :] < trig[:, None]) & (k[None, :] <= call_at[:, None])
    frac = mda_fraction(C[:, cidx], ms.mda_threshold, ms.combined_buffer)
    e_cpn = (live * frac).mean(axis=0) * cpn_amt

    # --- redemption, conversion, survival ---------------------------------
    t_grid = np.arange(n_steps + 1) * dt
    e_call = np.zeros(n_steps + 1)
    e_trig = np.zeros(n_steps + 1)
    e_end = np.zeros(n_steps + 1)
    called = call_at < big
    call_grid = cidx[np.minimum(call_at, t_cpn.size - 1)]
    np.add.at(e_call, call_grid[called], bond.notional / n_paths)
    np.add.at(e_end, call_grid[called], 1.0 / n_paths)

    conv = (~called) & (trig < big)
    rec = min(1.0, ms.share_stress * mkt.share_usd / bond.floor_price_usd)
    np.add.at(e_trig, trig[conv], rec * bond.notional / n_paths)
    np.add.at(e_end, trig[conv], 1.0 / n_paths)
    p_surv = float(np.mean(~called & ~conv))

    diag = {
        "p_call_first": float(np.mean(called & (call_at == (call_col[0] if call_col.size else -1)))),
        "p_called": float(called.mean()),
        "p_conversion": float(conv.mean()),
        "p_never_called": p_surv,
        "avg_coupon_cut": float(1 - (frac * live).sum() / max(live.sum(), 1)),
        "recovery": rec,
    }
    tail = (mkt.reset_reference + bond.reset_margin) * bond.notional
    return CashFlowProfile(t_cpn, e_cpn, t_grid, e_call, e_trig, e_end, p_surv, tail, diag)


def cont_rate(y_semiannual: float) -> float:
    return 2.0 * np.log1p(y_semiannual / 2.0)


def implied_ponv(bond: DatedAT1, mkt: MarketState, ms: ModelSettings, randoms: Randoms,
                 profile: CashFlowProfile | None = None) -> tuple[float, CashFlowProfile]:
    """PONV hazard that reprices the bond to its market dirty price."""
    prof = profile or cash_flow_profile(bond, mkt, ms, randoms)
    target = mkt.clean_price + bond.accrued(mkt.val_date)
    r = cont_rate(mkt.discount_yield)

    def f(lam):
        return prof.dirty_price(r, lam, ms.ponv_recovery) - target

    if f(0.0) < 0:          # model already too cheap without PONV risk
        return float("nan"), prof
    return brentq(f, 0.0, 1.0, xtol=1e-7), prof


def default_randoms(n_paths: int = 20_000, horizon: float = 40.0, dt: float = 1 / 12,
                    seed: int = 7) -> Randoms:
    return draw_randoms(n_paths, int(round(horizon / dt)), seed=seed)
