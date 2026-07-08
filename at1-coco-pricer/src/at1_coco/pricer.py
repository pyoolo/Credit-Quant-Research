"""Monte-Carlo pricing engine for the AT1 note."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instrument import AT1Note, mda_fraction
from .processes import CET1Params, CET1Paths, Randoms, draw_randoms, simulate_cet1


@dataclass
class PricingResult:
    """Price and path-level diagnostics returned by :meth:`MonteCarloPricer.price`."""

    price: float
    std_error: float
    pv_paths: np.ndarray            # discounted payoff per path

    prob_absorption: float          # P(mechanical or PONV before call/horizon)
    prob_mechanical: float
    prob_ponv: float
    prob_called: float
    prob_survived: float            # reached horizon without call/absorption

    expected_coupons_pv: float      # PV of coupon leg
    expected_coupon_restriction: float  # 1 - avg paid fraction on live coupons
    expected_time_to_absorption: float  # conditional on absorption (years)

    def summary(self) -> str:
        lines = [
            f"price                     {self.price:8.3f}  (± {self.std_error:.3f})",
            f"P(absorption)             {self.prob_absorption:8.2%}",
            f"  P(mechanical trigger)   {self.prob_mechanical:8.2%}",
            f"  P(PONV)                 {self.prob_ponv:8.2%}",
            f"P(called)                 {self.prob_called:8.2%}",
            f"P(survived to horizon)    {self.prob_survived:8.2%}",
            f"coupon-leg PV             {self.expected_coupons_pv:8.3f}",
            f"avg coupon restriction    {self.expected_coupon_restriction:8.2%}",
            f"E[time to absorption]     {self.expected_time_to_absorption:8.2f} y",
        ]
        return "\n".join(lines)


class MonteCarloPricer:
    """Prices an :class:`AT1Note` on simulated :class:`CET1Paths`.

    The discount curve is flat (continuously-compounded ``discount_rate``).
    Coupons are cancellable through the MDA schedule; the issuer calls at the
    first call date on which refinancing is cheaper than the reset coupon.
    """

    def __init__(
        self,
        note: AT1Note,
        params: CET1Params,
        discount_rate: float = 0.03,
        horizon: float = 40.0,
        dt: float = 1.0 / 12.0,
    ) -> None:
        self.note = note
        self.params = params
        self.discount_rate = discount_rate
        self.horizon = horizon
        self.dt = dt

    # -- grid helpers -----------------------------------------------------
    def _grid_index(self, t: float | np.ndarray) -> np.ndarray:
        return np.round(np.asarray(t) / self.dt).astype(np.int64)

    def simulate(self, randoms: Randoms) -> CET1Paths:
        return simulate_cet1(
            self.params, self.note.trigger_level, self.horizon, self.dt, randoms
        )

    # -- pricing ----------------------------------------------------------
    def price_on_paths(self, sim: CET1Paths) -> PricingResult:
        note, r, dt = self.note, self.discount_rate, self.dt
        C = sim.paths
        n_paths = C.shape[0]
        n_steps = C.shape[1] - 1
        horizon_idx = n_steps

        trig_idx = np.where(sim.trigger_idx < 0, np.iinfo(np.int64).max, sim.trigger_idx)

        times = np.arange(n_steps + 1) * dt
        disc = np.exp(-r * times)

        # --- issuer call / extension decision -------------------------------
        call_times = note.call_times(self.horizon)
        first_call_idx = self._grid_index(note.first_call_year)
        redemption_idx = np.full(n_paths, np.iinfo(np.int64).max, dtype=np.int64)
        if call_times.size:
            call_idx = self._grid_index(call_times)                 # (n_calls,)
            Ccall = C[:, call_idx]                                  # (n_paths, n_calls)
            # refinancing spread widens as the capital buffer erodes
            shortfall = np.maximum(0.0, (note.mda_threshold + note.combined_buffer) - Ccall)
            refi_spread = note.base_refi_spread + note.refi_widen_beta * shortfall
            # call when keeping the reset coupon costs more than refinancing,
            # and only if the note is still alive at that call date
            call_flag = (note.reset_spread > refi_spread) & (call_idx[None, :] <= trig_idx[:, None])
            has_call = call_flag.any(axis=1)
            first_col = np.argmax(call_flag, axis=1)                # first True column
            redemption_idx[has_call] = call_idx[first_col[has_call]]

        # --- coupon leg -----------------------------------------------------
        coupon_times = note.coupon_times(self.horizon)
        coupon_idx = self._grid_index(coupon_times)
        pv = np.zeros(n_paths)
        coupon_pv = np.zeros(n_paths)
        restriction_num = np.zeros(n_paths)   # sum of (1 - frac) over live coupons
        restriction_den = np.zeros(n_paths)   # count of live coupons

        for cidx in coupon_idx:
            rate = note.coupon_rate if cidx <= first_call_idx else note.reset_coupon_rate
            cpn = rate / note.coupon_freq * note.notional
            frac = mda_fraction(C[:, cidx], note.mda_threshold, note.combined_buffer)
            # coupon is due if strictly before absorption and not past redemption
            live = (cidx < trig_idx) & (cidx <= redemption_idx)
            paid = live * frac * cpn * disc[cidx]
            pv += paid
            coupon_pv += paid
            restriction_num += live * (1.0 - frac)
            restriction_den += live

        # --- redemption / absorption payoffs --------------------------------
        called = (redemption_idx < np.iinfo(np.int64).max) & (redemption_idx <= trig_idx)
        pv[called] += note.notional * disc[np.clip(redemption_idx[called], 0, horizon_idx)]

        absorbed = (sim.trigger_idx >= 0) & (~called)
        pv[absorbed] += note.trigger_recovery * note.notional * disc[sim.trigger_idx[absorbed]]

        survived = (~called) & (~absorbed)
        # perpetual tail: value remaining reset coupons as a perpetuity at the
        # horizon (approximation; horizon is chosen long enough that disc is small)
        tail_pv = note.reset_coupon_rate * note.notional / note.effective_tail_yield
        pv[survived] += tail_pv * disc[horizon_idx]

        # --- aggregate ------------------------------------------------------
        price = float(pv.mean())
        std_error = float(pv.std(ddof=1) / np.sqrt(n_paths))

        prob_mech = float(np.mean((sim.trigger_type == 1) & absorbed))
        prob_ponv = float(np.mean((sim.trigger_type == 2) & absorbed))
        prob_abs = float(np.mean(absorbed))
        prob_called = float(np.mean(called))
        prob_surv = float(np.mean(survived))

        with np.errstate(invalid="ignore"):
            restr = np.where(restriction_den > 0, restriction_num / restriction_den, 0.0)
        expected_restriction = float(restr.mean())

        if absorbed.any():
            e_tta = float((sim.trigger_idx[absorbed] * dt).mean())
        else:
            e_tta = float("nan")

        return PricingResult(
            price=price,
            std_error=std_error,
            pv_paths=pv,
            prob_absorption=prob_abs,
            prob_mechanical=prob_mech,
            prob_ponv=prob_ponv,
            prob_called=prob_called,
            prob_survived=prob_surv,
            expected_coupons_pv=float(coupon_pv.mean()),
            expected_coupon_restriction=expected_restriction,
            expected_time_to_absorption=e_tta,
        )

    def price(
        self, n_paths: int = 50_000, seed: int | None = 0, randoms: Randoms | None = None
    ) -> PricingResult:
        """Draw paths (or reuse ``randoms``) and price the note."""
        n_steps = int(round(self.horizon / self.dt))
        if randoms is None:
            randoms = draw_randoms(n_paths, n_steps, seed=seed)
        sim = self.simulate(randoms)
        return self.price_on_paths(sim)
