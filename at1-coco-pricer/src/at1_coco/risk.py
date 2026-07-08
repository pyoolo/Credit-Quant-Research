"""Risk analytics: sensitivities, absorption term structure, loss distribution.

Sensitivities are computed by *bump-and-reprice under common random numbers*:
the same :class:`~at1_coco.processes.Randoms` bundle drives the base and bumped
runs, so the Monte-Carlo noise cancels and the finite-difference estimate is
stable at moderate path counts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .instrument import AT1Note
from .pricer import MonteCarloPricer, PricingResult
from .processes import CET1Params, Randoms, draw_randoms


def _reprice(pricer: MonteCarloPricer, randoms: Randoms) -> float:
    return pricer.price_on_paths(pricer.simulate(randoms)).price


def greeks(
    pricer: MonteCarloPricer,
    randoms: Randoms | None = None,
    n_paths: int = 50_000,
    seed: int | None = 0,
    bumps: dict[str, float] | None = None,
) -> dict[str, float]:
    """Central-difference sensitivities of the price.

    Returns a dict keyed by risk factor:

    * ``delta_cet1``      — dP / dC0            (per pp of initial CET1)
    * ``vega``            — dP / dsigma         (per pp/sqrt-y of capital vol)
    * ``jump_sensitivity``— dP / d(jump_intensity)
    * ``ponv_sensitivity``— dP / d(ponv_intensity)
    * ``dv01``            — dP / d(discount_rate), per 1bp
    * ``trigger_sensitivity`` — dP / d(trigger_level)
    * ``buffer_sensitivity``  — dP / d(combined_buffer).  Negative in general:
      a *wider* combined buffer requirement raises the CET1 level needed for
      unrestricted coupons, so it increases coupon-cancellation risk.
    """
    n_steps = int(round(pricer.horizon / pricer.dt))
    if randoms is None:
        randoms = draw_randoms(n_paths, n_steps, seed=seed)

    b = {
        "cet1": 0.25,          # pp
        "sigma": 0.10,         # pp/sqrt-y
        "jump_intensity": 0.01,
        "ponv_intensity": 0.005,
        "rate": 1e-4,          # 1 bp
        "trigger": 0.25,       # pp
        "buffer": 0.25,        # pp
    }
    if bumps:
        b.update(bumps)

    p = pricer.params
    out: dict[str, float] = {}

    def deriv(down_pricer, up_pricer, h):
        return (_reprice(up_pricer, randoms) - _reprice(down_pricer, randoms)) / (2 * h)

    # delta to initial CET1
    out["delta_cet1"] = deriv(
        _with_params(pricer, replace(p, c0=p.c0 - b["cet1"])),
        _with_params(pricer, replace(p, c0=p.c0 + b["cet1"])),
        b["cet1"],
    )
    # vega to capital volatility
    out["vega"] = deriv(
        _with_params(pricer, replace(p, sigma=max(0.0, p.sigma - b["sigma"]))),
        _with_params(pricer, replace(p, sigma=p.sigma + b["sigma"])),
        b["sigma"],
    )
    # jump-intensity sensitivity
    out["jump_sensitivity"] = deriv(
        _with_params(pricer, replace(p, jump_intensity=max(0.0, p.jump_intensity - b["jump_intensity"]))),
        _with_params(pricer, replace(p, jump_intensity=p.jump_intensity + b["jump_intensity"])),
        b["jump_intensity"],
    )
    # PONV-intensity sensitivity
    out["ponv_sensitivity"] = deriv(
        _with_params(pricer, replace(p, ponv_intensity=max(0.0, p.ponv_intensity - b["ponv_intensity"]))),
        _with_params(pricer, replace(p, ponv_intensity=p.ponv_intensity + b["ponv_intensity"])),
        b["ponv_intensity"],
    )
    # discount-rate DV01 (per bp)
    out["dv01"] = deriv(
        _with_rate(pricer, pricer.discount_rate - b["rate"]),
        _with_rate(pricer, pricer.discount_rate + b["rate"]),
        b["rate"],
    ) * 1e-4
    # trigger-level sensitivity
    out["trigger_sensitivity"] = deriv(
        _with_note(pricer, replace(pricer.note, trigger_level=pricer.note.trigger_level - b["trigger"])),
        _with_note(pricer, replace(pricer.note, trigger_level=pricer.note.trigger_level + b["trigger"])),
        b["trigger"],
    )
    # combined-buffer sensitivity (coupon-cancellation channel)
    out["buffer_sensitivity"] = deriv(
        _with_note(pricer, replace(pricer.note, combined_buffer=max(0.1, pricer.note.combined_buffer - b["buffer"]))),
        _with_note(pricer, replace(pricer.note, combined_buffer=pricer.note.combined_buffer + b["buffer"])),
        b["buffer"],
    )
    return out


def _with_params(pricer: MonteCarloPricer, params: CET1Params) -> MonteCarloPricer:
    return MonteCarloPricer(pricer.note, params, pricer.discount_rate, pricer.horizon, pricer.dt)


def _with_note(pricer: MonteCarloPricer, note: AT1Note) -> MonteCarloPricer:
    return MonteCarloPricer(note, pricer.params, pricer.discount_rate, pricer.horizon, pricer.dt)


def _with_rate(pricer: MonteCarloPricer, rate: float) -> MonteCarloPricer:
    return MonteCarloPricer(pricer.note, pricer.params, rate, pricer.horizon, pricer.dt)


def trigger_term_structure(
    pricer: MonteCarloPricer,
    tenors: np.ndarray | None = None,
    n_paths: int = 50_000,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative probability of loss absorption by each tenor (years)."""
    if tenors is None:
        tenors = np.arange(1, int(pricer.horizon) + 1, dtype=float)
    n_steps = int(round(pricer.horizon / pricer.dt))
    randoms = draw_randoms(n_paths, n_steps, seed=seed)
    sim = pricer.simulate(randoms)
    tta = sim.trigger_time
    probs = np.array([np.mean(tta <= t) for t in tenors])
    return tenors, probs


def loss_distribution(
    result: PricingResult, par: float = 100.0, alpha: float = 0.99
) -> dict[str, float]:
    """Summary of the discounted-payoff loss distribution.

    Loss is defined relative to par: ``loss = par - PV``.  Positive VaR/ES are
    losses; a negative VaR means the α-quantile outcome is still a gain.
    """
    pv = result.pv_paths
    loss = par - pv
    var = float(np.quantile(loss, alpha))
    es = float(loss[loss >= var].mean()) if np.any(loss >= var) else var
    return {
        "expected_loss": float(loss.mean()),
        f"VaR_{int(alpha * 100)}": var,
        f"ES_{int(alpha * 100)}": es,
        "worst_loss": float(loss.max()),
        "prob_loss": float(np.mean(loss > 0)),
    }
