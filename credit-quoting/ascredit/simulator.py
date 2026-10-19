"""
Synthetic data generators for the credit market-making simulation.

Two pieces:
  1. Spread path: mid credit spread as (optionally mean-reverting) diffusion.
  2. Order flow: Poisson arrivals with intensity that decays in the quoted
     distance from mid, lambda(delta) = A * exp(-k * delta).

Everything is synthetic by construction; this module exists so the rest of the
codebase reads as a *study of optimal quoting behaviour*, not a historical
backtest.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SpreadModel:
    """
    Mid credit spread dynamics. Two regimes:

    * driftless arithmetic BM (the AS baseline):   ds = sigma dW
    * Ornstein-Uhlenbeck (more realistic for spreads, which are mean-reverting
      and strictly positive in practice):          ds = kappa(theta - s) dt + sigma dW

    Spread is floored at a small positive value to stay economically sensible.
    """
    s0: float = 120.0       # initial spread (bp)
    sigma: float = 4.0      # bp / sqrt(year) -> scaled by dt below
    kappa: float = 0.0      # mean-reversion speed (0 => pure BM)
    theta: float = 120.0    # long-run spread (bp)
    floor: float = 1.0      # minimum spread (bp)

    def simulate(self, T: float, n_steps: int, rng: np.random.Generator) -> np.ndarray:
        dt = T / n_steps
        s = np.empty(n_steps + 1)
        s[0] = self.s0
        sqrt_dt = np.sqrt(dt)
        for i in range(n_steps):
            dw = rng.standard_normal() * sqrt_dt
            ds = self.kappa * (self.theta - s[i]) * dt + self.sigma * dw
            s[i + 1] = max(self.floor, s[i] + ds)
        return s


def fill_probability(delta: float, A: float, k: float, dt: float) -> float:
    """
    Probability of at least one fill in [t, t+dt] for a quote at distance delta
    (bp) from mid, given intensity lambda = A exp(-k delta).

    P(fill) = 1 - exp(-lambda dt). delta is clipped at 0 (quoting through the
    mid does not give >A intensity in this reduced-form model).
    """
    lam = A * np.exp(-k * max(delta, 0.0))
    return 1.0 - np.exp(-lam * dt)


def calibrate_k_from_halflife(half_distance_bp: float) -> float:
    """
    Convenience: choose k so that intensity halves every `half_distance_bp` bp
    of quote distance.  k = ln(2) / half_distance.
    """
    return np.log(2.0) / half_distance_bp
