"""Dynamics of the bank CET1 ratio and the loss-absorption trigger times.

The CET1 ratio :math:`C_t` (in percentage points) follows a mean-reverting
jump-diffusion,

.. math::

    dC_t = \\kappa(\\theta - C_t)\\,dt + \\sigma\\,dW_t - J_t\\,dN_t,

where :math:`N_t` is a Poisson process of stress events and each jump
:math:`J_t > 0` is a lognormal *downward* shock to capital (recapitalisations
are handled by the mean reversion, not by upward jumps).  Two absorption
events are tracked per path:

* **mechanical trigger** — first time :math:`C_t \\le c_{\\text{trig}}`
  (contractual write-down / conversion trigger, e.g. 5.125% or 7%);
* **PONV** — an independent hazard ``ponv_intensity`` under which the
  resolution authority writes the instrument down before the mechanical
  trigger is reached.

The simulator is deterministic given a :class:`Randoms` bundle, so
bump-and-reprice risk uses *common random numbers* and is free of Monte-Carlo
noise on the sensitivity itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CET1Params:
    """Risk-neutral (pricing) parameters of the CET1 ratio process.

    All ratio quantities are in percentage points (e.g. ``12.0`` == 12%).
    """

    c0: float              # initial CET1 ratio
    kappa: float           # mean-reversion speed (per year)
    theta: float           # long-run capital target
    sigma: float           # diffusion volatility (pp / sqrt(year))
    jump_intensity: float  # annual Poisson rate of stress jumps
    jump_median: float     # median downward jump size (pp)
    jump_vol: float        # lognormal vol of the jump size
    ponv_intensity: float  # constant PONV hazard (per year)

    def __post_init__(self) -> None:
        for name in ("sigma", "jump_intensity", "jump_median", "jump_vol", "ponv_intensity", "kappa"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class Randoms:
    """Pre-drawn primitive randoms, shared across a bump-and-reprice run.

    Shapes are ``(n_paths, n_steps)``.  Keeping the *uniforms* fixed while a
    parameter (e.g. ``jump_intensity``) is bumped makes the event structure
    monotone in the parameter, which is what common-random-number Greeks need.
    """

    normals: np.ndarray       # diffusion Brownian increments (standard normal)
    jump_unif: np.ndarray     # jump-occurrence uniforms
    jump_norm: np.ndarray     # jump-size standard normals
    ponv_unif: np.ndarray     # PONV-occurrence uniforms

    @property
    def n_paths(self) -> int:
        return self.normals.shape[0]

    @property
    def n_steps(self) -> int:
        return self.normals.shape[1]


def draw_randoms(n_paths: int, n_steps: int, seed: int | None = 0) -> Randoms:
    """Draw a reusable :class:`Randoms` bundle."""
    rng = np.random.default_rng(seed)
    return Randoms(
        normals=rng.standard_normal((n_paths, n_steps)),
        jump_unif=rng.random((n_paths, n_steps)),
        jump_norm=rng.standard_normal((n_paths, n_steps)),
        ponv_unif=rng.random((n_paths, n_steps)),
    )


@dataclass
class CET1Paths:
    """Output of :func:`simulate_cet1`."""

    paths: np.ndarray          # (n_paths, n_steps + 1) CET1 ratio
    trigger_idx: np.ndarray    # first absorption step index; -1 if none
    trigger_type: np.ndarray   # 0 none, 1 mechanical, 2 PONV
    dt: float

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.paths.shape[1]) * self.dt

    @property
    def trigger_time(self) -> np.ndarray:
        """Absorption time per path; ``+inf`` where no absorption occurred."""
        t = np.where(self.trigger_idx < 0, np.inf, self.trigger_idx * self.dt)
        return t


def simulate_cet1(
    params: CET1Params,
    trigger_level: float,
    horizon: float,
    dt: float,
    randoms: Randoms,
) -> CET1Paths:
    """Simulate CET1 paths and record the first loss-absorption event.

    Parameters
    ----------
    params:
        CET1 process parameters.
    trigger_level:
        Mechanical trigger :math:`c_{\\text{trig}}` (percentage points).
    horizon:
        Simulation horizon in years.  The Euler grid has
        ``round(horizon / dt)`` steps.
    dt:
        Time step (years).
    randoms:
        Pre-drawn randoms; its ``n_steps`` must be at least the grid size.
    """
    n_steps = int(round(horizon / dt))
    if randoms.n_steps < n_steps:
        raise ValueError(
            f"randoms has {randoms.n_steps} steps but {n_steps} are required"
        )
    n_paths = randoms.n_paths
    sqrt_dt = np.sqrt(dt)

    p_jump = 1.0 - np.exp(-params.jump_intensity * dt)
    p_ponv = 1.0 - np.exp(-params.ponv_intensity * dt)

    C = np.empty((n_paths, n_steps + 1), dtype=float)
    C[:, 0] = params.c0
    trigger_idx = np.full(n_paths, -1, dtype=np.int64)
    trigger_type = np.zeros(n_paths, dtype=np.int64)

    for n in range(n_steps):
        cn = C[:, n]
        drift = params.kappa * (params.theta - cn) * dt
        diffusion = params.sigma * sqrt_dt * randoms.normals[:, n]

        occurs = randoms.jump_unif[:, n] < p_jump
        # lognormal jump size with median `jump_median`
        size = params.jump_median * np.exp(
            params.jump_vol * randoms.jump_norm[:, n] - 0.5 * params.jump_vol**2
        )
        jump = occurs * size

        cnext = cn + drift + diffusion - jump
        C[:, n + 1] = cnext

        alive = trigger_idx < 0
        mech = alive & (cnext <= trigger_level)
        ponv = alive & (~mech) & (randoms.ponv_unif[:, n] < p_ponv)

        idx = n + 1
        trigger_idx[mech] = idx
        trigger_type[mech] = 1
        trigger_idx[ponv] = idx
        trigger_type[ponv] = 2

    return CET1Paths(paths=C, trigger_idx=trigger_idx, trigger_type=trigger_type, dt=dt)
