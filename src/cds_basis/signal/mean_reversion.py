"""Signal layer: Ornstein-Uhlenbeck calibration and mean-reversion signals.

The basis time series is modelled as an OU process

    dX = kappa (theta - X) dt + sigma dW

calibrated per name by maximum likelihood (closed form via the AR(1)
representation).  Trading signals are z-scores of the basis relative to its
estimated equilibrium, with entry/exit bands.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    kappa: float       # mean-reversion speed (per year)
    theta: float       # long-run mean
    sigma: float       # diffusion (per sqrt-year)

    @property
    def half_life(self) -> float:
        """Time for a deviation to halve, in years."""
        return float(np.log(2.0) / self.kappa) if self.kappa > 0 else np.inf

    @property
    def stationary_std(self) -> float:
        return float(np.sqrt(self.sigma**2 / (2.0 * self.kappa)))


def calibrate_ou(series: pd.Series, dt: float = 1.0 / 252.0) -> OUParams:
    """Calibrate OU parameters by exact-likelihood AR(1) regression.

    X_{t+1} = a + b X_t + eps, with
        b = exp(-kappa dt)
        a = theta (1 - b)
        Var(eps) = sigma^2 (1 - b^2) / (2 kappa)
    """
    x = np.asarray(series.dropna(), dtype=float)
    if x.size < 10:
        raise ValueError("need at least 10 observations to calibrate OU")
    x0 = x[:-1]
    x1 = x[1:]
    # OLS slope/intercept
    b, a = np.polyfit(x0, x1, 1)
    b = float(np.clip(b, 1e-6, 1 - 1e-9))  # keep mean-reversion well-defined
    resid = x1 - (a + b * x0)
    var_resid = float(np.var(resid, ddof=2))

    kappa = -np.log(b) / dt
    theta = a / (1.0 - b)
    sigma = np.sqrt(var_resid * 2.0 * kappa / (1.0 - b**2))
    return OUParams(kappa=float(kappa), theta=float(theta), sigma=float(sigma))


@dataclass
class SignalConfig:
    lookback: int = 60          # rolling window (days) for z-score
    entry_z: float = 2.0        # enter when |z| exceeds this
    exit_z: float = 0.5         # exit when |z| falls back inside this
    use_ou_mean: bool = True    # center on OU theta vs rolling mean


def zscore_signal(basis: pd.Series, config: SignalConfig | None = None) -> pd.DataFrame:
    """Generate a mean-reversion signal from a single basis series.

    Returns a frame with the rolling z-score and a target position in
    {-1, 0, +1}, where +1 means "expect basis to rise" (long basis: the
    basis is below equilibrium, i.e. negative-basis entry).
    """
    cfg = config or SignalConfig()
    df = pd.DataFrame({"basis": basis})
    roll = df["basis"].rolling(cfg.lookback)
    mean = roll.mean()
    std = roll.std(ddof=1)
    df["z"] = (df["basis"] - mean) / std

    # State machine with hysteresis: hold until z crosses the exit band.
    pos = np.zeros(len(df))
    current = 0
    z = df["z"].to_numpy()
    for t in range(len(df)):
        if np.isnan(z[t]):
            pos[t] = 0
            current = 0
            continue
        if current == 0:
            if z[t] <= -cfg.entry_z:
                current = 1   # basis cheap -> expect rise
            elif z[t] >= cfg.entry_z:
                current = -1  # basis rich -> expect fall
        else:
            if abs(z[t]) <= cfg.exit_z:
                current = 0
        pos[t] = current
    df["position"] = pos
    return df


def build_signals(
    panel: pd.DataFrame, basis_col: str = "basis", config: SignalConfig | None = None
) -> pd.DataFrame:
    """Apply the z-score signal name-by-name across a long panel.

    The panel must contain ``date``, ``name`` and ``basis_col`` columns.
    Returns the panel with ``z`` and ``position`` columns added.
    """
    cfg = config or SignalConfig()
    out = []
    for _name, g in panel.sort_values("date").groupby("name"):
        sig = zscore_signal(g.set_index("date")[basis_col], cfg)
        g = g.set_index("date")
        g["z"] = sig["z"].to_numpy()
        g["position"] = sig["position"].to_numpy()
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)
