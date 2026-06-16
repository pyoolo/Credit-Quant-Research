"""Risk layer: VaR / Expected Shortfall, jump-to-default and DV01.

These are the risk metrics that matter for a basis book.  The market risk of
the *package* is small (it is basis-DV01, not credit-DV01), but the tail risk
is dominated by jump-to-default: if the name defaults, the bond and the CDS
do not perfectly offset because of recovery-rate uncertainty and the
cheapest-to-deliver option.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def historical_var(pnl: pd.Series, level: float = 0.99) -> float:
    """Historical VaR (positive number = loss) at confidence ``level``."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    q = np.quantile(pnl.dropna(), 1 - level)
    return float(-q)


def expected_shortfall(pnl: pd.Series, level: float = 0.99) -> float:
    """Historical Expected Shortfall (mean loss beyond VaR)."""
    x = pnl.dropna().to_numpy()
    threshold = np.quantile(x, 1 - level)
    tail = x[x <= threshold]
    if tail.size == 0:
        return float("nan")
    return float(-tail.mean())


def parametric_var(pnl: pd.Series, level: float = 0.99) -> float:
    """Gaussian parametric VaR."""
    from scipy.stats import norm

    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    z = norm.ppf(1 - level)
    return float(-(mu + sd * z))


@dataclass
class JumpToDefaultConfig:
    notional: float = 1.0e6
    bond_recovery: float = 0.40        # realised recovery on the bond
    cds_recovery: float = 0.40         # recovery assumed in CDS settlement
    bond_price: float = 100.0          # clean price held (per 100 face)


def jump_to_default_pnl(cfg: JumpToDefaultConfig) -> float:
    """Instantaneous P&L of a negative-basis package on an immediate default.

    Package = long bond (face = notional) + long protection (notional).
    On default:
        bond leg  -> recovers ``bond_recovery`` * face, lost (price - recovery)
        CDS leg   -> pays (1 - cds_recovery) * notional
    Net P&L (per notional) = (1 - cds_recovery) - (bond_price/100 - bond_recovery)

    The residual is driven by the recovery-basis (bond vs CDS recovery) and by
    the premium paid above recovery for the bond.  A perfectly hedged package
    has zero JTD only if bond_price/100 - bond_recovery == 1 - cds_recovery.
    """
    per_unit = (1.0 - cfg.cds_recovery) - (cfg.bond_price / 100.0 - cfg.bond_recovery)
    return float(per_unit * cfg.notional)


def basis_dv01_exposure(positions: pd.DataFrame, dv01_per_mm: float, notional: float) -> pd.Series:
    """Aggregate basis-DV01 of the book per date (P&L per 1bp basis move)."""
    gross = positions.abs().sum(axis=1)
    return gross * dv01_per_mm * (notional / 1e6)


def summarise_risk(
    pnl: pd.Series,
    positions: pd.DataFrame | None = None,
    dv01_per_mm: float = 450.0,
    notional: float = 1.0e6,
    jtd_cfg: JumpToDefaultConfig | None = None,
    level: float = 0.99,
) -> dict:
    """One-shot risk summary for a backtest result."""
    out = {
        f"hist_var_{int(level*100)}": historical_var(pnl, level),
        f"es_{int(level*100)}": expected_shortfall(pnl, level),
        f"param_var_{int(level*100)}": parametric_var(pnl, level),
    }
    if positions is not None:
        dv01 = basis_dv01_exposure(positions, dv01_per_mm, notional)
        out["mean_basis_dv01"] = float(dv01.mean())
        out["max_basis_dv01"] = float(dv01.max())
    if jtd_cfg is not None:
        out["jtd_pnl_per_name"] = jump_to_default_pnl(jtd_cfg)
    return out
