"""Backtest engine for a CDS-cash basis strategy.

A basis position is the negative-basis package (long bond + long protection)
or its reverse.  We model the daily P&L of the package as driven by changes
in the *basis* itself (the package is, by construction, first-order neutral
to the common credit and rates moves), plus a carry term, minus transaction
costs charged on position turnover.

This is a research backtest, not an execution simulator: it captures the
economics that matter for evaluating the signal (mean-reversion capture,
carry, costs, turnover) while remaining transparent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Backtest economics.

    The package P&L is driven by basis moves through ``basis_dv01``: the dollar
    P&L per 1bp basis change per $1mm of notional.  For a ~5y IG package this
    is the risky annuity times notional times 1bp, i.e. roughly

        DV01 ≈ notional * annuity(5y) * 1e-4 ≈ 1e6 * 4.5 * 1e-4 ≈ $450/bp.

    Transaction costs are charged as ``tcost_basis_bps`` *basis points of
    spread* per unit of turnover, converted to dollars through the same DV01 —
    this is the correct unit (you cross a bid/offer measured in spread, not in
    notional), and keeps costs commensurate with the P&L they erode.
    """

    notional: float = 1.0e6           # per-name notional
    basis_dv01: float = 450.0         # $ P&L per 1bp basis move per $1mm (≈ 5y annuity)
    tcost_basis_bps: float = 0.5      # round-trip cost in bps of *spread* per turnover unit
    carry_bps_per_year: float = 0.0   # optional carry on the package (bps of spread/yr)
    trading_days: int = 252


@dataclass
class BacktestResult:
    pnl: pd.Series                   # daily P&L summed across names
    equity: pd.Series                # cumulative P&L
    by_name_pnl: pd.DataFrame        # daily P&L per name
    positions: pd.DataFrame          # daily position per name
    stats: dict = field(default_factory=dict)


def _per_name_pnl(
    g: pd.DataFrame, basis_col: str, cfg: BacktestConfig
) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    # Trade on next day's move to avoid look-ahead: position known at close t
    # earns the basis change from t to t+1.
    basis = g[basis_col].to_numpy()
    pos = g["position"].to_numpy()
    # Next-day move in bps; final row has no forward move (set to 0, not repeat).
    dbasis_bps = np.empty_like(basis)
    dbasis_bps[:-1] = np.diff(basis) * 1e4
    dbasis_bps[-1] = 0.0

    scale = cfg.basis_dv01 * (cfg.notional / 1e6)

    # P&L sign: position +1 means we expect basis to RISE, so we profit from a
    # positive basis change.
    pnl = pos * dbasis_bps * scale

    # Carry accrues (in bps of spread per year) while in a position
    carry = np.abs(pos) * cfg.carry_bps_per_year / cfg.trading_days * scale

    # Transaction costs: bps of spread per turnover unit, valued through DV01
    turnover = np.abs(np.diff(pos, prepend=0.0))
    tcost = turnover * cfg.tcost_basis_bps * scale

    g["pnl"] = pnl + carry - tcost
    g["turnover"] = turnover
    return g[["date", "name", "position", "pnl", "turnover"]]


def run_backtest(
    panel: pd.DataFrame, basis_col: str = "basis", config: BacktestConfig | None = None
) -> BacktestResult:
    """Run the basis strategy backtest over a signalled panel.

    The panel must contain ``date``, ``name``, ``position`` and ``basis_col``.
    """
    cfg = config or BacktestConfig()
    required = {"date", "name", "position", basis_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns for backtest: {sorted(missing)}")

    parts = [
        _per_name_pnl(g, basis_col, cfg)
        for _, g in panel.groupby("name")
    ]
    detail = pd.concat(parts, ignore_index=True)

    pnl = detail.groupby("date")["pnl"].sum().sort_index()
    equity = pnl.cumsum()
    by_name = detail.pivot_table(index="date", columns="name", values="pnl")
    positions = detail.pivot_table(index="date", columns="name", values="position")

    stats = _performance_stats(pnl, cfg)
    return BacktestResult(
        pnl=pnl,
        equity=equity,
        by_name_pnl=by_name,
        positions=positions,
        stats=stats,
    )


def _performance_stats(pnl: pd.Series, cfg: BacktestConfig) -> dict:
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    ann_factor = np.sqrt(cfg.trading_days)
    sharpe = float(mu / sd * ann_factor) if sd > 0 else float("nan")
    cum = pnl.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    return {
        "total_pnl": float(pnl.sum()),
        "ann_pnl": float(mu * cfg.trading_days),
        "ann_vol": float(sd * ann_factor),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((pnl > 0).mean()),
        "n_days": int(len(pnl)),
    }
