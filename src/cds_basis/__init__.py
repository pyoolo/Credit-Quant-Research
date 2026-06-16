"""cds_basis: a quantitative toolkit for trading the CDS-cash basis (IG).

Pipeline: pricing -> signal -> backtest -> risk.

Quick start
-----------
>>> from cds_basis import run_pipeline
>>> result = run_pipeline()
>>> print(result.backtest.stats["sharpe"])
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import data, pricing, risk, signal
from .backtest import BacktestConfig, BacktestResult, run_backtest
from .pricing import CDS, Bond, CashSpread, DiscountCurve, compute_basis
from .signal import SignalConfig, build_signals

__version__ = "0.1.0"

__all__ = [
    "data",
    "pricing",
    "signal",
    "risk",
    "run_pipeline",
    "PipelineResult",
    "add_basis_column",
]


def add_basis_column(
    panel: pd.DataFrame,
    curve: DiscountCurve | None = None,
    cash_spread_type: CashSpread = CashSpread.ASW,
) -> pd.DataFrame:
    """Compute the per-row CDS-cash basis and append it as a ``basis`` column.

    Uses the pricing layer to turn each (bond_price, cds_spread) observation
    into an asset-swap (or Z-) spread and the resulting basis.
    """
    curve = curve or DiscountCurve.flat(0.03)
    out = panel.copy()
    bases = []
    for row in out.itertuples(index=False):
        bond = Bond(maturity=row.maturity, coupon=row.coupon)
        cds = CDS(maturity=row.maturity, recovery=row.recovery)
        res = compute_basis(
            curve, bond, row.bond_price, cds, row.cds_spread, cash_spread_type
        )
        bases.append(res.basis)
    out["basis"] = bases
    return out


@dataclass
class PipelineResult:
    panel: pd.DataFrame
    backtest: BacktestResult
    risk: dict


def run_pipeline(
    panel: pd.DataFrame | None = None,
    curve: DiscountCurve | None = None,
    signal_config: SignalConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    recompute_basis: bool = False,
) -> PipelineResult:
    """Run the full pricing -> signal -> backtest -> risk pipeline.

    If ``panel`` is None a synthetic IG panel is generated.  By default the
    synthetic panel already carries a clean ``basis`` series; set
    ``recompute_basis=True`` to re-derive it through the pricing layer
    (slower, exercises the full stack).
    """
    if panel is None:
        panel = data.generate_panel()
    data.validate_panel(panel)

    if "basis" not in panel.columns or recompute_basis:
        panel = add_basis_column(panel, curve)

    signalled = build_signals(panel, basis_col="basis", config=signal_config)
    bt = run_backtest(signalled, basis_col="basis", config=backtest_config)
    risk_summary = risk.summarise_risk(
        bt.pnl,
        positions=bt.positions,
        jtd_cfg=risk.JumpToDefaultConfig(),
    )
    return PipelineResult(panel=signalled, backtest=bt, risk=risk_summary)
