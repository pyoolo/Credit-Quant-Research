"""End-to-end example: run the CDS-cash basis pipeline and report results.

Run with:
    python examples/run_example.py

Generates a synthetic IG panel, computes the basis through the pricing layer,
builds mean-reversion signals, backtests the strategy and prints risk metrics.
If matplotlib is installed it also saves an equity-curve plot.
"""
from __future__ import annotations

import pandas as pd

from cds_basis import run_pipeline
from cds_basis.backtest import BacktestConfig
from cds_basis.signal import SignalConfig


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    result = run_pipeline(
        signal_config=SignalConfig(lookback=60, entry_z=2.0, exit_z=0.5),
        backtest_config=BacktestConfig(notional=1e6, tcost_basis_bps=0.5),
    )

    print("=" * 60)
    print("CDS-CASH BASIS STRATEGY — INVESTMENT GRADE (synthetic)")
    print("=" * 60)

    print("\nBacktest performance")
    print("-" * 60)
    for k, v in result.backtest.stats.items():
        print(f"  {k:<16}: {v:,.4f}")

    print("\nRisk summary")
    print("-" * 60)
    for k, v in result.risk.items():
        print(f"  {k:<18}: {v:,.2f}")

    # Optional plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        result.backtest.equity.plot(ax=ax, color="#1f4e79", lw=1.5)
        ax.set_title("Cumulative P&L — CDS-cash basis (IG)")
        ax.set_ylabel("P&L ($)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("equity_curve.png", dpi=120)
        print("\nSaved equity_curve.png")
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
