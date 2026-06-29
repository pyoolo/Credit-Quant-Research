"""
scripts/run_analysis.py
-----------------------
End-to-end runner: generates all figures and prints a summary table.

Usage:
    cd convertible-bond-pricer
    python scripts/run_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from models import (
    price_tsiveriotis_fernandes,
    price_bs_benchmark,
    ConvertibleBond, MarketParams,
)
from synthetic_data import (
    make_vanilla_cb,
    make_callable_cb,
    make_putable_cb,
    make_distressed_cb,
    make_equity_sensitive_cb,
    make_scenario_grid,
    make_convergence_data,
)
from plots import (
    plot_price_vs_stock,
    plot_price_vs_spread,
    plot_delta_vs_stock,
    plot_regime_scatter,
    plot_convergence,
    plot_price_vs_vol,
)


def run_summary_table():
    """
    Price all representative bonds under TF and BS, print comparison.
    """
    cases = {
        "Vanilla"          : make_vanilla_cb(),
        "Callable"         : make_callable_cb(),
        "Putable"          : make_putable_cb(),
        "Distressed"       : make_distressed_cb(),
        "Equity-Sensitive" : make_equity_sensitive_cb(),
    }

    rows = []
    for name, (cb, mkt) in cases.items():
        tf = price_tsiveriotis_fernandes(cb, mkt, N=300)
        bs = price_bs_benchmark(cb, mkt)
        rows.append({
            "Case"            : name,
            "S₀"              : mkt.S0,
            "σ"               : f"{mkt.sigma:.0%}",
            "Spread"          : f"{mkt.credit_spread:.0%}",
            "Conv. Value"     : f"{cb.conversion_ratio * mkt.S0:.1f}",
            "TF Price"        : f"{tf['price']:.2f}",
            "BS Price"        : f"{bs['price']:.2f}",
            "TF − BS"         : f"{tf['price'] - bs['price']:+.2f}",
            "TF Delta"        : f"{tf['delta']:.4f}",
            "Cash Fraction"   : f"{tf['bond_floor_proxy']:.2%}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  CONVERTIBLE BOND PRICING — SUMMARY TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print("\nNote: TF ≤ BS always (TF discounts equity cash flows at r+h).")
    print("Divergence largest in 'balanced' regime; small in equity/distressed extremes.\n")
    return df


def run_all_plots():
    print("\nGenerating figures...")
    cb, mkt = make_vanilla_cb()

    print("  [1/6] Price vs Stock (S-curve)...")
    plot_price_vs_stock(cb, mkt)

    print("  [2/6] Price vs Credit Spread...")
    plot_price_vs_spread(cb, mkt)

    print("  [3/6] Delta vs Stock...")
    plot_delta_vs_stock(cb, mkt)

    print("  [4/6] Regime scatter (synthetic grid)...")
    df_grid = make_scenario_grid(n_scenarios=200)
    plot_regime_scatter(df_grid)
    df_grid.to_csv(
        Path(__file__).parent.parent / "paper" / "figures" / "scenario_grid.csv",
        index=False,
    )
    print(f"  Regime counts:\n{df_grid['regime'].value_counts().to_string()}")

    print("  [5/6] Tree convergence...")
    df_conv = make_convergence_data(cb, mkt)
    plot_convergence(df_conv)

    print("  [6/6] Price vs Volatility...")
    plot_price_vs_vol(cb, mkt)

    print("\nAll figures saved to paper/figures/")


if __name__ == "__main__":
    run_summary_table()
    run_all_plots()
