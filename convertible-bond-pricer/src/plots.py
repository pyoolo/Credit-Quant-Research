"""
plots.py
--------
All analysis charts for the convertible bond pricer.

Run:  python scripts/run_analysis.py
Output goes to: paper/figures/
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

FIGURE_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "tf" : {"color": "#1f77b4", "linewidth": 2.0, "label": "Tsiveriotis-Fernandes"},
    "bs" : {"color": "#ff7f0e", "linewidth": 2.0, "label": "Black-Scholes benchmark",
             "linestyle": "--"},
    "diff": {"color": "#d62728", "linewidth": 1.5, "label": "TF − BS (credit discount)"},
}


def _save(fig: plt.Figure, name: str):
    path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Price vs Stock Price (S-curve)
# ---------------------------------------------------------------------------

def plot_price_vs_stock(cb, base_mkt, N: int = 200):
    """
    Classic CB S-curve: shows equity regime, balanced, and distressed regimes.
    TF vs BS comparison — divergence widens as stock falls (credit risk matters).
    """
    from models import sensitivity_surface
    S_range = np.linspace(10, 150, 80)

    _, tf_prices = sensitivity_surface(cb, base_mkt, "S0", S_range, model="tf")
    _, bs_prices = sensitivity_surface(cb, base_mkt, "S0", S_range, model="bs")

    conv_line = cb.conversion_ratio * S_range
    bond_floor_approx = cb.face_value * np.exp(
        -(base_mkt.r + base_mkt.credit_spread) * cb.maturity
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(S_range, tf_prices, **STYLE["tf"])
    ax1.plot(S_range, bs_prices, **STYLE["bs"])
    ax1.plot(S_range, conv_line, color="green", linewidth=1.2, linestyle=":",
             label=f"Conversion value (×{cb.conversion_ratio})")
    ax1.axhline(bond_floor_approx, color="grey", linewidth=1.0, linestyle=":",
                label=f"Bond floor proxy (≈{bond_floor_approx:.0f})")
    ax1.axvline(base_mkt.S0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5,
                label=f"Base S₀ = {base_mkt.S0}")

    ax1.set_ylabel("CB Price (€)", fontsize=11)
    ax1.set_title("Convertible Bond Price vs Stock Price\n"
                  "Tsiveriotis-Fernandes vs Black-Scholes", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Lower panel: divergence TF - BS
    diff = tf_prices - bs_prices
    ax2.fill_between(S_range, diff, 0, alpha=0.3, color="#d62728")
    ax2.plot(S_range, diff, **STYLE["diff"])
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Stock Price S₀ (€)", fontsize=11)
    ax2.set_ylabel("TF − BS", fontsize=10)
    ax2.set_title("Credit Discount (TF is always ≤ BS)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "01_price_vs_stock")


# ---------------------------------------------------------------------------
# 2. Price vs Credit Spread
# ---------------------------------------------------------------------------

def plot_price_vs_spread(cb, base_mkt, N: int = 200):
    """
    Show how TF is sensitive to credit spread while BS is not
    (BS only uses spread for bond floor discounting, not the equity option).
    """
    from models import sensitivity_surface
    cs_range = np.linspace(0.001, 0.25, 60)

    _, tf_prices = sensitivity_surface(cb, base_mkt, "credit_spread", cs_range, "tf")
    _, bs_prices = sensitivity_surface(cb, base_mkt, "credit_spread", cs_range, "bs")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs_range * 100, tf_prices, **STYLE["tf"])
    ax.plot(cs_range * 100, bs_prices, **STYLE["bs"])
    ax.axvline(base_mkt.credit_spread * 100, color="grey", linestyle="--",
               linewidth=0.8, label=f"Base spread = {base_mkt.credit_spread*100:.0f}bps")
    ax.set_xlabel("Credit Spread (bps)", fontsize=11)
    ax.set_ylabel("CB Price (€)", fontsize=11)
    ax.set_title("CB Price vs Credit Spread\n"
                 "TF correctly discounts equity component; BS does not", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "02_price_vs_spread")


# ---------------------------------------------------------------------------
# 3. Delta vs Stock Price (equity sensitivity)
# ---------------------------------------------------------------------------

def plot_delta_vs_stock(cb, base_mkt, N: int = 200):
    """
    Delta = ∂Price/∂S. Shows transition from bond-like (δ≈0) to equity-like (δ≈cr).
    """
    from models import price_tsiveriotis_fernandes, price_bs_benchmark, MarketParams

    S_range = np.linspace(15, 150, 50)
    tf_deltas, bs_deltas = [], []

    for S in S_range:
        mkt_copy = MarketParams(
            S0=S, r=base_mkt.r, sigma=base_mkt.sigma,
            credit_spread=base_mkt.credit_spread, recovery=base_mkt.recovery
        )
        tf_deltas.append(price_tsiveriotis_fernandes(cb, mkt_copy, N=150)["delta"])
        bs_deltas.append(price_bs_benchmark(cb, mkt_copy)["delta"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_range, tf_deltas, **STYLE["tf"])
    ax.plot(S_range, bs_deltas, **STYLE["bs"])
    ax.axhline(0, color="grey", linewidth=0.7, linestyle=":")
    ax.axvline(base_mkt.S0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Stock Price S₀ (€)", fontsize=11)
    ax.set_ylabel("Delta (∂Price / ∂S)", fontsize=11)
    ax.set_title("CB Delta vs Stock Price\n"
                 "Bond-like at low S, equity-like at high S", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "03_delta_vs_stock")


# ---------------------------------------------------------------------------
# 4. Regime scatter: TF-BS divergence
# ---------------------------------------------------------------------------

def plot_regime_scatter(df):
    """
    Scatter of (credit_spread, S0) coloured by regime.
    Size = |TF - BS| divergence.
    """
    colors = {"equity": "#2ca02c", "balanced": "#1f77b4", "distressed": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for regime, grp in df.groupby("regime"):
        ax.scatter(
            grp["credit_spread"] * 100,
            grp["S0"],
            s=np.abs(grp["tf_minus_bs"]) * 0.5 + 5,
            alpha=0.6,
            color=colors[regime],
            label=regime.capitalize(),
        )
    ax.set_xlabel("Credit Spread (bps)", fontsize=11)
    ax.set_ylabel("Stock Price S₀ (€)", fontsize=11)
    ax.set_title("CB Regime Classification (200 synthetic scenarios)\n"
                 "Dot size ∝ |TF − BS| divergence", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "04_regime_scatter")


# ---------------------------------------------------------------------------
# 5. Tree convergence
# ---------------------------------------------------------------------------

def plot_convergence(df_conv):
    """
    Price and delta as a function of tree steps N.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df_conv["N"], df_conv["price"], "o-", color="#1f77b4", markersize=4)
    ax1.axhline(df_conv["price"].iloc[-1], color="grey", linewidth=0.8, linestyle="--",
                label=f'Limit ≈ {df_conv["price"].iloc[-1]:.2f}')
    ax1.set_xlabel("Tree Steps N", fontsize=11)
    ax1.set_ylabel("CB Price (€)", fontsize=11)
    ax1.set_title("Tree Convergence — Price", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_conv["N"], df_conv["delta"], "s-", color="#ff7f0e", markersize=4)
    ax2.set_xlabel("Tree Steps N", fontsize=11)
    ax2.set_ylabel("Delta", fontsize=11)
    ax2.set_title("Tree Convergence — Delta", fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Tsiveriotis-Fernandes Trinomial Tree Convergence", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "05_convergence")


# ---------------------------------------------------------------------------
# 6. Vol surface (sigma sensitivity)
# ---------------------------------------------------------------------------

def plot_price_vs_vol(cb, base_mkt):
    from models import sensitivity_surface
    sig_range = np.linspace(0.05, 0.80, 60)
    _, tf_prices = sensitivity_surface(cb, base_mkt, "sigma", sig_range, "tf")
    _, bs_prices = sensitivity_surface(cb, base_mkt, "sigma", sig_range, "bs")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sig_range * 100, tf_prices, **STYLE["tf"])
    ax.plot(sig_range * 100, bs_prices, **STYLE["bs"])
    ax.axvline(base_mkt.sigma * 100, color="grey", linestyle="--", linewidth=0.8,
               label=f"Base σ = {base_mkt.sigma*100:.0f}%")
    ax.set_xlabel("Equity Volatility σ (%)", fontsize=11)
    ax.set_ylabel("CB Price (€)", fontsize=11)
    ax.set_title("CB Price vs Equity Volatility\n"
                 "Convexity increases with vol (positive vega)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_price_vs_vol")
