"""
Publication-quality figures replicating the paper's exhibits (Figs. 1-8).

All functions return the matplotlib ``Figure`` so callers can save or embed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 7,
    }
)


def plot_price_dynamics(prices: pd.DataFrame, title: str = "Price dynamics (cents)") -> plt.Figure:
    """Fig. 1 / Fig. 2 analogue."""
    fig, ax = plt.subplots(figsize=(9, 4))
    prices.plot(ax=ax, lw=0.9)
    ax.set_ylabel("Price (cents)")
    ax.set_title(title)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def plot_curve_snapshots(
    prices: pd.DataFrame,
    maturities: pd.Series,
    snapshot_dates: list[pd.Timestamp],
    labels: list[str] | None = None,
) -> plt.Figure:
    """Fig. 3 analogue: price vs maturity at selected dates."""
    fig, ax = plt.subplots(figsize=(7, 4))
    order = maturities.reindex(prices.columns).sort_values()
    for k, d in enumerate(snapshot_dates):
        row = prices.reindex([pd.Timestamp(d)], method="nearest").iloc[0]
        lbl = labels[k] if labels else str(pd.Timestamp(d).date())
        ax.plot(order.values, row[order.index].values, marker="o", label=lbl)
    ax.set_xlabel("Maturity year")
    ax.set_ylabel("Price (cents)")
    ax.set_title("Price curve snapshots")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_level_dispersion(summary: pd.DataFrame) -> plt.Figure:
    """Fig. 4 analogue: cross-sectional level and dispersion."""
    fig, ax = plt.subplots(figsize=(9, 4))
    summary["level"].plot(ax=ax, label="level (mean price)")
    summary["dispersion"].plot(ax=ax, label="dispersion (std across maturities)")
    ax.set_title("Curve diagnostics")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_car(car: pd.Series, title: str = "Cumulative abnormal returns") -> plt.Figure:
    """Figs. 5-6 analogue."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(car.index, car.values, marker=".")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Event time (trading days)")
    ax.set_ylabel("CAR")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_implied_probabilities(
    probs: pd.DataFrame,
    default_date: pd.Timestamp | None = None,
    latent: pd.Series | None = None,
) -> plt.Figure:
    """Fig. 7 analogue: implied normalization probabilities across maturities."""
    fig, ax = plt.subplots(figsize=(9, 4))
    probs.plot(ax=ax, lw=0.9)
    if latent is not None:
        latent.plot(ax=ax, color="k", ls=":", lw=1.4, label="latent p(t) (synthetic truth)")
    if default_date is not None:
        ax.axvline(pd.Timestamp(default_date), color="tab:blue", ls="--", lw=1, label="default")
    ax.set_ylabel("Implied probability")
    ax.set_title("Implied normalization probabilities")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def plot_violation_mass(v: pd.DataFrame, default_date: pd.Timestamp | None = None) -> plt.Figure:
    """Fig. 8 analogue: monotonicity violation mass V(t)."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    v["V"].plot(ax=ax, lw=0.6, alpha=0.5, label="V(t) — raw")
    if "V_ma" in v:
        v["V_ma"].plot(ax=ax, lw=1.4, label="V(t) — moving average")
    if default_date is not None:
        ax.axvline(pd.Timestamp(default_date), color="tab:blue", ls="--", lw=1, label="default")
    ax.set_ylabel("V(t)")
    ax.set_title("Monotonicity violation mass")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
