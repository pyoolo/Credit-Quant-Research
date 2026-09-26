"""Descriptive analysis of the BNP Paribas USD AT1 dataset.

Produces three figures in ``results/`` and prints the numbers quoted in
``RESEARCH.md``:

1. ``bnp_cet1_history.png``   CET1 ratio vs MDA requirement and AT1 trigger
2. ``bnp_at1_prices.png``     evaluated mid prices of the three AT1s
3. ``bnp_at1_yield_split.png`` yield change split into Treasury vs spread

Run after ``scripts/build_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"

# reference palette (light mode), categorical slots in fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"US05602XQR25": "#2a78d6", "US05602XQS08": "#eb6834", "US05602XQQ42": "#1baf7a"}
LABEL = {
    "US05602XQR25": "6.875% NC2033 (reset +285bp)",
    "US05602XQS08": "7.20% NC2036 (reset +294bp)",
    "US05602XQQ42": "7.45% NC2035 (reset +313bp)",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def load():
    prices = pd.read_csv(DATA / "bond_prices.csv", parse_dates=["date"])
    q = pd.read_csv(DATA / "bnp_cet1_quarterly.csv", parse_dates=["date"], index_col="date")
    a = pd.read_csv(DATA / "bnp_capital_annual.csv", parse_dates=["date"], index_col="date")
    share = pd.read_csv(DATA / "bnp_share.csv", parse_dates=["date"], index_col="date")
    static = pd.read_csv(DATA / "bonds_static.csv", index_col="isin_144a")
    return prices, q, a, share, static


def cet1_history(q: pd.DataFrame, a: pd.DataFrame) -> pd.Series:
    """Annual points up to 2023, quarterly from 2024 (the finer series wins)."""
    ann = a["cet1_ratio"][a.index < q.index.min()]
    return pd.concat([ann, q["cet1_ratio"]]).sort_index()


def fig_cet1(q, a):
    c = cet1_history(q, a)
    req = a["cet1_requirement"].dropna()
    # latest requirement (30-Jun-2026) published by BNP, not yet in the annual export
    req_pts = pd.concat([req, pd.Series({pd.Timestamp("2026-06-30"): 10.43})])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(c.index, c.values, color=SERIES["US05602XQR25"], lw=2, marker="o", ms=4, label="CET1 ratio")
    ax.step(req_pts.index, req_pts.values, where="post", color=INK2, lw=1.5, ls="--",
            label="CET1 requirement (MDA threshold)")
    ax.axhline(5.125, color="#e34948", lw=1.5, label="AT1 trigger 5.125%")
    ax.set_ylim(4, 14.5)
    ax.set_ylabel("% of RWA")
    ax.set_title("BNP Paribas: CET1 ratio vs MDA requirement and AT1 trigger")
    last = c.index[-1]
    ax.annotate(f"{c.iloc[-1]:.2f}%", (last, c.iloc[-1]), xytext=(6, 4), textcoords="offset points", color=INK)
    ax.annotate("cushion to MDA ≈ 2.5pp", (last, 11.7), xytext=(-150, 0), textcoords="offset points", color=INK2)
    ax.annotate("distance to trigger ≈ 7.8pp", (last, 8.5), xytext=(-165, 0), textcoords="offset points", color=INK2)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "bnp_cet1_history.png", dpi=150)
    plt.close(fig)


def fig_prices(prices):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for isin, g in prices.groupby("isin"):
        ax.plot(g["date"], g["mid_price"], color=SERIES[isin], lw=2, label=LABEL[isin])
    ax.axhline(100, color=INK2, lw=0.8)
    ax.set_ylabel("Evaluated mid price (clean)")
    ax.set_title("BNP Paribas USD AT1s: evaluated mid prices")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "bnp_at1_prices.png", dpi=150)
    plt.close(fig)


def with_common_treasury(prices):
    """CIQ switches the benchmark of the 6.875% twice (Jan and Aug 2026), which
    creates fake spread jumps. Use one Treasury series for all three bonds: the
    benchmark of the 7.45% (the ~10y on-the-run, continuous over the sample)."""
    ust = prices[prices["isin"] == "US05602XQQ42"].set_index("date")["benchmark_yield"]
    out = prices.copy()
    out["ust10"] = out["date"].map(ust)
    out["spread_vs_ust10_bp"] = 100 * (out["mid_yield"] - out["ust10"])
    return out


def yield_split(prices, start, end):
    prices = with_common_treasury(prices)
    rows = []
    for isin, g in prices.groupby("isin"):
        g = g.set_index("date").sort_index()
        s = g.loc[:start].iloc[-1] if g.index.min() <= pd.Timestamp(start) else g.iloc[0]
        e = g.loc[:end].iloc[-1]
        rows.append(
            {
                "isin": isin, "from": s.name.date(), "to": e.name.date(),
                "price_chg": e["mid_price"] - s["mid_price"],
                "yield_chg_bp": 100 * (e["mid_yield"] - s["mid_yield"]),
                "treasury_chg_bp": 100 * (e["ust10"] - s["ust10"]),
                "spread_chg_bp": e["spread_vs_ust10_bp"] - s["spread_vs_ust10_bp"],
            }
        )
    return pd.DataFrame(rows).set_index("isin")


def fig_split(ytd, sep):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    order = ["US05602XQR25", "US05602XQS08", "US05602XQQ42"]
    for ax, df, title in zip(axes, [ytd, sep], ["From 31-Dec-2025 (or issue)", "September 2026 sell-off"]):
        df = df.loc[order]
        x = np.arange(len(order))
        ax.bar(x - 0.18, df["treasury_chg_bp"], 0.34, color=INK2, label="10y Treasury")
        ax.bar(x + 0.18, df["spread_chg_bp"], 0.34, color="#2a78d6", label="Spread over 10y")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, ["6.875% '33", "7.20% '36", "7.45% '35"])
        ax.set_title(title, fontsize=11)
        for xi, (t, s) in enumerate(zip(df["treasury_chg_bp"], df["spread_chg_bp"])):
            ax.annotate(f"{t:+.0f}", (xi - 0.18, t), ha="center", va="bottom" if t > 0 else "top", fontsize=8)
            ax.annotate(f"{s:+.0f}", (xi + 0.18, s), ha="center", va="bottom" if s > 0 else "top", fontsize=8)
    axes[0].set_ylabel("Change in yield (bp)")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Where the AT1 price moves came from: rates vs spread", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "bnp_at1_yield_split.png", dpi=150)
    plt.close(fig)


def main():
    prices, q, a, share, static = load()
    RES.mkdir(exist_ok=True)
    fig_cet1(q, a)
    fig_prices(prices)
    ytd = yield_split(prices, "2025-12-31", "2026-09-25")
    sep = yield_split(prices, "2026-08-28", "2026-09-25")
    fig_split(ytd, sep)

    last = prices[prices["date"] == prices["date"].max()].set_index("isin")
    cross = last[["mid_price", "mid_yield", "mid_spread_bp", "benchmark_yield"]].join(
        static[["coupon_initial", "reset_margin", "first_call_date", "floor_price_usd"]]
    )
    print("== cross-section on", prices["date"].max().date())
    print(cross.round(3).to_string())
    print("\n== yield split since 31-Dec-2025\n", ytd.round(1).to_string())
    print("\n== yield split Sep-2026\n", sep.round(1).to_string())

    c = cet1_history(q, a)
    dc = c.diff().dropna()
    print("\n== CET1: last", c.iloc[-1].round(2), "min since 2019", c["2019":].min().round(2),
          "max", c.max().round(2))
    print("quarterly changes 2024-26: mean %.3f sd %.3f" % (q["cet1_ratio"].diff().mean(), q["cet1_ratio"].diff().std()))
    print("annual changes 2014-25: sd %.3f" % a["cet1_ratio"].diff().std())
    rwa = q["rwa_eur_k"].iloc[-1] / 1e6
    print(f"RWA €{rwa:.0f}bn -> MDA cushion (13.0-10.43)={2.54*rwa/100:.1f}bn, "
          f"trigger distance {(q['cet1_ratio'].iloc[-1]-5.125)*rwa/100:.1f}bn")
    s = share["close_eur"]
    print("\n== share: last %.2f on %s, 52w high %.2f, min since Dec-23 %.2f"
          % (s.iloc[-1], s.index[-1].date(), s[-252:].max(), s.min()))
    print("share since 28-Aug-2026: %.1f%%" % (100 * (s.iloc[-1] / s.loc[:"2026-08-28"].iloc[-1] - 1)))


if __name__ == "__main__":
    main()
