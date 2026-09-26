"""What the BNP AT1 spread pays for, whether the 6.875% gets called, and stress P&L.

Uses the calibrated model of ``scripts/calibrate_bnp.py`` on the last date
(PONV hazard implied per bond, 100bp refinancing-spread vol) and the same
random numbers everywhere, so every difference below is free of Monte-Carlo
noise.

1. Spread decomposition.  Features are switched on one at a time and each
   intermediate price is turned into a yield-to-first-call spread over the
   Treasury used for discounting.  The steps add up exactly to the market spread:
       risk-free bullet to call  (0bp)
     + extension (issuer's call option)
     + MDA coupon cuts
     + mechanical conversion at the trigger
     + PONV / tail / liquidity premium (the implied hazard)  = market spread
   The order matters a little (interactions); it follows the order above.

2. Call vs extension: P(call at first reset) as a function of the new-issue
   AT1 spread, and the price if the market prices "called" vs "extended".

3. Stress scenarios and the price-CET1 profile.

Outputs in ``results/``: spread_decomposition.(png|csv), call_probability.png,
price_vs_cet1.png, stress_scenarios.csv.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_bnp import (  # noqa: E402
    COLOR, DATA, ORDER, RES, SHORT, fit_cet1_ou, load_cet1_series, market_panel, settings,
)

from at1_coco.market import (  # noqa: E402
    DatedAT1, MarketState, cash_flow_profile, cont_rate, default_randoms, implied_ponv,
)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COMP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # palette slots 1-4
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
def ytc(bond: DatedAT1, val_date: pd.Timestamp, dirty: float) -> float:
    """Semi-annual yield to the first reset date implied by a dirty price."""
    t, is_reset, _ = bond.schedule(val_date, 40)
    n = int(np.argmax(is_reset)) + 1
    t = t[:n]
    cf = np.full(n, bond.coupon / bond.coupon_freq * bond.notional)
    cf[-1] += bond.notional

    def f(y):
        return (cf * (1 + y / 2) ** (-2 * t)).sum() - dirty

    return brentq(f, -0.05, 0.5)


def price(prof, mkt, lam):
    return prof.dirty_price(cont_rate(mkt.discount_yield), lam)


def main():
    static = pd.read_csv(DATA / "bonds_static.csv", index_col="isin_144a")
    bonds = {i: DatedAT1.from_row(i, static.loc[i]) for i in ORDER}
    kappa, sigma = fit_cet1_ou(load_cet1_series())
    panel = market_panel(curve="ust10")
    long_run = float(panel.groupby("date")["refi"].first().mean())
    last = panel[panel["date"] == panel["date"].max()].set_index("isin")
    ms = settings(kappa, sigma, 0.01, long_run)
    randoms = default_randoms(n_paths=20_000)

    mkts = {i: MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
            for i, r in last.iterrows()}
    lam = {}
    for i in ORDER:
        lam[i], _ = implied_ponv(bonds[i], mkts[i], ms, randoms)
    val_date = last["date"].iloc[0]
    print("valuation date", val_date.date(), "| implied PONV", {SHORT[i]: round(100 * lam[i], 2) for i in ORDER})

    # 1. spread decomposition -------------------------------------------------
    # order: premium first (so every later feature is valued with risky
    # discounting), extension last (valued with all other features in place)
    stages = [
        ("risk-free to call", dict(call_rule="first", mda=False, conversion=False), False),
        ("PONV / tail / liquidity", dict(call_rule="first", mda=False, conversion=False), True),
        ("MDA coupon cuts", dict(call_rule="first", mda=True, conversion=False), True),
        ("conversion", dict(call_rule="first", mda=True, conversion=True), True),
        ("extension", dict(call_rule="model", mda=True, conversion=True), True),
    ]
    rows = []
    for i in ORDER:
        b, m = bonds[i], mkts[i]
        base_y = None
        prev = 0.0
        for name, kw, with_lam in stages:
            prof = cash_flow_profile(b, m, ms, randoms, **kw)
            d = price(prof, m, lam[i] if with_lam else 0.0)
            y = ytc(b, val_date, d)
            if base_y is None:
                base_y = y
            cum = 1e4 * (y - base_y)
            rows.append(dict(isin=i, component=name, dirty=d, clean=d - b.accrued(val_date),
                             ytc=y, cum_spread_bp=cum, contribution_bp=cum - prev))
            prev = cum
        mkt_y = ytc(b, val_date, m.clean_price + b.accrued(val_date))
        rows.append(dict(isin=i, component="market", dirty=m.clean_price + b.accrued(val_date),
                         clean=m.clean_price, ytc=mkt_y, cum_spread_bp=1e4 * (mkt_y - base_y),
                         contribution_bp=np.nan))
    dec = pd.DataFrame(rows)
    dec.to_csv(RES / "spread_decomposition.csv", index=False)
    tab = dec[dec.component != "risk-free to call"].pivot(index="component", columns="isin",
                                                         values="contribution_bp")[ORDER]
    tab = tab.reindex([st[0] for st in stages[1:]])
    tab.loc["total model"] = tab.sum()
    tab.loc["market YTC spread"] = dec[dec.component == "market"].set_index("isin")["cum_spread_bp"][ORDER]
    print("\nspread decomposition (bp of yield to first call over the Treasury):")
    print(tab.round(1).to_string())
    fig_decomposition(tab)

    # 2. call vs extension ------------------------------------------------------
    grid = np.arange(0.015, 0.0451, 0.0025)
    call_rows = []
    for i in ORDER:
        b, m = bonds[i], mkts[i]
        for lvl in grid:
            ms_l = replace(ms, spread=replace(ms.spread, long_run=lvl))
            prof = cash_flow_profile(b, replace(m, refi_spread=lvl), ms_l, randoms)
            call_rows.append(dict(isin=i, new_issue_spread_bp=1e4 * lvl,
                                  p_call_first=prof.diagnostics["p_call_first"]))
    calls = pd.DataFrame(call_rows)
    calls.to_csv(RES / "call_probability.csv", index=False)
    fig_calls(calls, bonds, long_run)

    print("\ncall vs extension (clean prices, same implied PONV):")
    ext_rows = []
    for i in ORDER:
        b, m = bonds[i], mkts[i]
        base = cash_flow_profile(b, m, ms, randoms)
        p_model = price(base, m, lam[i]) - b.accrued(val_date)
        p_called = price(cash_flow_profile(b, m, ms, randoms, call_rule="first"), m, lam[i]) - b.accrued(val_date)
        # extension scenario: AT1 spreads settle 50bp higher -> how much more
        # does each bond lose than it would if it were certain to be called?
        up = replace(ms, spread=replace(ms.spread, long_run=ms.spread.long_run + 0.005))
        m_up = replace(m, refi_spread=m.refi_spread + 0.005)
        lam_up = lam[i] * (m.refi_spread + 0.005) / m.refi_spread
        d_model = price(cash_flow_profile(b, m_up, up, randoms), m_up, lam_up) - b.accrued(val_date) - p_model
        d_called = (price(cash_flow_profile(b, m_up, up, randoms, call_rule="first"), m_up, lam_up)
                    - b.accrued(val_date) - p_called)
        ext_rows.append(dict(isin=i, p_call_first=base.diagnostics["p_call_first"],
                             p_called_ever=base.diagnostics["p_called"],
                             price_model=p_model, price_if_surely_called=p_called,
                             call_option_cost_pts=p_called - p_model,
                             spread_up50_pnl=d_model, spread_up50_pnl_if_called=d_called,
                             extension_amplification_pts=d_model - d_called,
                             margin_vs_new_issue_bp=1e4 * (b.reset_margin - m.refi_spread)))
    ext = pd.DataFrame(ext_rows).set_index("isin").loc[ORDER]
    ext.to_csv(RES / "call_vs_extension.csv")
    print(ext.round(3).to_string())

    # 3. stress scenarios -------------------------------------------------------
    def reprice(i, mkt=None, ms_=None, lam_=None):
        b = bonds[i]
        mkt = mkt or mkts[i]
        prof = cash_flow_profile(b, mkt, ms_ or ms, randoms)
        return price(prof, mkt, lam[i] if lam_ is None else lam_) - b.accrued(val_date)

    def spreads_up(ms_, d):
        return replace(ms_, spread=replace(ms_.spread, long_run=ms_.spread.long_run + d))

    scen = {
        "Rates +100bp": lambda i: reprice(i, replace(mkts[i], discount_yield=mkts[i].discount_yield + 0.01,
                                                     reset_reference=mkts[i].reset_reference + 0.01)),
        "Rates -100bp": lambda i: reprice(i, replace(mkts[i], discount_yield=mkts[i].discount_yield - 0.01,
                                                     reset_reference=mkts[i].reset_reference - 0.01)),
        "AT1 spreads +200bp": lambda i: reprice(i, replace(mkts[i], refi_spread=mkts[i].refi_spread + 0.02),
                                                spreads_up(ms, 0.02), lam[i] * (mkts[i].refi_spread + 0.02) / mkts[i].refi_spread),
        "CET1 -2pp (capital only)": lambda i: reprice(i, replace(mkts[i], cet1=mkts[i].cet1 - 2)),
        "CET1 -4.2pp = EBA adverse (capital only)": lambda i: reprice(i, replace(mkts[i], cet1=mkts[i].cet1 - 4.2)),
        "EBA adverse + spreads +200bp": lambda i: reprice(
            i, replace(mkts[i], cet1=mkts[i].cet1 - 4.2, refi_spread=mkts[i].refi_spread + 0.02),
            spreads_up(ms, 0.02), lam[i] * (mkts[i].refi_spread + 0.02) / mkts[i].refi_spread),
        "PONV scare (hazard x3)": lambda i: reprice(i, lam_=3 * lam[i]),
        "March-2023 style: spreads +200bp, rates -50bp": lambda i: reprice(
            i, replace(mkts[i], discount_yield=mkts[i].discount_yield - 0.005,
                       reset_reference=mkts[i].reset_reference - 0.005,
                       refi_spread=mkts[i].refi_spread + 0.02), spreads_up(ms, 0.02), lam[i] * (mkts[i].refi_spread + 0.02) / mkts[i].refi_spread),
    }
    base_px = {i: reprice(i) for i in ORDER}
    st = pd.DataFrame({SHORT[i]: {k: f(i) - base_px[i] for k, f in scen.items()} for i in ORDER})
    st.loc["base clean price (model)"] = [base_px[i] for i in ORDER]
    st.to_csv(RES / "stress_scenarios.csv")
    print("\nstress P&L (clean price points, vs model base):")
    print(st.round(2).to_string())

    # capital delta and price-CET1 profile
    levels = np.arange(6.0, 15.01, 0.5)
    prof_rows = []
    for i in ORDER:
        for c in levels:
            prof_rows.append(dict(isin=i, cet1=c, clean=reprice(i, replace(mkts[i], cet1=c))))
    pc = pd.DataFrame(prof_rows)
    pc.to_csv(RES / "price_vs_cet1.csv", index=False)
    d = pc.pivot(index="cet1", columns="isin", values="clean")[ORDER].diff() / 0.5
    print("\ncapital delta (price points per +1pp CET1):")
    print(d.loc[[7.0, 9.0, 10.5, 11.0, 12.0, 13.0]].round(2).to_string())
    fig_cet1_profile(pc, ms, mkts)


# ---------------------------------------------------------------------------
def fig_decomposition(tab):
    comps = ["PONV / tail / liquidity", "MDA coupon cuts", "conversion", "extension"]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(ORDER))[::-1]
    left = np.zeros(len(ORDER))
    for comp, col in zip(comps, COMP_COLORS):
        v = tab.loc[comp, ORDER].values
        ax.barh(y, v, left=left, color=col, height=0.55, label=comp, edgecolor=SURFACE, linewidth=2)
        for yi, (l, w) in enumerate(zip(left, v)):
            if w >= 18:
                ax.text(l + w / 2, y[yi], f"{w:.0f}", ha="center", va="center", color="white" if col != "#eda100" else INK, fontsize=9)
        left += v
    mkt = tab.loc["market YTC spread", ORDER].values
    for yi, m in enumerate(mkt):
        ax.text(m + 4, y[yi], f"{m:.0f}bp", va="center", color=INK, fontsize=9, fontweight="bold")
    ax.set_yticks(y, [SHORT[i] for i in ORDER])
    ax.set_xlabel("bp of yield to first call over Treasury")
    ax.set_title("What the BNP AT1 spread pays for")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(mkt) * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "spread_decomposition.png", dpi=150)
    plt.close(fig)


def fig_calls(calls, bonds, current):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in ORDER:
        g = calls[calls["isin"] == i]
        ax.plot(g["new_issue_spread_bp"], 100 * g["p_call_first"], color=COLOR[i], lw=2,
                label=f"{SHORT[i]} (margin {1e4*bonds[i].reset_margin:.0f}bp)")
        ax.axvline(1e4 * bonds[i].reset_margin, color=COLOR[i], lw=0.8, ls=":")
    ax.axvline(1e4 * current, color=INK2, lw=1.2, ls="--")
    ax.annotate(f"sample-average\nAT1 spread ≈ {1e4*current:.0f}bp", (1e4 * current, 8), xytext=(-110, 0),
                textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("New-issue AT1 spread level (bp), today and long-run")
    ax.set_ylabel("P(called at first reset) %")
    ax.set_ylim(0, 100)
    ax.set_title("Will BNP call? Call probability vs AT1 spread level")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "call_probability.png", dpi=150)
    plt.close(fig)


def fig_cet1_profile(pc, ms, mkts):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i in ORDER:
        g = pc[pc["isin"] == i]
        ax.plot(g["cet1"], g["clean"], color=COLOR[i], lw=2, marker="o", ms=3, label=SHORT[i])
    top = ms.mda_threshold + ms.combined_buffer
    ax.axvline(top, color=INK2, ls="--", lw=1.2)
    ax.annotate(f"MDA threshold {top:.2f}%", (top, ax.get_ylim()[0]), xytext=(-122, 8), textcoords="offset points", color=INK2)
    today = mkts[ORDER[0]].cet1
    ax.axvline(today, color=INK2, lw=1)
    ax.annotate(f"today {today:.2f}%", (today, ax.get_ylim()[0]), xytext=(5, 8), textcoords="offset points", color=INK2)
    ax.set_xlabel("CET1 ratio (%), everything else unchanged")
    ax.set_ylabel("Model clean price")
    ax.set_title("AT1 price vs BNP capital")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "price_vs_cet1.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
