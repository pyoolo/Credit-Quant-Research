"""How much can we trust the CET1 parameters, and does it matter?

1. Parametric bootstrap of the CET1 dynamics.  Simulate the exact OU process on
   the observed quarterly dates (2014Q1-2026Q2) with the fitted (kappa, sigma),
   re-estimate on every simulated history, and read confidence intervals from
   the distribution of the re-estimates.

2. Sensitivity of the headline results to kappa.  Take kappa at the 5th, 50th
   and 95th percentile of the bootstrap (sigma re-estimated given kappa) and
   recompute on the last date:
     - the implied PONV hazard of each bond,
     - P(call at first reset),
     - the MDA coupon-cut component of the spread,
     - the capital delta at today's CET1 and just below the MDA threshold,
     - the EBA-adverse capital shock (-4.2pp CET1).

3. Market-implied kappa.  The three bonds must imply the same PONV hazard.
   Scan kappa (sigma re-fitted to history given kappa) and find the value that
   makes the three implied hazards most consistent: what bond prices say about
   how fast BNP rebuilds capital, to compare with the historical estimate.

4. Capital cliff under each kappa: price of the most extension-exposed bond
   (6.875% NC33) against CET1, with its implied hazard re-solved per scenario.

Outputs: results/cet1_bootstrap.csv, results/parameter_sensitivity.csv,
         results/kappa_scan.csv, results/parameter_uncertainty.png,
         results/capital_cliff_uncertainty.(csv|png)

Run after calibrate_bnp.py (it reuses its data loaders):
    python scripts/parameter_uncertainty.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402

from calibrate_bnp import (  # noqa: E402
    BASE_KAPPA, BASE_SPREAD_VOL, COLOR, DATA, ONE_OFF_QUARTERS, ORDER, RES, SHORT, fit_cet1_ou, load_cet1_series,
    market_panel, settings,
)
from at1_coco.market import (  # noqa: E402
    DatedAT1, MarketState, cash_flow_profile, cont_rate, default_randoms, implied_ponv,
)

THETA = 13.0
N_BOOT = 1000


# ---------------------------------------------------------------------------
# 1. bootstrap
# ---------------------------------------------------------------------------
def ou_nll(x, t, keep, k, s, theta=THETA):
    dt = np.diff(t)
    m = theta + (x[:-1] - theta) * np.exp(-k * dt)
    v = s * s * (1 - np.exp(-2 * k * dt)) / (2 * k)
    return 0.5 * np.sum((np.log(2 * np.pi * v) + (x[1:] - m) ** 2 / v)[keep])


def fit(x, t, keep):
    r = minimize(lambda p: ou_nll(x, t, keep, *np.exp(p)), np.log([0.3, 0.4]), method="Nelder-Mead")
    return np.exp(r.x)


def sigma_given_kappa(x, t, keep, k):
    r = minimize_scalar(lambda ls: ou_nll(x, t, keep, k, np.exp(ls)), bounds=(np.log(0.05), np.log(3)),
                        method="bounded")
    return float(np.exp(r.x))


def bootstrap(c: pd.Series, k_hat: float, s_hat: float, n: int = N_BOOT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.asarray((c.index - c.index[0]).days / 365.25)
    dt = np.diff(t)
    keep = np.ones(len(dt), bool)                  # simulated histories have no one-off
    e = np.exp(-k_hat * dt)
    sd = s_hat * np.sqrt((1 - e * e) / (2 * k_hat))
    out = []
    for _ in range(n):
        x = np.empty(len(t))
        x[0] = c.iloc[0]
        z = rng.standard_normal(len(dt))
        for i in range(len(dt)):
            x[i + 1] = THETA + (x[i] - THETA) * e[i] + sd[i] * z[i]
        out.append(fit(x, t, keep))
    return pd.DataFrame(out, columns=["kappa", "sigma"])


# ---------------------------------------------------------------------------
# 2. sensitivity of the headline results
# ---------------------------------------------------------------------------
def headline(kappa, sigma, bonds, last, long_run, randoms):
    ms = settings(kappa, sigma, BASE_SPREAD_VOL, long_run)
    rows = []
    for isin in ORDER:
        r = last.loc[isin]
        b = bonds[isin]
        mkt = MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
        lam, prof = implied_ponv(b, mkt, ms, randoms)
        rr = cont_rate(mkt.discount_yield)
        acc = b.accrued(mkt.val_date)

        def px(m, **kw):
            return cash_flow_profile(b, m, ms, randoms, **kw).dirty_price(rr, lam) - acc

        base = px(mkt)
        # MDA component: price with vs without the MDA restriction, in points
        mda_pts = px(mkt, mda=False) - base
        up = px(replace(mkt, cet1=mkt.cet1 + 0.5))
        dn = px(replace(mkt, cet1=mkt.cet1 - 0.5))
        mda = ms.mda_threshold + ms.combined_buffer
        at_mda_up = px(replace(mkt, cet1=mda + 0.25))
        at_mda_dn = px(replace(mkt, cet1=mda - 0.25))
        eba = px(replace(mkt, cet1=mkt.cet1 - 4.2)) - base
        rows.append(dict(isin=isin, implied_ponv_pct=100 * lam,
                         p_call_first=prof.diagnostics["p_call_first"],
                         mda_cost_pts=mda_pts,
                         capital_delta_today=(up - dn) / 1.0,
                         capital_delta_at_mda=(at_mda_up - at_mda_dn) / 0.5,
                         eba_shock_pts=eba))
    return pd.DataFrame(rows)


def main():
    c = load_cet1_series()
    t = np.asarray((c.index - c.index[0]).days / 365.25)
    keep = ~np.isin(c.index[1:], pd.to_datetime(ONE_OFF_QUARTERS))
    k_hat, s_hat = fit_cet1_ou(c)
    print(f"point estimate ({len(c)} quarters): kappa={k_hat:.3f}  sigma={s_hat:.3f}")

    boot = bootstrap(c, k_hat, s_hat)
    boot.to_csv(RES / "cet1_bootstrap.csv", index=False)
    q = boot.quantile([0.05, 0.5, 0.95])
    print("\nbootstrap (", N_BOOT, "histories ) 5% / 50% / 95%:")
    print(q.round(3).to_string())
    print(f"half-life of a capital shock: {np.log(2)/k_hat:.1f} years "
          f"(90% CI {np.log(2)/q.loc[0.95,'kappa']:.1f}-{np.log(2)/q.loc[0.05,'kappa']:.1f})")

    # scenarios: kappa at its 5/50/95 percentile, sigma re-fitted given kappa
    x = c.values
    scen = {}
    for name, k in [("kappa low (5%)", q.loc[0.05, "kappa"]), ("historical MLE", k_hat),
                    ("base case", BASE_KAPPA), ("kappa high (95%)", q.loc[0.95, "kappa"])]:
        scen[name] = (float(k), sigma_given_kappa(x, t, keep, float(k)))

    static = pd.read_csv(DATA / "bonds_static.csv", index_col="isin_144a")
    bonds = {i: DatedAT1.from_row(i, static.loc[i]) for i in ORDER}
    panel = market_panel(curve="fred")
    long_run = float(panel.groupby("date")["refi"].first().mean())
    last = panel[panel["date"] == panel["date"].max()].set_index("isin")
    randoms = default_randoms(n_paths=20_000)

    res = []
    for name, (k, s) in scen.items():
        h = headline(k, s, bonds, last, long_run, randoms)
        h.insert(0, "scenario", name)
        h.insert(1, "kappa", k)
        h.insert(2, "sigma", s)
        res.append(h)
        print(f"  done: {name}  kappa={k:.3f} sigma={s:.3f}")
    res = pd.concat(res, ignore_index=True)
    res.to_csv(RES / "parameter_sensitivity.csv", index=False)

    summary = res.groupby("scenario", sort=False)[
        ["kappa", "sigma", "implied_ponv_pct", "p_call_first", "mda_cost_pts",
         "capital_delta_today", "capital_delta_at_mda", "eba_shock_pts"]].mean()
    print("\nheadline results, average of the three bonds:")
    print(summary.round(3).to_string())

    # 3. market-implied kappa ------------------------------------------------
    scan = []
    for k in [0.06, 0.10, 0.165, 0.25, 0.35, 0.48, 0.65, 0.9]:
        s_k = sigma_given_kappa(x, t, keep, k)
        ms = settings(k, s_k, BASE_SPREAD_VOL, long_run)
        lams = []
        for isin in ORDER:
            r = last.loc[isin]
            mkt = MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
            lams.append(implied_ponv(bonds[isin], mkt, ms, randoms)[0])
        scan.append(dict(kappa=k, sigma=s_k, dispersion_bp=1e4 * (max(lams) - min(lams)),
                         mean_ponv_pct=100 * np.mean(lams),
                         loglik_history=-ou_nll(x, t, keep, k, s_k)))
    scan = pd.DataFrame(scan)
    scan.to_csv(RES / "kappa_scan.csv", index=False)
    print("\nmarket-implied kappa scan (last date):")
    print(scan.round(3).to_string(index=False))
    # the gap usually keeps shrinking with kappa and flattens out: report the
    # smallest kappa whose gap is within 0.5bp of the best one on the grid
    near = scan[scan["dispersion_bp"] <= scan["dispersion_bp"].min() + 0.5]
    k_mkt = float(near["kappa"].min())
    print(f"bond prices favour kappa >= {k_mkt} (gap within 0.5bp of the best); "
          f"history: {k_hat:.3f}, 90% CI {q.loc[0.05,'kappa']:.3f}-{q.loc[0.95,'kappa']:.3f}")

    # figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.scatter(boot["kappa"], boot["sigma"], s=6, color="#2a78d6", alpha=0.35, lw=0)
    ax.scatter([k_hat], [s_hat], s=60, color="#0b0b0b", zorder=3, label="estimate")
    for v in q["kappa"].loc[[0.05, 0.95]]:
        ax.axvline(v, color="#52514e", ls="--", lw=1)
    ax.set_xlabel("kappa (mean reversion, per year)")
    ax.set_ylabel("sigma (pp / sqrt(year))")
    ax.set_title("Bootstrap of the CET1 dynamics", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax = axes[1]
    metrics = ["implied_ponv_pct", "mda_cost_pts", "capital_delta_at_mda", "eba_shock_pts"]
    labels = ["implied PONV\n(%/yr)", "MDA coupon\ncost (pts)", "capital delta\nat MDA (pts/pp)", "EBA shock\n(pts)"]
    xpos = np.arange(len(metrics))
    for j, (name, col) in enumerate(zip(summary.index, ["#eb6834", "#52514e", "#2a78d6", "#1baf7a"])):
        ax.bar(xpos + (j - 1.5) * 0.2, summary.loc[name, metrics].abs(), 0.18, color=col, label=name)
    ax.set_xticks(xpos, labels, fontsize=9)
    ax.set_title("Headline results across the kappa range", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    ax.plot(scan["kappa"], scan["dispersion_bp"], "o-", color="#2a78d6", lw=2)
    ax.axvspan(q.loc[0.05, "kappa"], q.loc[0.95, "kappa"], color="#52514e", alpha=0.12, lw=0,
               label="90% CI from history")
    ax.axvline(k_hat, color="#0b0b0b", lw=1, ls="--", label="historical estimate")
    ax.axvline(BASE_KAPPA, color="#2a78d6", lw=1.5, label="base case")
    ax.set_xlabel("kappa")
    ax.set_ylabel("gap between the 3 implied PONV (bp)")
    ax.set_title("What bond prices say about kappa", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(RES / "parameter_uncertainty.png", dpi=150)
    print("\nsaved results/parameter_uncertainty.png")

    # 4. capital cliff under each kappa scenario --------------------------------
    isin = ORDER[0]
    b = bonds[isin]
    r = last.loc[isin]
    mkt = MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
    rr = cont_rate(mkt.discount_yield)
    acc = b.accrued(mkt.val_date)
    levels = np.arange(6.0, 15.01, 0.5)
    rows = []
    for name, (k, s_k) in scen.items():
        ms = settings(k, s_k, BASE_SPREAD_VOL, long_run)
        lam, _ = implied_ponv(b, mkt, ms, randoms)            # every curve reprices today's market
        for lvl in levels:
            px = cash_flow_profile(b, replace(mkt, cet1=float(lvl)), ms, randoms).dirty_price(rr, lam) - acc
            rows.append(dict(scenario=name, kappa=k, cet1=lvl, clean=px))
    cc = pd.DataFrame(rows)
    cc.to_csv(RES / "capital_cliff_uncertainty.csv", index=False)
    mda = ms.mda_threshold + ms.combined_buffer
    fig, ax = plt.subplots(figsize=(8, 4.4))
    colors = {"kappa low (5%)": "#eb6834", "historical MLE": "#52514e", "base case": "#2a78d6",
              "kappa high (95%)": "#1baf7a"}
    lo = cc[cc["scenario"] == "kappa low (5%)"].set_index("cet1")["clean"]
    hi = cc[cc["scenario"] == "kappa high (95%)"].set_index("cet1")["clean"]
    ax.fill_between(lo.index, lo.values, hi.values, color="#2a78d6", alpha=0.10, lw=0,
                    label="90% range of κ")
    for name, g in cc.groupby("scenario", sort=False):
        ax.plot(g["cet1"], g["clean"], color=colors[name], lw=2.6 if name == "base case" else 1.4,
                ls="-" if name in ("base case", "historical MLE") else "--",
                label=f"{name}, κ={g['kappa'].iloc[0]:.2f}")
    ax.axvline(mda, color="#52514e", ls="--", lw=1)
    ax.axvline(mkt.cet1, color="#52514e", lw=1)
    ax.annotate("MDA threshold", (mda, ax.get_ylim()[0]), xytext=(-6, 8), textcoords="offset points",
                ha="right", fontsize=8.5, color="#52514e")
    ax.annotate(f"today {mkt.cet1:.2f}%", (mkt.cet1, ax.get_ylim()[0]), xytext=(6, 8),
                textcoords="offset points", fontsize=8.5, color="#52514e")
    ax.set_xlabel("CET1 ratio (%), everything else unchanged")
    ax.set_ylabel("model clean price")
    ax.set_title(f"How uncertain is the capital cliff? {SHORT[isin]}", loc="left", fontweight="bold",
                 pad=22)
    ax.text(0, 1.015, "each curve re-calibrated to today's price (implied PONV re-solved per κ)",
            transform=ax.transAxes, fontsize=8.5, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 0.08))
    fig.tight_layout()
    fig.savefig(RES / "capital_cliff_uncertainty.png", dpi=150)
    print("saved results/capital_cliff_uncertainty.png")


if __name__ == "__main__":
    main()
