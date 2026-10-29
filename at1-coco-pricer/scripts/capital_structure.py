"""Anchor the AT1 hazard to BNP's own credit risk: Tier 2, senior non-preferred and CDS.

The AT1 implied PONV hazard (calibrate_bnp.py) absorbs everything the model's
mechanical features do not explain.  Under EU resolution the creditor hierarchy
holds: at the point of non-viability CET1 absorbs losses first, then AT1, then
Tier 2.  BNP's Tier 2 bonds are therefore exposed to the same resolution event
as the AT1s, without coupon discretion, MDA, trigger or perpetual extension.

Pricing the T2 (and the SNP) on the same simulated hazard driver gives the
part of the AT1 hazard that is plain bank credit risk, ``lam_res``; the rest,
``lam_spec = lam_AT1 - lam_res``, is what the AT1 pays for being an AT1.

Outputs in ``results/``:
  capital_structure_ladder.csv      CDS -> SNP -> T2 -> AT1 on the valuation date
  recovery_sensitivity.csv          lam_res and its share of the AT1 hazard vs T2 recovery
  resolution_hazard_timeseries.csv  weekly lam_res (each senior bond) and lam_spec (each AT1)
  hazard_vs_bidask.csv              cross-AT1 hazard dispersion in price points vs bid-ask
  capital_structure.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_bnp import (  # noqa: E402
    ANCHOR, BASE_SPREAD_VOL, ORDER, RECOVERY, RES, SENIOR_SHORT, SHORT,
    base_cet1_dynamics, implied_for_row, market_panel, resolution_hazards, senior_bonds, senior_panel,
    settings,
)

from at1_coco import data as D  # noqa: E402
from at1_coco.market import (  # noqa: E402
    DatedAT1, MarketState, cont_rate, default_randoms, hazard_state, implied_resolution_hazard,
)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RES_BLUE, SPEC_BLUE = "#1f4e8c", "#7fb0e6"
SENIOR_COLOR = {"US09659T2D24": RES_BLUE, "US09659T2B67": "#8a8986", "US09659W2X16": "#c98500"}
CDS_RECOVERY = 0.40
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})


def state_for(day_row, ms, randoms):
    mkt = MarketState(day_row.date, np.nan, np.nan, day_row.reset_ref, day_row.refi, day_row.cet1,
                      day_row.share_usd)
    return hazard_state(mkt, ms, randoms)


def main():
    static = D.bonds_static()
    at1 = {i: DatedAT1.from_row(i, static.loc[i]) for i in ORDER}
    sen = senior_bonds()
    kappa, sigma = base_cet1_dynamics(verbose=True)
    panel = market_panel(curve="fred")
    long_run = float(panel.groupby("date")["refi"].first().mean())
    ms = settings(kappa, sigma, BASE_SPREAD_VOL, long_run)
    randoms = default_randoms(n_paths=20_000)
    sp = senior_panel(panel)
    val_date = panel["date"].max()
    last = panel[panel["date"] == val_date]
    sp_last = sp[sp["date"] == val_date].set_index("isin")
    md = D.market_daily()
    md_last = md[md["date"] == val_date].set_index("isin")

    # 1. hazards on the valuation date --------------------------------------
    lam_at1, prof_at1 = {}, {}
    for r in last.itertuples():
        lam_at1[r.isin], prof_at1[r.isin] = implied_for_row(r, at1, ms, randoms)
    state = state_for(last.iloc[0], ms, randoms)
    lam_sen = resolution_hazards(sp_last.reset_index(), sen, ms, randoms, state=state)
    lam_res = lam_sen[ANCHOR]
    lam_at1_avg = float(np.mean(list(lam_at1.values())))

    curves = D.cds_curves()
    cds_day = curves.index[curves.index >= val_date][0]          # first snapshot on/after the date
    cds5 = float(md_last["cds_snr_eur_5y_bp"].iloc[0])
    ladder = [
        dict(instrument="CDS senior EUR 5Y", spread_bp=cds5, recovery=CDS_RECOVERY,
             hazard_pct=cds5 / 100 / (1 - CDS_RECOVERY), method="credit triangle s/(1-R)", bid_ask_pts=np.nan),
        dict(instrument=f"CDS senior EUR 7Y ({cds_day:%d-%b})", spread_bp=float(curves.at[cds_day, "7Y"]),
             recovery=CDS_RECOVERY, hazard_pct=float(curves.at[cds_day, "7Y"]) / 100 / (1 - CDS_RECOVERY),
             method="credit triangle s/(1-R)", bid_ask_pts=np.nan),
        dict(instrument=f"CDS senior EUR 10Y ({cds_day:%d-%b})", spread_bp=float(curves.at[cds_day, "10Y"]),
             recovery=CDS_RECOVERY, hazard_pct=float(curves.at[cds_day, "10Y"]) / 100 / (1 - CDS_RECOVERY),
             method="credit triangle s/(1-R)", bid_ask_pts=np.nan),
    ]
    for i in ["US09659W2X16", "US09659T2B67", "US09659T2D24"]:
        ladder.append(dict(instrument=SENIOR_SHORT[i], spread_bp=float(md_last.at[i, "spread_ytc_vs_fred_bp"]),
                           recovery=RECOVERY[sen[i].rank], hazard_pct=100 * lam_sen[i],
                           method="model, same hazard driver as the AT1s", bid_ask_pts=float(md_last.at[i, "bid_ask_pts"])))
    for i in ORDER:
        ladder.append(dict(instrument=f"AT1 {SHORT[i]}", spread_bp=float(md_last.at[i, "spread_ytc_vs_fred_bp"]),
                           recovery=0.0, hazard_pct=100 * lam_at1[i], method="model (implied PONV)",
                           bid_ask_pts=float(md_last.at[i, "bid_ask_pts"])))
    ladder = pd.DataFrame(ladder)
    ladder["share_of_at1_hazard"] = ladder["hazard_pct"] / (100 * lam_at1_avg)
    ladder.to_csv(RES / "capital_structure_ladder.csv", index=False)
    print(f"\ncapital-structure ladder on {val_date.date()} (CDS curve snapshot {cds_day.date()}):")
    print(ladder.round(3).to_string(index=False))
    print(f"\nAT1 hazard {100*lam_at1_avg:.2f}% = resolution {100*lam_res:.2f}% (T2 anchor, R=20%) "
          f"+ AT1-specific {100*(lam_at1_avg-lam_res):.2f}%  -> T2 share {lam_res/lam_at1_avg:.0%}")

    # 2. recovery sensitivity -------------------------------------------------
    rows = []
    for rec in [0.0, 0.1, 0.2, 0.3, 0.4]:
        lr = resolution_hazards(sp_last.reset_index(), sen, ms, randoms,
                                recovery={"T2": rec, "SNP": RECOVERY["SNP"]}, state=state)[ANCHOR]
        rows.append(dict(t2_recovery=rec, lam_res_pct=100 * lr, lam_spec_pct=100 * (lam_at1_avg - lr),
                         res_share_of_at1_hazard=lr / lam_at1_avg))
    rs = pd.DataFrame(rows)
    # T2 recovery at which T2 and SNP (R=40%) imply the same resolution hazard
    snp = sen["US09659W2X16"]
    t2 = sen[ANCHOR]

    def gap(rec):
        a, _ = implied_resolution_hazard(t2, sp_last.at[ANCHOR, "clean"], sp_last.at[ANCHOR, "disc_yield"],
                                         state, val_date, ms.dt, rec)
        return a - lam_sen["US09659W2X16"]
    try:
        r_star = brentq(gap, 0.0, 0.6)
    except ValueError:
        r_star = np.nan
    rs.attrs["r_star"] = r_star
    rs.to_csv(RES / "recovery_sensitivity.csv", index=False)
    print("\nrecovery sensitivity (T2 anchor; SNP at 40%):")
    print(rs.round(3).to_string(index=False))
    print(f"T2 recovery that equalises T2 and SNP hazards: {r_star:.1%}")
    _ = snp

    # 3. dispersion vs bid-ask ------------------------------------------------
    rows = []
    disp = 1e4 * (max(lam_at1.values()) - min(lam_at1.values()))
    for i in ORDER:
        r = last.set_index("isin").loc[i]
        rr = cont_rate(r.disc_yield)
        d1 = prof_at1[i].dirty_price(rr, lam_at1[i] + 1e-4) - prof_at1[i].dirty_price(rr, lam_at1[i])
        rows.append(dict(isin=i, bond=SHORT[i], pts_per_bp_hazard=-d1, dispersion_bp=disp,
                         dispersion_pts=-d1 * disp, bid_ask_pts=float(md_last.at[i, "bid_ask_pts"])))
    hb = pd.DataFrame(rows)
    hb.to_csv(RES / "hazard_vs_bidask.csv", index=False)
    print("\ncross-AT1 hazard dispersion in price terms:")
    print(hb.round(3).to_string(index=False))

    # 4. weekly series ----------------------------------------------------------
    ts_file = RES / "calibration_timeseries.csv"
    if ts_file.exists():
        ts = pd.read_csv(ts_file, parse_dates=["date"])
        ts = ts[ts["model"].str.startswith("stochastic")]
        at1_w = ts.pivot_table(index="date", columns="isin", values="implied_ponv")
    else:                                   # quick mode: valuation date only
        at1_w = pd.DataFrame({i: [lam_at1[i]] for i in ORDER}, index=[val_date])
    out = []
    for day, rows_ in sp.groupby("date"):
        if day not in at1_w.index:
            continue
        st_ = state_for(panel[panel["date"] == day].iloc[0], ms, randoms)
        lr = resolution_hazards(rows_, sen, ms, randoms, state=st_)
        rec = {"date": day, **{f"lam_res_{i}": v for i, v in lr.items()}}
        for i in ORDER:
            rec[f"lam_at1_{i}"] = at1_w.at[day, i] if i in at1_w.columns else np.nan
        out.append(rec)
    w = pd.DataFrame(out).set_index("date").sort_index()
    w["lam_at1_avg"] = w[[f"lam_at1_{i}" for i in ORDER]].mean(axis=1)
    w["lam_spec_avg"] = w["lam_at1_avg"] - w[f"lam_res_{ANCHOR}"]
    w["res_share"] = w[f"lam_res_{ANCHOR}"] / w["lam_at1_avg"]
    w.to_csv(RES / "resolution_hazard_timeseries.csv")
    print("\nweekly: resolution share of the AT1 hazard",
          w["res_share"].describe()[["mean", "min", "max"]].round(3).to_dict())

    fig_capital_structure(ladder, w, val_date, lam_res, lam_at1_avg)


def fig_capital_structure(ladder, w, val_date, lam_res, lam_at1_avg):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})
    # left: the ladder of implied hazards
    lad = ladder[~ladder["instrument"].str.startswith("CDS senior EUR 10Y")].copy()
    lad = lad[~lad["instrument"].str.contains("2.588")]
    labels = [s.replace("CDS senior EUR ", "CDS ") for s in lad["instrument"]]
    y = np.arange(len(lad))[::-1]
    cols = [INK2 if s.startswith("CDS") else (SENIOR_COLOR["US09659W2X16"] if s.startswith("SNP")
            else (RES_BLUE if s.startswith("T2") else SPEC_BLUE)) for s in lad["instrument"]]
    a1.barh(y, lad["hazard_pct"], color=cols, height=0.6)
    for yi, (h, s, rec) in enumerate(zip(lad["hazard_pct"], lad["spread_bp"], lad["recovery"])):
        a1.text(h + 0.03, y[yi], f"{h:.2f}%  ({s:.0f}bp, R={rec:.0%})", va="center", fontsize=8, color=INK)
    a1.axvline(100 * lam_res, color=RES_BLUE, ls=":", lw=1)
    a1.set_yticks(y, labels, fontsize=8.5)
    a1.set_xlabel("implied hazard (% per year)")
    a1.set_xlim(0, max(lad["hazard_pct"]) * 1.6)
    a1.grid(axis="y", visible=False)
    a1.set_title(f"BNP capital structure, {val_date:%d %b %Y}", loc="left", fontweight="bold", fontsize=10)

    # right: AT1 hazard over time = T2-priced resolution + AT1-specific
    res = 100 * w[f"lam_res_{ANCHOR}"]
    tot = 100 * w["lam_at1_avg"]
    a2.fill_between(w.index, 0, res, color=RES_BLUE, alpha=0.85, lw=0, label="resolution risk priced in T2 2035NC34")
    a2.fill_between(w.index, res, tot, color=SPEC_BLUE, alpha=0.85, lw=0, label="AT1-specific premium")
    a2.plot(w.index, 100 * w["lam_res_US09659W2X16"], color=SENIOR_COLOR["US09659W2X16"], lw=1.4,
            label="SNP 2035NC34 (R=40%)")
    a2.plot(w.index, 100 * w["lam_res_US09659T2B67"], color=SENIOR_COLOR["US09659T2B67"], lw=1.2, ls="--",
            label="T2 2035NC30 (R=20%)")
    a2.set_ylabel("hazard (% per year)")
    a2.set_ylim(0, tot.max() * 1.3)
    a2.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    a2.tick_params(axis="x", labelrotation=30)
    a2.set_title(f"AT1 hazard = {100*lam_res:.2f}% bank resolution + {100*(lam_at1_avg-lam_res):.2f}% AT1-specific"
                 " (last date)", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "capital_structure.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
