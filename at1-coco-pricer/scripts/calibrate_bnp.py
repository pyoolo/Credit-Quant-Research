"""Calibrate the AT1 model to BNP Paribas's three USD AT1s.

Steps
-----
1. Historical CET1 dynamics: exact-OU maximum likelihood on the CET1 series
   (annual 2014-2023, quarterly 2024-2026), long-run level fixed at the 13%
   management target.
2. Market inputs per date: prices, the FRED Treasury curve (5y CMT for the
   reset, call-matched yield for discounting; file ``data/fred_treasury.csv``),
   the current AT1 spread, latest CET1, share price.
3. Implied PONV hazard for each bond on each date.  If the model captured the
   bonds' relative value, the three implied hazards would coincide (same issuer,
   same capital, same trigger).  The gap between them is the test.
4. The cross-section identifies the refinancing-spread volatility: the value
   that makes the three hazards closest is the market-implied "extension vol".

Outputs: ``results/calibration_*.csv`` and ``results/implied_ponv.png``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find at1_coco without install

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from at1_coco import CET1Params
from at1_coco import data as D
from at1_coco.market import (
    DatedAT1, DatedBullet, MarketState, ModelSettings, SpreadParams, cash_flow_profile,
    default_randoms, hazard_state, implied_ponv, implied_resolution_hazard,
)

ROOT = D.ROOT
DATA, RES = D.DATA, D.RES
ORDER = ["US05602XQR25", "US05602XQS08", "US05602XQQ42"]
SHORT = {"US05602XQR25": "6.875% NC33", "US05602XQS08": "7.20% NC36", "US05602XQQ42": "7.45% NC35"}
COLOR = {"US05602XQR25": "#2a78d6", "US05602XQS08": "#eb6834", "US05602XQQ42": "#1baf7a"}

# EUR/USD: approximation implied by CIQ's EUR amounts outstanding at 30-Jun-2026
# ($1.5bn = EUR 1.304bn). Replace with FRED DEXUSEU for a daily series.
EURUSD = 1.15


# ---------------------------------------------------------------------------
BASEL3_START = "2014-01-01"      # CET1 as defined today exists from Basel III (2014)
# quarters whose change is a disposal, not a capital shock: excluded from the fit
ONE_OFF_QUARTERS = ["2023-03-31"]  # sale of Bank of the West (+1.2pp)


def load_cet1_series() -> pd.Series:
    """Quarterly CET1 ratio since Basel III (2014Q1-2026Q2, ~50 observations)."""
    q = D.cet1_quarterly()["cet1_ratio"]          # prints published by the valuation date
    return q.dropna()[q.dropna().index >= BASEL3_START]


def fit_cet1_ou(c: pd.Series, theta: float = 13.0,
                skip: list[str] = ONE_OFF_QUARTERS) -> tuple[float, float]:
    """Exact-OU maximum likelihood; transitions ending on ``skip`` dates are ignored."""
    t = np.asarray((c.index - c.index[0]).days / 365.25)
    x = c.values
    keep = ~np.isin(c.index[1:], pd.to_datetime(skip))

    def nll(p):
        k, s = np.exp(p)
        dt = np.diff(t)
        m = theta + (x[:-1] - theta) * np.exp(-k * dt)
        v = s * s * (1 - np.exp(-2 * k * dt)) / (2 * k)
        return 0.5 * np.sum((np.log(2 * np.pi * v) + (x[1:] - m) ** 2 / v)[keep])

    r = minimize(nll, np.log([0.5, 0.5]), method="Nelder-Mead")
    k, s = np.exp(r.x)
    return float(k), float(s)


# Base-case speed of capital mean reversion.  History alone gives 0.165
# (90% bootstrap CI 0.06-0.48); the three bond prices are only mutually
# consistent for faster mean reversion (see parameter_uncertainty.py).  0.35 is
# inside the historical CI and close to the market plateau: the value both
# sources of information can live with.  sigma is re-estimated given kappa.
BASE_KAPPA = 0.35

# Base-case volatility of the new-issue AT1 spread (decimal).  With the FRED
# curve the cross-bond gap keeps shrinking as this vol rises (no interior
# minimum), so the data do not pin it down: 100bp is an assumption, shown with
# its sensitivity grid in the calibration output.
BASE_SPREAD_VOL = 0.01


def sigma_given_kappa(c: pd.Series, kappa: float, theta: float = 13.0,
                      skip: list[str] = ONE_OFF_QUARTERS) -> float:
    """Maximum-likelihood sigma of the CET1 OU process for a fixed kappa."""
    t = np.asarray((c.index - c.index[0]).days / 365.25)
    x = c.values
    keep = ~np.isin(c.index[1:], pd.to_datetime(skip))
    dt = np.diff(t)
    m = theta + (x[:-1] - theta) * np.exp(-kappa * dt)
    w = (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)

    def nll(ls):
        v = np.exp(2 * ls) * w
        return 0.5 * np.sum((np.log(2 * np.pi * v) + (x[1:] - m) ** 2 / v)[keep])

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(3.0)), method="bounded")
    return float(np.exp(r.x))


def base_cet1_dynamics(verbose: bool = False) -> tuple[float, float]:
    """(kappa, sigma) used in the base case."""
    c = load_cet1_series()
    k_hist, s_hist = fit_cet1_ou(c)
    sigma = sigma_given_kappa(c, BASE_KAPPA)
    if verbose:
        print(f"CET1 history ({len(c)} quarters): MLE kappa={k_hist:.3f} sigma={s_hist:.3f} | "
              f"base case kappa={BASE_KAPPA:.3f} sigma={sigma:.3f}")
    return BASE_KAPPA, sigma


def load_fred_curve() -> pd.DataFrame:
    """Daily constant-maturity Treasury yields (%) from FRED, tenors 5/7/10y."""
    return D.fred_curve()


def curve_yield(curve_row: pd.Series, tenor: float) -> float:
    """Linear interpolation on the 5/7/10y points, flat outside."""
    return float(np.interp(tenor, [5.0, 7.0, 10.0], curve_row[["DGS5", "DGS7", "DGS10"]].values))


def market_panel(freq: str = "W-FRI", curve: str = "fred") -> pd.DataFrame:
    """One row per (date, isin) with all model inputs.

    ``curve="fred"`` (baseline): FRED constant-maturity Treasuries. The reset
    reference is the 5y CMT (as in the term sheets); each bond is discounted at
    the Treasury yield interpolated at its time to first call; the AT1 spread is
    the average of (bond yield - call-matched Treasury).

    Sensitivities using only Capital IQ data (no FRED): ``curve="ust10"`` uses one
    continuous ~10y series for everything; ``curve="matched"`` discounts the
    6.875% on its own CIQ benchmark (a 7y note since Jan-2026).
    """
    p = D.bond_prices()
    wide = {c: p.pivot(index="date", columns="isin", values=c) for c in ["mid_price", "mid_yield", "benchmark_yield"]}
    static = D.bonds_static()
    static["first_call_date"] = pd.to_datetime(static["first_call_date"])
    cet1 = D.cet1_quarterly()["cet1_ratio"].dropna()
    share = D.share_eur()

    def years_to_call(isin, day):
        return (static.at[isin, "first_call_date"] - day).days / 365.25

    if curve == "fred":
        fred = load_fred_curve()

        def treasuries(day):
            row = fred[fred.index <= day].iloc[-1]          # last FRED print on or before the day
            disc = {i: curve_yield(row, years_to_call(i, day)) for i in ORDER}
            return disc, float(row["DGS5"])
    else:
        # Capital IQ only: ~10y = benchmark of the 7.45% (continuous); shorter =
        # benchmark of the 6.875% (a 2033 note), used as the 5y-CMT proxy
        ust10 = wide["benchmark_yield"]["US05602XQQ42"]
        ust_short = wide["benchmark_yield"]["US05602XQR25"].fillna(ust10)
        if curve == "ust10":
            ust_short = ust10

        def treasuries(day):
            disc = {i: (ust_short.at[day] if i == "US05602XQR25" else ust10.at[day]) for i in ORDER}
            return disc, float(ust_short.at[day])

    dates = wide["mid_price"].resample(freq).last().index
    rows = []
    for d in dates:
        day = wide["mid_price"].index[wide["mid_price"].index <= d].max()
        # CET1 is known ~1 month after quarter end: use the last quarter published
        pub = cet1[cet1.index + pd.Timedelta(days=35) <= day]
        if pub.empty:
            continue
        disc, ref = treasuries(day)
        quoted = [i for i in ORDER if not np.isnan(wide["mid_price"].at[day, i])]
        # current AT1 spread: average over quoted bonds of yield minus the Treasury
        # used to discount that bond
        refi = float(np.mean([wide["mid_yield"].at[day, i] - disc[i] for i in quoted]))
        for isin in quoted:
            rows.append(
                dict(date=day, isin=isin, clean=wide["mid_price"].at[day, isin], disc_yield=disc[isin] / 100,
                     reset_ref=ref / 100, refi=refi / 100, cet1=pub.iloc[-1],
                     share_usd=share[share.index <= day].iloc[-1] * EURUSD)
            )
    return pd.DataFrame(rows)


# MDA geometry.  Floor = P1 + P2R (CET1 part) from the 30-Sep-2025 stack; the
# combined buffer is set so that floor + buffer equals the latest published CET1
# requirement (10.43% at 30-Jun-2026).  The O-SII buffer then steps up by 25bp on
# 1-Jan-2027 and 1-Jan-2028 (the higher of G-SIB and O-SII applies).
MDA_FLOOR = 5.64
OSII_STEPS = (("2027-01-01", 0.25), ("2028-01-01", 0.25))


def mda_requirement_today() -> float:
    return float(D.requirement_series().iloc[-1])


def settings(kappa: float, sigma: float, spread_vol: float, long_run: float,
             buffer_steps: tuple = OSII_STEPS) -> ModelSettings:
    cet1 = CET1Params(c0=13.0, kappa=kappa, theta=13.0, sigma=sigma, jump_intensity=0.12,
                      jump_median=2.2, jump_vol=0.5, ponv_intensity=0.0)
    return ModelSettings(cet1=cet1, spread=SpreadParams(kappa=0.5, long_run=long_run, vol=spread_vol),
                         mda_threshold=MDA_FLOOR, combined_buffer=mda_requirement_today() - MDA_FLOOR,
                         buffer_steps=buffer_steps)


# ---------------------------------------------------------------------------
# capital-structure anchor: BNP's Tier 2 and senior non-preferred bonds
# ---------------------------------------------------------------------------
# T2 5.906% 2035NC34 is the anchor: its call (Nov-2034) sits inside the AT1 call
# window (2033-2036).  The 2.588% T2 (call 2030) and the SNP (call 2034) are checks.
SENIOR_ORDER = ["US09659T2D24", "US09659T2B67", "US09659W2X16"]
SENIOR_SHORT = {"US09659T2D24": "T2 5.906% 2035NC34", "US09659T2B67": "T2 2.588% 2035NC30",
                "US09659W2X16": "SNP 5.738% 2035NC34"}
ANCHOR = "US09659T2D24"
# recovery at resolution: Markit conventions, subordinated 20% and senior 40%
RECOVERY = {"T2": 0.20, "SNP": 0.40}


def senior_bonds() -> dict:
    st = D.senior_t2_static()
    return {i: DatedBullet.from_row(i, st.loc[i]) for i in SENIOR_ORDER}


def senior_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """T2/SNP rows on the AT1 panel dates, with the same state inputs (AT1 spread, CET1)."""
    m = D.market_daily()
    m = m[m["isin"].isin(SENIOR_ORDER)]
    st = D.senior_t2_static()
    fred = load_fred_curve()
    state = panel.groupby("date")[["refi", "cet1", "reset_ref", "share_usd"]].first()
    rows = []
    for day, s_ in state.iterrows():
        q = m[m["date"] == day]
        row = fred[fred.index <= day].iloc[-1]
        for r in q.itertuples():
            ttc = (pd.Timestamp(st.at[r.isin, "first_call_date"]) - day).days / 365.25
            rows.append(dict(date=day, isin=r.isin, rank=r.rank, clean=r.mid_price,
                             disc_yield=curve_yield(row, ttc) / 100, years_to_call=ttc,
                             refi=s_.refi, cet1=s_.cet1, reset_ref=s_.reset_ref, share_usd=s_.share_usd,
                             bid_ask=r.bid_ask_pts))
    return pd.DataFrame(rows)


def resolution_hazards(day_rows: pd.DataFrame, bonds: dict, ms, randoms, recovery: dict | None = None,
                       state=None) -> dict:
    """Implied resolution hazard of each T2/SNP bond on one date (shared hazard state)."""
    recovery = recovery or RECOVERY
    r0 = day_rows.iloc[0]
    mkt = MarketState(r0.date, np.nan, np.nan, r0.reset_ref, r0.refi, r0.cet1, r0.share_usd)
    state = state or hazard_state(mkt, ms, randoms)
    out = {}
    for r in day_rows.itertuples():
        b = bonds[r.isin]
        out[r.isin], _ = implied_resolution_hazard(b, r.clean, r.disc_yield, state, r.date, ms.dt,
                                                   recovery[b.rank])
    return out


def implied_for_row(row, bonds, ms, randoms):
    bond = bonds[row.isin]
    mkt = MarketState(row.date, row.clean, row.disc_yield, row.reset_ref, row.refi, row.cet1, row.share_usd)
    lam, prof = implied_ponv(bond, mkt, ms, randoms)
    return lam, prof


def main():
    RES.mkdir(exist_ok=True)
    static = D.bonds_static()
    bonds = {i: DatedAT1.from_row(i, static.loc[i]) for i in ORDER}

    # 1. historical CET1 dynamics ------------------------------------------
    kappa, sigma = base_cet1_dynamics(verbose=True)

    panel = market_panel(curve="fred")
    long_run = float(panel.groupby("date")["refi"].first().mean())
    print(f"panel: {panel['date'].nunique()} weekly dates, mean AT1 spread {1e4*long_run:.0f}bp")

    randoms = default_randoms(n_paths=20_000)

    # 2. last date: implied PONV per bond for a grid of spread vols ----------
    last = panel[panel["date"] == panel["date"].max()]
    grid = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
    rows = []
    for v in grid:
        ms = settings(kappa, sigma, v, long_run)
        for r in last.itertuples():
            lam, prof = implied_for_row(r, bonds, ms, randoms)
            rows.append(dict(spread_vol_bp=1e4 * v, isin=r.isin, implied_ponv=lam, **prof.diagnostics))
    xs = pd.DataFrame(rows)
    xs.to_csv(RES / "calibration_cross_section.csv", index=False)
    piv = xs.pivot(index="spread_vol_bp", columns="isin", values="implied_ponv")[ORDER]
    piv["dispersion_bp"] = 1e4 * (piv.max(axis=1) - piv.min(axis=1))
    print("\nimplied PONV hazard (per year) on", last["date"].iloc[0].date())
    print(piv.round(4).to_string())
    # the vol that makes the three bonds most consistent (U-shaped dispersion);
    # the base case uses the grid point closest to it
    best_v = BASE_SPREAD_VOL
    print(f"base-case spread vol: {1e4 * best_v:.0f}bp (grid above = sensitivity; "
          f"lowest gap on the grid at {piv['dispersion_bp'].idxmin():.0f}bp)")

    # sensitivity of the last-date cross-section to the curve and refi level
    sens = []
    ms = settings(kappa, sigma, 0.01, long_run)
    for curve in ["fred", "ust10", "matched"]:
        lp = market_panel(curve=curve)
        lp = lp[lp["date"] == lp["date"].max()]
        for shift in [-0.002, 0.0, 0.002]:
            for r in lp.assign(refi=lp["refi"] + shift).itertuples():
                lam, prof = implied_for_row(r, bonds, ms, randoms)
                sens.append(dict(curve=curve, refi_shift_bp=1e4 * shift, isin=r.isin, implied_ponv=lam,
                                 p_call_first=prof.diagnostics["p_call_first"]))
    sens = pd.DataFrame(sens)
    sens.to_csv(RES / "calibration_sensitivity.csv", index=False)
    sp = sens.pivot_table(index=["curve", "refi_shift_bp"], columns="isin", values="implied_ponv")[ORDER]
    sp["dispersion_bp"] = 1e4 * (sp.max(axis=1) - sp.min(axis=1))
    print("\nsensitivity (100bp spread vol):\n", sp.round(4).to_string())

    # 3. time series of implied PONV: deterministic vs best stochastic -------
    out = []
    for label, v in [("deterministic refi spread", 0.0), (f"stochastic refi spread ({1e4*best_v:.0f}bp vol)", best_v)]:
        ms = settings(kappa, sigma, v, long_run)
        # one date at a time: path-wise profiles are large, keep only numbers.
        # Relative value: price every bond with the SAME hazard (average of the
        # bonds quoted that day); the pricing error is the bond's rich/cheapness.
        from at1_coco.market import cont_rate
        for d, day in panel.groupby("date"):
            res = []
            for r in day.itertuples():
                lam, prof = implied_for_row(r, bonds, ms, randoms)
                res.append((r, lam, prof))
            common = float(np.nanmean([x[1] for x in res]))
            for r, lam, prof in res:
                acc = bonds[r.isin].accrued(r.date)
                model_clean = prof.dirty_price(cont_rate(r.disc_yield), common) - acc
                out.append(dict(model=label, date=r.date, isin=r.isin, implied_ponv=lam,
                                common_ponv=common, clean=r.clean, model_clean=model_clean,
                                cheapness_pts=model_clean - r.clean,   # >0: market below model = cheap
                                p_call_first=prof.diagnostics["p_call_first"],
                                p_conversion=prof.diagnostics["p_conversion"]))
            del res
    ts = pd.DataFrame(out)
    print("\nrich(-)/cheap(+) vs common-hazard model, price points")
    print(ts.groupby(["model", "isin"])["cheapness_pts"].agg(["mean", "min", "max", "last"]).round(2).to_string())
    ts.to_csv(RES / "calibration_timeseries.csv", index=False)

    summ = (ts.pivot_table(index=["model", "date"], columns="isin", values="implied_ponv")
              .assign(disp=lambda d: d.max(axis=1) - d.min(axis=1))
              .groupby("model")["disp"].agg(["mean", "median", "max"]) * 1e4)
    print("\ncross-bond dispersion of implied PONV (bp of hazard), all weekly dates\n", summ.round(1))

    # figure
    models = ts["model"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, m in zip(axes, models):
        g = ts[ts["model"] == m]
        for isin in ORDER:
            h = g[g["isin"] == isin]
            ax.plot(h["date"], 100 * h["implied_ponv"], color=COLOR[isin], lw=2, label=SHORT[isin])
        ax.set_title(m, fontsize=11, loc="left", fontweight="bold")
        ax.grid(True, color="#e4e3df", lw=0.6)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("Implied PONV hazard (% per year)")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Same issuer, same capital: do the three AT1s imply the same PONV risk?",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RES / "implied_ponv.png", dpi=150, facecolor="#fcfcfb")

    # 4. headline numbers on the last date under the best model ---------------
    ms = settings(kappa, sigma, best_v, long_run)
    ms0 = settings(kappa, sigma, best_v, long_run)
    print("\nlast date, best model:")
    for r in last.itertuples():
        bond = bonds[r.isin]
        mkt = MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
        lam, prof = implied_ponv(bond, mkt, ms, randoms)
        from at1_coco.market import cont_rate
        pv0 = prof.dirty_price(cont_rate(r.disc_yield), 0.0) - bond.accrued(r.date)
        print(f"  {SHORT[r.isin]:12s} clean {r.clean:7.3f}  price with no PONV {pv0:7.2f}  "
              f"implied PONV {100*lam:5.2f}%/y  P(called at 1st reset) {prof.diagnostics['p_call_first']:.0%}  "
              f"P(never called) {prof.diagnostics['p_never_called']:.0%}  P(conversion) {prof.diagnostics['p_conversion']:.2%}")
    _ = ms0


if __name__ == "__main__":
    main()
