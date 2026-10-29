"""Numerical diagnostics for the extended paper (valuation date).

1. Monte Carlo standard error of each AT1 price at its implied hazard, and of the
   implied hazard itself (delta method: se(lam) = se(V) / |dV/dlam|).
2. Convergence of the implied hazards in the number of paths (5k-40k, independent
   seeds) and in the time step (monthly vs fortnightly grid).
3. Analytic moments of the CET1 process and of the spread process, and the
   probability that the Gaussian spread is negative (where the hazard is floored).
4. Price sensitivity to the hazard (points per bp), used to translate hazard
   dispersion into price units.

Output: results/numerics.csv (long format: block, item, bond, value).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_bnp import (  # noqa: E402
    BASE_SPREAD_VOL, ORDER, RES, SHORT, base_cet1_dynamics, market_panel, settings,
)

from at1_coco import data as D  # noqa: E402
from at1_coco.market import (  # noqa: E402
    DatedAT1, MarketState, cont_rate, default_randoms, implied_ponv,
)


def main():
    static = D.bonds_static()
    bonds = {i: DatedAT1.from_row(i, static.loc[i]) for i in ORDER}
    kappa, sigma = base_cet1_dynamics(verbose=True)
    panel = market_panel(curve="fred")
    long_run = float(panel.groupby("date")["refi"].first().mean())
    ms = settings(kappa, sigma, BASE_SPREAD_VOL, long_run)
    val_date = panel["date"].max()
    last = panel[panel["date"] == val_date].set_index("isin")
    mkts = {i: MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
            for i, r in last.iterrows()}
    rows = []

    def add(block, item, bond, value):
        rows.append(dict(block=block, item=item, bond=bond, value=value))

    # 1. standard errors and price sensitivity at 20k paths --------------------
    randoms = default_randoms(n_paths=20_000)
    for i in ORDER:
        lam, prof = implied_ponv(bonds[i], mkts[i], ms, randoms)
        r = cont_rate(mkts[i].discount_yield)
        pv = (prof.cpn * np.exp(-r * prof.t_cpn[None, :] - lam * prof.i_cpn)).sum(axis=1)
        pv += prof.pay_end * np.exp(-r * prof.t_end - lam * prof.i_end)
        tail = prof.tail_coupon / (r + lam * prof.tail_driver)
        pv += np.where(prof.survive, tail * np.exp(-r * prof.t_end - lam * prof.i_end), 0.0)
        se_v = pv.std(ddof=1) / np.sqrt(pv.size)
        dv = (prof.dirty_price(r, lam + 1e-4) - prof.dirty_price(r, lam - 1e-4)) / 2e-4   # per unit of hazard
        add("mc", "implied_hazard_pct", SHORT[i], 100 * lam)
        add("mc", "price_se_pts", SHORT[i], se_v)
        add("mc", "hazard_se_bp", SHORT[i], 1e4 * se_v / abs(dv))
        add("mc", "pts_per_bp_hazard", SHORT[i], -dv * 1e-4)
        add("mc", "pathwise_price_sd_pts", SHORT[i], pv.std(ddof=1))
        print(f"{SHORT[i]}: lam {100*lam:.3f}%  se(V) {se_v:.3f}pt  se(lam) {1e4*se_v/abs(dv):.2f}bp  "
              f"dV/dlam {-dv*1e-4:.3f}pt/bp")

    # 2a. convergence in paths (independent seeds) ------------------------------
    for n in [5_000, 10_000, 20_000, 40_000]:
        for seed in [11, 23]:
            rnd = default_randoms(n_paths=n, seed=seed)
            for i in ORDER:
                lam, _ = implied_ponv(bonds[i], mkts[i], ms, rnd)
                add("paths", f"n={n},seed={seed}", SHORT[i], 100 * lam)
            del rnd
        print("paths", n, "done")
    # 2b. time step: fortnightly grid
    ms_f = replace(ms, dt=1 / 24)
    rnd_f = default_randoms(n_paths=20_000, horizon=40.0, dt=1 / 24)
    for i in ORDER:
        lam, _ = implied_ponv(bonds[i], mkts[i], ms_f, rnd_f)
        add("dt", "fortnightly", SHORT[i], 100 * lam)
    del rnd_f
    for i in ORDER:
        lam, _ = implied_ponv(bonds[i], mkts[i], ms, randoms)
        add("dt", "monthly", SHORT[i], 100 * lam)
    print("dt done")

    # 3. analytic moments ------------------------------------------------------
    c = ms.cet1
    lj, mj, sj = c.jump_intensity, c.jump_median, c.jump_vol
    # J = mJ exp(sJ Z - sJ^2/2): E[J] = mJ, E[J^2] = mJ^2 exp(sJ^2)
    ej, ej2 = mj, mj ** 2 * np.exp(sj ** 2)
    mean_inf = c.theta - lj * ej / kappa
    var_inf = (sigma ** 2 + lj * ej2) / (2 * kappa)
    add("moments", "cet1_stationary_mean", "", mean_inf)
    add("moments", "cet1_stationary_sd", "", np.sqrt(var_inf))
    add("moments", "cet1_diffusion_only_sd", "", sigma / np.sqrt(2 * kappa))
    add("moments", "jump_mean_pp", "", ej)
    add("moments", "jump_p95_pp", "", mj * np.exp(sj * norm.ppf(0.95) - sj ** 2 / 2))
    sp = ms.spread
    sd_s = sp.vol / np.sqrt(2 * sp.kappa)
    add("moments", "spread_stationary_sd_bp", "", 1e4 * sd_s)
    add("moments", "p_spread_negative_stationary", "", norm.cdf(-sp.long_run / sd_s))
    add("moments", "half_life_cet1_years", "", np.log(2) / kappa)
    add("moments", "half_life_spread_years", "", np.log(2) / sp.kappa)
    print(f"CET1 stationary mean {mean_inf:.2f}, sd {np.sqrt(var_inf):.2f} (diffusion only "
          f"{sigma/np.sqrt(2*kappa):.2f}); spread sd {1e4*sd_s:.0f}bp, P(S<0) {norm.cdf(-sp.long_run/sd_s):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "numerics.csv", index=False)
    piv = out[out.block.isin(["paths", "dt"])].pivot_table(index=["block", "item"], columns="bond", values="value")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
