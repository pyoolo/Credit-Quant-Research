"""Robustness of the headline results on the valuation date.

1. Reputational call.  The base rule calls only when the reset margin exceeds
   the new-issue spread.  With a tolerance delta the issuer also calls when the
   margin is up to delta below it.  For delta = 0 / 25 / 50 / 100bp: P(call at
   the first reset), the implied hazard, the extension component of the spread
   and the share of the hazard priced in Tier 2.

2. Order of the spread decomposition.  The base decomposition switches features
   on in a fixed order.  Here: the reverse order; the full Shapley value (the
   average contribution over all 5! orders, from the 2^5 subsets); and a nested
   Shapley value, symmetric within the hazard block (resolution vs AT1-specific)
   and within the contract features (MDA, conversion, extension), with the hazard
   block first.  Orders that switch extension on before any hazard value the
   option on a riskless bond, where extending at a fixed reset margin is a gift to
   the investor; the nested value keeps extension priced with risky discounting.

3. Term structure of the resolution hazard.  The two Tier 2 bonds call in 2030
   and 2034.  Bootstrapping a two-piece hazard (lam1 to Aug-2030 from the short
   T2, lam2 after from the long one) instead of one flat level, and re-solving
   the AT1-specific hazard on top of it.

Outputs in ``results/``: robustness_call.csv, robustness_decomposition.csv,
robustness_term_structure.csv.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_bnp import (  # noqa: E402
    ANCHOR, BASE_SPREAD_VOL, ORDER, RECOVERY, RES, SHORT, base_cet1_dynamics, market_panel,
    resolution_hazards, senior_bonds, senior_panel, settings,
)
from spread_and_stress import ytc  # noqa: E402

from at1_coco import data as D  # noqa: E402
from at1_coco.market import (  # noqa: E402
    DatedAT1, MarketState, at1_price_piecewise, bullet_price_piecewise, bullet_profile,
    cash_flow_profile, cont_rate, default_randoms, hazard_state, implied_ponv,
)

SHORT_T2 = "US09659T2B67"
FEATURES = ["R", "S", "M", "C", "E"]   # resolution, AT1-specific, MDA, conversion, extension
LABEL = {"R": "resolution risk (priced in T2)", "S": "AT1-specific premium", "M": "MDA coupon cuts",
         "C": "conversion", "E": "extension"}


def main():
    static = D.bonds_static()
    bonds = {i: DatedAT1.from_row(i, static.loc[i]) for i in ORDER}
    kappa, sigma = base_cet1_dynamics(verbose=True)
    panel = market_panel(curve="fred")
    long_run = float(panel.groupby("date")["refi"].first().mean())
    ms = settings(kappa, sigma, BASE_SPREAD_VOL, long_run)
    randoms = default_randoms(n_paths=20_000)
    val_date = panel["date"].max()
    last = panel[panel["date"] == val_date].set_index("isin")
    mkts = {i: MarketState(r.date, r.clean, r.disc_yield, r.reset_ref, r.refi, r.cet1, r.share_usd)
            for i, r in last.iterrows()}
    sp = senior_panel(panel)
    sp_last = sp[sp["date"] == val_date]
    state = hazard_state(mkts[ORDER[0]], ms, randoms)
    lam_res_all = resolution_hazards(sp_last, senior_bonds(), ms, randoms, state=state)
    lam_res = lam_res_all[ANCHOR]
    acc = {i: bonds[i].accrued(val_date) for i in ORDER}

    def spread_bp(i, dirty):
        r0 = ytc(bonds[i], val_date, rf_dirty[i])
        return 1e4 * (ytc(bonds[i], val_date, dirty) - r0)

    # risk-free reference (certain call, full coupons, no conversion, no hazard)
    rf_dirty = {}
    for i in ORDER:
        p = cash_flow_profile(bonds[i], mkts[i], ms, randoms, call_rule="first", mda=False, conversion=False)
        rf_dirty[i] = p.dirty_price(cont_rate(mkts[i].discount_yield), 0.0)

    # 1. reputational call --------------------------------------------------
    rows = []
    for d_bp in [0, 25, 50, 100]:
        ms_d = replace(ms, call_tolerance=d_bp / 1e4)
        for i in ORDER:
            lam, prof = implied_ponv(bonds[i], mkts[i], ms_d, randoms)
            r = cont_rate(mkts[i].discount_yield)
            first = cash_flow_profile(bonds[i], mkts[i], ms_d, randoms, call_rule="first")
            ext = spread_bp(i, prof.dirty_price(r, lam)) - spread_bp(i, first.dirty_price(r, lam))
            rows.append(dict(tolerance_bp=d_bp, isin=i, bond=SHORT[i], implied_hazard_pct=100 * lam,
                             p_call_first=prof.diagnostics["p_call_first"], extension_bp=ext,
                             res_share=lam_res / lam))
    rc = pd.DataFrame(rows)
    rc.to_csv(RES / "robustness_call.csv", index=False)
    print("\n1. reputational call tolerance:")
    print(rc.pivot_table(index="tolerance_bp", columns="bond",
                         values=["p_call_first", "implied_hazard_pct", "extension_bp"]).round(3).to_string())
    print("T2 share of the AT1 hazard (avg):", rc.groupby("tolerance_bp")["res_share"].mean().round(3).to_dict())

    # 2. order of the decomposition ----------------------------------------
    lam = {i: implied_ponv(bonds[i], mkts[i], ms, randoms)[0] for i in ORDER}
    rows = []
    for i in ORDER:
        r = cont_rate(mkts[i].discount_yield)
        profs = {(m, c, e): cash_flow_profile(bonds[i], mkts[i], ms, randoms, mda=m, conversion=c,
                                             call_rule="model" if e else "first")
                 for m in (False, True) for c in (False, True) for e in (False, True)}

        def v(subset):
            hz = (lam_res if "R" in subset else 0.0) + ((lam[i] - lam_res) if "S" in subset else 0.0)
            p = profs[("M" in subset, "C" in subset, "E" in subset)]
            return spread_bp(i, p.dirty_price(r, hz))

        vals = {frozenset(s): v(set(s)) for k in range(6) for s in itertools.combinations(FEATURES, k)}
        n = len(FEATURES)
        shap = {}
        for j in FEATURES:
            others = [f for f in FEATURES if f != j]
            tot = 0.0
            for k in range(n):
                for s in itertools.combinations(others, k):
                    w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                    tot += w * (vals[frozenset(s) | {j}] - vals[frozenset(s)])
            shap[j] = tot

        def sequential(order):
            out, cur = {}, set()
            for f in order:
                out[f] = vals[frozenset(cur | {f})] - vals[frozenset(cur)]
                cur.add(f)
            return out
        def shapley(players, given):
            """Shapley values of ``players`` with the features in ``given`` always on."""
            n_, out = len(players), {}
            for j in players:
                others = [f for f in players if f != j]
                tot = 0.0
                for k in range(n_):
                    for s_ in itertools.combinations(others, k):
                        w = math.factorial(k) * math.factorial(n_ - k - 1) / math.factorial(n_)
                        base_set = frozenset(s_) | frozenset(given)
                        tot += w * (vals[base_set | {j}] - vals[base_set])
                out[j] = tot
            return out
        # nested: the hazard block first (R vs S symmetric), then the contract features
        # (M, C, E symmetric) valued with the full hazard in place
        nested = {**shapley(["R", "S"], []), **shapley(["M", "C", "E"], ["R", "S"])}
        base_o = sequential(["R", "S", "M", "C", "E"])
        rev_o = sequential(["E", "C", "M", "S", "R"])
        for f in FEATURES:
            rows.append(dict(isin=i, bond=SHORT[i], component=LABEL[f], base_order_bp=base_o[f],
                             reverse_order_bp=rev_o[f], shapley_bp=shap[f], nested_shapley_bp=nested[f]))
        total = vals[frozenset(FEATURES)]
        rows.append(dict(isin=i, bond=SHORT[i], component="total", base_order_bp=total,
                         reverse_order_bp=total, shapley_bp=total, nested_shapley_bp=total))
    rd = pd.DataFrame(rows)
    rd.to_csv(RES / "robustness_decomposition.csv", index=False)
    print("\n2. spread decomposition by method (bp):")
    print(rd.pivot_table(index="component", columns="bond",
                         values=["base_order_bp", "nested_shapley_bp", "shapley_bp", "reverse_order_bp"]).round(1).to_string())

    # 3. term structure of the resolution hazard ---------------------------
    sen = senior_bonds()
    sl = sp_last.set_index("isin")
    t1 = sen[SHORT_T2].schedule(val_date)[-1]                    # years to the short T2's call
    n1 = int(round(t1 / ms.dt))
    i_t1 = state.I[:, n1]
    rec = RECOVERY["T2"]
    prof_s = bullet_profile(sen[SHORT_T2], state, val_date, ms.dt)
    prof_l = bullet_profile(sen[ANCHOR], state, val_date, ms.dt)
    tgt_s = sl.at[SHORT_T2, "clean"] + sen[SHORT_T2].accrued(val_date)
    tgt_l = sl.at[ANCHOR, "clean"] + sen[ANCHOR].accrued(val_date)
    r_s, r_l = cont_rate(sl.at[SHORT_T2, "disc_yield"]), cont_rate(sl.at[ANCHOR, "disc_yield"])
    lam1 = brentq(lambda x: bullet_price_piecewise(prof_s, r_s, x, x, i_t1, rec) - tgt_s, 0, 1)
    lam2 = brentq(lambda x: bullet_price_piecewise(prof_l, r_l, lam1, x, i_t1, rec) - tgt_l, 0, 1)
    rows = []
    for i in ORDER:
        prof = cash_flow_profile(bonds[i], mkts[i], ms, randoms)
        r = cont_rate(mkts[i].discount_yield)
        tgt = mkts[i].clean_price + acc[i]
        ls_flat = lam[i] - lam_res
        ls_pw = brentq(lambda x: at1_price_piecewise(prof, r, x, lam1, lam2, i_t1) - tgt, -0.05, 0.5)
        first = cash_flow_profile(bonds[i], mkts[i], ms, randoms, call_rule="first", mda=False, conversion=False)
        res_flat = spread_bp(i, first.dirty_price(r, lam_res))
        res_pw = spread_bp(i, at1_price_piecewise(first, r, 0.0, lam1, lam2, i_t1))
        rows.append(dict(isin=i, bond=SHORT[i], lam_res_flat_pct=100 * lam_res, lam1_pct=100 * lam1,
                         lam2_pct=100 * lam2, lam_spec_flat_pct=100 * ls_flat, lam_spec_piecewise_pct=100 * ls_pw,
                         res_component_flat_bp=res_flat, res_component_piecewise_bp=res_pw,
                         t1_years=t1))
    rt = pd.DataFrame(rows)
    rt.to_csv(RES / "robustness_term_structure.csv", index=False)
    print(f"\n3. two-piece resolution hazard: lam1 = {100*lam1:.2f}% to {t1:.1f}y, lam2 = {100*lam2:.2f}% after "
          f"(flat: {100*lam_res:.2f}%)")
    print(rt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
