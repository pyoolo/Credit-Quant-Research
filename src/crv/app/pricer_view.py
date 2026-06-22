"""Pricer view: interactive new-issue pricer over the fitted artefacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from crv.pricing import analytics, engine


@st.cache_resource
def _artefacts() -> engine.Artefacts:
    return engine.load_artefacts()


def render() -> None:
    st.header("New-Issue Pricer")
    art = _artefacts()
    if not art.t1_ok:
        st.warning("Track-1 model artefact not found. Track-1 pricing is disabled until "
                   "the Track-1 pipeline has been run.")

    sub_single, sub_scenario, sub_sens = st.tabs(["Single bond", "Scenario grid", "Sensitivity"])

    with sub_single:
        _single(art)
    with sub_scenario:
        _scenario(art)
    with sub_sens:
        _sensitivity(art)


def _spec_form() -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        rating = st.selectbox("Rating bucket", ["HY_BB", "HY_B", "HY_CCC"])
        coupon_type = st.selectbox("Coupon type", ["FIXED", "FLOATING", "PERP"])
        coupon = st.number_input("Coupon (%)", 0.0, 20.0, 5.5, 0.25)
    with c2:
        duration = st.number_input("Duration (yrs)", 0.1, 15.0, 4.2, 0.1)
        px = st.number_input("Price", 1.0, 130.0, 100.0, 0.5)
        yrs_to_call = st.number_input("Years to call", 0.0, 15.0, 3.0, 0.5)
    with c3:
        net_lev = st.number_input("Net leverage (x)", 0.0, 15.0, 3.8, 0.1)
        icr = st.number_input("Interest coverage (x)", 0.0, 20.0, 3.0, 0.1)
        proposed = st.number_input("Proposed Z-spread (bps, 0 = none)", 0.0, 3000.0, 0.0, 5.0)

    spec = dict(
        rating_bucket=rating,
        coupon_type=coupon_type,
        is_perp=(coupon_type == "PERP"),
        coupon=coupon, duration=duration, px=px,
        yrs_to_call=yrs_to_call, is_callable=yrs_to_call > 0,
        net_lev=net_lev, icr=icr,
        sector="OTHER",
    )
    if proposed > 0:
        spec["proposed_zs"] = proposed
    return spec


def _single(art: engine.Artefacts) -> None:
    spec = _spec_form()
    if st.button("Price", type="primary"):
        r = engine.route_and_price(spec, art)
        st.subheader(f"Track {r.get('track', '?')}")
        if r.get("track") == "1":
            cols = st.columns(4)
            cols[0].metric("Fair Z (bps)", f"{r['fair']:.0f}")
            if not np.isnan(r.get("lo", np.nan)):
                cols[1].metric("Low (q10)", f"{r['lo']:.0f}")
                cols[2].metric("High (q90)", f"{r['hi']:.0f}")
            if "verdict" in r:
                cols[3].metric("Verdict", r["verdict"])
            if r.get("npeers"):
                st.caption(f"Peer Z-spread quartiles ({r['npeers']} peers): "
                           f"p25 {r['p25']:.0f} / p50 {r['p50']:.0f} / p75 {r['p75']:.0f} bps")
        elif r.get("track") == "2":
            st.write(f"Family **{r['family']}** ({r['metric']})")
            if "verdict" in r:
                st.metric("Placement", r["verdict"], help=f"{r.get('pct', float('nan')):.0f}th percentile")
            elif r.get("note"):
                st.info(r["note"])
        elif r.get("track") == "3":
            st.write("Price-implied cumulative default probability:")
            scen = r.get("pd_scenarios", {})
            st.dataframe(pd.DataFrame(
                [{"Recovery": f"{k}%", "Cumulative PD": (f"{v * 100:.0f}%" if v is not None else "—")}
                 for k, v in scen.items()]), hide_index=True)


def _scenario(art: engine.Artefacts) -> None:
    st.caption("Vary two drivers and price every combination (Track-1 base bond).")
    base = dict(coupon_type="FIXED", rating_bucket="HY_BB", coupon=5.5, duration=4.2,
                px=100, yrs_to_call=3.0, is_callable=True, net_lev=3.8, sector="OTHER")
    grid = {
        "coupon": [4.5, 5.0, 5.5, 6.0, 6.5],
        "duration": [3.0, 4.0, 5.0],
    }
    df = analytics.scenario_grid(base, grid, art)
    if not df.empty and {"coupon", "duration", "fair"}.issubset(df.columns):
        st.dataframe(df.pivot(index="coupon", columns="duration", values="fair"),
                     use_container_width=True)


def _sensitivity(art: engine.Artefacts) -> None:
    st.caption("Fair Z-spread as one driver is swept (steeper = more sensitive).")
    base = dict(coupon_type="FIXED", rating_bucket="HY_BB", coupon=5.5, duration=4.2,
                px=100, yrs_to_call=3.0, is_callable=True, net_lev=3.8, sector="OTHER")
    driver = st.selectbox("Driver", ["coupon", "duration", "net_lev", "px"])
    ranges = {"coupon": (3.0, 9.0), "duration": (1.0, 8.0),
              "net_lev": (1.0, 8.0), "px": (90.0, 108.0)}
    lo, hi = ranges[driver]
    xs, ys = analytics.sweep(base, driver, lo, hi, 25, art)
    st.line_chart(pd.DataFrame({driver: xs, "fair_zs": ys}).set_index(driver))
