"""Monitor view: cross-track relative-value tables + distressed frontier."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from crv.app import data


def _signal_from_alpha(alpha) -> str:
    if alpha is None or pd.isna(alpha):
        return "—"
    if alpha > 1.0:
        return "CHEAP"
    if alpha < -1.0:
        return "RICH"
    return "FAIR"


def render() -> None:
    st.header("Relative-Value Monitor")
    md = data.load_monitor()
    st.caption(f"As of {md.as_of}")

    tab1, tab2, tab3 = st.tabs([md.tracks[1]["name"], md.tracks[2]["name"], md.tracks[3]["name"]])

    # ---- Track 1 ----
    with tab1:
        df = data.bonds_dataframe(md.tracks[1])
        if df.empty:
            st.info("No Track-1 output found. Run the Track-1 pipeline first.")
        else:
            df = df.assign(signal=df["alpha"].map(_signal_from_alpha))
            show = df[["issuer", "sector", "rating", "zspread", "fair", "resid",
                       "alpha", "signal", "lev", "cov"]].sort_values("alpha", ascending=False)
            st.dataframe(show, use_container_width=True, hide_index=True)
            _cheap_rich_summary(df)

    # ---- Track 2 ----
    with tab2:
        df = data.bonds_dataframe(md.tracks[2])
        if df.empty:
            st.info("No Track-2 output found. Run the Track-2 pipeline first.")
        else:
            df = df.assign(signal=df["alpha"].map(_signal_from_alpha))
            show = df[["issuer", "sector", "rating", "zspread", "fair", "resid",
                       "alpha", "signal", "dur"]].sort_values("alpha", ascending=False)
            st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- Track 3 ----
    with tab3:
        df = data.bonds_dataframe(md.tracks[3])
        if df.empty:
            st.info("No Track-3 output found. Run the Track-3 pipeline first.")
        else:
            show = df[["issuer", "rating", "zspread", "regime", "pd40", "conf",
                       "alpha"]].sort_values("alpha", ascending=False)
            st.dataframe(show, use_container_width=True, hide_index=True)
            _distressed_detail(df, md.trajectories)


def _cheap_rich_summary(df: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3)
    sig = df["alpha"].map(_signal_from_alpha)
    c1.metric("Cheap", int((sig == "CHEAP").sum()))
    c2.metric("Fair", int((sig == "FAIR").sum()))
    c3.metric("Rich", int((sig == "RICH").sum()))


def _distressed_detail(df: pd.DataFrame, trajectories: dict) -> None:
    st.subheader("Bond detail")
    names = df["issuer"].tolist()
    pick = st.selectbox("Select a distressed name", names, key="ds_pick")
    row = df[df["issuer"] == pick].iloc[0]

    fr = data.parse_frontier(row.get("frontier", "[]"))
    if not fr.empty:
        st.caption("PD–recovery frontier (cumulative default probability vs assumed recovery)")
        st.line_chart(fr.set_index("recovery"))

    traj = trajectories.get(str(row["isin"]))
    if traj:
        tdf = pd.DataFrame(traj, columns=["date", "px"]).set_index("date")
        st.caption("Price trajectory")
        st.line_chart(tdf)
