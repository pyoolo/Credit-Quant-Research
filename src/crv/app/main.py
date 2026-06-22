"""Streamlit entry point. Run via ``crv app`` or ``streamlit run`` on this file."""

from __future__ import annotations

import streamlit as st

from crv import __version__
from crv.app import monitor_view, pricer_view


def main() -> None:
    st.set_page_config(page_title="CRV — HY EUR Relative Value", layout="wide")
    st.sidebar.title("CRV")
    st.sidebar.caption(f"HY EUR relative-value engine · v{__version__}")
    page = st.sidebar.radio("View", ["Monitor", "Pricer"])
    if page == "Monitor":
        monitor_view.render()
    else:
        pricer_view.render()


if __name__ == "__main__":
    main()
