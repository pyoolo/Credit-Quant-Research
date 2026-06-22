"""Streamlit application layer.

A single Streamlit app exposes two views over the persisted track outputs:

    Monitor  — the cross-track relative-value table (CHEAP/RICH per track),
               with the distressed PD frontier and price trajectory
    Pricer   — the interactive new-issue pricer (router + per-track pricers)

Run with::

    crv app          # -> streamlit run .../app/main.py
    # or directly:
    streamlit run src/crv/app/main.py

The data-loading helpers in :mod:`crv.app.data` are UI-agnostic and reused by
both views.
"""

from crv.app import data

__all__ = ["data"]
