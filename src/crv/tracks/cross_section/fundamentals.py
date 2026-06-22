"""Build per-ISIN fundamentals from Financials.xlsx.

9Fin is the primary source; the Back-Up sheet provides a per-field fallback.
Provenance is recorded per ratio so downstream consumers can see whether a value
came from 9Fin, the backup, or neither.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crv.utils.logging import get_logger

log = get_logger(__name__)


def build_fundamentals(path: Path | str) -> pd.DataFrame:
    """One row per ISIN: 9Fin primary + Back-Up per-field fallback, with provenance."""
    nine = pd.read_excel(path, sheet_name="9Fin", header=6)
    back = pd.read_excel(path, sheet_name="Back-Up", header=0)
    back.columns = [str(c).strip() for c in back.columns]

    # --- 9Fin (primary) ---
    nine["isin"] = nine["ISIN"].astype(str).str.strip()
    nine["ebitda_used"] = nine["EBITDA (IFRS 16)"].fillna(nine["EBITDA"])   # prefer IFRS-16
    nine["net_lev"] = nine["Net debt"] / nine["ebitda_used"]
    nine["icr"] = nine["ebitda_used"] / nine["Cash Interest"].abs()
    nine["ebitda_margin"] = nine["ebitda_used"] / nine["Rev"]
    nine["capex_pct"] = pd.to_numeric(nine["Capex %"], errors="coerce")
    nine["ebitda_growth"] = pd.to_numeric(nine["EBITDA growth"], errors="coerce")
    nine["issuer_key"] = nine["Ticker"].astype(str).str.split().str[0]
    n9 = nine[["isin", "Company", "issuer_key", "net_lev", "icr",
               "ebitda_margin", "capex_pct", "ebitda_growth"]].drop_duplicates("isin")

    # --- Back-Up (per-field fallback) ---
    back["isin"] = back["ISIN"].astype(str).str.strip()
    bk = back.rename(columns={
        "Net Leverage": "net_lev_bk", "Interest Coverage Ratio": "icr_bk",
        "Ebitda Margin": "ebitda_margin_bk", "ISSUER": "issuer_bk",
    })
    for c in ["net_lev_bk", "icr_bk", "ebitda_margin_bk"]:
        bk[c] = pd.to_numeric(bk[c], errors="coerce")
    bk = bk[["isin", "issuer_bk", "net_lev_bk", "icr_bk", "ebitda_margin_bk"]].drop_duplicates("isin")

    # --- merge + provenance ---
    m = n9.merge(bk, on="isin", how="outer")
    for r in ["net_lev", "icr", "ebitda_margin"]:
        m[f"src_{r}"] = np.where(
            m[r].notna(), "9fin",
            np.where(m[f"{r}_bk"].notna(), "backup", "none"))
        m[r] = m[r].fillna(m[f"{r}_bk"])
    m["company"] = m["Company"].fillna(m["issuer_bk"])
    m["issuer_key"] = m["issuer_key"].fillna(m["company"].astype(str).str.split().str[0])

    out = m[["isin", "company", "issuer_key", "net_lev", "icr", "ebitda_margin",
             "capex_pct", "ebitda_growth", "src_net_lev", "src_icr", "src_ebitda_margin"]]
    out = out.copy()
    out["isin"] = out["isin"].astype(str).str.strip()
    log.info("fundamentals: %d ISIN | all 3 ratios: %d",
             len(out), out.dropna(subset=["net_lev", "icr", "ebitda_margin"]).shape[0])
    return out
