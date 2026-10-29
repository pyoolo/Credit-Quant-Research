"""Paths and loaders for the BNP Paribas dataset.

Layout (see ``data/README.md``)::

    data/raw/         S&P Capital IQ exports (licensed, gitignored)
    data/reference/   public inputs: term sheets, requirements, FRED Treasuries
    data/processed/   built by ``scripts/build_bnp_dataset.py`` (gitignored)

Every analysis script reads through this module, so the layout lives in one place.
Loaders return frames in the shapes the analysis scripts were written for.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW, REF, PROC = DATA / "raw", DATA / "reference", DATA / "processed"
RES = ROOT / "results"

# Valuation date of the study.  Market data after it are ignored, and a CET1
# print counts only once it has been published (``available_from``).
VALUATION_DATE = pd.Timestamp("2026-09-25")

AT1_ISINS = ["US05602XQR25", "US05602XQS08", "US05602XQQ42"]


def _check(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python scripts/build_bnp_dataset.py` first")
    return path


def market_daily(asof: pd.Timestamp | None = VALUATION_DATE) -> pd.DataFrame:
    """One row per business day and bond (AT1, T2, SNP), plus date-level series."""
    m = pd.read_csv(_check(PROC / "bnp_market_daily.csv"), parse_dates=["date", "cet1_quarter"])
    return m if asof is None else m[m["date"] <= asof]


def bond_prices(ranks: tuple[str, ...] = ("AT1",), asof=VALUATION_DATE) -> pd.DataFrame:
    """Long price panel in the original ``bond_prices.csv`` format.

    ``benchmark_yield`` is the yield of the Treasury Capital IQ quotes the bond
    against (yield minus CIQ spread), used by the CIQ-only curve sensitivities.
    """
    m = market_daily(asof)
    m = m[m["rank"].isin(ranks)]
    out = pd.DataFrame({
        "date": m["date"], "isin": m["isin"], "bid_price": m["bid_price"], "mid_price": m["mid_price"],
        "ask_price": m["ask_price"], "mid_yield": m["mid_yield_ciq_pct"], "mid_spread_bp": m["mid_spread_ciq_bp"],
    })
    out["benchmark_yield"] = out["mid_yield"] - out["mid_spread_bp"] / 100.0
    return out.reset_index(drop=True)


def capital_quarterly(asof=VALUATION_DATE) -> pd.DataFrame:
    """Quarterly CET1 and requirement, keeping only prints published by ``asof``."""
    q = pd.read_csv(_check(PROC / "bnp_capital_quarterly.csv"), parse_dates=["date", "available_from"])
    if asof is not None:
        q = q[q["available_from"] <= asof]
    return q.set_index("date")


def cet1_quarterly(asof=VALUATION_DATE) -> pd.DataFrame:
    """Original ``bnp_cet1_quarterly.csv`` shape: index date, column cet1_ratio, rwa_eur_k, ..."""
    return capital_quarterly(asof)


def requirement_series(asof=VALUATION_DATE) -> pd.Series:
    """CET1 (MDA) requirement as a step series: dates where it changes."""
    r = capital_quarterly(asof)["cet1_requirement_mda"].dropna()
    return r[r.ne(r.shift())]


def share_eur(asof=VALUATION_DATE) -> pd.Series:
    """Daily BNP share close (EUR) on the panel's business days."""
    m = market_daily(asof)
    return m.groupby("date")["share_close_eur"].first().dropna()


def instruments() -> pd.DataFrame:
    return pd.read_csv(_check(PROC / "bnp_instruments.csv"), index_col="isin_144a")


def bonds_static() -> pd.DataFrame:
    """AT1 term sheets (public reference file)."""
    return pd.read_csv(REF / "bonds_static.csv", index_col="isin_144a")


def senior_t2_static() -> pd.DataFrame:
    """T2 and SNP term sheets (public reference file)."""
    return pd.read_csv(REF / "senior_t2_static.csv", index_col="isin_144a")


def fred_curve() -> pd.DataFrame:
    """Daily constant-maturity Treasury yields (%) from FRED, tenors 5/7/10y."""
    f = pd.read_csv(REF / "fred_treasury.csv", parse_dates=["observation_date"], index_col="observation_date")
    f = f.apply(pd.to_numeric, errors="coerce")[["DGS5", "DGS7", "DGS10"]]
    return f.dropna(how="all").ffill()


def mda_requirement_path() -> pd.DataFrame:
    """Requirement stack today and the scheduled O-SII step-ups (reference file)."""
    return pd.read_csv(REF / "bnp_cet1_requirement_breakdown.csv")


def cds_curves() -> pd.DataFrame:
    """Senior CDS curves (EUR, MM14, conventional spread mid, bp) on the export's snapshot dates.

    Read straight from the raw Capital IQ export: one row per curve date, one
    column per tenor.  Rows with a missing tenor are dropped, since the export
    does not say which tenor is missing.
    """
    raw = pd.read_excel(_check(RAW / "ciq_cds_bnp_snrfor_eur.xlsx"), header=None)
    hdr = raw.index[raw[0] == "Issuer Name"][0]
    cols = raw.iloc[hdr].dropna().tolist()
    body = raw.iloc[hdr + 1:hdr + 20, :len(cols)]
    body = body[body[0] == raw.iloc[hdr + 1, 0]]
    body.columns = cols
    complete = body.notna().all(axis=1)
    out = body[complete].set_index(pd.to_datetime(body.loc[complete, "Curve Date"], format="%m/%d/%Y"))
    return out.drop(columns=["Issuer Name", "Curve Description", "Curve Date"]).astype(float).sort_index()
