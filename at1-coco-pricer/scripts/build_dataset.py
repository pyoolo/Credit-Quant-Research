"""Turn the raw S&P Capital IQ exports in ``data/raw/`` into tidy CSVs.

Outputs (all in ``data/``):

* ``bond_prices.csv``   date, isin, bid/mid/ask price, mid yield, spread to
                         benchmark (bp), implied benchmark Treasury yield
* ``bnp_cet1_quarterly.csv``  quarter-end CET1 ratio (transitional + fully loaded),
                         CET1 capital, RWA, Tier 1 ratio, leverage ratio
* ``bnp_capital_annual.csv``  year-end CET1 ratio and the full CET1 requirement
                         stack (P1, P2R, buffers) used for the MDA geometry
* ``bnp_share.csv``     daily BNP share price (EUR)

Run:  python scripts/build_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

BONDS = ["US05602XQR25", "US05602XQQ42", "US05602XQS08"]


def _read_ciq_table(path: Path, sheet: str, label_col: int = 0) -> pd.DataFrame:
    """CIQ statement exports: row labels in col 0, one column per period.

    Returns a frame indexed by row label with the period-end dates as columns.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = raw.index[raw[label_col] == "Fiscal Period Ended"][0]
    dates = pd.to_datetime(raw.iloc[hdr, 1:])
    body = raw.iloc[hdr + 1 :].set_index(label_col)
    body.columns = dates
    body = body[body.index.notna()]
    body.index = body.index.astype(str).str.strip()
    # keep the first occurrence of duplicated labels (same values repeated)
    return body[~body.index.duplicated()]


def build_bond_prices() -> pd.DataFrame:
    frames = []
    for isin in BONDS:
        raw = pd.read_excel(RAW / f"ciq_bond_{isin}.xlsx", sheet_name="Chart", header=None)
        i = raw.index[raw[0] == "Date"][0]
        df = raw.iloc[i + 1 :].copy()
        df.columns = raw.iloc[i].tolist()
        df = df.dropna(subset=["Date"])
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop(columns="Date").set_index("date").astype(float).sort_index()
        # CIQ repeats Friday's evaluation on Sat/Sun: keep business days only
        df = df[df.index.dayofweek < 5]
        out = pd.DataFrame(
            {
                "isin": isin,
                "bid_price": df["Bid Price"],
                "mid_price": df["Mid Price"],
                "ask_price": df["Ask Price"],
                "mid_yield": df["Mid Yield"],
                "mid_spread_bp": df["Mid Spread"],
            }
        )
        # spread is quoted over the on-the-run Treasury benchmark chosen by CIQ,
        # so yield - spread recovers that benchmark's yield
        out["benchmark_yield"] = out["mid_yield"] - out["mid_spread_bp"] / 100.0
        frames.append(out.reset_index())
    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(OUT / "bond_prices.csv", index=False, float_format="%.4f")
    return prices


def build_cet1_quarterly() -> pd.DataFrame:
    t = _read_ciq_table(RAW / "ciq_capital_adequacy_quarterly.xlsx", "Capital Adequacy")
    rows = {
        "cet1_ratio": "Tier 1 Common Capital (CET1) Ratio (%)",
        "cet1_ratio_fully_loaded": "Fully Loaded: Common Equity Tier 1 Ratio (%)",
        "tier1_ratio": "Tier 1 Ratio (%)",
        "total_capital_ratio": "Total Capital Ratio (%)",
        "leverage_ratio": "Basel III Leverage Ratio, as Reported (%)",
        "cet1_capital_eur_k": "Tier 1 Common Capital (CET1)",
        "rwa_eur_k": "Total Risk-weighted Assets",
        "at1_eligible_eur_k": "T1: Tier 1 Eligible Hybrid Capital Securities",
    }
    df = pd.DataFrame({k: pd.to_numeric(t.loc[v], errors="coerce") for k, v in rows.items()})
    df.index.name = "date"
    df.to_csv(OUT / "bnp_cet1_quarterly.csv", float_format="%.4f")
    return df


def build_capital_annual() -> pd.DataFrame:
    t = _read_ciq_table(RAW / "ciq_basel_annual.xlsx", "Basel III IV")
    rows = {
        "cet1_ratio": "Tier 1 Common Capital (CET1) Ratio",
        "tier1_ratio": "Tier 1 Ratio",
        "leverage_ratio": "Basel III Leverage Ratio, as Reported",
        "p1": "Pillar 1 Minimum Requirement",
        "p2r_cet1": "Pillar 2 Minimum Requirement",
        "ccob": "Capital Conservation Buffer",
        "ccyb": "Counter Cyclical Buffer",
        "gsib": "Global and Other Systematically Impo Instn Buffer",
        "srb": "Systemic Risk Buffer",
        "cet1_requirement": "Total CET1 Requirement",
        "tier1_requirement": "Total Tier 1 Requirement",
        "leverage_requirement": "Minimum Basel III Leverage Requirement Total",
    }
    df = pd.DataFrame({k: pd.to_numeric(t.loc[v], errors="coerce") for k, v in rows.items()})
    df.index.name = "date"
    # model geometry: floor = P1 + P2R (CET1 part); buffer = combined buffer requirement
    df["mda_floor"] = df["p1"] + df["p2r_cet1"]
    df["combined_buffer"] = df[["ccob", "ccyb", "gsib", "srb"]].sum(axis=1, min_count=1)
    df["cet1_cushion"] = df["cet1_ratio"] - df["cet1_requirement"]
    df.to_csv(OUT / "bnp_capital_annual.csv", float_format="%.4f")
    return df


def build_share() -> pd.DataFrame:
    raw = pd.read_excel(RAW / "ciq_bnp_share.xlsx", sheet_name="Data")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[:, 0]),
            "close_eur": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),
            "volume": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
        }
    ).dropna(subset=["close_eur"]).sort_values("date")
    df.to_csv(OUT / "bnp_share.csv", index=False)
    return df


if __name__ == "__main__":
    p = build_bond_prices()
    q = build_cet1_quarterly()
    a = build_capital_annual()
    s = build_share()
    print(p.groupby("isin")["date"].agg(["min", "max", "count"]))
    print(q.round(2).to_string())
    print(a.round(2).to_string())
    print(s["date"].min().date(), s["date"].max().date(), len(s))
