"""Build the consolidated BNP Paribas dataset from data/raw + data/reference.

Inputs
  data/raw/        S&P Capital IQ exports (licensed, gitignored)
  data/reference/  public, hand-maintained terms and FRED Treasuries
Outputs (data/processed/, gitignored because they contain CIQ data)
  bnp_market_daily.csv, bnp_instruments.csv, bnp_capital_quarterly.csv,
  bnp_capital_structure_dataset.xlsx (same data in one workbook, with README)

Run from the at1-coco-pricer folder:  python scripts/build_bnp_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
RAW, REF, OUT = ROOT / "data" / "raw", ROOT / "data" / "reference", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def _read_ciq_table(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = raw.index[raw[0] == "Fiscal Period Ended"][0]
    dates = pd.to_datetime(raw.iloc[hdr, 1:])
    b = raw.iloc[hdr + 1:].set_index(0)
    b.columns = dates
    b = b[b.index.notna()]
    b.index = b.index.astype(str).str.strip()
    return b[~b.index.duplicated()]


def cet1_quarterly():
    t = _read_ciq_table(RAW / "ciq_capital_adequacy_quarterly.xlsx", "Capital Adequacy")
    rows = {"cet1_ratio": "Tier 1 Common Capital (CET1) Ratio (%)",
            "cet1_ratio_fully_loaded": "Fully Loaded: Common Equity Tier 1 Ratio (%)",
            "tier1_ratio": "Tier 1 Ratio (%)", "total_capital_ratio": "Total Capital Ratio (%)",
            "leverage_ratio": "Basel III Leverage Ratio, as Reported (%)",
            "cet1_capital_eur_k": "Tier 1 Common Capital (CET1)", "rwa_eur_k": "Total Risk-weighted Assets",
            "at1_eligible_eur_k": "T1: Tier 1 Eligible Hybrid Capital Securities"}
    df = pd.DataFrame({k: pd.to_numeric(t.loc[v], errors="coerce") for k, v in rows.items()})
    df.index.name = "date"
    return df.reset_index()


def capital_annual():
    t = _read_ciq_table(RAW / "ciq_basel_annual.xlsx", "Basel III IV")
    df = pd.DataFrame({"cet1_requirement": pd.to_numeric(t.loc["Total CET1 Requirement"], errors="coerce")})
    df.index.name = "date"
    return df.reset_index()

START = pd.Timestamp("2025-06-24")   # first AT1 print (7.45% when-issued)
END = pd.Timestamp("2026-10-27")
BASEL3_START = pd.Timestamp("2014-01-01")
PUB_LAG_DAYS = 35

# ---------------------------------------------------------------------------
# 1. instruments
# ---------------------------------------------------------------------------
at1 = pd.read_csv(REF / "bonds_static.csv")
rows = []
for r in at1.itertuples():
    rows.append(dict(
        isin_144a=r.isin_144a, isin_regs=r.isin_regs, short_name={"US05602XQR25": "AT1 6.875% NC33",
                                                                   "US05602XQQ42": "AT1 7.45% NC35",
                                                                   "US05602XQS08": "AT1 7.20% NC36"}[r.isin_144a],
        rank="AT1", rank_order=4, currency=r.currency, amount_mn=r.amount_mn, issue_date=r.issue_date,
        coupon_initial=r.coupon_initial, coupon_freq=r.coupon_freq, coupon_dates=r.coupon_dates,
        day_count=r.day_count, first_call_date=r.first_call_date, maturity_date="perpetual",
        call_rule=r.call_rule, after_first_call="reset every 5y", reset_reference=r.reset_reference,
        reset_margin=r.reset_margin, trigger_level_pct=r.trigger_level_pct,
        loss_absorption=r.loss_absorption, floor_price_usd=r.floor_price_usd,
        model_treatment="economic call on reset dates (model)", source=r.source))

new = pd.read_csv(REF / "senior_t2_static.csv").to_dict("records")
for n in new:
    n["rank_order"] = {"T2": 3, "SNP": 2}[n["rank"]]
    n.update(currency="USD", coupon_freq=2, day_count="30/360", trigger_level_pct=np.nan,
             loss_absorption="bail-in / PONV write-down (statutory, BRRD)", floor_price_usd=np.nan,
             model_treatment=("call assumed certain (market convention)" if n["rank"] != "T2" or "2D24" in n["isin_144a"]
                              else "reprice with own cash flows; CIQ yield is to maturity"))
    rows.append(n)
instruments = pd.DataFrame(rows)
cds_row = dict(isin_144a="CDS_BNP_SNRFOR_EUR_MM14", isin_regs="", short_name="CDS senior 5Y EUR", rank="CDS senior",
               rank_order=1, currency="EUR", coupon_initial=0.01, coupon_freq=4, day_count="ACT/360",
               first_call_date="", maturity_date="5Y rolling", call_rule="", after_first_call="",
               reset_reference="", reset_margin=np.nan, loss_absorption="ISDA 2014 credit events, MM14",
               model_treatment="conv. spread mid, standard coupon 100bp; Markit recovery convention 40% senior",
               source="S&P Capital IQ (Markit) - Credit Default Swaps export 28-Oct-2026")
instruments = pd.concat([instruments, pd.DataFrame([cds_row])], ignore_index=True)

# ---------------------------------------------------------------------------
# 2. prices
# ---------------------------------------------------------------------------
def read_chart(path):
    c = pd.read_excel(path, sheet_name="Chart", header=None)
    i = c.index[c[0] == "Date"][0]
    b = c.iloc[i + 1:].copy()
    b.columns = c.iloc[i].tolist()
    b = b.dropna(subset=["Date"])
    b["date"] = pd.to_datetime(b["Date"], format="%m/%d/%Y")
    b = b.drop(columns="Date").set_index("date").apply(pd.to_numeric, errors="coerce").sort_index()
    return b[b.index.dayofweek < 5]


def accrual_of(path):
    d = pd.read_excel(path, sheet_name="BOND DETAIL", header=None)
    row = d[d[0] == "T0 Accrued Interest"]
    return float(row[1].iloc[0])


files = {isin: RAW / f"ciq_bond_{isin}.xlsx" for isin in instruments.loc[instruments["rank"] != "CDS senior", "isin_144a"]}
inst = instruments.set_index("isin_144a")


def d30360(d1, d2):
    return 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (min(d2.day, 30) - min(d1.day, 30))


def coupon_grid(isin, until):
    anchor = pd.Timestamp(inst.at[isin, "first_call_date"])
    start = anchor - pd.DateOffset(months=6 * 80)
    return pd.date_range(start, until, freq=pd.DateOffset(months=6))


def accrued(isin, day):
    g = coupon_grid(isin, day + pd.DateOffset(years=1))
    last = g[g <= day][-1]
    return inst.at[isin, "coupon_initial"] * 100 * d30360(last, day) / 360.0


def ytc(isin, day, dirty):
    """Yield to first call (s.a., 30/360), bullet at par on the first call date."""
    call = pd.Timestamp(inst.at[isin, "first_call_date"])
    g = coupon_grid(isin, call)
    fut = g[g > day]
    prev = g[g <= day][-1]
    frac = 1 - d30360(prev, day) / 180.0
    c = inst.at[isin, "coupon_initial"] * 100 / 2
    n = np.arange(len(fut))
    cf = np.full(len(fut), c)
    cf[-1] += 100

    def f(y):
        return (cf / (1 + y / 2) ** (frac + n)).sum() - dirty
    return brentq(f, -0.5, 2.0)


fred = pd.read_csv(REF / "fred_treasury.csv", parse_dates=["observation_date"], index_col="observation_date")
fred = fred.apply(pd.to_numeric, errors="coerce")[["DGS5", "DGS7", "DGS10"]].dropna(how="all")
FRED_END = fred.index.max()

cds_raw = pd.read_excel(RAW / "ciq_cds_bnp_snrfor_eur.xlsx", header=None)
cds = cds_raw.iloc[60:, :3].dropna(subset=[1])
cds = cds[pd.to_datetime(cds[0], format="%m/%d/%Y", errors="coerce").notna()]
cds = pd.Series(pd.to_numeric(cds[1]).values, index=pd.to_datetime(cds[0], format="%m/%d/%Y")).sort_index()
cds = cds[cds.index.dayofweek < 5]

share = pd.read_excel(RAW / "ciq_bnp_share.xlsx", sheet_name="Data")
share = pd.Series(pd.to_numeric(share.iloc[:, 1]).values, index=pd.to_datetime(share.iloc[:, 0])).sort_index()

capq = cet1_quarterly()
q3 = dict(date=pd.Timestamp("2026-09-30"), cet1_ratio=12.9)
capq = pd.concat([capq, pd.DataFrame([q3])], ignore_index=True).sort_values("date")
capq["available_from"] = capq["date"] + pd.Timedelta(days=PUB_LAG_DAYS)
capq.loc[capq["date"] == "2026-09-30", "available_from"] = pd.Timestamp("2026-10-28")
capq["source"] = "S&P Capital IQ, capital adequacy (quarterly)"
capq.loc[capq["date"] == "2026-09-30", "source"] = ("BNP Paribas 3Q26 results, 28-Oct-2026 (reported 12.9%, rounded; "
                                                    "Euronews 28-10-2026) - replace with CIQ value when available")

srep = pd.read_csv(REF / "bnp_srep_requirement.csv", parse_dates=["date"])
ann = capital_annual()
req = pd.concat([ann[["date", "cet1_requirement"]].dropna().assign(req_source="CIQ Basel annual (year-end)"),
                 srep[["date", "cet1_requirement"]].assign(req_source=srep["source"])])
req = req.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
capq = capq.set_index("date")
capq["cet1_requirement_mda"] = req["cet1_requirement"].reindex(capq.index.union(req.index)).ffill().reindex(capq.index)
capq["requirement_source"] = req["req_source"].reindex(capq.index.union(req.index)).ffill().reindex(capq.index)
capq.loc[capq.index == "2026-09-30", ["cet1_requirement_mda", "requirement_source"]] = [
    10.43, "carried forward from 30-Jun-2026 (no update published) - check 3Q26 results"]
capq["cushion_to_mda_pp"] = capq["cet1_ratio"] - capq["cet1_requirement_mda"]
capq["distance_to_trigger_pp"] = capq["cet1_ratio"] - 5.125
capq["basel3_sample"] = capq.index >= BASEL3_START
capq = capq.reset_index()

panel = []
for isin, path in files.items():
    b = read_chart(path)
    # grey-market prints before the issue date are dropped (accrued is undefined there)
    b = b[(b.index >= max(START, pd.Timestamp(inst.at[isin, "issue_date"]))) & (b.index <= END)]
    call = pd.Timestamp(inst.at[isin, "first_call_date"])
    for day, r in b.iterrows():
        if np.isnan(r["Mid Price"]):
            continue
        acc = accrued(isin, day)
        dirty = r["Mid Price"] + acc
        ttc = (call - day).days / 365.25
        y = ytc(isin, day, dirty)
        frow = fred[fred.index <= day]
        fr = frow.iloc[-1] if (len(frow) and day <= FRED_END + pd.Timedelta(days=3)) else None
        ust = float(np.interp(ttc, [5, 7, 10], fr.values)) if fr is not None else np.nan
        cet = capq[capq["available_from"] <= day].iloc[-1]
        sh = share[share.index <= day]
        cd = cds[cds.index <= day]
        panel.append(dict(
            date=day.date(), isin=isin, short_name=inst.at[isin, "short_name"], rank=inst.at[isin, "rank"],
            bid_price=r["Bid Price"], mid_price=r["Mid Price"], ask_price=r["Ask Price"],
            bid_ask_pts=r["Ask Price"] - r["Bid Price"], accrued_30360=round(acc, 4), dirty_mid=round(dirty, 4),
            mid_yield_ciq_pct=r["Mid Yield"], mid_spread_ciq_bp=r["Mid Spread"],
            years_to_first_call=round(ttc, 4), ytc_mid_pct=round(100 * y, 4),
            ust_call_matched_pct=round(ust, 4) if not np.isnan(ust) else np.nan,
            spread_ytc_vs_fred_bp=round(1e4 * y - 100 * ust, 2) if not np.isnan(ust) else np.nan,
            ust_interp_flag=("extrapolated flat below 5y" if ttc < 5 else ""),
            liquidity_score_ciq=r.get("Liquidity Score", np.nan),
            ust5y_cmt_pct=fr["DGS5"] if fr is not None else np.nan,
            ust7y_cmt_pct=fr["DGS7"] if fr is not None else np.nan,
            ust10y_cmt_pct=fr["DGS10"] if fr is not None else np.nan,
            cds_snr_eur_5y_bp=float(cd.iloc[-1]) if len(cd) else np.nan,
            share_close_eur=float(sh.iloc[-1]) if (len(sh) and (day - sh.index[-1]).days <= 4) else np.nan,
            cet1_published_pct=cet["cet1_ratio"], cet1_quarter=cet["date"].date(),
            cet1_requirement_mda_pct=cet["cet1_requirement_mda"],
        ))
market = pd.DataFrame(panel).sort_values(["date", "rank", "isin"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. reference tables
# ---------------------------------------------------------------------------
breakdown = pd.read_csv(REF / "bnp_cet1_requirement_breakdown.csv")
eba = pd.read_csv(REF / "bnp_eba_stress_2025.csv")
cur = cds_raw.iloc[42:49].dropna(how="all", axis=1)
cur.columns = cur.iloc[0].tolist()
curves = cur.iloc[1:].reset_index(drop=True)
# 7/27/2026 has one tenor fewer: flag it rather than guess which one is missing
curves["note"] = ["" if curves.iloc[i].notna().sum() == curves.shape[1] else "one tenor missing in CIQ export: do not use"
                  for i in range(len(curves))]
mda_path = pd.DataFrame([
    dict(effective="2026-06-30", cet1_requirement_mda=10.43, note="reported, BNP 2Q26"),
    dict(effective="2027-01-01", cet1_requirement_mda=10.68,
         note="+25bp O-SII step-up (higher of G-SIB/O-SII applies); other components held constant"),
    dict(effective="2028-01-01", cet1_requirement_mda=10.93,
         note="+25bp further O-SII step-up; other components held constant"),
])

market.to_csv(OUT / "bnp_market_daily.csv", index=False)
instruments.to_csv(OUT / "bnp_instruments.csv", index=False)
capq.to_csv(OUT / "bnp_capital_quarterly.csv", index=False)

# checks printed for the log
print(market.groupby("short_name")["date"].agg(["min", "max", "count"]))
chk = market[market["date"] == pd.Timestamp("2026-10-27").date()][["short_name", "mid_yield_ciq_pct", "ytc_mid_pct", "accrued_30360"]]
print(chk)
last_at1 = market[market["date"] == pd.Timestamp("2026-09-25").date()][["short_name", "dirty_mid", "ytc_mid_pct", "mid_yield_ciq_pct", "spread_ytc_vs_fred_bp", "bid_ask_pts"]]
print(last_at1)


# ---------------------------------------------------------------------------
# 4. workbook
# ---------------------------------------------------------------------------
m, ins, cq = market, instruments, capq

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEAD = Font(name="Arial", size=10, bold=True, color="FFFFFF")
TITLE = Font(name="Arial", size=14, bold=True)
FILL = PatternFill("solid", fgColor="1F3A5F")
MANUAL = PatternFill("solid", fgColor="FFF2CC")

wb = Workbook()


def write_table(ws, df, r0=1, c0=1, formats=None, widths=None):
    formats = formats or {}
    for j, col in enumerate(df.columns):
        c = ws.cell(row=r0, column=c0 + j, value=col)
        c.font, c.fill = HEAD, FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            if isinstance(v, float) and pd.isna(v):
                v = None
            elif isinstance(v, pd.Timestamp):
                v = v.date()
            c = ws.cell(row=r0 + i, column=c0 + j, value=v)
            c.font = FONT
            fmt = formats.get(df.columns[j])
            if fmt:
                c.number_format = fmt
    for j, col in enumerate(df.columns):
        w = (widths or {}).get(col) or min(max(len(str(col)), 10) + 2, 42)
        ws.column_dimensions[get_column_letter(c0 + j)].width = w
    return r0 + len(df) + 1


# README ---------------------------------------------------------------
ws = wb.active
ws.title = "README"
cov = m.groupby(["short_name", "rank"])["date"].agg(["min", "max", "count"]).reset_index()
cov.columns = ["instrument", "rank", "first date", "last date", "business days"]
lines = [
    ("BNP Paribas capital structure dataset", TITLE),
    ("Built 28-Oct-2026 for the at1-coco-pricer project (P. Mastrogiacomo). Contains S&P Capital IQ data: licensed, keep out of the public repository.", FONT),
    ("", FONT),
    ("Sheets", BOLD),
    ("instruments - one row per instrument (3 AT1, 2 T2, 1 SNP, senior CDS): terms from BNP prospectuses / debt issuance list.", FONT),
    ("market_daily - one row per business day and bond: CIQ evaluated bid/mid/ask, own accrued and yield-to-first-call, spread over FRED Treasuries, plus date-level CMT, CDS, share and published CET1.", FONT),
    ("capital_quarterly - CET1 history, MDA requirement, cushion, publication date (35-day lag convention).", FONT),
    ("reference - CET1 requirement stack, projected MDA path (O-SII step-ups), EBA 2025 stress, CDS curve snapshots.", FONT),
    ("", FONT),
    ("Derived columns (computed in Python, not formulas)", BOLD),
    ("accrued_30360: 30/360 on the trade date (= CIQ 'T0 Accrued Interest'). dirty_mid = mid_price + accrued.", FONT),
    ("ytc_mid_pct: semi-annual yield to first call, bullet at par, from dirty_mid. Matches CIQ mid yield to <0.3bp on 27-Oct-2026 for the T2/SNP.", FONT),
    ("ust_call_matched_pct: FRED CMT interpolated linearly at years_to_first_call on the 5/7/10y points, flat outside. spread_ytc_vs_fred_bp = ytc - that yield (same definition as the paper: 279/274/269bp for the AT1s on 25-Sep-2026).", FONT),
    ("cet1_published_pct: last quarter whose available_from date is on or before the row date.", FONT),
    ("", FONT),
    ("Caveats", BOLD),
    ("1. AT1 prices end 25-Sep-2026; T2, SNP and CDS run to 27-Oct-2026. To extend the AT1s, re-export the three CIQ charts.", FONT),
    (f"2. FRED Treasuries end {FRED_END.date()}: spread_ytc_vs_fred_bp and the CMT columns are empty after that. Update data/fred_treasury.csv (DGS5, DGS7, DGS10) to fill them.", FONT),
    ("3. BNP share price ends 24-Sep-2026 (CIQ export); empty after that.", FONT),
    ("4. T2 2.588% 2035NC30 resets on the 5y CMT in 2030 (like the AT1s): its ytc treats the call as certain and its first call is < 5y away (Treasury extrapolated flat). Reprice it with its own reset cash flows in the model.", FONT),
    ("5. T2/SNP exports carry no ISIN: matched by T0 accrued at 27-Oct-2026 (2.592 / 1.068 / 0.539 = 158 / 67 / 75 days at 5.906% / 5.738% / 2.588%, 30/360).", FONT),
    ("6. The CDS is EUR senior (SNRFOR, MM14); the bonds are USD. No subordinated CDS (SUBLT2) in the export.", FONT),
    ("7. 3Q26 CET1 (12.9%, rounded) entered by hand from the 28-Oct-2026 results release (highlighted). The CIQ 'capital structure' export labels the AT1s as Senior Debt: not used for ranking.", FONT),
    ("8. Grey-market prints before each bond's issue date are dropped.", FONT),
    ("", FONT),
    ("Coverage", BOLD),
]
for i, (t, f) in enumerate(lines, start=1):
    c = ws.cell(row=i, column=1, value=t)
    c.font = f
write_table(ws, cov, r0=len(lines) + 1, formats={"first date": "yyyy-mm-dd", "last date": "yyyy-mm-dd"},
            widths={"instrument": 26})
ws.column_dimensions["A"].width = 26

# instruments ------------------------------------------------------------
ws = wb.create_sheet("instruments")
write_table(ws, ins, formats={"coupon_initial": "0.000%", "reset_margin": "0.000%", "amount_mn": "#,##0"},
            widths={"isin_144a": 26, "short_name": 22, "after_first_call": 36, "model_treatment": 44,
                    "loss_absorption": 34, "source": 40, "call_rule": 30})
ws.freeze_panes = "C2"

# market_daily -----------------------------------------------------------
ws = wb.create_sheet("market_daily")
fm = {c: "0.000" for c in ["bid_price", "mid_price", "ask_price", "bid_ask_pts", "accrued_30360", "dirty_mid",
                           "mid_yield_ciq_pct", "ytc_mid_pct", "ust_call_matched_pct", "ust5y_cmt_pct",
                           "ust7y_cmt_pct", "ust10y_cmt_pct", "cet1_published_pct", "cet1_requirement_mda_pct",
                           "years_to_first_call"]}
fm.update({c: "0.0" for c in ["mid_spread_ciq_bp", "spread_ytc_vs_fred_bp", "cds_snr_eur_5y_bp"]})
fm.update({"date": "yyyy-mm-dd", "cet1_quarter": "yyyy-mm-dd", "share_close_eur": "0.00"})
write_table(ws, m, formats=fm, widths={"date": 12, "isin": 15, "short_name": 22, "ust_interp_flag": 24})
ws.freeze_panes = "D2"
ws.auto_filter.ref = ws.dimensions

# capital_quarterly ------------------------------------------------------
ws = wb.create_sheet("capital_quarterly")
fq = {c: "0.00" for c in ["cet1_ratio", "cet1_ratio_fully_loaded", "tier1_ratio", "total_capital_ratio",
                          "leverage_ratio", "cet1_requirement_mda", "cushion_to_mda_pp", "distance_to_trigger_pp"]}
fq.update({c: "#,##0" for c in ["cet1_capital_eur_k", "rwa_eur_k", "at1_eligible_eur_k"]})
fq.update({"date": "yyyy-mm-dd", "available_from": "yyyy-mm-dd"})
end = write_table(ws, cq, formats=fq, widths={"source": 50, "requirement_source": 44})
for r in range(2, end):
    if str(ws.cell(row=r, column=1).value) == "2026-09-30":
        for c in range(1, ws.max_column + 1):
            ws.cell(row=r, column=c).fill = MANUAL
ws.freeze_panes = "B2"

# reference --------------------------------------------------------------
ws = wb.create_sheet("reference")
r = 1
for title, df, fmt in [
    ("CET1 requirement stack (MDA threshold), 30-Sep-2025", breakdown, {"pct": "0.00"}),
    ("Projected MDA threshold (assumption: only the O-SII buffer changes)", mda_path, {"cet1_requirement_mda": "0.00"}),
    ("EBA 2025 stress test, BNP Paribas (%)", eba, None),
    ("Senior CDS curves, EUR MM14, conventional spread mid (bp)", curves, None),
]:
    ws.cell(row=r, column=1, value=title).font = BOLD
    r = write_table(ws, df, r0=r + 1, formats=fmt) + 2
for col in range(1, 20):
    ws.column_dimensions[get_column_letter(col)].width = 16
ws.column_dimensions["A"].width = 34
ws.column_dimensions["D"].width = 30

wb.save(OUT / "bnp_capital_structure_dataset.xlsx")

print("written to", OUT)
