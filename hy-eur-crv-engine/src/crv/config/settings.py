"""All model and pipeline parameters.

Centralises every magic number that used to live in the ``CONFIG`` cell of each
notebook. Grouped by concern so each track imports only what it needs.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Universe / fetch                                                            #
# --------------------------------------------------------------------------- #
CURRENCY = "EUR"
PANEL_WEEKS = 26

N_RETRY = 3
BASE_WAIT = 5
CHUNK_SIZE = 1500
SAVE_EVERY_DATE = True

# Z-spread band (bps)
SPREAD_FLOOR = 50        # below: mis-classified FRN / broken price -> DROPPED
DISTRESS_CUTOFF = 1200   # above: Track 3 distressed (no cross-section)

# Universe: corporates ONLY (drop sovereigns / banks / financials)
NON_CORP = {
    "Sovereign", "Banks", "Insurance", "Diversified Finan Serv",
    "Regional(state/provnc)", "Municipal", "Savings&Loans", "REITS",
    "Government Regional", "Government Sponsored", "Multi-National",
}

# Buckets / coupon types eligible for Track 1
HY_BUCKETS_T1 = {"HY_BB", "HY_B", "HY_CCC"}
TRACK1_CPN = {"FIXED", "STEP CPN", "FIX-TO-FIX", "ZERO COUPON"}

# --------------------------------------------------------------------------- #
# Cross-section model (Track 1)                                               #
# --------------------------------------------------------------------------- #
MIN_SEC = 8              # min bonds for a sector to keep its own dummy
WINSOR = 0.02            # winsor quantile (train-only bounds)
N_SPLITS = 5             # GroupKFold splits
MIN_XS_PER_DATE = 50     # min cross-section width for a date to enter training

# near-maturity filter: near-zero-duration bonds have numerically unstable
# Z-spreads (small price moves -> exploding spread) -> spurious huge residuals.
MIN_TTM_YRS = 0.75
MIN_DURATION = 0.50

HORIZON_W = 4            # backtest forward horizon (weeks)
Z_CUT = 1.0             # CHEAP/RICH rv_z threshold
MIN_GRP = 50            # min bonds per plotted rating group (rating-curve diagnostic)

# Two-sided fundamental scores
QUALITY_RATIOS = ["icr", "ebitda_margin", "ebitda_growth"]   # higher = safer
RISK_RATIOS = ["net_lev", "capex_pct"]                       # higher = riskier
FUND_FEATS = ["net_lev", "icr", "ebitda_margin", "capex_pct", "ebitda_growth"]

# Final feature set for the fair-spread Lasso
SEL_FEATS = [
    "rating_notch", "duration", "coupon", "fund_quality", "fund_risk",
    "age", "yrs_to_call", "is_callable",
]
# Path-diagnostic feature set (superset used in the Lasso-path selection plot)
SET_FEATS = [
    "rating_notch", "duration", "coupon", "age", "log_amt",
    "yrs_to_call", "is_callable", "fund_quality", "fund_risk",
]

# --------------------------------------------------------------------------- #
# Distressed model (Track 3)                                                  #
# --------------------------------------------------------------------------- #
RISK_FREE = 0.03         # flat risk-free for the PD/recovery solver

# --------------------------------------------------------------------------- #
# Rating maps (shared)                                                        #
# --------------------------------------------------------------------------- #
RATING_NUM = {
    "HY_BB": 4, "HY_B": 5, "HY_CCC": 6, "NR": 5,
    "IG_BBB": 3, "IG_A": 2, "IG_AAA_AA": 1,
}

# Granular agency rating -> coarse bucket (Moody's + S&P/Fitch on one map)
RATING_BUCKET = {
    "Aaa": "IG_AAA_AA", "Aa1": "IG_AAA_AA", "Aa2": "IG_AAA_AA", "Aa3": "IG_AAA_AA",
    "A1": "IG_A", "A2": "IG_A", "A3": "IG_A",
    "Baa1": "IG_BBB", "Baa2": "IG_BBB", "Baa3": "IG_BBB",
    "Ba1": "HY_BB", "Ba2": "HY_BB", "Ba3": "HY_BB",
    "B1": "HY_B", "B2": "HY_B", "B3": "HY_B",
    "Caa1": "HY_CCC", "Caa2": "HY_CCC", "Caa3": "HY_CCC", "Ca": "HY_CCC", "C": "HY_CCC",
    "AAA": "IG_AAA_AA", "AA+": "IG_AAA_AA", "AA": "IG_AAA_AA", "AA-": "IG_AAA_AA",
    "A+": "IG_A", "A": "IG_A", "A-": "IG_A",
    "BBB+": "IG_BBB", "BBB": "IG_BBB", "BBB-": "IG_BBB",
    "BB+": "HY_BB", "BB": "HY_BB", "BB-": "HY_BB",
    "B+": "HY_B", "B": "HY_B", "B-": "HY_B",
    "CCC+": "HY_CCC", "CCC": "HY_CCC", "CCC-": "HY_CCC", "CC": "HY_CCC",
}

# HY-only rating codes, used to filter the Bloomberg universe down to high yield
HY_RATING_CODES = [
    "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca", "C",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC",
]

# Granular notch scales (lower = safer)
SP_NOTCH = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4, "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10, "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16, "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC": 20, "C": 21, "D": 22,
}
MOODY_NOTCH = {
    "Aaa": 1, "Aa1": 2, "Aa2": 3, "Aa3": 4, "A1": 5, "A2": 6, "A3": 7,
    "Baa1": 8, "Baa2": 9, "Baa3": 10, "Ba1": 11, "Ba2": 12, "Ba3": 13,
    "B1": 14, "B2": 15, "B3": 16, "Caa1": 17, "Caa2": 18, "Caa3": 19,
    "Ca": 20, "C": 21,
}

# --------------------------------------------------------------------------- #
# Bloomberg field maps (shared by fetch)                                      #
# --------------------------------------------------------------------------- #
COL_MAP = {
    "SECURITY_DES": "name", "MATURITY": "maturity", "CPN": "coupon",
    "CPN_TYP": "coupon_type", "AMT_OUTSTANDING": "amount", "CALLABLE": "callable",
    "INDUSTRY_GROUP": "industry", "RTG_MOODY": "rating_moody", "RTG_SP": "rating_sp",
    "RTG_FITCH": "rating_fitch", "MIFID_SENIORITY_OF_BOND": "payment_rank",
    "CRNCY": "currency", "ISSUE_DT": "issue_date", "ISSUER": "issuer",
    "ID_BB_COMPANY": "issuer_id", "MTY_DUR_MID": "duration", "SPREAD": "z_spread",
    "YIELD": "ytm", "PX_BID": "px_bid", "PX_ASK": "px_ask",
    "NXT_CALL_DT": "next_call_dt", "NXT_CALL_PX": "next_call_px",
    "RISK_OAS_MID": "oas", "WORKOUT_DATE_MID_TO_WORST": "workout_dt", "ID_ISIN": "isin",
}
STATIC_COLS = [
    "isin", "name", "maturity", "coupon", "coupon_type", "amount", "callable",
    "industry", "rating_moody", "rating_sp", "rating_fitch", "payment_rank",
    "currency", "issue_date", "issuer", "issuer_id",
]
DYNAMIC_COLS = [
    "duration", "z_spread", "ytm", "px_bid", "px_ask",
    "next_call_dt", "next_call_px", "oas", "workout_dt",
]


def summary() -> str:
    return (
        f"config OK | {PANEL_WEEKS} weeks | distressed > {DISTRESS_CUTOFF} "
        f"| {len(SEL_FEATS)} model features"
    )
