"""Text feature extraction and point-in-time assembly.

Extraction never touches latent variables: it reads the rendered text only.
The lexicon is a deliberately small, illustrative word list in the spirit of
Loughran & McDonald (2011) — in production you would license and use the real
LM dictionary. Two practices encoded here matter more than the word list:

* uncertainty/negativity is scored on management-authored sections ONLY
  (Business review, Liquidity, Outlook). RISK FACTORS is boilerplate and is
  tracked separately as a length feature — scoring it would poison the tone
  measure with lawyer language.
* every model-matrix row carries ``max_pub_q``, the latest publication
  quarter that contributed to it. The no-leakage test asserts
  ``max_pub_q <= quarter`` for every row, and a truncation-invariance test
  asserts that recomputing features on data censored at T leaves all rows
  with as-of <= T unchanged.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

LEXICON = {
    "uncertainty": {
        "may", "might", "could", "uncertain", "uncertainty", "uncertainties",
        "depends", "depending", "subject", "volatile", "volatility",
        "cautious", "caution", "limited", "visibility", "pressure",
        "possibly", "perhaps", "risk", "risks",
    },
    "negative": {
        "weakened", "weak", "declined", "decline", "deteriorated",
        "deteriorate", "deterioration", "reduced", "tighten", "tightened",
        "amendment", "adverse", "adversely", "loss", "losses", "impairment",
        "breach", "default", "restructuring", "downgrade",
    },
}

_SECTIONS = ("BUSINESS REVIEW", "LIQUIDITY AND CAPITAL RESOURCES",
             "OUTLOOK", "RISK FACTORS")
_MGMT_SECTIONS = _SECTIONS[:3]
_TOKEN = re.compile(r"[a-z']+")


def split_sections(text: str) -> dict[str, str]:
    out, current, buf = {}, None, []
    for line in text.split("\n"):
        s = line.strip()
        if s in _SECTIONS:
            if current is not None:
                out[current] = "\n".join(buf)
            current, buf = s, []
        elif current is not None:
            buf.append(s)
    if current is not None:
        out[current] = "\n".join(buf)
    return out


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def score_document(text: str) -> dict:
    sec = split_sections(text)
    mgmt_toks: list[str] = []
    for name in _MGMT_SECTIONS:
        mgmt_toks.extend(_tokens(sec.get(name, "")))
    n = max(len(mgmt_toks), 1)
    unc = sum(t in LEXICON["uncertainty"] for t in mgmt_toks) / n
    neg = sum(t in LEXICON["negative"] for t in mgmt_toks) / n
    rf_len = len(_tokens(sec.get("RISK FACTORS", "")))
    liq = sec.get("LIQUIDITY AND CAPITAL RESOURCES", "")
    liq_detail_missing = int(len(_tokens(liq)) < 55)   # detail block absent
    tf: dict[str, int] = {}
    for t in mgmt_toks:
        tf[t] = tf.get(t, 0) + 1
    return {"unc": unc, "neg": neg, "rf_len": rf_len,
            "liq_missing": liq_detail_missing, "_tf": tf}


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return np.nan
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = np.sqrt(sum(v * v for v in a.values()))
    nb = np.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else np.nan


def build_filing_features(filings_with_text: pd.DataFrame) -> pd.DataFrame:
    """Per-filing features + within-issuer deltas and YoY similarity.

    Deltas compare consecutive *filings* of the same issuer; the YoY
    similarity compares to the filing four periods earlier. All quantities
    are functions of the filing history only, so the as-of join below is the
    single place where time discipline must hold.
    """
    df = filings_with_text.sort_values(["issuer", "period_q"]).reset_index(drop=True)
    scored = [score_document(t) for t in df.text]
    for key in ("unc", "neg", "rf_len", "liq_missing"):
        df[key] = [s[key] for s in scored]
    tfs = [s["_tf"] for s in scored]

    d_unc = np.zeros(len(df)); d_neg = np.zeros(len(df))
    d_rf = np.zeros(len(df)); yoy = np.full(len(df), np.nan)
    for _, idx in df.groupby("issuer").indices.items():
        idx = np.asarray(idx)
        for j, row in enumerate(idx):
            if j >= 1:
                prev = idx[j - 1]
                d_unc[row] = df.unc.iat[row] - df.unc.iat[prev]
                d_neg[row] = df.neg.iat[row] - df.neg.iat[prev]
                d_rf[row] = df.rf_len.iat[row] - df.rf_len.iat[prev]
            if j >= 4:
                yoy[row] = _cosine(tfs[row], tfs[idx[j - 4]])
    df["d_unc"], df["d_neg"], df["d_rf_len"] = d_unc, d_neg, d_rf
    df["yoy_sim"] = yoy
    df["sim_missing"] = df.yoy_sim.isna().astype(int)
    df["yoy_sim"] = df.yoy_sim.fillna(0.90)
    return df


TONE_FEATURES = ["unc", "neg", "d_unc", "d_neg", "liq_missing"]
BLOAT_FEATURES = ["rf_len", "d_rf_len", "yoy_sim", "sim_missing"]
TEXT_FEATURES = TONE_FEATURES + BLOAT_FEATURES   # all columns carried in the join
FUND_FEATURES = ["leverage", "coverage", "margin", "staleness"]
MARKET_FEATURES = ["log_spread", "d_log_spread", "rating", "d_rating",
                   "trailing_event_rate"]

FEATURE_SETS = {
    "market": MARKET_FEATURES,
    "market+fund": MARKET_FEATURES + FUND_FEATURES,
    # headline: disciplined tone block only. Adding the boilerplate-length
    # and similarity features ("kitchen sink") demonstrably destroys the
    # incremental signal — kept as a named set to show exactly that.
    "full": MARKET_FEATURES + FUND_FEATURES + TONE_FEATURES,
    "kitchen_sink": MARKET_FEATURES + FUND_FEATURES + TEXT_FEATURES,
    "text_only": TEXT_FEATURES,
}


def build_model_matrix(panel: pd.DataFrame, filing_feats: pd.DataFrame,
                       horizon: int = 4) -> pd.DataFrame:
    """As-of join at quarter granularity + label construction.

    For each alive issuer-quarter t, attach the latest filing with
    pub_q <= t. Label: event in (t, t+horizon]. Rows whose label window is
    truncated by the sample end are dropped (unresolved labels).
    """
    panel = panel.sort_values(["issuer", "quarter"]).reset_index(drop=True)
    T = int(panel.quarter.max())

    # market dynamics
    panel["log_spread"] = np.log(panel.spread)
    panel["d_log_spread"] = panel.groupby("issuer").log_spread.diff().fillna(0.0)
    panel["d_rating"] = panel.groupby("issuer").rating.diff().fillna(0.0)

    agg = panel.groupby("quarter").event.mean()
    trailing = agg.rolling(4, min_periods=1).mean().shift(1).fillna(agg.iloc[0])
    panel["trailing_event_rate"] = panel.quarter.map(trailing)

    # label: event within (t, t+h]
    ev_q = panel.loc[panel.event, ["issuer", "quarter"]]
    ev_map = {iss: g.quarter.values for iss, g in ev_q.groupby("issuer")}
    labels = np.zeros(len(panel), dtype=int)
    for i, (iss, q) in enumerate(zip(panel.issuer.values, panel.quarter.values)):
        evs = ev_map.get(iss)
        if evs is not None and ((evs > q) & (evs <= q + horizon)).any():
            labels[i] = 1
    panel["y"] = labels
    panel = panel[panel.quarter + horizon <= T].copy()   # resolved labels only

    # as-of join of filing features
    cols = TEXT_FEATURES + ["leverage", "coverage", "margin",
                            "period_q", "pub_q"]
    ff = filing_feats.sort_values(["issuer", "pub_q", "period_q"])
    grouped = {iss: g for iss, g in ff.groupby("issuer")}
    rows = []
    for iss, g in panel.groupby("issuer"):
        f = grouped.get(iss)
        if f is None:
            continue
        pub = f.pub_q.values
        for r in g.itertuples():
            k = np.searchsorted(pub, r.quarter, side="right") - 1
            if k < 0:
                continue                       # nothing published yet
            frow = f.iloc[k]
            rows.append((r.Index, frow.period_q, frow.pub_q,
                         *[frow[c] for c in cols[:-2]]))
    joined = pd.DataFrame(rows, columns=["_idx", "period_q", "max_pub_q",
                                         *cols[:-2]]).set_index("_idx")
    out = panel.join(joined, how="inner")
    out["staleness"] = out.quarter - out.period_q
    return out.reset_index(drop=True)
