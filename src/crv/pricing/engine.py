"""Pricing engine: artefact loader, three track pricers, router.

The engine is built around an immutable :class:`Artefacts` bundle loaded once
from the persisted track outputs. Each pricer is a pure function of
``(spec, artefacts)`` so they are straightforward to test with a stub bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crv.config import paths, settings
from crv.tracks.distressed.solver import implied_lambda
from crv.utils.io import load_json, load_pickle
from crv.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_T3_PARAMS = {
    "risk_free": 0.03, "r_scenarios": [0.2, 0.4, 0.6],
    "rank_order": settings.RANK_ORDER if hasattr(settings, "RANK_ORDER") else {},
}
_FUND_RAW_KEYS = ["net_lev", "icr", "ebitda_margin", "ebitda_growth", "capex_pct"]


@dataclass
class Artefacts:
    """Everything the pricers need, loaded from the persisted track outputs."""
    t1_ok: bool = False
    model: object = None
    scaler: object = None
    xcols: list = field(default_factory=list)
    feats: list = field(default_factory=list)
    resid_q: dict = field(default_factory=dict)
    peer_snap: dict = field(default_factory=dict)
    feat_med: dict = field(default_factory=dict)
    latest_wk: str | None = None
    sectors: list = field(default_factory=lambda: ["OTHER"])
    fund_std: dict = field(default_factory=dict)
    fund_map: dict = field(default_factory=lambda: {"quality": [], "risk": []})
    bucket_notch: dict = field(default_factory=lambda: {"HY_BB": 12.0, "HY_B": 15.0, "HY_CCC": 18.0})
    t2_peers: dict | None = None
    t3_params: dict = field(default_factory=lambda: dict(_DEFAULT_T3_PARAMS))


def load_artefacts() -> Artefacts:
    """Load the fitted Track-1 model + Track-2 peers + Track-3 params."""
    art = Artefacts()
    a = load_pickle(paths.MODEL_T1)
    if a is not None:
        art.t1_ok = True
        art.model = a["model"]
        art.scaler = a["scaler"]
        art.xcols = list(a["xcols"])
        art.feats = list(a["feats"])
        art.resid_q = a.get("resid_q", {})
        art.peer_snap = a.get("peer_snap", {})
        art.feat_med = a.get("feat_median", {})
        art.latest_wk = a.get("latest_wk")
        art.sectors = a.get("sectors", []) or ["OTHER"]
        art.fund_std = a.get("fund_std_params", {})
        art.fund_map = a.get("fund_score_map", {"quality": [], "risk": []})
        art.bucket_notch = a.get("bucket_notch", art.bucket_notch)
        log.info("Track-1 model loaded | feats %s", art.feats)
    else:
        log.warning("Track-1 model not found at %s — pricer 1 disabled", paths.MODEL_T1)

    art.t2_peers = load_json(paths.T2_PEERS)
    art.t3_params = load_json(paths.T3_PARAMS) or dict(_DEFAULT_T3_PARAMS)
    return art


# --------------------------------------------------------------------------- #
# Fundamental-score reconstruction (raw ratios -> z-scores)                   #
# --------------------------------------------------------------------------- #
def raw_ratio_to_z(art: Artefacts, col: str, raw) -> float | None:
    """Convert a raw ratio to the model's z-score using saved train params."""
    p = art.fund_std.get(col)
    if p is None or raw is None:
        return None
    v = min(max(float(raw), p["lo"]), p["hi"])
    return (v - p["mu"]) / p["sd"] if p["sd"] else 0.0


def build_fund_scores(art: Artefacts, raw: dict) -> tuple[float, float]:
    """Mean of available oriented z-scores per side — matches the training build."""
    out = {}
    for side in ("quality", "risk"):
        zs = [z for col in art.fund_map.get(side, [])
              if (z := raw_ratio_to_z(art, col, raw.get(col))) is not None]
        out[f"fund_{side}"] = float(np.mean(zs)) if zs else art.feat_med.get(f"fund_{side}", 0.0)
    return out["fund_quality"], out["fund_risk"]


# --------------------------------------------------------------------------- #
# Router                                                                      #
# --------------------------------------------------------------------------- #
def classify_bond(spec: dict) -> str:
    """Route a spec to track ``"1"`` / ``"2"`` / ``"3"``."""
    ct = str(spec.get("coupon_type", "FIXED")).upper()
    zs = spec.get("expected_zs")
    rb = spec.get("rating_bucket", "HY_BB")
    is_float = ("FLOAT" in ct) or ("VARIABLE" in ct) or ("PIK" in ct)
    is_perp = bool(spec.get("is_perp", False))
    if zs is not None and zs > settings.DISTRESS_CUTOFF:
        return "3"
    if is_float or is_perp:
        return "2"
    if rb in settings.HY_BUCKETS_T1:
        return "1"
    return "2"


# --------------------------------------------------------------------------- #
# Track 1 — fair-spread Lasso                                                 #
# --------------------------------------------------------------------------- #
def price_t1(spec: dict, art: Artefacts, ci=("q10", "q90")) -> dict:
    """Fair Z-spread from the notch+fundamentals Lasso (raw ratios accepted)."""
    if not art.t1_ok:
        return {"error": "Track-1 model not loaded"}
    rb = spec.get("rating_bucket", "HY_BB")
    raw = {k: spec.get(k) for k in _FUND_RAW_KEYS}
    fq, fr = build_fund_scores(art, raw)

    feat = {}
    for f in art.feats:
        if f == "rating_notch":
            feat[f] = spec.get("rating_notch", art.bucket_notch.get(rb, 15.0))
        elif f == "fund_quality":
            feat[f] = fq
        elif f == "fund_risk":
            feat[f] = fr
        elif f == "is_callable":
            feat[f] = 1.0 if spec.get("is_callable") else 0.0
        elif f == "age":
            feat[f] = spec.get("issue_age", art.feat_med.get("age", 0.0))
        elif f == "yrs_to_call":
            feat[f] = spec.get("yrs_to_call", art.feat_med.get("yrs_to_call", 0.0))
        else:
            feat[f] = spec.get(f, art.feat_med.get(f))

    d = pd.DataFrame([feat])[art.feats].fillna(pd.Series(art.feat_med))
    Xn = pd.DataFrame(art.scaler.transform(d), columns=art.feats, index=d.index)
    Xc = pd.DataFrame(0.0, index=[0], columns=[c for c in art.xcols if c not in art.feats])
    sec = f"sector_{str(spec.get('sector', 'OTHER')).replace(' ', '_')}"
    wkc = f"wk_{art.latest_wk}"
    if sec in Xc.columns:
        Xc.loc[0, sec] = 1.0
    if wkc in Xc.columns:
        Xc.loc[0, wkc] = 1.0
    Xf = pd.concat([Xn, Xc], axis=1).reindex(columns=art.xcols, fill_value=0)
    fair = float(np.exp(art.model.predict(Xf)[0]))

    q = art.resid_q.get(rb)
    lo, hi = (fair * np.exp(q[ci[0]]), fair * np.exp(q[ci[1]])) if q else (np.nan, np.nan)
    inner = ((fair * np.exp(-0.5 * q["std"]), fair * np.exp(0.5 * q["std"]))
             if q and "std" in q else (np.nan, np.nan))
    peers = np.array(art.peer_snap.get(rb, []), float)

    res = dict(track="1", fair=fair, lo=lo, hi=hi, inner_lo=inner[0], inner_hi=inner[1],
               bucket=rb, fq=fq, fr=fr, npeers=len(peers), peers=peers,
               p25=np.quantile(peers, .25) if len(peers) else np.nan,
               p50=np.quantile(peers, .50) if len(peers) else np.nan,
               p75=np.quantile(peers, .75) if len(peers) else np.nan)
    prop = spec.get("proposed_zs")
    if prop:
        if prop > settings.DISTRESS_CUTOFF:
            res["distress_warn"] = True
        res["proposed"] = prop
        res["gap"] = prop - fair
        res["verdict"] = ("CHEAP" if (not np.isnan(hi) and prop > hi)
                          else "RICH" if (not np.isnan(lo) and prop < lo) else "FAIR")
    return res


# --------------------------------------------------------------------------- #
# Track 2 — peer-distribution placement                                       #
# --------------------------------------------------------------------------- #
def price_t2(spec: dict, art: Artefacts) -> dict:
    fam = "PERP" if spec.get("is_perp") else "FRN"
    metric = "spread-to-worst" if fam == "PERP" else "discount margin"
    prop = spec.get("proposed_zs")
    d = (art.t2_peers or {}).get(fam)
    res = dict(track="2", family=fam, metric=metric, proposed=prop, flags=[])
    if not d:
        res["note"] = "no saved peer distribution"
        return res
    vals = np.array(d["values"], float)
    res.update(p25=d["p25"], p50=d["p50"], p75=d["p75"], npeers=d["n"])
    if prop is not None and len(vals):
        res["pct"] = (vals < prop).mean() * 100
        res["verdict"] = ("WIDE/CHEAP" if prop > d["p75"]
                          else "TIGHT/RICH" if prop < d["p25"] else "IN-RANGE")
    return res


# --------------------------------------------------------------------------- #
# Track 3 — price-implied default                                             #
# --------------------------------------------------------------------------- #
def price_t3(spec: dict, art: Artefacts) -> dict:
    rf = art.t3_params["risk_free"]
    Rs = art.t3_params["r_scenarios"]
    px = spec.get("px", 60.0)
    cpn = spec.get("coupon", 5.0)
    yrs = max(0.25, spec.get("years", 3.0))
    freq = spec.get("cpn_freq", 2.0)
    res = dict(track="3", px=px, years=yrs, rank=spec.get("rank_txt", "Sr Unsecured"), flags=[])
    scen = {}
    for R in Rs:
        lam = implied_lambda(px, cpn, freq, yrs, R, rf)
        scen[int(R * 100)] = (1 - np.exp(-lam * yrs)) if not np.isnan(lam) else None
    res["pd_scenarios"] = scen
    res["pd_40"] = scen.get(40)
    return res


def route_and_price(spec: dict, art: Artefacts) -> dict:
    """Classify then price a single spec."""
    trk = classify_bond(spec)
    if trk == "1" and art.t1_ok:
        return price_t1(spec, art)
    if trk == "3":
        return price_t3(spec, art)
    return price_t2(spec, art)
