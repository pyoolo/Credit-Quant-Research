"""Track-3 model levels L1-L4.

Each level is a standalone function so it can be tested and reasoned about in
isolation; :mod:`crv.tracks.distressed.signal` composes them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.tracks.distressed import config as ds_cfg
from crv.tracks.distressed.solver import implied_lambda, max_implied_recovery


def level1_pd_frontier(static: pd.DataFrame) -> pd.DataFrame:
    """L1 — price-implied cumulative default at fixed recovery scenarios."""
    rows = []
    for r in static.itertuples(index=False):
        d = r._asdict()
        px, cpn = d.get("px"), d.get("coupon")
        mat = d.get("workout_dt") or d.get("maturity")
        typ = str(d.get("cpn_typ", "")).upper()
        freq = 2.0
        rec = {"isin": d["isin"]}
        bad = (pd.isna(px) or pd.isna(cpn) or pd.isna(mat) or "FLAT" in typ
               or "FLOAT" in typ or px >= 99 or px <= 0)
        if bad:
            note = ("flat/float/perp/par"
                    if ("FLAT" in typ or "FLOAT" in typ or (not pd.isna(px) and px >= 99))
                    else "missing data")
            rec.update(l1_applicable=False, l1_note=note)
            for R in ds_cfg.R_SCENARIOS:
                rec[f"pd_{int(R * 100)}"] = np.nan
            rec["max_impl_recovery"] = np.nan
            rows.append(rec)
            continue
        years = max(0.25, (pd.to_datetime(mat) - pd.Timestamp.today()).days / 365.25)
        rec.update(l1_applicable=True, l1_note="ok", years=round(years, 2))
        any_ok = False
        for R in ds_cfg.R_SCENARIOS:
            lam = implied_lambda(px, cpn, freq, years, R)
            pd_cum = 1 - np.exp(-lam * years) if not np.isnan(lam) else np.nan
            rec[f"pd_{int(R * 100)}"] = pd_cum
            any_ok |= not np.isnan(pd_cum)
        rec["max_impl_recovery"] = np.nan if any_ok else max_implied_recovery(px, cpn, freq, years)
        rows.append(rec)
    return pd.DataFrame(rows)


def level2_capstructure(static: pd.DataFrame, l1: pd.DataFrame) -> pd.DataFrame:
    """L2 — on multi-bond issuers, implied recovery/PD should respect seniority."""
    m = static.merge(l1[["isin", "l1_applicable", "pd_40"]], on="isin", how="left")
    m["rank_num"] = m["rank_txt"].map(ds_cfg.RANK_ORDER)
    out = []
    for iss, g in m.groupby("issuer_id"):
        if g["isin"].nunique() < 2 or g["rank_num"].notna().sum() < 2:
            continue
        g = g.sort_values("rank_num")
        gg = g.dropna(subset=["px", "rank_num"])
        if len(gg) < 2:
            continue
        # more senior (low rank_num) should be priced higher; positive corr = violation
        viol = (np.corrcoef(gg["rank_num"], gg["px"])[0, 1] > 0) if gg["px"].nunique() > 1 else False
        out.append({"issuer_id": iss, "issuer": g["issuer"].iloc[0],
                    "n_bond": g["isin"].nunique(), "ranks": list(g["rank_txt"]),
                    "prices": [round(x, 1) for x in g["px"]],
                    "seniority_violation": bool(viol)})
    return pd.DataFrame(out)


def level3_trajectory(long: pd.DataFrame) -> pd.DataFrame:
    """L3 — recent price trend + acceleration per bond -> regime + confidence."""
    long = long.copy()
    long["date"] = pd.to_datetime(long["date"])
    rows = []
    for isin, g in long.sort_values("date").groupby("isin"):
        g = g.dropna(subset=["px"])
        n = len(g)
        if n < 4:
            rows.append({"isin": isin, "l3_n": n, "slope": np.nan, "accel": np.nan,
                         "regime": "insufficient", "l3_conf": 0.0})
            continue
        t = (g["date"] - g["date"].min()).dt.days.values / 7.0
        y = g["px"].values
        c1 = np.polyfit(t, y, 1)
        slope = c1[0]
        c2 = np.polyfit(t, y, 2) if n >= 6 else [0, c1[0], c1[1]]
        accel = 2 * c2[0]
        rmse = np.sqrt(np.mean((y - np.polyval(c1, t)) ** 2))
        if slope < -0.5:
            reg = "deteriorating"
        elif slope > 0.5:
            reg = "recovering"
        else:
            reg = "stabilizing"
        if accel > 0.05 and slope < 0:
            reg = "deteriorating_slowing"
        conf = float(np.clip(n / 26.0, 0, 1) * (1 / (1 + rmse / 5)))
        rows.append({"isin": isin, "l3_n": n, "slope": round(slope, 3),
                     "accel": round(accel, 4), "regime": reg,
                     "l3_conf": round(conf, 2), "last_px": round(y[-1], 1)})
    return pd.DataFrame(rows)


def level4_composite(l1: pd.DataFrame, l2: pd.DataFrame, l3: pd.DataFrame,
                     static: pd.DataFrame) -> pd.DataFrame:
    """L4 — combine the levels into a [-1,+1] judgement score with an uncertainty band."""
    df = (static[["isin", "issuer", "rank_txt", "px", "amount"]]
          .merge(l1, on="isin", how="left").merge(l3, on="isin", how="left"))
    if len(l2):
        flagged = l2[l2.get("seniority_violation", False).astype(bool)]
        viol = set(flagged["issuer_id"])
    else:
        viol = set()
    df["cap_struct_flag"] = static.set_index("isin")["issuer_id"].reindex(df["isin"]).isin(viol).values

    s1 = np.where(df["pd_40"].notna(), (df["pd_40"] - 0.5) * 2, 0.0)
    reg_map = {"recovering": 1, "stabilizing": 0.3, "deteriorating_slowing": -0.3,
               "deteriorating": -1, "insufficient": 0}
    s3 = df["regime"].map(reg_map).fillna(0).values * df["l3_conf"].fillna(0).values
    s2 = np.where(df["cap_struct_flag"], 1.0, 0.0)

    conf = (0.4 * df["l1_applicable"].fillna(False).astype(float)
            + 0.4 * df["l3_conf"].fillna(0)
            + 0.2 * df["cap_struct_flag"].astype(float)).clip(0, 1)
    score = 0.4 * s1 + 0.4 * s3 + 0.2 * s2
    df["ds_score"] = np.round(score, 3)
    df["ds_conf"] = np.round(conf, 2)
    df["ds_score_lo"] = np.round(score - (1 - conf), 3)
    df["ds_score_hi"] = np.round(score + (1 - conf), 3)
    return df
