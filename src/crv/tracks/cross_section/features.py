"""Track-1 feature engineering.

Three responsibilities, in order of application:

1. :func:`add_structural_features` — call / age / liquidity / rating features.
2. :func:`apply_near_maturity_filter` — drop near-zero-duration bonds whose
   Z-spreads are numerically unstable (small price moves -> exploding spread).
3. :func:`build_fundamental_scores` — compress ragged credit ratios into two
   *direction-split* scores (``fund_quality`` higher=safer, ``fund_risk``
   higher=riskier), z-scored on TRAIN stats only (no leakage). The fitted
   standardisation params are returned so the pricer can reproduce the transform
   from raw ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crv.config import settings
from crv.data import ratings
from crv.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FundScoreParams:
    """Persisted artefacts the app needs to rebuild fund scores from raw ratios."""
    std_params: dict = field(default_factory=dict)   # col -> {lo, hi, mu, sd}
    score_map: dict = field(default_factory=dict)     # {"quality": [...], "risk": [...]}


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive call / age / liquidity / rating structure features."""
    df = df.copy()
    df["next_call_dt"] = pd.to_datetime(df["next_call_dt"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["yrs_to_call"] = ((df["next_call_dt"] - df["date"]).dt.days / 365.25).clip(lower=0)
    df["is_callable"] = df["next_call_dt"].notna().astype(int)
    if "px" in df.columns and "next_call_px" in df.columns:
        df["px_vs_call"] = (
            ((df["px"] - df["next_call_px"]) / df["next_call_px"])
            .replace([np.inf, -np.inf], np.nan).fillna(0.0))
    else:
        df["px_vs_call"] = 0.0
    df["log_amt"] = np.log(df["amount"].clip(lower=1)) if "amount" in df.columns else np.nan
    df["rating_num"] = df["rating_bucket"].map(settings.RATING_NUM)

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["age"] = ((df["date"] - df["issue_date"]).dt.days / 365.25).clip(lower=0)
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")

    df = ratings.add_rating_notch(df)
    return df


def attach_fundamentals(df: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Merge fundamentals and a stable ``sector`` (rare industries -> OTHER)."""
    df = df.merge(fund[["isin", "issuer_key"] + settings.FUND_FEATS], on="isin", how="left")
    if "issuer" in df.columns:
        df["issuer_key"] = df["issuer_key"].fillna(df["issuer"])
    df["issuer_key"] = df["issuer_key"].fillna(df["isin"])
    sec_n = df.groupby("industry")["isin"].transform("nunique")
    df["sector"] = np.where(sec_n >= settings.MIN_SEC, df["industry"], "OTHER")
    return df


def apply_near_maturity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-zero-duration / near-maturity bonds with unstable Z-spreads."""
    df = df.copy()
    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["ttm_yrs"] = (df["maturity"] - df["date"]).dt.days / 365.25
    n0, b0 = len(df), df["isin"].nunique()
    dur = pd.to_numeric(df["duration"], errors="coerce")
    keep = ((df["ttm_yrs"] >= settings.MIN_TTM_YRS) & (dur >= settings.MIN_DURATION))
    # keep rows where ttm is unknown but duration is fine
    keep = keep | (df["ttm_yrs"].isna() & (dur >= settings.MIN_DURATION))
    df = df[keep].copy()
    log.info("near-maturity filter: dropped %d rows (%d bonds) -> %d rows, %d bonds",
             n0 - len(df), b0 - df["isin"].nunique(), len(df), df["isin"].nunique())
    return df


def build_fundamental_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, FundScoreParams]:
    """Compress credit ratios into two direction-split, train-standardised scores.

    Each ratio is winsorised on TRAIN bounds then z-scored with TRAIN mean/std
    (global, not within sector/rating — see the methodology notes). Components
    are averaged over whatever is present (>=1), making the scores robust to
    missing ratios. Returns the augmented frame and the fitted std params.
    """
    df = df.copy()
    train_mask = df["in_train"].fillna(False)
    params = FundScoreParams()

    def _global_z(col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        lo, hi = s[train_mask].quantile(settings.WINSOR), s[train_mask].quantile(1 - settings.WINSOR)
        s = s.clip(lo, hi)
        mu, sd = s[train_mask].mean(), s[train_mask].std()
        if not sd or np.isnan(sd):
            return pd.Series(np.nan, index=df.index, name=col)
        params.std_params[col] = {"lo": float(lo), "hi": float(hi),
                                  "mu": float(mu), "sd": float(sd)}
        return ((s - mu) / sd).rename(col)

    def _bucket(ratios):
        parts = [_global_z(c) for c in ratios
                 if c in df.columns and pd.to_numeric(df[c], errors="coerce")[train_mask].std()]
        if not parts:
            return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index)
        Z = pd.concat(parts, axis=1)
        n = Z.notna().sum(axis=1)
        return Z.mean(axis=1).where(n >= 1), n

    df["fund_quality"], df["fund_n_q"] = _bucket(settings.QUALITY_RATIOS)
    df["fund_risk"], df["fund_n_r"] = _bucket(settings.RISK_RATIOS)

    params.score_map = {
        "quality": [r for r in settings.QUALITY_RATIOS if r in params.std_params],
        "risk": [r for r in settings.RISK_RATIOS if r in params.std_params],
    }
    log.info("fund_quality coverage %.1f%% | fund_risk coverage %.1f%%",
             100 * df["fund_quality"].notna().mean(), 100 * df["fund_risk"].notna().mean())
    return df, params
