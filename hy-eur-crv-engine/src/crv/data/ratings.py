"""Rating-bucket and granular-notch derivation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import settings


def add_rating_bucket(static_snap: pd.DataFrame) -> pd.DataFrame:
    """Assign a coarse rating bucket per bond.

    Uses the first available of Moody's / S&P / Fitch, falling back to ``NR``.
    """
    def _bucket(row) -> str:
        for c in ("rating_moody", "rating_sp", "rating_fitch"):
            r = str(row.get(c, "")).strip()
            if r in settings.RATING_BUCKET:
                return settings.RATING_BUCKET[r]
        return "NR"

    out = static_snap.copy()
    out["rating_bucket"] = out.apply(_bucket, axis=1)
    return out


def _clean_rtg(s: pd.Series) -> pd.Series:
    """Normalise raw agency strings (strip ``*``/spaces, drop Bloomberg ``u`` suffix)."""
    return (
        s.astype(str).str.strip()
        .str.replace(r"[ \*]", "", regex=True)
        .str.replace(r"u$", "", regex=True)        # 'Ba2u' -> 'Ba2'
        .replace({"N.A.": np.nan, "NA": np.nan, "nan": np.nan, "None": np.nan})
    )


def add_rating_notch(df: pd.DataFrame) -> pd.DataFrame:
    """Add a continuous consensus notch rating (mean of available agency notches).

    The 3-level ``rating_bucket`` is left unchanged (it is the intra-rating peer
    axis); the notch refines the fair-value model only. Falls back to
    ``rating_num * 3`` where no agency rating parses.
    """
    df = df.copy()
    sp = _clean_rtg(df["rating_sp"]).map(settings.SP_NOTCH)
    mo = _clean_rtg(df["rating_moody"]).map(settings.MOODY_NOTCH)
    ft = _clean_rtg(df["rating_fitch"]).map(settings.SP_NOTCH)   # Fitch on S&P scale
    df["rating_notch"] = pd.concat([sp, mo, ft], axis=1).mean(axis=1)
    if "rating_num" not in df.columns:
        df["rating_num"] = df["rating_bucket"].map(settings.RATING_NUM)
    df["rating_notch"] = df["rating_notch"].fillna(df["rating_num"] * 3)
    df["rating_notch_sq"] = df["rating_notch"] ** 2
    return df
