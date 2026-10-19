"""Rating-bucket and notch derivation."""

from __future__ import annotations

import pandas as pd

from crv.data import ratings


def test_bucket_from_first_available_agency():
    df = pd.DataFrame([
        {"rating_moody": "Ba2", "rating_sp": "", "rating_fitch": ""},     # -> HY_BB
        {"rating_moody": "", "rating_sp": "B-", "rating_fitch": ""},       # -> HY_B
        {"rating_moody": "", "rating_sp": "", "rating_fitch": ""},          # -> NR
    ])
    out = ratings.add_rating_bucket(df)
    assert list(out["rating_bucket"]) == ["HY_BB", "HY_B", "NR"]


def test_notch_orders_buckets():
    df = pd.DataFrame([
        {"rating_moody": "Ba2", "rating_sp": "BB", "rating_fitch": "", "rating_bucket": "HY_BB"},
        {"rating_moody": "Caa1", "rating_sp": "CCC+", "rating_fitch": "", "rating_bucket": "HY_CCC"},
    ])
    out = ratings.add_rating_notch(df)
    # CCC must carry a higher (riskier) notch than BB
    assert out.iloc[1]["rating_notch"] > out.iloc[0]["rating_notch"]


def test_notch_strips_bloomberg_suffix():
    df = pd.DataFrame([{"rating_moody": "Ba2u", "rating_sp": "", "rating_fitch": "",
                        "rating_bucket": "HY_BB"}])
    out = ratings.add_rating_notch(df)
    assert out.iloc[0]["rating_notch"] == 12  # 'Ba2u' parsed as 'Ba2'
