"""Router must produce three disjoint panels and respect the spread bands."""

from __future__ import annotations

import pandas as pd

from crv.config import settings
from crv.data import router


def _raw_panel() -> pd.DataFrame:
    return pd.DataFrame([
        # clean fixed HY -> Track 1
        dict(isin="A", date="2025-01-03", industry="Retail", z_spread=400,
             coupon_type="FIXED", rating_bucket="HY_BB", duration=4, px_bid=99, px_ask=101),
        # clean floating -> Track 2
        dict(isin="B", date="2025-01-03", industry="Retail", z_spread=380,
             coupon_type="FLOATING", rating_bucket="HY_BB", duration=3, px_bid=99, px_ask=101),
        # distressed -> Track 3
        dict(isin="C", date="2025-01-03", industry="Retail", z_spread=1500,
             coupon_type="FIXED", rating_bucket="HY_CCC", duration=2, px_bid=55, px_ask=57),
        # broken (below floor) -> dropped
        dict(isin="D", date="2025-01-03", industry="Retail", z_spread=10,
             coupon_type="FIXED", rating_bucket="HY_BB", duration=4, px_bid=99, px_ask=101),
        # non-corporate -> dropped before routing
        dict(isin="E", date="2025-01-03", industry="Banks", z_spread=300,
             coupon_type="FIXED", rating_bucket="HY_BB", duration=4, px_bid=99, px_ask=101),
    ])


def test_tracks_are_disjoint():
    p = router.route(_raw_panel())
    s1 = set(p.cross_section["isin"])
    s2 = set(p.self_ref["isin"])
    s3 = set(p.distressed["isin"])
    assert s1.isdisjoint(s2) and s1.isdisjoint(s3) and s2.isdisjoint(s3)


def test_routing_assignment():
    p = router.route(_raw_panel())
    assert set(p.cross_section["isin"]) == {"A"}
    assert set(p.self_ref["isin"]) == {"B"}
    assert set(p.distressed["isin"]) == {"C"}


def test_distress_cutoff_boundary():
    p = router.route(_raw_panel())
    assert (p.distressed["z_spread"] > settings.DISTRESS_CUTOFF).all()
    assert (p.cross_section["z_spread"] >= settings.SPREAD_FLOOR).all()
