"""Pricing router, fund-score reconstruction and per-track pricers."""

from __future__ import annotations

import numpy as np

from crv.pricing import engine


def _stub_artefacts() -> engine.Artefacts:
    art = engine.Artefacts()
    art.t3_params = {"risk_free": 0.03, "r_scenarios": [0.2, 0.4, 0.6]}
    art.t2_peers = {"FRN": {"values": [300, 350, 400, 450, 500],
                            "p25": 350, "p50": 400, "p75": 450, "n": 5}}
    art.fund_std = {"net_lev": {"lo": 1.0, "hi": 8.0, "mu": 4.0, "sd": 1.5}}
    art.fund_map = {"quality": [], "risk": ["net_lev"]}
    return art


def test_router_distress_by_expected_zs():
    _stub_artefacts()
    spec = dict(coupon_type="FIXED", rating_bucket="HY_CCC", expected_zs=1500)
    assert engine.classify_bond(spec) == "3"


def test_router_floating_to_track2():
    spec = dict(coupon_type="FLOATING", rating_bucket="HY_BB")
    assert engine.classify_bond(spec) == "2"


def test_router_fixed_hy_to_track1():
    spec = dict(coupon_type="FIXED", rating_bucket="HY_BB")
    assert engine.classify_bond(spec) == "1"


def test_track3_pricer_returns_pd_scenarios():
    art = _stub_artefacts()
    spec = dict(coupon_type="FIXED", rating_bucket="HY_CCC", coupon=6.0, px=55,
                years=3.0, expected_zs=1500)
    r = engine.route_and_price(spec, art)
    assert r["track"] == "3"
    assert 0.0 <= r["pd_40"] <= 1.0


def test_track2_placement_verdict():
    art = _stub_artefacts()
    spec = dict(coupon_type="FLOATING", rating_bucket="HY_BB", proposed_zs=480)
    r = engine.route_and_price(spec, art)
    assert r["track"] == "2"
    assert r["verdict"] == "WIDE/CHEAP"


def test_raw_ratio_to_z_winsorises():
    art = _stub_artefacts()
    # value far above the high bound clips to (hi - mu)/sd
    z = engine.raw_ratio_to_z(art, "net_lev", 100.0)
    assert np.isclose(z, (8.0 - 4.0) / 1.5)


def test_track1_disabled_without_model():
    art = _stub_artefacts()  # t1_ok is False
    spec = dict(coupon_type="FIXED", rating_bucket="HY_BB", coupon=5.0, duration=4.0)
    r = engine.route_and_price(spec, art)
    # routes to track 1 but the pricer is disabled -> falls through to track 2 path
    assert r["track"] in {"1", "2"}
