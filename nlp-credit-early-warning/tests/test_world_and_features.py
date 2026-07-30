import numpy as np
import pandas as pd

from synthcredit.world import WorldConfig, simulate_world, WORLDS
from synthcredit.documents import render_all, render_document
from synthcredit.features import (build_filing_features, build_model_matrix,
                                  score_document)


def _small(world, seed=11):
    cfg = WorldConfig(world=world, seed=seed)
    cfg.n_issuers, cfg.n_quarters = 80, 44
    return cfg


# ---------------------------------------------------------------- world ----

def test_worlds_share_identical_dynamics():
    """Same seed => bit-identical states, events and prices across worlds.
    The text channel is the only difference — this is what makes the
    three-world comparison a controlled experiment."""
    ref = simulate_world(_small("signal"))
    for wname in ("coincident", "noise"):
        w = simulate_world(_small(wname))
        for col in ("x", "spread", "rating"):
            assert np.allclose(w["panel"][col], ref["panel"][col])
        assert (w["panel"].event == ref["panel"].event).all()
        assert not np.allclose(w["filings"].mgmt, ref["filings"].mgmt)


def test_event_rates_plausible():
    w = simulate_world(_small("signal"))
    p, cyc = w["panel"], w["cycle"]
    er = p.groupby("quarter").event.mean()
    calm = er[cyc.stress.values == 0].mean()
    stress = er[cyc.stress.values == 1].mean()
    assert 0.005 < calm < 0.06
    assert stress > calm            # events cluster in stress


def test_exit_and_entry():
    w = simulate_world(_small("signal"))
    p = w["panel"]
    gone = p[p.event].issuer.unique()
    for iss in gone[:10]:
        g = p[p.issuer == iss]
        assert g.quarter.max() == g[g.event].quarter.max()   # no zombie rows
    assert p.issuer.nunique() > 80   # entrants replaced the departed


# ------------------------------------------------------------- documents ----

def test_hedging_monotone_in_mgmt():
    lo = [score_document(render_document(-2.0, 8, 0, s))["unc"]
          for s in range(60)]
    hi = [score_document(render_document(+2.0, 8, 0, s))["unc"]
          for s in range(60)]
    assert np.mean(hi) > np.mean(lo) + 0.02


def test_risk_factors_grow_append_only():
    a = score_document(render_document(0.0, 4, 0, 7))["rf_len"]
    b = score_document(render_document(0.0, 40, 0, 7))["rf_len"]
    assert b > a


def test_liquidity_detail_flag():
    docs = [render_document(3.0, 8, 0, s) for s in range(80)]
    flags = [score_document(d)["liq_missing"] for d in docs]
    assert np.mean(flags) > 0.3      # bad signal => detail often missing


# ---------------------------------------------------------------- PIT -------

def test_no_future_publication_in_matrix():
    w = simulate_world(_small("signal"))
    ff = build_filing_features(render_all(w["filings"]))
    mm = build_model_matrix(w["panel"], ff, horizon=4)
    assert (mm.max_pub_q <= mm.quarter).all()
    assert (mm.staleness >= 1).all()


def test_truncation_invariance():
    """Recomputing everything on data censored at T must leave every row
    with as-of quarter <= T - horizon unchanged. If this fails, some feature
    is peeking past its publication date."""
    w = simulate_world(_small("signal"))
    ff = build_filing_features(render_all(w["filings"]))
    full = build_model_matrix(w["panel"], ff, horizon=4)

    T_cut = 30
    panel_c = w["panel"][w["panel"].quarter <= T_cut]
    filings_c = w["filings"][w["filings"].pub_q <= T_cut]
    ff_c = build_filing_features(render_all(filings_c))
    trunc = build_model_matrix(panel_c, ff_c, horizon=4)

    key = ["issuer", "quarter"]
    cols = ["y", "unc", "neg", "d_unc", "yoy_sim", "leverage",
            "log_spread", "max_pub_q"]
    a = full[full.quarter + 4 <= T_cut].set_index(key)[cols].sort_index()
    b = trunc.set_index(key)[cols].sort_index()
    b = b.loc[a.index]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)
