import numpy as np

from synthcredit.world import WorldConfig, simulate_world
from synthcredit.documents import render_all
from synthcredit.features import build_filing_features, build_model_matrix
from synthcredit import model as M

H = 4


def _matrix(world, seed=11, n=110, T=52):
    cfg = WorldConfig(world=world, seed=seed)
    cfg.n_issuers, cfg.n_quarters = n, T
    w = simulate_world(cfg)
    ff = build_filing_features(render_all(w["filings"]))
    return build_model_matrix(w["panel"], ff, horizon=H)


# ------------------------------------------------------------- validation ---

def test_walk_forward_embargo():
    mm = _matrix("noise")
    preds = M.walk_forward(mm, "market", horizon=H, first_train=20)
    for cutoff, g in preds.groupby("fold_cutoff"):
        # test rows start only after the embargo
        assert g.quarter.min() >= cutoff + H
    # training purge is structural: rows with quarter + H > cutoff are
    # excluded inside walk_forward; verify no test quarter ever overlaps
    # a label window that a training row could have seen.
    assert (preds.quarter - preds.fold_cutoff).min() >= H


def test_walk_forward_needs_history():
    mm = _matrix("noise")
    try:
        M.walk_forward(mm[mm.quarter < 8], "market", first_train=40)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on a too-short sample")


# ----------------------------------------------------------- falsification --

_CACHE = {}


def deltas():
    if _CACHE:
        return _CACHE
    out = _CACHE
    for world in ("signal", "coincident", "noise"):
        mm = _matrix(world, seed=11, n=130, T=56)
        base = M.walk_forward(mm, "market+fund", horizon=H, first_train=20)
        full = M.walk_forward(mm, "full", horizon=H, first_train=20)
        text = M.walk_forward(mm, "text_only", horizon=H, first_train=20)
        boot = M.bootstrap_delta_auc(full, base, n_boot=150, seed=1)
        out[world] = {"delta": boot["delta"], "lo": boot["lo"],
                      "hi": boot["hi"],
                      "auc_text": M.pooled_auc(text)}
    return out


def test_text_alone_ranking():
    d_all = deltas()
    """Text-alone AUC: informative in signal and coincident worlds,
    coin-flip in the noise world."""
    assert d_all["signal"]["auc_text"] > 0.62
    assert d_all["coincident"]["auc_text"] > 0.60
    assert abs(d_all["noise"]["auc_text"] - 0.5) < 0.06


def test_incremental_only_where_it_exists():
    d_all = deltas()
    """The falsification core: the SAME pipeline must find incremental
    signal in the signal world and (approximately) nothing elsewhere."""
    assert d_all["signal"]["delta"] > d_all["coincident"]["delta"]
    assert d_all["signal"]["delta"] > d_all["noise"]["delta"]
    assert d_all["signal"]["delta"] > 0.0
    assert abs(d_all["coincident"]["delta"]) < 0.008
    assert abs(d_all["noise"]["delta"]) < 0.008


def test_tautology_trap_demonstrated():
    d_all = deltas()
    """Coincident world: text predicts events on its own AND adds ~nothing
    given the spread — 'true and useless'."""
    d = d_all["coincident"]
    assert d["auc_text"] > 0.60 and abs(d["delta"]) < 0.008
