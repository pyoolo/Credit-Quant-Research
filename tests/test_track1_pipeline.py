"""End-to-end Track-1 model on the synthetic panel (no Bloomberg required)."""

from __future__ import annotations

from crv.tracks.cross_section import features, model, signal


def _prepare(panel, fund):
    panel = features.add_structural_features(panel)
    panel = features.attach_fundamentals(panel, fund)
    panel = features.apply_near_maturity_filter(panel)
    panel, params = features.build_fundamental_scores(panel)
    return panel, params


def test_fit_recovers_rating_effect(synthetic_xs_panel, synthetic_fundamentals):
    panel, _ = _prepare(synthetic_xs_panel, synthetic_fundamentals)
    fitted, train = model.fit(panel)
    # the synthetic data has a strong rating effect -> notch should lead
    ct = model.contribution_table(fitted)
    assert ct.index[0] == "rating_notch"
    assert fitted.r2_oof > 0.2


def test_score_produces_signal(synthetic_xs_panel, synthetic_fundamentals):
    panel, _ = _prepare(synthetic_xs_panel, synthetic_fundamentals)
    fitted, _ = model.fit(panel)
    live, snap = signal.score(panel, fitted)
    assert "signal" in snap.columns
    assert set(snap["signal"].unique()).issubset({"CHEAP", "FAIR", "RICH"})
    assert {"fair_zs", "resid_bps", "rv_z_4w"}.issubset(snap.columns)


def test_fund_scores_have_std_params(synthetic_xs_panel, synthetic_fundamentals):
    _, params = _prepare(synthetic_xs_panel, synthetic_fundamentals)
    # the persisted params must let the pricer rebuild scores from raw ratios
    assert params.std_params
    assert set(params.score_map) == {"quality", "risk"}
