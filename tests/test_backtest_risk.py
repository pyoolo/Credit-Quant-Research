"""Tests for backtest, risk and the end-to-end pipeline."""
import pytest

from cds_basis import run_pipeline
from cds_basis.backtest import run_backtest
from cds_basis.data import generate_panel, validate_panel
from cds_basis.risk import JumpToDefaultConfig, jump_to_default_pnl, summarise_risk


def test_synthetic_panel_is_valid():
    panel = generate_panel()
    validate_panel(panel)
    assert set(["date", "name", "cds_spread", "bond_price"]).issubset(panel.columns)
    assert panel["name"].nunique() == 12


def test_backtest_requires_position_column():
    panel = generate_panel()
    panel["basis"] = panel["cds_spread"] - 0.01
    with pytest.raises(ValueError):
        run_backtest(panel, basis_col="basis")


def test_jtd_zero_when_perfectly_matched():
    # If bond_price/100 - bond_recovery == 1 - cds_recovery, JTD nets to zero
    cfg = JumpToDefaultConfig(
        notional=1e6, bond_recovery=0.4, cds_recovery=0.4, bond_price=100.0
    )
    # per_unit = (1-0.4) - (1.0 - 0.4) = 0
    assert jump_to_default_pnl(cfg) == pytest.approx(0.0, abs=1e-9)


def test_jtd_recovery_basis_creates_loss():
    # Bond recovers less than the CDS assumes -> residual loss on default
    cfg = JumpToDefaultConfig(
        notional=1e6, bond_recovery=0.30, cds_recovery=0.40, bond_price=100.0
    )
    assert jump_to_default_pnl(cfg) < 0


def test_pipeline_runs_end_to_end():
    result = run_pipeline()
    assert "sharpe" in result.backtest.stats
    assert "hist_var_99" in result.risk
    assert len(result.backtest.equity) > 0
    # Equity series should be cumulative of daily pnl
    assert result.backtest.equity.iloc[-1] == pytest.approx(
        result.backtest.pnl.sum(), rel=1e-9
    )


def test_pipeline_with_recompute_basis():
    # Exercises the full pricing stack on a small panel
    from cds_basis.data import SyntheticConfig

    panel = generate_panel(SyntheticConfig(n_names=3, n_days=120))
    panel = panel.drop(columns=[c for c in panel.columns if c == "basis"], errors="ignore")
    result = run_pipeline(panel=panel, recompute_basis=True)
    assert "basis" in result.panel.columns


def test_summarise_risk_keys():
    result = run_pipeline()
    summary = summarise_risk(result.backtest.pnl, positions=result.backtest.positions)
    assert "hist_var_99" in summary
    assert "es_99" in summary
    assert "mean_basis_dv01" in summary
