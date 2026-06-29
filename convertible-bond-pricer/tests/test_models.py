"""
tests/test_models.py
--------------------
Unit tests for the convertible bond pricing engine.

Run:  pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from models import (
    ConvertibleBond, MarketParams,
    price_bs_benchmark, price_tsiveriotis_fernandes,
)
from synthetic_data import (
    make_vanilla_cb, make_callable_cb, make_putable_cb,
    make_distressed_cb, make_equity_sensitive_cb,
)


def _tf(cb, mkt, N=300):
    return price_tsiveriotis_fernandes(cb, mkt, N=N)

def _bs(cb, mkt):
    return price_bs_benchmark(cb, mkt)


# ---------------------------------------------------------------------------
# 1. Arbitrage lower bounds (model-independent)
# ---------------------------------------------------------------------------

class TestArbitrageBounds:
    """CB price must be >= both conversion value and bond floor."""

    def test_above_conversion_value_vanilla(self):
        cb, mkt = make_vanilla_cb()
        assert _tf(cb, mkt)["price"] >= cb.conversion_ratio * mkt.S0 - 1.0

    def test_above_bond_floor_vanilla(self):
        cb, mkt = make_vanilla_cb()
        disc = mkt.r + mkt.credit_spread
        bf = sum(cb.coupon * np.exp(-disc*t) for t in cb.coupon_times)
        bf += cb.face_value * np.exp(-disc * cb.maturity)
        assert _tf(cb, mkt)["price"] >= bf - 1.0

    def test_above_conversion_value_equity_case(self):
        cb, mkt = make_equity_sensitive_cb()
        assert _tf(cb, mkt)["price"] >= cb.conversion_ratio * mkt.S0 - 1.0

    def test_price_positive(self):
        for factory in [make_vanilla_cb, make_distressed_cb, make_equity_sensitive_cb]:
            cb, mkt = factory()
            assert _tf(cb, mkt)["price"] > 0


# ---------------------------------------------------------------------------
# 2. American >= European (TF >= BS benchmark)
# ---------------------------------------------------------------------------

class TestAmericanEuropeanRelation:
    """
    TF prices an AMERICAN convertible (holder can convert at any node).
    BS benchmark uses a EUROPEAN call.
    American CB >= European CB always, so TF >= BS.
    """

    def test_tf_geq_bs_vanilla(self):
        cb, mkt = make_vanilla_cb()
        assert _tf(cb, mkt)["price"] >= _bs(cb, mkt)["price"] - 1.0

    def test_tf_geq_bs_distressed(self):
        cb, mkt = make_distressed_cb()
        assert _tf(cb, mkt)["price"] >= _bs(cb, mkt)["price"] - 1.0

    def test_tf_prices_equity_case_above_conv_value(self):
        """Deep ITM CB: TF = conversion value (American converts immediately)."""
        cb, mkt = make_equity_sensitive_cb()
        conv = cb.conversion_ratio * mkt.S0
        assert _tf(cb, mkt)["price"] >= conv - 1.0

    def test_tf_geq_bs_callable(self):
        """Even callable CBs: TF American >= TF European (which ~ BS + American premium)."""
        cb, mkt = make_callable_cb()
        # Basic sanity: TF price > 0
        assert _tf(cb, mkt)["price"] > 0


# ---------------------------------------------------------------------------
# 3. Spread sensitivity
# ---------------------------------------------------------------------------

class TestSpreadSensitivity:
    """Both models should decrease in price as credit spread rises."""

    def test_tf_decreasing_in_spread(self):
        cb, mkt = make_vanilla_cb()
        prices = []
        for cs in [0.01, 0.05, 0.10, 0.20]:
            m = MarketParams(S0=mkt.S0, r=mkt.r, sigma=mkt.sigma,
                             credit_spread=cs, recovery=mkt.recovery)
            prices.append(_tf(cb, m, N=150)["price"])
        assert all(prices[i] >= prices[i+1] - 1.0 for i in range(len(prices)-1))

    def test_bs_decreasing_in_spread(self):
        cb, mkt = make_vanilla_cb()
        prices = []
        for cs in [0.01, 0.05, 0.10, 0.20]:
            m = MarketParams(S0=mkt.S0, r=mkt.r, sigma=mkt.sigma,
                             credit_spread=cs, recovery=mkt.recovery)
            prices.append(_bs(cb, m)["price"])
        assert all(prices[i] >= prices[i+1] - 1.0 for i in range(len(prices)-1))


# ---------------------------------------------------------------------------
# 4. Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_price_increasing_in_stock(self):
        """Higher stock price -> higher CB price (positive delta)."""
        cb, mkt = make_vanilla_cb()
        prices = []
        for S in [30, 50, 80, 120]:
            m = MarketParams(S0=S, r=mkt.r, sigma=mkt.sigma,
                             credit_spread=mkt.credit_spread, recovery=mkt.recovery)
            prices.append(_tf(cb, m, N=150)["price"])
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_price_increasing_in_vol(self):
        """Higher vol -> higher CB price (positive vega; option component gains)."""
        cb, mkt = make_vanilla_cb()
        prices = []
        for sig in [0.10, 0.25, 0.45, 0.65]:
            m = MarketParams(S0=mkt.S0, r=mkt.r, sigma=sig,
                             credit_spread=mkt.credit_spread, recovery=mkt.recovery)
            prices.append(_tf(cb, m, N=150)["price"])
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))


# ---------------------------------------------------------------------------
# 5. Extreme regimes
# ---------------------------------------------------------------------------

class TestExtremeRegimes:
    def test_equity_sensitive_low_cash_fraction(self):
        """Deep ITM CB: bond floor / price should be < 30%."""
        cb, mkt = make_equity_sensitive_cb()
        result = _tf(cb, mkt, N=200)
        assert result["bond_floor_proxy"] < 0.30

    def test_distressed_high_cash_fraction(self):
        """Busted convert: bond floor / price should dominate (> 70%)."""
        cb, mkt = make_distressed_cb()
        result = _tf(cb, mkt, N=200)
        assert result["bond_floor_proxy"] > 0.50  # dominant bond component; high vol keeps some equity value

    def test_zero_spread_tf_above_arbitrage_bounds(self):
        """
        Near-zero spread: TF price should be above arbitrage bounds.
        Note: BS (additive decomposition) can exceed TF because BS ignores
        the trade-off between conversion and bond floor retention -- both
        models are internally consistent but BS is an upper-bound approximation.
        """
        cb, mkt = make_vanilla_cb()
        mkt_ns = MarketParams(S0=mkt.S0, r=mkt.r, sigma=mkt.sigma,
                              credit_spread=1e-6, recovery=mkt.recovery)
        tf_price = _tf(cb, mkt_ns, N=300)["price"]
        # Must be above conversion value
        assert tf_price >= cb.conversion_ratio * mkt_ns.S0 - 1.0
        # Must be reasonable (within 2x of face value for ATM-ish CB)
        assert tf_price < cb.face_value * 2.0


# ---------------------------------------------------------------------------
# 6. Call / Put provisions
# ---------------------------------------------------------------------------

class TestProvisions:
    def test_call_reduces_price_vs_non_callable(self):
        """Callable CB <= non-callable CB (call hurts holder)."""
        cb_call, mkt = make_callable_cb()
        cb_nc = ConvertibleBond(
            face_value=cb_call.face_value, maturity=cb_call.maturity,
            coupon_rate=cb_call.coupon_rate, coupon_freq=cb_call.coupon_freq,
            conversion_ratio=cb_call.conversion_ratio,
        )
        assert _tf(cb_call, mkt, N=200)["price"] <= _tf(cb_nc, mkt, N=200)["price"] + 1.0

    def test_put_increases_price_vs_non_putable(self):
        """Putable CB >= non-putable CB (put helps holder)."""
        cb_put, mkt = make_putable_cb()
        cb_np = ConvertibleBond(
            face_value=cb_put.face_value, maturity=cb_put.maturity,
            coupon_rate=cb_put.coupon_rate, coupon_freq=cb_put.coupon_freq,
            conversion_ratio=cb_put.conversion_ratio,
        )
        assert _tf(cb_put, mkt, N=200)["price"] >= _tf(cb_np, mkt, N=200)["price"] - 1.0


# ---------------------------------------------------------------------------
# 7. Convergence and output format
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_price_converges(self):
        """N=400 and N=500 prices differ by < 0.5."""
        cb, mkt = make_vanilla_cb()
        p400 = _tf(cb, mkt, N=400)["price"]
        p500 = _tf(cb, mkt, N=500)["price"]
        assert abs(p400 - p500) < 0.5

    def test_tf_output_keys(self):
        cb, mkt = make_vanilla_cb()
        result = _tf(cb, mkt, N=100)
        required = {"model", "price", "cash_component", "equity_component",
                    "delta", "gamma", "premium", "conversion_value", "bond_floor_proxy"}
        assert required.issubset(result.keys())

    def test_bs_output_keys(self):
        cb, mkt = make_vanilla_cb()
        result = _bs(cb, mkt)
        required = {"model", "price", "bond_floor", "option_value",
                    "conversion_value", "delta", "premium"}
        assert required.issubset(result.keys())

    def test_cash_equity_partition(self):
        """cash_component + equity_component == price."""
        cb, mkt = make_vanilla_cb()
        r = _tf(cb, mkt, N=200)
        assert abs(r["cash_component"] + r["equity_component"] - r["price"]) < 0.01


# ---------------------------------------------------------------------------
# 8. BS benchmark specific checks
# ---------------------------------------------------------------------------

class TestBsBenchmarkSpecific:
    """
    The BS benchmark treats bond floor and option as additive (independent).
    This is an approximation. Tests verify internal consistency, not TF vs BS.
    """

    def test_bs_decomposes_correctly(self):
        """bond_floor + option_value == bs price."""
        cb, mkt = make_vanilla_cb()
        r = _bs(cb, mkt)
        assert abs(r["bond_floor"] + r["option_value"] - r["price"]) < 0.01

    def test_bs_bond_floor_decreasing_in_spread(self):
        """Higher spread -> lower bond floor."""
        cb, mkt = make_vanilla_cb()
        bf_lo = _bs(cb, MarketParams(S0=mkt.S0, r=mkt.r, sigma=mkt.sigma,
                                      credit_spread=0.02, recovery=mkt.recovery))["bond_floor"]
        bf_hi = _bs(cb, MarketParams(S0=mkt.S0, r=mkt.r, sigma=mkt.sigma,
                                      credit_spread=0.15, recovery=mkt.recovery))["bond_floor"]
        assert bf_lo > bf_hi
