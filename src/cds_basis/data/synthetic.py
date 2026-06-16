"""Data layer.

Defines the panel schema the rest of the library consumes and a synthetic
generator so the repo runs out-of-the-box.  To plug in real data, produce a
DataFrame with the same columns (see ``PANEL_COLUMNS``) and feed it directly
to the signal/backtest layers.

Panel schema (long format, one row per name per date):
    date         : datetime64
    name         : str            issuer identifier
    cds_spread   : float          CDS par spread (decimal)
    bond_price   : float          dirty price of the reference bond
    coupon       : float          bond annual coupon (decimal)
    maturity     : float          years to maturity
    recovery     : float          assumed recovery
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PANEL_COLUMNS = [
    "date",
    "name",
    "cds_spread",
    "bond_price",
    "coupon",
    "maturity",
    "recovery",
]


@dataclass
class SyntheticConfig:
    n_names: int = 12
    n_days: int = 504  # ~2y of business days
    seed: int = 42
    base_rate: float = 0.03
    # IG credit: spreads roughly 0.4%-1.5%
    spread_min: float = 0.004
    spread_max: float = 0.015
    # OU process for the *basis* (in bps, mean-reverting around a small neg level)
    ou_kappa: float = 12.0      # mean-reversion speed (per year)
    ou_theta_bps: float = -8.0  # long-run basis level (bps), mildly negative for IG
    ou_sigma_bps: float = 25.0  # vol of basis (bps per sqrt-year)


def _simulate_ou(
    n_days: int, dt: float, kappa: float, theta: float, sigma: float,
    x0: float, rng: np.random.Generator,
) -> np.ndarray:
    """Exact discretisation of an Ornstein-Uhlenbeck path."""
    x = np.empty(n_days)
    x[0] = x0
    exp_kdt = np.exp(-kappa * dt)
    var = sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt))
    std = np.sqrt(var)
    shocks = rng.standard_normal(n_days)
    for t in range(1, n_days):
        x[t] = theta + (x[t - 1] - theta) * exp_kdt + std * shocks[t]
    return x


def generate_panel(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a synthetic IG panel with a mean-reverting CDS-cash basis.

    The CDS spread follows a positive process; the cash spread is set so that
    the basis (cds - cash) follows an OU path.  This gives the signal layer a
    genuine, statistically tradable mean-reverting target.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    dates = pd.bdate_range("2022-01-03", periods=cfg.n_days)
    dt = 1.0 / 252.0
    theta = cfg.ou_theta_bps * 1e-4
    sigma = cfg.ou_sigma_bps * 1e-4

    rows = []
    for i in range(cfg.n_names):
        name = f"IG_{i:02d}"
        # Static bond characteristics for this name
        maturity = float(rng.uniform(4.5, 5.5))
        coupon = float(rng.uniform(0.025, 0.06))
        recovery = 0.4

        # CDS spread: mean-reverting positive process around a per-name level
        level = rng.uniform(cfg.spread_min, cfg.spread_max)
        cds_path = _simulate_ou(
            cfg.n_days, dt, kappa=6.0, theta=level,
            sigma=0.0015, x0=level, rng=rng,
        )
        cds_path = np.clip(cds_path, 5e-4, None)

        # Basis OU path (in decimal), per-name idiosyncratic
        basis_path = _simulate_ou(
            cfg.n_days, dt, cfg.ou_kappa, theta, sigma,
            x0=theta, rng=rng,
        )
        cash_spread_path = cds_path - basis_path

        # Convert cash spread to an approximate bond price via a flat-curve
        # par relationship: price ~ par - spread * duration * 100.
        duration = maturity * 0.9  # rough modified duration proxy
        bond_price = 100.0 - cash_spread_path * duration * 100.0
        # Add the coupon-vs-rate carry so prices are realistic (near par for IG)
        bond_price += (coupon - cfg.base_rate) * duration * 100.0

        for d, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "name": name,
                    "cds_spread": float(cds_path[d]),
                    "bond_price": float(bond_price[d]),
                    "coupon": coupon,
                    "maturity": maturity,
                    "recovery": recovery,
                }
            )

    df = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """Raise if a DataFrame does not conform to the panel schema."""
    missing = set(PANEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")
    if df["cds_spread"].lt(0).any():
        raise ValueError("cds_spread contains negative values")
    if df["bond_price"].le(0).any():
        raise ValueError("bond_price contains non-positive values")
