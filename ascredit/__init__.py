"""
ascredit — Avellaneda-Stoikov optimal market-making, adapted to credit (spread space).

A study of optimal dealer quoting under inventory risk, on synthetic data.
See paper/ for the write-up and figures/ for generated charts.
"""

from .model import (
    ASParams,
    reservation_spread,
    optimal_half_spread,
    optimal_quotes,
    symmetric_quotes,
)
from .simulator import SpreadModel, fill_probability, calibrate_k_from_halflife
from .engine import SimConfig, SimResult, run_session, monte_carlo

__version__ = "0.1.0"

__all__ = [
    "ASParams",
    "reservation_spread",
    "optimal_half_spread",
    "optimal_quotes",
    "symmetric_quotes",
    "SpreadModel",
    "fill_probability",
    "calibrate_k_from_halflife",
    "SimConfig",
    "SimResult",
    "run_session",
    "monte_carlo",
    "__version__",
]
