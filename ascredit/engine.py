"""
Simulation engine: runs a market-making session over a synthetic spread path
and synthetic Poisson order flow, for either the inventory-aware (Avellaneda-
Stoikov) strategy or the inventory-agnostic symmetric benchmark.

P&L accounting (credit / spread space)
--------------------------------------
The dealer's mark-to-market on inventory q at mid spread s is approximated by a
first-order (DV01) expansion around the entry spread:

    dPnL_inventory ~= - q * dv01 * ds

i.e. a long-credit dealer (q>0) loses dv01 per bp of spread *widening*. On each
fill the dealer captures the half-spread edge |quote - s| * dv01 (the distance
of the executed quote from the contemporaneous mid, converted to price via
DV01), and inventory updates by +/- one clip (in DV01 units).

We track:
  * realized spread capture (edge earned on fills)
  * inventory MtM (running)
  * total P&L = capture + MtM
  * inventory path, number of fills, terminal inventory
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .model import ASParams, optimal_quotes, symmetric_quotes
from .simulator import SpreadModel, fill_probability


@dataclass
class SimConfig:
    T: float = 1.0
    n_steps: int = 1000
    clip: float = 1.0           # trade size in DV01 units per fill
    inventory_cap: float = 25.0 # hard risk limit on |q|
    strategy: str = "as"        # "as" or "symmetric"


@dataclass
class SimResult:
    pnl: float
    spread_capture: float
    inventory_mtm: float
    terminal_q: float
    n_fills: int
    max_abs_q: float
    pnl_path: np.ndarray = field(repr=False)
    q_path: np.ndarray = field(repr=False)
    s_path: np.ndarray = field(repr=False)


def run_session(p: ASParams, spread_model: SpreadModel, cfg: SimConfig,
                rng: np.random.Generator) -> SimResult:
    dt = cfg.T / cfg.n_steps
    s = spread_model.simulate(cfg.T, cfg.n_steps, rng)

    q = 0.0
    capture = 0.0
    mtm = 0.0
    n_fills = 0
    max_abs_q = 0.0

    pnl_path = np.zeros(cfg.n_steps + 1)
    q_path = np.zeros(cfg.n_steps + 1)

    for i in range(cfg.n_steps):
        t = i * dt
        mid = s[i]

        if cfg.strategy == "as":
            bid_q, ask_q = optimal_quotes(mid, q, t, p)
        else:
            bid_q, ask_q = symmetric_quotes(mid, t, p)

        # distances from mid (bp). bid_q > mid (wider), ask_q < mid (tighter).
        bid_delta = bid_q - mid     # >=0 expected
        ask_delta = mid - ask_q     # >=0 expected

        # client SELLS bond to dealer at the bid -> dealer inventory +clip
        if q + cfg.clip <= cfg.inventory_cap:
            if rng.random() < fill_probability(bid_delta, p.A, p.k, dt):
                edge = abs(bid_delta) * p.dv01 * cfg.clip
                capture += edge
                q += cfg.clip
                n_fills += 1

        # client BUYS bond from dealer at the ask -> dealer inventory -clip
        if q - cfg.clip >= -cfg.inventory_cap:
            if rng.random() < fill_probability(ask_delta, p.A, p.k, dt):
                edge = abs(ask_delta) * p.dv01 * cfg.clip
                capture += edge
                q -= cfg.clip
                n_fills += 1

        # inventory mark-to-market over the step: long credit loses on widening
        ds = s[i + 1] - s[i]
        mtm += -q * p.dv01 * ds

        max_abs_q = max(max_abs_q, abs(q))
        pnl_path[i + 1] = capture + mtm
        q_path[i + 1] = q

    return SimResult(
        pnl=capture + mtm,
        spread_capture=capture,
        inventory_mtm=mtm,
        terminal_q=q,
        n_fills=n_fills,
        max_abs_q=max_abs_q,
        pnl_path=pnl_path,
        q_path=q_path,
        s_path=s,
    )


def monte_carlo(p: ASParams, spread_model: SpreadModel, cfg: SimConfig,
                n_paths: int, seed: int = 0) -> dict:
    """Run many sessions; return summary stats for the chosen strategy."""
    rng = np.random.default_rng(seed)
    pnls = np.empty(n_paths)
    term_q = np.empty(n_paths)
    fills = np.empty(n_paths)
    caps = np.empty(n_paths)
    for j in range(n_paths):
        res = run_session(p, spread_model, cfg, rng)
        pnls[j] = res.pnl
        term_q[j] = res.terminal_q
        fills[j] = res.n_fills
        caps[j] = res.spread_capture
    return {
        "strategy": cfg.strategy,
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "sharpe_like": float(np.mean(pnls) / (np.std(pnls) + 1e-12)),
        "mean_terminal_q": float(np.mean(term_q)),
        "std_terminal_q": float(np.std(term_q)),
        "mean_fills": float(np.mean(fills)),
        "mean_capture": float(np.mean(caps)),
        "pnls": pnls,
        "terminal_q": term_q,
    }
