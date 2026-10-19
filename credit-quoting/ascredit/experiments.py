"""
Reproduce the paper's experiments:
  (1) Single-bond Monte Carlo: AS vs symmetric (P&L distribution, terminal q).
  (2) Quote/inventory illustration over one path.
  (3) Risk-aversion sweep: efficient frontier of mean P&L vs P&L std.
  (4) Small basket (independent bonds) aggregate behaviour.

Run:  python -m src.experiments
Outputs go to figures/ and results/.
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import ASParams, optimal_quotes, symmetric_quotes
from .simulator import SpreadModel, calibrate_k_from_halflife
from .engine import SimConfig, run_session, monte_carlo

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ---- Baseline credit calibration (illustrative IG corporate) -------------
# spread vol ~ 4 bp/sqrt(year) of *daily-ish* moves scaled into a session;
# k chosen so intensity halves every ~3 bp of quote distance; DV01 normalized.
PARAMS = ASParams(
    gamma=0.05,
    sigma=4.0,
    k=calibrate_k_from_halflife(3.0),
    A=140.0,
    T=1.0,
    dv01=1.0,
)
SPREAD = SpreadModel(s0=120.0, sigma=PARAMS.sigma, kappa=1.0, theta=120.0)
CFG = SimConfig(T=1.0, n_steps=1000, clip=1.0, inventory_cap=25.0)
N_PATHS = 4000


def exp_quote_illustration():
    rng = np.random.default_rng(7)
    cfg = SimConfig(**{**CFG.__dict__, "strategy": "as"})
    res = run_session(PARAMS, SPREAD, cfg, rng)

    t = np.linspace(0, cfg.T, cfg.n_steps + 1)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, res.s_path, color="black", lw=1.0, label="mid spread $s_t$")
    ax[0].set_ylabel("spread (bp)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Mid spread path")

    ax[1].plot(t, res.q_path, color="C3", lw=1.2)
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].set_ylabel("inventory $q$ (DV01)")
    ax[1].set_xlabel("time (session fraction)")
    ax[1].set_title("Inventory under AS optimal quoting")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "quote_illustration.png"), dpi=140)
    plt.close(fig)


def exp_quote_skew():
    """Show how quotes skew with inventory at fixed t."""
    s, t = 120.0, 0.5
    qs = np.linspace(-15, 15, 200)
    bids = [optimal_quotes(s, q, t, PARAMS)[0] for q in qs]
    asks = [optimal_quotes(s, q, t, PARAMS)[1] for q in qs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qs, bids, label="bid quote (client sells)")
    ax.plot(qs, asks, label="ask quote (client buys)")
    ax.axhline(s, color="grey", lw=0.7, ls="--", label="mid")
    ax.set_xlabel("inventory $q$ (DV01)")
    ax.set_ylabel("quoted spread (bp)")
    ax.set_title("Inventory-dependent quote skew (AS, credit)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "quote_skew.png"), dpi=140)
    plt.close(fig)


def exp_monte_carlo():
    as_cfg = SimConfig(**{**CFG.__dict__, "strategy": "as"})
    sym_cfg = SimConfig(**{**CFG.__dict__, "strategy": "symmetric"})
    as_res = monte_carlo(PARAMS, SPREAD, as_cfg, N_PATHS, seed=42)
    sym_res = monte_carlo(PARAMS, SPREAD, sym_cfg, N_PATHS, seed=42)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(sym_res["pnls"], bins=60, alpha=0.55, label="symmetric", color="C0")
    ax[0].hist(as_res["pnls"], bins=60, alpha=0.55, label="AS", color="C3")
    ax[0].set_title("Session P&L distribution")
    ax[0].set_xlabel("P&L"); ax[0].legend()

    ax[1].hist(sym_res["terminal_q"], bins=40, alpha=0.55, label="symmetric", color="C0")
    ax[1].hist(as_res["terminal_q"], bins=40, alpha=0.55, label="AS", color="C3")
    ax[1].set_title("Terminal inventory $q_T$")
    ax[1].set_xlabel("$q_T$ (DV01)"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "montecarlo.png"), dpi=140)
    plt.close(fig)

    summary = {
        "as": {k: v for k, v in as_res.items() if k not in ("pnls", "terminal_q")},
        "symmetric": {k: v for k, v in sym_res.items() if k not in ("pnls", "terminal_q")},
    }
    with open(os.path.join(RES, "montecarlo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def exp_risk_aversion_frontier():
    gammas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    means, stds = [], []
    for g in gammas:
        p = ASParams(gamma=g, sigma=PARAMS.sigma, k=PARAMS.k, A=PARAMS.A,
                     T=PARAMS.T, dv01=PARAMS.dv01)
        cfg = SimConfig(**{**CFG.__dict__, "strategy": "as"})
        r = monte_carlo(p, SPREAD, cfg, 2000, seed=11)
        means.append(r["mean_pnl"]); stds.append(r["std_pnl"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(stds, means, "-o")
    for g, x, y in zip(gammas, stds, means):
        ax.annotate(f"$\\gamma$={g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("P&L std (risk)")
    ax.set_ylabel("mean P&L (reward)")
    ax.set_title("Risk-aversion frontier (AS, credit)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "frontier.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(RES, "frontier.json"), "w") as f:
        json.dump({"gamma": gammas, "mean_pnl": means, "std_pnl": stds}, f, indent=2)


def exp_basket():
    """Small basket of independent bonds; aggregate inventory & P&L."""
    rng = np.random.default_rng(99)
    bonds = [
        SpreadModel(s0=90.0, sigma=3.0, kappa=1.2, theta=90.0),
        SpreadModel(s0=120.0, sigma=4.0, kappa=1.0, theta=120.0),
        SpreadModel(s0=180.0, sigma=6.0, kappa=0.8, theta=180.0),
    ]
    cfg = SimConfig(**{**CFG.__dict__, "strategy": "as"})
    agg = np.zeros(cfg.n_steps + 1)
    for b in bonds:
        p = ASParams(gamma=PARAMS.gamma, sigma=b.sigma, k=PARAMS.k, A=PARAMS.A,
                     T=PARAMS.T, dv01=1.0)
        res = run_session(p, b, cfg, rng)
        agg += res.pnl_path
    t = np.linspace(0, cfg.T, cfg.n_steps + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, agg, color="C2")
    ax.set_title("Aggregate basket P&L (3 independent bonds, AS)")
    ax.set_xlabel("time (session fraction)"); ax.set_ylabel("cumulative P&L")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "basket.png"), dpi=140)
    plt.close(fig)


def main():
    exp_quote_illustration()
    exp_quote_skew()
    summary = exp_monte_carlo()
    exp_risk_aversion_frontier()
    exp_basket()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
