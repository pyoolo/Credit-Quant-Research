"""Absorption term structure and price sensitivity to the starting CET1 ratio.

Run from the project root::

    python scripts/trigger_term_structure.py
"""

from __future__ import annotations

import pathlib
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from at1_coco import AT1Note, CET1Params, MonteCarloPricer
from at1_coco.risk import trigger_term_structure

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    note = AT1Note(coupon_rate=0.06, first_call_year=5.0, trigger_level=5.125,
                   mda_threshold=9.0, combined_buffer=3.5)
    base = CET1Params(c0=14.5, kappa=0.40, theta=14.5, sigma=1.3,
                      jump_intensity=0.12, jump_median=2.2, jump_vol=0.5, ponv_intensity=0.005)

    # ---- absorption term structure for three capitalisation regimes -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for label, c in [("strong (16%)", 16.0), ("base (14.5%)", 14.5), ("weak (11%)", 11.0)]:
        p = replace(base, c0=c, theta=c)
        pr = MonteCarloPricer(note, p, discount_rate=0.03, horizon=30.0, dt=1 / 12)
        tenors, probs = trigger_term_structure(pr, n_paths=40_000, seed=0)
        ax1.plot(tenors, probs * 100, marker="o", ms=3, label=label)
    ax1.set_xlabel("tenor (years)")
    ax1.set_ylabel("cumulative P(absorption) (%)")
    ax1.set_title("Loss-absorption term structure")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- price and coupon restriction vs starting capital ---------------
    c_grid = np.linspace(8.0, 18.0, 21)
    prices, restr, pabs = [], [], []
    for c in c_grid:
        p = replace(base, c0=c, theta=c)
        res = MonteCarloPricer(note, p, 0.03, 40.0, 1 / 12).price(n_paths=30_000, seed=0)
        prices.append(res.price)
        restr.append(res.expected_coupon_restriction * 100)
        pabs.append(res.prob_absorption * 100)

    ax2.plot(c_grid, prices, color="C0", marker="o", ms=3, label="price (LHS)")
    ax2.axhline(100, color="grey", ls=":", lw=1)
    ax2.axvline(note.mda_threshold + note.combined_buffer, color="C2", ls=":", lw=1.2)
    ax2.set_xlabel("starting CET1 ratio (%)")
    ax2.set_ylabel("price")
    ax2b = ax2.twinx()
    ax2b.plot(c_grid, restr, color="C3", ls="--", marker="s", ms=3, label="coupon restriction (RHS)")
    ax2b.set_ylabel("avg coupon restriction (%)")
    ax2.set_title("Price and coupon-cancellation vs capital")
    lines = ax2.get_lines()[:1] + ax2b.get_lines()[:1]
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(RESULTS / "absorption_and_price_curves.png", dpi=130)
    print(f"saved {RESULTS / 'absorption_and_price_curves.png'}")


if __name__ == "__main__":
    main()
