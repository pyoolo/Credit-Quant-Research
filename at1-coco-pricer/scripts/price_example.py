"""Price a representative AT1 note and report valuation, risk and sensitivities.

Run from the project root::

    python scripts/price_example.py
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from at1_coco import AT1Note, CET1Params, MonteCarloPricer, draw_randoms
from at1_coco.risk import greeks, loss_distribution

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def main() -> None:
    # ---- a plausible, healthy European bank AT1 -------------------------
    note = AT1Note(
        notional=100.0,
        coupon_rate=0.06,
        coupon_freq=2,
        first_call_year=5.0,
        reference_rate=0.03,
        reset_spread=0.045,
        base_refi_spread=0.040,
        trigger_level=5.125,
        trigger_recovery=0.0,
        mda_threshold=9.0,
        combined_buffer=3.5,
    )
    params = CET1Params(
        c0=14.5, kappa=0.40, theta=14.5, sigma=1.3,
        jump_intensity=0.12, jump_median=2.2, jump_vol=0.5, ponv_intensity=0.005,
    )
    pricer = MonteCarloPricer(note, params, discount_rate=0.03, horizon=40.0, dt=1 / 12)

    randoms = draw_randoms(80_000, int(round(pricer.horizon / pricer.dt)), seed=0)
    sim = pricer.simulate(randoms)
    res = pricer.price_on_paths(sim)

    print("=" * 60)
    print("AT1 valuation — representative healthy issuer")
    print("=" * 60)
    print(res.summary())

    print("\nSensitivities (bump-and-reprice, common random numbers)")
    for k, v in greeks(pricer, randoms=randoms).items():
        print(f"  {k:22s} {v:+.4f}")

    print("\nDiscounted-payoff risk")
    for k, v in loss_distribution(res).items():
        print(f"  {k:22s} {v:8.3f}")

    # ---- CET1 fan chart -------------------------------------------------
    t = sim.times
    qs = np.percentile(sim.paths, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(t, qs[0], qs[4], alpha=0.15, color="C0", label="5–95%")
    ax.fill_between(t, qs[1], qs[3], alpha=0.30, color="C0", label="25–75%")
    ax.plot(t, qs[2], color="C0", lw=1.6, label="median")
    ax.axhline(note.trigger_level, color="C3", ls="--", lw=1.2, label="mechanical trigger")
    ax.axhline(note.mda_threshold + note.combined_buffer, color="C2", ls=":", lw=1.2,
               label="MDA buffer top")
    ax.set_xlim(0, 15)
    ax.set_xlabel("years")
    ax.set_ylabel("CET1 ratio (%)")
    ax.set_title("Simulated CET1 distribution")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "cet1_fan_chart.png", dpi=130)
    print(f"\nsaved {RESULTS / 'cet1_fan_chart.png'}")


if __name__ == "__main__":
    main()
