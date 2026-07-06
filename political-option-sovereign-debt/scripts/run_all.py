"""
Reproduce all paper exhibits end-to-end on the synthetic panel.

    python scripts/run_all.py [--outdir results]

Real data: swap `simulate_default_panel()` for `load_price_panel("data/venz.csv")`
plus your own macro CSV; everything downstream is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from political_option import (
    calibrate_benchmarks,
    curve_summary,
    event_study,
    hazard_term_structure,
    implied_probability_panel,
    level_factor,
    macro_falsification,
    post_default_interactions,
    simulate_default_panel,
    violation_series,
)
from political_option.plotting import (
    plot_car,
    plot_curve_snapshots,
    plot_implied_probabilities,
    plot_level_dispersion,
    plot_price_dynamics,
    plot_violation_mass,
)

R_LOW, R_HIGH = 5.0, 50.0  # paper baseline (Sec. 9.1)


def main(outdir: str = "results") -> None:
    out = Path(outdir)
    out.mkdir(exist_ok=True)

    panel = simulate_default_panel()
    px, macro = panel.prices, panel.macro
    post = px.loc[panel.default_date :]

    # -- Figs. 1-2: price dynamics ---------------------------------------- #
    plot_price_dynamics(px, "Sovereign & PDVSA bonds — full sample").savefig(out / "fig1_prices_full.png")
    plot_price_dynamics(px.loc[px.index[-1] - pd.Timedelta(days=180):], "Last 6 months").savefig(out / "fig2_prices_6m.png")

    # -- Fig. 3: curve snapshots ------------------------------------------ #
    snaps = [px.index[200], panel.default_date + pd.Timedelta(days=45), px.index[-1]]
    plot_curve_snapshots(
        px, panel.maturities, snaps, labels=["pre-default", "default window", "latest"]
    ).savefig(out / "fig3_curve_snapshots.png")

    # -- Fig. 4: level & dispersion ---------------------------------------- #
    summ = curve_summary(post, panel.maturities)
    plot_level_dispersion(summ.loc[summ.index[-1] - pd.Timedelta(days=180):]).savefig(out / "fig4_level_dispersion.png")

    # -- Figs. 5-6: event study around the salient political shock --------- #
    salient = panel.event_dates[-1]
    lvl_post = level_factor(post)
    es = event_study(lvl_post, salient)
    plot_car(es.car, f"Level factor CAR around political event ({salient.date()})").savefig(
        out / "fig5_event_car.png"
    )
    print(f"[event study] impact AR = {es.jump_on_impact:+.3f}  "
          f"pre-drift = {es.pre_event_drift:+.4f}  t = {es.t_stat_impact:.1f}")

    # -- Fig. 7: implied normalization probabilities ----------------------- #
    probs = implied_probability_panel(post, R_LOW, R_HIGH)
    plot_implied_probabilities(probs, panel.default_date, panel.latent_prob).savefig(
        out / "fig7_implied_probs.png"
    )

    # -- Fig. 8: monotonicity violation mass ------------------------------- #
    v = violation_series(probs, panel.maturities, smoothing_window=60)
    plot_violation_mass(v, panel.default_date).savefig(out / "fig8_violation_mass.png")
    print(f"[monotonicity] mean V(t) = {v['V'].mean():.4f}")

    # -- Extensions beyond the paper --------------------------------------- #
    cal = calibrate_benchmarks(post, panel.maturities)
    print(f"[calibration] data-driven benchmarks: R_low = {cal.r_low:.1f}, "
          f"R_high = {cal.r_high:.1f} (paper baseline: {R_LOW:.0f}/{R_HIGH:.0f})")

    last = probs.iloc[-1]
    horizons = panel.maturities - (post.index[-1].year + post.index[-1].dayofyear / 365.25)
    ts = hazard_term_structure(last, horizons)
    ts.to_csv(out / "hazard_term_structure.csv")
    print("[hazard] forward hazard term structure:\n", ts.round(4).to_string())

    # -- App. B: macro falsification regressions --------------------------- #
    daily = macro_falsification(lvl_post, macro.loc[panel.default_date :])
    weekly = macro_falsification(lvl_post, macro.loc[panel.default_date :], freq="W-FRI")
    inter = post_default_interactions(level_factor(px), macro, panel.default_date)
    daily.table().to_csv(out / "tabB1_daily.csv")
    weekly.table().to_csv(out / "tabB2_weekly.csv")
    inter.table().to_csv(out / "tabB3_interactions.csv")
    print(f"[App. B] daily R2 = {daily.r_squared:.3f} (n={daily.nobs}); "
          f"weekly R2 = {weekly.r_squared:.3f}; "
          f"post-default dummy p = {inter.pvalues['post_default_jump']:.2e}")

    print(f"\nAll exhibits written to ./{outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    main(**vars(ap.parse_args()))
