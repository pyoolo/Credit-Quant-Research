"""End-to-end demo: prior from spread -> LR update -> the correlation trap ->
hierarchical Bayesian fit -> holdout calibration. Saves a calibration figure
used by the paper.

Run:  python scripts/run_demo.py
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bcu import priors, likelihood, updating, data, backtest, bayesian_model

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)
SIGNALS = ["downgrade", "earnings_miss", "spread_widening"]


def main():
    rng = np.random.default_rng(0)
    df = data.generate(n=6000, seed=0)
    train = df.sample(frac=0.6, random_state=0)
    test = df.drop(train.index).reset_index(drop=True)

    print("=" * 64)
    print("1) Prior from a spread (credit triangle)")
    s_bbb = float(priors.spread_from_pd(0.02))     # spread for a 2% prior
    print(f"   A 2.0% one-year PD <-> CDS spread ~ {s_bbb*1e4:.0f} bp")
    prior_pd = priors.pd_from_spread(s_bbb)
    print(f"   Recovered prior PD from that spread: {prior_pd:.3%}")

    print("\n2) Likelihood ratios estimated from history")
    marg = likelihood.all_marginal_lrs(train, SIGNALS)
    for k, v in marg.items():
        print(f"   LR[{k:16s}] = {v:5.2f}")
    joint = likelihood.joint_likelihood_ratio(train, SIGNALS)
    print(f"   joint LR (all three) = {joint:5.2f}")
    print(f"   product of marginals = {np.prod(list(marg.values())):5.2f}")

    print("\n3) The correlation trap: a name with all three signals")
    naive = updating.naive_update(prior_pd, list(marg.values()))
    aware = updating.joint_update(prior_pd, joint)
    print(f"   naive (independent) posterior PD = {naive:.1%}")
    print(f"   joint (correlation-aware)     PD = {aware:.1%}")
    print(f"   naive overstates by {(naive-aware)*100:.1f} pts")

    print("\n4) Hierarchical Bayesian fit (PyMC)")
    model, idata, ratings, sig = bayesian_model.fit(
        train, draws=400, tune=400, chains=2, seed=0)
    bs = bayesian_model.beta_summary(idata, sig)
    print(bs.to_string(index=False))
    print("   (each beta is a rating- and co-signal-adjusted log-odds effect;")
    print("    exp(beta) is the model's correlation-aware odds multiplier.)")

    print("\n5) Holdout calibration: prior vs naive vs Bayesian")
    prior_test = priors.pd_from_spread(test["cds_spread"].to_numpy())

    def naive_row(row):
        lrs = [marg[s] for s in SIGNALS if row[s] == 1]
        p0 = priors.pd_from_spread(row["cds_spread"])
        return updating.naive_update(p0, lrs) if lrs else p0
    naive_test = test.apply(naive_row, axis=1).to_numpy()
    bayes_test = bayesian_model.posterior_pd(idata, ratings, sig, test)

    y = test["default"].to_numpy()
    for name, pred in [("prior-only", prior_test),
                       ("naive-LR", naive_test),
                       ("bayesian", bayes_test)]:
        m = backtest.metrics(pred, y)
        print(f"   {name:11s}  AUC={m['AUC']:.3f}  "
              f"Brier={m['Brier']:.4f}  LogLoss={m['LogLoss']:.4f}")

    # calibration figure
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, pred, c in [("naive-LR", naive_test, "#C9402E"),
                          ("bayesian", bayes_test, "#1F7A53")]:
        ct = backtest.calibration_table(pred, y, bins=8)
        ax.plot(ct["predicted"], ct["realized"], "o-", color=c, label=name)
    ax.set_xlabel("predicted PD"); ax.set_ylabel("realized default rate")
    ax.set_title("Holdout calibration"); ax.legend(); ax.set_xlim(0, None)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "calibration.png")
    fig.savefig(out, dpi=140)
    print(f"\n   saved {out}")

    print("\n6) Where correlation bites: names with 2+ active signals")
    nsig = test[SIGNALS].sum(axis=1).to_numpy()
    sub = nsig >= 2
    if sub.sum() > 0:
        realized = y[sub].mean()
        print(f"   n = {int(sub.sum())} names; realized default rate = {realized:.1%}")
        print(f"   mean predicted PD  naive-LR = {naive_test[sub].mean():.1%}")
        print(f"   mean predicted PD  bayesian = {bayes_test[sub].mean():.1%}")
        print("   naive runs hotter than the correlation-aware model: treating")
        print("   redundant signals as independent confirmations inflates the PD.")


if __name__ == "__main__":
    main()
