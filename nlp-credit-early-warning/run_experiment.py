#!/usr/bin/env python3
"""Run the three-world experiment and write outputs/results.md.

Usage:
    python run_experiment.py                     # full run, 3 seeds
    python run_experiment.py --fast              # small, for CI / smoke
    python run_experiment.py --model hgb         # gradient-boosting check
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from synthcredit.world import WorldConfig, simulate_world, WORLDS
from synthcredit.documents import render_all
from synthcredit.features import build_filing_features, build_model_matrix
from synthcredit import model as M

HORIZON = 4
K = 15


def run_one(world: str, seed: int, fast: bool, mdl: str) -> dict:
    cfg = WorldConfig(world=world, seed=seed)
    if fast:
        cfg.n_issuers, cfg.n_quarters = 120, 56
    w = simulate_world(cfg)
    ff = build_filing_features(render_all(w["filings"]))
    mm = build_model_matrix(w["panel"], ff, horizon=HORIZON)

    first_train = 20 if fast else 28
    preds = {fs: M.walk_forward(mm, fs, horizon=HORIZON,
                                first_train=first_train, model=mdl)
             for fs in ("market", "market+fund", "full", "kitchen_sink",
                        "text_only")}

    boot = M.bootstrap_delta_auc(preds["full"], preds["market+fund"],
                                 n_boot=200 if fast else 400, seed=seed)
    boot_k = M.bootstrap_delta_auc(preds["kitchen_sink"],
                                   preds["market+fund"],
                                   n_boot=200 if fast else 400, seed=seed)
    res = {
        "world": world, "seed": seed,
        "events": int(w["panel"].event.sum()),
        "auc_market": M.pooled_auc(preds["market"]),
        "auc_mkt_fund": M.pooled_auc(preds["market+fund"]),
        "auc_full": M.pooled_auc(preds["full"]),
        "auc_text_only": M.pooled_auc(preds["text_only"]),
        "d_auc_text": boot["delta"], "d_lo": boot["lo"], "d_hi": boot["hi"],
        "d_auc_kitchen": boot_k["delta"],
        "p_at_k_base": M.precision_at_k(preds["market+fund"], K),
        "p_at_k_full": M.precision_at_k(preds["full"], K),
    }

    hold_full, ep = M.cycle_holdout(mm, w["cycle"], "full", HORIZON, mdl)
    hold_base, _ = M.cycle_holdout(mm, w["cycle"], "market+fund", HORIZON, mdl)
    if hold_full is not None and hold_base is not None:
        res["holdout_episode"] = f"q{ep[0]}-q{ep[1]}"
        res["holdout_auc_full"] = M.pooled_auc(hold_full)
        res["holdout_d_auc"] = (M.pooled_auc(hold_full)
                                - M.pooled_auc(hold_base))
    return res


def summarise(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.groupby("world").agg(
        seeds=("seed", "count"),
        auc_text_only=("auc_text_only", "mean"),
        auc_mkt_fund=("auc_mkt_fund", "mean"),
        auc_full=("auc_full", "mean"),
        d_auc_text=("d_auc_text", "mean"),
        d_lo=("d_lo", "mean"), d_hi=("d_hi", "mean"),
        d_auc_kitchen=("d_auc_kitchen", "mean"),
        p_at_k_base=("p_at_k_base", "mean"),
        p_at_k_full=("p_at_k_full", "mean"),
    ).reindex(list(WORLDS))
    return agg


def to_markdown(agg: pd.DataFrame, rows: list[dict], mdl: str) -> str:
    lines = [
        "# Three-world experiment — results",
        "", f"Model: `{mdl}` | horizon: {HORIZON}q | watchlist k={K} | "
        f"seeds per world: {int(agg.seeds.iloc[0])}", "",
        "| world | AUC text-only | AUC mkt+fund | AUC full | ΔAUC text "
        "(5–95% CI) | ΔAUC kitchen-sink | P@15 base → full |",
        "|---|---|---|---|---|---|---|",
    ]
    for wname, r in agg.iterrows():
        lines.append(
            f"| {wname} | {r.auc_text_only:.3f} | {r.auc_mkt_fund:.3f} | "
            f"{r.auc_full:.3f} | {r.d_auc_text:+.3f} "
            f"({r.d_lo:+.3f}, {r.d_hi:+.3f}) | {r.d_auc_kitchen:+.3f} | "
            f"{r.p_at_k_base:.3f} → {r.p_at_k_full:.3f} |")
    hold = [x for x in rows if "holdout_d_auc" in x]
    if hold:
        lines += ["", "## Out-of-cycle holdout (largest stress episode)", "",
                  "| world | seed | episode | AUC full | ΔAUC text |",
                  "|---|---|---|---|---|"]
        for x in hold:
            lines.append(f"| {x['world']} | {x['seed']} | "
                         f"{x['holdout_episode']} | "
                         f"{x['holdout_auc_full']:.3f} | "
                         f"{x['holdout_d_auc']:+.3f} |")
    return "\n".join(lines) + "\n"


def make_figure(outdir: Path, seed: int = 7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = simulate_world(WorldConfig(world="signal", seed=seed))
    cyc, panel = w["cycle"], w["panel"]
    er = panel.groupby("quarter").event.mean()
    sp = panel.groupby("quarter").spread.median()
    fig, ax1 = plt.subplots(figsize=(9, 3.4))
    for s, e in M.find_stress_episodes(cyc):
        ax1.axvspan(s, e, color="0.85", zorder=0)
    ax1.plot(er.index, 100 * er.values, lw=1.4, label="event rate (%, q)")
    ax1.set_ylabel("quarterly event rate, %")
    ax1.set_xlabel("quarter")
    ax2 = ax1.twinx()
    ax2.plot(sp.index, sp.values, lw=1.2, ls="--", color="tab:red",
             label="median spread (bp)")
    ax2.set_ylabel("median spread, bp")
    ax1.set_title("Synthetic HY world: events cluster in stress episodes "
                  "(shaded)")
    fig.tight_layout()
    fig.savefig(outdir / "world_overview.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS),
                    choices=list(WORLDS))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--model", choices=["logit", "hgb"], default="logit")
    ap.add_argument("--figure", action="store_true")
    args = ap.parse_args()

    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)
    rows = []
    partial = out / "results_raw.csv"
    if partial.exists() and set(args.worlds) != set(WORLDS):
        rows = pd.read_csv(partial).to_dict("records")
        rows = [r for r in rows if r["world"] not in args.worlds]
    for world in args.worlds:
        for s in range(args.seeds):
            t0 = time.time()
            r = run_one(world, seed=7 + s, fast=args.fast, mdl=args.model)
            rows.append(r)
            print(f"{world:<11} seed {7+s}  AUC full {r['auc_full']:.3f}  "
                  f"ΔAUC text {r['d_auc_text']:+.3f}  "
                  f"({time.time()-t0:.0f}s)")
    agg = summarise(rows)
    md = to_markdown(agg, rows, args.model)
    (out / "results.md").write_text(md, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "results_raw.csv", index=False)
    print("\n" + md)
    if args.figure:
        make_figure(Path(__file__).parent / "docs")
        print("figure written to docs/world_overview.png")


if __name__ == "__main__":
    main()
