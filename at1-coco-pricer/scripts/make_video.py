"""Animated charts of the BNP AT1 case study, for LinkedIn (no voice, no text cards).

Four charts, each drawn progressively, with cross-fades in between:

1. CET1 fan chart     - simulated capital paths and percentile bands, with the
                        MDA threshold and the AT1 trigger
2. Spread breakdown   - what the yield-to-call spread pays for, bar by bar
3. Price vs CET1      - the capital cliff at the MDA threshold
4. Call probability   - P(call at first reset) vs the AT1 spread level

Reads ``results/spread_decomposition.csv``, ``price_vs_cet1.csv`` and
``call_probability.csv`` (written by ``spread_and_stress.py``) and simulates
the CET1 paths with the package, so the video always matches the latest run.

Output: ``results/at1_story.mp4`` (1080x1350, 4:5 portrait, 30 fps, ~40 s)

Run:  pip install imageio-ffmpeg        (once, provides ffmpeg)
      python scripts/make_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # find at1_coco without install

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from at1_coco import CET1Params  # noqa: E402
from at1_coco.processes import draw_randoms, simulate_cet1  # noqa: E402

try:  # ffmpeg bundled with imageio-ffmpeg, if no system ffmpeg
    import imageio_ffmpeg

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "at1_story.mp4"
FPS = 30
W, H = 10.8, 13.5                 # inches at dpi=100 -> 1080x1350
SHOW_SOURCE = True                # small source line under each chart

# ---- style: dark theme, reference palette (dark steps) ---------------------
BG, PANEL = "#141413", "#1a1a19"
INK, INK2, INK3, GRID = "#f5f4ef", "#c3c2b7", "#8a897f", "#2c2c29"
BLUE, ORANGE, AQUA, YELLOW, RED = "#3987e5", "#d95926", "#199e70", "#c98500", "#e66767"
ORDER = ["US05602XQR25", "US05602XQS08", "US05602XQQ42"]
SHORT = {"US05602XQR25": "6.875% NC33", "US05602XQS08": "7.20% NC36", "US05602XQQ42": "7.45% NC35"}
BOND_COLOR = {"US05602XQR25": BLUE, "US05602XQS08": ORANGE, "US05602XQQ42": AQUA}
COMP_COLOR = [BLUE, ORANGE, AQUA, YELLOW]
COMPS = ["PONV / tail / liquidity", "MDA coupon cuts", "conversion", "extension"]

MDA, TRIGGER, CET1_TODAY = 10.51, 5.125, 12.97

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK3, "xtick.color": INK3, "ytick.color": INK3,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "font.size": 17,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.labelsize": 16,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 8, "ytick.major.pad": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def ease(x: float) -> float:
    """Smooth 0 -> 1 (cubic ease in-out)."""
    x = min(max(x, 0.0), 1.0)
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def window(p: float, a: float, b: float) -> float:
    """Eased progress of a sub-animation that runs from a to b (in scene time 0..1)."""
    return ease((p - a) / (b - a))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    dec = pd.read_csv(RES / "spread_decomposition.csv")
    tab = dec[dec["component"].isin(COMPS)].pivot(index="component", columns="isin",
                                                  values="contribution_bp").loc[COMPS, ORDER]
    total = dec[dec["component"] == "market"].set_index("isin")["cum_spread_bp"][ORDER]
    pvc = pd.read_csv(RES / "price_vs_cet1.csv")
    calls = pd.read_csv(RES / "call_probability.csv")

    params = CET1Params(c0=CET1_TODAY, kappa=0.32, theta=13.0, sigma=0.41, jump_intensity=0.12,
                        jump_median=2.2, jump_vol=0.5, ponv_intensity=0.0)
    years, dt, n = 10, 1 / 12, 4000
    sim = simulate_cet1(params, TRIGGER, years, dt, draw_randoms(n, int(years / dt), seed=11))
    bands = np.percentile(sim.paths, [5, 25, 50, 75, 95], axis=0)
    return tab, total, pvc, calls, sim.paths, sim.times, bands


# ---------------------------------------------------------------------------
# chart frame: title, subtitle, axes, source
# ---------------------------------------------------------------------------
def frame(fig, title, subtitle, alpha, rect=(0.12, 0.16, 0.8, 0.66)):
    fig.text(0.075, 0.925, title, fontsize=34, fontweight="bold", color=INK, alpha=alpha)
    fig.text(0.075, 0.888, subtitle, fontsize=18, color=INK2, alpha=alpha)
    if SHOW_SOURCE:
        fig.text(0.075, 0.035, "BNP Paribas USD AT1s · Sep 2026 · Source: S&P Capital IQ, BNP Paribas, "
                 "EBA, FRED; author's model", fontsize=12, color=INK3, alpha=alpha)
    ax = fig.add_axes(list(rect))
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    return ax


# ---------------------------------------------------------------------------
# scenes: draw(fig, p, alpha) with p in [0, 1]
# ---------------------------------------------------------------------------
def scene_fan(paths, times, bands):
    sample = paths[:14]

    def draw(fig, p, a):
        ax = frame(fig, "Simulating BNP's capital",
                   "CET1 ratio, 4,000 paths · mean-reverting with stress jumps", a)
        g = window(p, 0.05, 0.75)
        k = max(2, int(g * len(times)))
        t = times[:k]
        ax.fill_between(t, bands[0][:k], bands[4][:k], color=BLUE, alpha=0.16 * a, lw=0)
        ax.fill_between(t, bands[1][:k], bands[3][:k], color=BLUE, alpha=0.30 * a, lw=0)
        for path in sample:
            ax.plot(t, path[:k], color=BLUE, lw=1.1, alpha=0.45 * a)
        ax.plot(t, bands[2][:k], color=INK, lw=2.4, alpha=a)
        ax.plot(t[-1], bands[2][k - 1], "o", color=INK, ms=8, alpha=a)

        la = window(p, 0.0, 0.2) * a
        ax.axhline(MDA, color=YELLOW, lw=2, ls=(0, (6, 4)), alpha=la)
        ax.axhline(TRIGGER, color=RED, lw=2, alpha=la)
        ax.text(9.95, MDA + 0.25, "MDA threshold", ha="right", color=YELLOW, fontsize=16, alpha=la)
        ax.text(9.95, TRIGGER + 0.25, "AT1 trigger", ha="right", color=RED, fontsize=16, alpha=la)
        la2 = window(p, 0.75, 0.9) * a
        ax.text(0.15, 16.2, "5–95% range", color=INK3, fontsize=14, alpha=la2)
        ax.set_xlim(0, 10)
        ax.set_ylim(4, 17)
        ax.set_xlabel("years ahead")
        ax.set_ylabel("CET1 ratio (%)")
    return draw


def scene_breakdown(tab, total):
    def draw(fig, p, a):
        ax = frame(fig, "What the spread pays for",
                   "Yield-to-call spread over Treasuries, bp", a, rect=(0.22, 0.22, 0.7, 0.6))
        y = np.arange(len(ORDER))[::-1] * 1.25
        left = np.zeros(len(ORDER))
        for j, comp in enumerate(COMPS):
            g = window(p, 0.05 + 0.16 * j, 0.25 + 0.16 * j)
            v = tab.loc[comp].values * g
            if g > 0:
                ax.barh(y, v, left=left, color=COMP_COLOR[j], height=0.72, edgecolor=BG, lw=2.5,
                        alpha=a, label=comp)
                for yi, (l, w) in enumerate(zip(left, v)):
                    if w > 35:
                        ax.text(l + w / 2, y[yi], f"{w:.0f}", ha="center", va="center",
                                color="white", fontsize=18, fontweight="bold", alpha=a)
            left += v
        ta = window(p, 0.72, 0.85) * a
        for yi, t in enumerate(total.values):
            ax.text(t + 5, y[yi], f"{t:.0f}", va="center", fontsize=20, fontweight="bold", alpha=ta)
        ax.set_yticks(y, [SHORT[i] for i in ORDER], fontsize=17, color=INK2)
        ax.set_xlim(0, total.max() * 1.15)
        ax.set_ylim(y.min() - 0.8, y.max() + 0.8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_xlabel("bp")
        if ax.get_legend_handles_labels()[1]:
            leg = ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.09), ncol=2, frameon=False,
                            fontsize=15, handlelength=1.2, columnspacing=2)
            for txt in leg.get_texts():
                txt.set_color(INK2)
                txt.set_alpha(a)
    return draw


def scene_cliff(pvc):
    def draw(fig, p, a):
        ax = frame(fig, "The capital cliff",
                   "Model price as BNP's CET1 ratio falls, all else equal", a)
        ra = window(p, 0.0, 0.2) * a
        ax.axvspan(pvc["cet1"].min(), MDA, color=YELLOW, alpha=0.07 * ra, lw=0)
        ax.axvline(MDA, color=YELLOW, lw=2, ls=(0, (6, 4)), alpha=ra)
        ax.axvline(CET1_TODAY, color=INK3, lw=1.5, alpha=ra)
        ymin, ymax = pvc["clean"].min() - 3, pvc["clean"].max() + 3
        ax.text(MDA - 0.15, ymax - 1.2, "below MDA:\ncoupons cut", ha="left", va="top",
                color=YELLOW, fontsize=15, alpha=ra)
        ax.text(CET1_TODAY + 0.15, ymax - 1.2, "today", ha="right", va="top", color=INK3, fontsize=15, alpha=ra)
        g = window(p, 0.1, 0.8)
        for isin in ORDER:
            d = pvc[pvc["isin"] == isin].sort_values("cet1", ascending=False)
            x, yv = d["cet1"].values, d["clean"].values
            xs = np.linspace(x[0], x[-1], 300)                     # smooth sweep from high to low CET1
            ys = np.interp(xs, x[::-1], yv[::-1])
            k = max(2, int(g * len(xs)))
            ax.plot(xs[:k], ys[:k], color=BOND_COLOR[isin], lw=3.2, alpha=a, label=SHORT[isin])
            ax.plot(xs[k - 1], ys[k - 1], "o", color=BOND_COLOR[isin], ms=9, alpha=a)
        ax.set_xlim(pvc["cet1"].max(), pvc["cet1"].min())          # capital falls to the right
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("CET1 ratio (%)")
        ax.set_ylabel("price")
        leg = ax.legend(loc="lower left", frameon=False, fontsize=15)
        for txt in leg.get_texts():
            txt.set_color(INK2)
    return draw


def scene_calls(calls):
    def draw(fig, p, a):
        ax = frame(fig, "Will BNP call?",
                   "Probability of a call at the first reset vs the AT1 spread level", a)
        g = window(p, 0.1, 0.8)
        for isin in ORDER:
            d = calls[calls["isin"] == isin].sort_values("new_issue_spread_bp")
            xs = np.linspace(d["new_issue_spread_bp"].min(), d["new_issue_spread_bp"].max(), 300)
            ys = 100 * np.interp(xs, d["new_issue_spread_bp"], d["p_call_first"])
            k = max(2, int(g * len(xs)))
            ax.plot(xs[:k], ys[:k], color=BOND_COLOR[isin], lw=3.2, alpha=a, label=SHORT[isin])
            ax.plot(xs[k - 1], ys[k - 1], "o", color=BOND_COLOR[isin], ms=9, alpha=a)
        ax.set_ylim(0, 100)
        ax.set_xlim(calls["new_issue_spread_bp"].min(), calls["new_issue_spread_bp"].max())
        ax.set_xlabel("AT1 spread level (bp)")
        ax.set_ylabel("P(called at first reset), %")
        leg = ax.legend(loc="upper right", frameon=False, fontsize=15)
        for txt in leg.get_texts():
            txt.set_color(INK2)
    return draw


# ---------------------------------------------------------------------------
def main():
    tab, total, pvc, calls, paths, times, bands = load()
    scenes = [
        (10, scene_fan(paths, times, bands)),
        (10, scene_breakdown(tab, total)),
        (10, scene_cliff(pvc)),
        (9, scene_calls(calls)),
    ]
    fade = 0.6                                          # seconds of fade in / out per scene
    fig = plt.figure(figsize=(W, H), dpi=100)
    writer = FFMpegWriter(fps=FPS, bitrate=8000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "18"])
    with writer.saving(fig, str(OUT), dpi=100):
        for dur, draw in scenes:
            n = int(dur * FPS)
            for i in range(n):
                t = i / FPS
                alpha = min(1.0, t / fade, (dur - t) / fade)
                fig.clf()
                draw(fig, i / (n - 1), max(alpha, 0.0))
                writer.grab_frame()
    print("saved", OUT)


if __name__ == "__main__":
    main()