"""Descriptive analysis of the BNP Paribas USD AT1 dataset.

Every number drawn or printed is computed from the data files, so the figures
update by themselves when the data are refreshed.

Figures in ``results/``:

1. ``bnp_cet1_history.png``    CET1 since Basel III, the CET1 requirement (MDA
                                threshold), the AT1 trigger, and the base-case
                                simulated outlook (percentile fan)
2. ``bnp_at1_prices.png``      evaluated mid prices of the three AT1s, with the
                                10y Treasury (FRED) on a second panel
3. ``bnp_at1_yield_split.png`` yield change split into Treasury (FRED 10y) and
                                spread: year to date, and the last 4 weeks

Run after ``scripts/build_dataset.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from at1_coco import CET1Params  # noqa: E402
from at1_coco.processes import draw_randoms, simulate_cet1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
ORDER = ["US05602XQR25", "US05602XQS08", "US05602XQQ42"]

# reference palette (light mode), categorical slots in fixed order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, RED, YELLOW = "#2a78d6", "#e34948", "#b07a00"
SERIES = {"US05602XQR25": "#2a78d6", "US05602XQS08": "#eb6834", "US05602XQQ42": "#1baf7a"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)

# dated capital events worth labelling (drawn only if the quarter is in the data)
EVENTS = [("2014-06-30", "US sanctions\nsettlement"), ("2023-03-31", "Bank of the West sale")]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load():
    prices = pd.read_csv(DATA / "bond_prices.csv", parse_dates=["date"])
    q = pd.read_csv(DATA / "bnp_cet1_quarterly.csv", parse_dates=["date"], index_col="date")
    a = pd.read_csv(DATA / "bnp_capital_annual.csv", parse_dates=["date"], index_col="date")
    share = pd.read_csv(DATA / "bnp_share.csv", parse_dates=["date"], index_col="date")
    static = pd.read_csv(DATA / "bonds_static.csv", index_col="isin_144a")
    def pct(x):
        t = f"{100 * x:.3f}".rstrip("0")
        return t + "0" if t.split(".")[1] == "" or len(t.split(".")[1]) < 2 else t

    for isin, row in static.iterrows():
        prices.loc[prices["isin"] == isin, "label"] = (
            f"{pct(row['coupon_initial'])}% NC{str(row['first_call_date'])[:4]} "
            f"(reset +{1e4*row['reset_margin']:.0f}bp)")
    return prices, q, a, share, static


def load_ust10(prices: pd.DataFrame) -> pd.Series:
    """10y Treasury (%): FRED DGS10 if available, else the CIQ benchmark of the 7.45%."""
    f = DATA / "fred_treasury.csv"
    if f.exists():
        s = pd.read_csv(f, parse_dates=["observation_date"], index_col="observation_date")["DGS10"]
        return pd.to_numeric(s, errors="coerce").dropna()
    return prices[prices["isin"] == "US05602XQQ42"].set_index("date")["benchmark_yield"]


def cet1_history(q: pd.DataFrame, a: pd.DataFrame) -> pd.Series:
    """Quarterly CET1 since Basel III (2014Q1); annual points fill any earlier gap.

    Before 2014 the reported ratio follows Basel II/2.5 definitions and is not
    comparable, so it is left out."""
    qq = q["cet1_ratio"].dropna()
    ann = a["cet1_ratio"][a.index < qq.index.min()]
    c = pd.concat([ann, qq]).sort_index()
    return c[c.index >= "2014-01-01"]


def requirement_series(a: pd.DataFrame) -> pd.Series:
    """CET1 requirement: annual CIQ stack plus the latest quarterly figures BNP publishes."""
    req = a["cet1_requirement"].dropna()
    f = DATA / "bnp_srep_requirement.csv"
    if f.exists():
        latest = pd.read_csv(f, parse_dates=["date"], index_col="date")["cet1_requirement"]
        req = pd.concat([req[req.index < latest.index.min()], latest])
    return req.sort_index()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_cet1(q, a, static, years_ahead=5):
    from calibrate_bnp import base_cet1_dynamics   # same base case as the analysis

    c = cet1_history(q, a)
    req = requirement_series(a)
    trigger = float(static["trigger_level_pct"].iloc[0])
    last_d, last_c, last_req = c.index[-1], float(c.iloc[-1]), float(req.iloc[-1])
    rwa_bn = float(q["rwa_eur_k"].dropna().iloc[-1]) / 1e6

    # base-case outlook from the latest CET1 print
    kappa, sigma = base_cet1_dynamics()
    params = CET1Params(c0=last_c, kappa=kappa, theta=13.0, sigma=sigma, jump_intensity=0.12,
                        jump_median=2.2, jump_vol=0.5, ponv_intensity=0.0)
    dt = 1 / 12
    sim = simulate_cet1(params, trigger, years_ahead, dt, draw_randoms(5000, int(years_ahead / dt), seed=5))
    fut = last_d + pd.to_timedelta(sim.times * 365.25, unit="D")
    bands = np.percentile(sim.paths, [5, 25, 50, 75, 95], axis=0)
    p_below_mda = float(np.mean((sim.paths < last_req).any(axis=1)))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(fut, bands[0], bands[4], color=BLUE, alpha=0.12, lw=0, label="outlook 5–95%")
    ax.fill_between(fut, bands[1], bands[3], color=BLUE, alpha=0.25, lw=0, label="outlook 25–75%")
    ax.plot(fut, bands[2], color=BLUE, lw=1.5, ls=":")
    ax.plot(c.index, c.values, color=BLUE, lw=2, marker="o", ms=3.5, label="CET1 ratio")
    req_x = list(req.index) + [fut[-1]]
    req_y = list(req.values) + [req.values[-1]]
    ax.step(req_x, req_y, where="post", color=INK2, lw=1.5, ls="--", label="CET1 requirement (MDA)")
    ax.axhline(trigger, color=RED, lw=1.5, label=f"AT1 trigger {trigger:g}%")
    ax.axvline(last_d, color=GRID, lw=1)

    ax.annotate(f"{last_c:.2f}%", (last_d, last_c), xytext=(-8, 8), textcoords="offset points",
                ha="right", color=INK, fontweight="bold")
    ax.annotate(f"cushion to MDA {last_c - last_req:.1f}pp (≈€{(last_c - last_req) * rwa_bn / 100:.0f}bn)",
                (last_d, (last_c + last_req) / 2), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", color=INK2, fontsize=9)
    ax.annotate(f"distance to trigger {last_c - trigger:.1f}pp (≈€{(last_c - trigger) * rwa_bn / 100:.0f}bn)",
                (last_d, (last_req + trigger) / 2), xytext=(-10, 0), textcoords="offset points",
                ha="right", va="center", color=INK2, fontsize=9)
    ax.text(fut[len(fut) // 2], bands[0].min() - 0.35,
            f"P(CET1 below MDA within {years_ahead}y) ≈ {p_below_mda:.0%}",
            ha="center", va="top", color=BLUE, fontsize=9)
    for d, txt in EVENTS:
        d = pd.Timestamp(d)
        if d in c.index:
            above = c[d] > c.shift(1).get(d, c[d])
            ax.annotate(txt, (d, c[d]), xytext=(0, 18 if above else -26), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_ylim(trigger - 2.6, max(c.max(), bands[4].max()) + 1)
    ax.set_ylabel("% of RWA")
    ax.set_title(f"BNP Paribas CET1: history and base-case outlook (κ={kappa:.2f}, σ={sigma:.2f})")
    ax.legend(loc="lower left", ncol=3, fontsize=8, bbox_to_anchor=(0, 0.0))
    fig.tight_layout()
    fig.savefig(RES / "bnp_cet1_history.png", dpi=150)
    plt.close(fig)
    return dict(cet1=last_c, requirement=last_req, cushion_pp=last_c - last_req,
                cushion_bn=(last_c - last_req) * rwa_bn / 100, trigger_bn=(last_c - trigger) * rwa_bn / 100,
                p_below_mda_5y=p_below_mda)


def fig_prices(prices, ust10):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, height_ratios=[3, 1.3])
    for isin in ORDER:
        g = prices[prices["isin"] == isin]
        ax.plot(g["date"], g["mid_price"], color=SERIES[isin], lw=2, label=g["label"].iloc[0])
        ax.annotate(f"{g['mid_price'].iloc[-1]:.2f}", (g["date"].iloc[-1], g["mid_price"].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=SERIES[isin])
    ax.axhline(100, color=INK2, lw=0.8)
    ax.set_ylabel("evaluated mid price (clean)")
    ax.set_title(f"BNP Paribas USD AT1s, to {prices['date'].max():%d %b %Y}")
    ax.legend(loc="lower left", fontsize=8)
    u = ust10[(ust10.index >= prices["date"].min()) & (ust10.index <= prices["date"].max())]
    ax2.plot(u.index, u.values, color=INK2, lw=1.5)
    ax2.set_ylabel("10y UST (%)")
    fig.tight_layout()
    fig.savefig(RES / "bnp_at1_prices.png", dpi=150)
    plt.close(fig)


def yield_split(prices, ust10, start, end):
    rows = []
    for isin in ORDER:
        g = prices[prices["isin"] == isin].set_index("date").sort_index()
        s = g.loc[:start].iloc[-1] if g.index.min() <= pd.Timestamp(start) else g.iloc[0]
        e = g.loc[:end].iloc[-1]
        u_s = float(ust10[ust10.index <= s.name].iloc[-1])
        u_e = float(ust10[ust10.index <= e.name].iloc[-1])
        rows.append({
            "isin": isin, "from": s.name.date(), "to": e.name.date(),
            "price_chg": e["mid_price"] - s["mid_price"],
            "yield_chg_bp": 100 * (e["mid_yield"] - s["mid_yield"]),
            "treasury_chg_bp": 100 * (u_e - u_s),
            "spread_chg_bp": 100 * ((e["mid_yield"] - u_e) - (s["mid_yield"] - u_s)),
        })
    return pd.DataFrame(rows).set_index("isin")


def fig_split(panels, labels):
    fig, axes = plt.subplots(1, len(panels), figsize=(9.5, 3.9), sharey=True)
    short = {i: labels[i].split(" (")[0] for i in ORDER}
    for ax, (title, df) in zip(axes, panels):
        df = df.loc[ORDER]
        x = np.arange(len(ORDER))
        ax.bar(x - 0.18, df["treasury_chg_bp"], 0.34, color=INK2, label="10y Treasury (FRED)")
        ax.bar(x + 0.18, df["spread_chg_bp"], 0.34, color=BLUE, label="spread over 10y")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [f"{short[i]}\nprice {p:+.1f}" for i, p in zip(ORDER, df["price_chg"])], fontsize=9)
        ax.set_title(title, fontsize=10)
        for xi, (t, s_) in enumerate(zip(df["treasury_chg_bp"], df["spread_chg_bp"])):
            ax.annotate(f"{t:+.0f}", (xi - 0.18, t), ha="center", va="bottom" if t > 0 else "top", fontsize=8)
            ax.annotate(f"{s_:+.0f}", (xi + 0.18, s_), ha="center", va="bottom" if s_ > 0 else "top", fontsize=8)
    axes[0].set_ylabel("change in yield (bp)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Where the AT1 price moves came from: rates vs spread", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(RES / "bnp_at1_yield_split.png", dpi=150)
    plt.close(fig)


def main():
    prices, q, a, share, static = load()
    RES.mkdir(exist_ok=True)
    ust10 = load_ust10(prices)
    last = prices["date"].max()
    year_start = pd.Timestamp(year=last.year - (1 if last.month == 1 else 0), month=1, day=1) - pd.Timedelta(days=1)
    four_weeks = last - pd.Timedelta(days=28)
    labels = prices.groupby("isin")["label"].first().to_dict()

    cap = fig_cet1(q, a, static)
    fig_prices(prices, ust10)
    ytd = yield_split(prices, ust10, year_start, last)
    recent = yield_split(prices, ust10, four_weeks, last)
    fig_split([(f"Since {year_start:%d %b %Y} (or issue)", ytd),
               (f"Last 4 weeks ({four_weeks:%d %b} – {last:%d %b %Y})", recent)], labels)

    cross = prices[prices["date"] == last].set_index("isin")[["mid_price", "mid_yield", "mid_spread_bp"]]
    cross["ust10"] = float(ust10[ust10.index <= last].iloc[-1])
    print("== cross-section on", last.date())
    print(cross.join(static[["coupon_initial", "reset_margin", "first_call_date"]]).round(3).to_string())
    print(f"\n== yield split since {year_start.date()}\n", ytd.round(1).to_string())
    print(f"\n== yield split last 4 weeks\n", recent.round(1).to_string())
    print(f"\n== capital: CET1 {cap['cet1']:.2f}% | requirement {cap['requirement']:.2f}% | "
          f"cushion {cap['cushion_pp']:.2f}pp (€{cap['cushion_bn']:.0f}bn) | "
          f"trigger distance €{cap['trigger_bn']:.0f}bn | P(below MDA in 5y, base case) {cap['p_below_mda_5y']:.1%}")
    s = share["close_eur"]
    print("\n== share: last %.2f on %s, 52w high %.2f, 4-week change %.1f%%"
          % (s.iloc[-1], s.index[-1].date(), s[-252:].max(),
             100 * (s.iloc[-1] / s[s.index <= four_weeks].iloc[-1] - 1)))


if __name__ == "__main__":
    main()
