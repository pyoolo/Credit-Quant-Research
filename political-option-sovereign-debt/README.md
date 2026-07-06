# political-option-sovereign-debt

**Defaulted sovereign bonds as political contingent claims — reference implementation.**

Companion code for *"When Sovereign Debt Becomes a Political Option"* (Mastrogiacomo, EDHEC Business School, 2026). Once sovereign default becomes prolonged and contractual enforceability breaks down, bond prices stop reflecting discount rates or credit risk and start pricing a single sovereign-level event: **political normalization**. This repo implements the full empirical and interpretative pipeline of the paper — plus several extensions — on Venezuelan sovereign / PDVSA-style data.

```
P_i(t) = R_low + (R_high,i(t) − R_low) · P_t(τ ≤ T_i)
```

Every bond is a baseline distressed claim `R_low` plus a **cash-or-nothing digital payoff** on the normalization event `τ`, with contractual maturity `T_i` acting as an *exposure horizon* rather than a payoff cutoff. Inverting the identity yields market-implied normalization probabilities; a piecewise-constant bootstrap turns the cross-section into an implied **hazard term structure**.

## What's implemented

| Paper section | Module | Content |
|---|---|---|
| §4.1–4.2, §5.2 | `curves.py` | Model-free curve measures: normalization by cross-sectional mean, level factor, dispersion, slope (curve-compression diagnostics) |
| §5.3 | `events.py` | Event-study machinery: abnormal returns vs. constant-mean benchmark, CARs, on-impact jump vs. pre-event drift decomposition with t-stats |
| §5.4, App. B | `regressions.py` | Falsification regressions with Newey–West HAC errors: daily, weekly aggregation, and post-default dummy + macro interactions (Tables 1–3) |
| §6–8 | `contingent.py` | Price ↔ probability inversion, scalar or **bond-specific** `R_high,i` (the general framework of §6.1, not just the fixed 5/50 baseline of §9.1) |
| §8.3 | `hazard.py` | Constant implied hazard **and** piecewise-constant forward-hazard bootstrap across maturities (CDS-style), where negative forward hazards flag monotonicity violations |
| §8.2, §9.2 | `monotonicity.py` | Violation mass V(t) with rolling smoothing (Fig. 8) |
| App. A | `data.py` | Synthetic DGP reproducing the post-default stylised facts + CSV loader for real vendor data |
| Figs. 1–8 | `plotting.py` | Publication-quality replications of all exhibits |

### Extensions beyond the paper

1. **Data-driven benchmark calibration** (`calibrate_benchmarks`). The paper fixes `R_low = 5`, `R_high = 50` a priori. Here (R_low, R_high) are selected by minimising monotonicity violations + unit-interval truncation + a *tightest-bracket* slack term. The docstring documents the identification issue honestly: the price→probability map is affine, so monotonicity alone is invariant to the benchmarks — they are **set-identified**, and the estimate should be read as sensitivity analysis around the baseline.
2. **Hazard term structure.** The paper reports a single constant λ; the bootstrap here recovers forward hazards on maturity buckets, giving a term structure of normalization intensity.
3. **Bond-specific recoveries.** `implied_probability_panel` accepts `{bond: R_high_i}`, restoring the cross-sectional heterogeneity (seniority, governing law, CACs) that the paper's empirical section flattens away.
4. **Synthetic laboratory with known truth.** The simulator generates prices *from the model* with a latent probability path, so every estimator can be validated against ground truth (see `fig7_implied_probs.png`: recovered probabilities track the latent p(t)).

## Quickstart

```bash
pip install -e ".[dev]"          # or: pip install -r requirements.txt
pytest                            # 20 tests
python scripts/run_all.py         # writes all figures + tables to results/
```

Sixty seconds later `results/` contains the analogues of Figures 1–8 and Appendix-B Tables 1–3, plus the hazard term structure. Sample console output:

```
[event study]  impact AR = +0.401   pre-drift = -0.013   t = 37.8
[monotonicity] mean V(t) = 0.0001
[App. B]       daily R² = 0.004 (n=2142); post-default dummy p ≈ 0
```

— i.e., exactly the paper's signature: discrete on-impact repricing with no anticipation, near-perfect cross-maturity consistency, and no macro-financial channel.

### Using real data

Vendor prices for defaulted Venezuelan debt cannot be redistributed. Drop a CSV in `data/` (a `date` column + one column per bond, cents on the dollar) and swap one line:

```python
from political_option import load_price_panel, implied_probability_panel

prices = load_price_panel("data/venz.csv")
probs = implied_probability_panel(prices, r_low=5, r_high=50)
```

Everything downstream — event studies, V(t), hazards, App.-B regressions — is unchanged.

## Minimal API tour

```python
from political_option import (
    simulate_default_panel, implied_probability_panel, calibrate_benchmarks,
    hazard_term_structure, violation_series, event_study, macro_falsification,
    level_factor,
)

panel = simulate_default_panel()
post  = panel.prices.loc[panel.default_date:]

# §9.1 — implied normalization probabilities
probs = implied_probability_panel(post, r_low=5, r_high=50)

# §9.2 — cross-maturity consistency
V = violation_series(probs, panel.maturities)          # V(t) + 60d MA

# §8.3 (extended) — forward hazard term structure at the last date
ts = hazard_term_structure(probs.iloc[-1], horizons=panel.maturities - 2026.0)

# §5.3 — discrete political repricing
es = event_study(level_factor(post), panel.event_dates[-1])
print(es.jump_on_impact, es.pre_event_drift, es.t_stat_impact)

# App. B — macro falsification (HAC)
tab1 = macro_falsification(level_factor(post), panel.macro.loc[panel.default_date:]).table()
```

## Project layout

```
political-option-sovereign-debt/
├── src/political_option/     # installable package (src layout)
│   ├── contingent.py         # pricing identity, inversion, calibration
│   ├── hazard.py             # constant + piecewise-constant hazards
│   ├── monotonicity.py       # V(t)
│   ├── curves.py             # level / dispersion / slope
│   ├── events.py             # AR, CAR, event diagnostics
│   ├── regressions.py        # HAC falsification regressions
│   ├── data.py               # synthetic DGP + loaders
│   └── plotting.py           # Figs. 1–8
├── scripts/run_all.py        # end-to-end reproduction
├── tests/                    # 20 unit/integration tests (pytest)
├── data/                     # your CSVs go here (see data/README.md)
└── results/                  # generated exhibits (gitignored)
```

## Known limitations (inherited from the paper, documented in code)

- Implied probabilities are **pricing** probabilities: beliefs × political risk premia, not physical probabilities (§8.4).
- With a common scalar `R_high`, implied probabilities are an affine transform of prices — co-movement and monotonicity are then partly mechanical. Use bond-specific `R_high,i` or the calibration diagnostics to assess how much structure the model adds beyond raw prices.
- The event study uses a constant-mean benchmark (justified by the null macro exposure in App. B) and, as in the paper, a single salient event; a systematic event catalogue is future work.
- No liquidity / microstructure modelling: low dispersion may partly reflect stale quotes on illiquid distressed bonds.

## Citation

```bibtex
@unpublished{mastrogiacomo2026political,
  title  = {When Sovereign Debt Becomes a Political Option},
  author = {Mastrogiacomo, Paolo},
  year   = {2026},
  note   = {Working paper, EDHEC Business School}
}
```

MIT License.
