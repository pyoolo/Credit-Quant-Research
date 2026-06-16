# cds-basis

A quantitative toolkit for trading the **CDS-cash basis** in investment-grade (IG) credit.

The CDS-cash basis is the difference between a name's CDS par spread and the
spread of its cash bond (asset-swap or Z-spread):

```
basis = CDS_par_spread − cash_spread
```

A **negative basis** means the bond trades cheap relative to the CDS — the
classic *negative-basis trade* is to buy the bond and buy protection, locking
in the discrepancy while being first-order hedged against default. This
library implements the full research workflow for such a strategy:

```
pricing  →  signal  →  backtest  →  risk
```

## Why investment grade

The design targets IG rather than high yield deliberately. In IG the bond/CDS
relationship is more stable, single-name CDS are more liquid, the
cheapest-to-deliver option is worth less, and jump-to-default is remote — so
the statistical signals (mean-reversion / OU) are far more reliable and the
strategy is suited to systematic, multi-name deployment. HY offers wider and
more frequent dislocations but they are largely idiosyncratic risk premia
(recovery, CTD, illiquidity) rather than clean mispricing, which is a poor fit
for a *general* model.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Runtime dependencies: numpy, pandas, scipy.

## Quick start

```python
from cds_basis import run_pipeline

result = run_pipeline()
print(result.backtest.stats)   # sharpe, ann_pnl, max_drawdown, hit_rate, ...
print(result.risk)             # VaR / ES / basis-DV01 / jump-to-default
```

Or run the worked example (also produces an equity-curve plot):

```bash
python examples/run_example.py
```

## Architecture

The package is organised as four composable layers under `src/cds_basis/`.

### `pricing/` — turn quotes into a basis
- `curves.DiscountCurve` — zero curve with log-linear discount-factor interpolation.
- `bond.Bond` — fixed-rate bond pricing, Z-spread solver, par-par asset-swap spread.
- `cds.CDS` — reduced-form (flat-hazard) CDS: par spread, risky annuity with
  accrued-on-default, and hazard bootstrapping from a quoted spread.
- `basis.compute_basis` — combines the two into the CDS-cash basis.

### `data/` — schema + synthetic generator
- A documented long-format panel schema (`PANEL_COLUMNS`).
- `generate_panel` simulates an IG universe where the basis follows a calibrated
  **Ornstein–Uhlenbeck** process, so the signal layer has a genuinely
  mean-reverting target. Swap in real data by producing a DataFrame with the
  same columns and skipping the generator.

### `signal/` — mean-reversion
- `calibrate_ou` — exact-likelihood OU calibration via the AR(1) representation
  (recovers κ, θ, σ, half-life).
- `zscore_signal` / `build_signals` — rolling z-score with entry/exit bands and
  a hysteresis state machine (enter beyond `entry_z`, hold until inside
  `exit_z`), applied name-by-name across the panel.

### `backtest/` — P&L engine
- `run_backtest` — vectorised daily P&L of the basis package, driven by basis
  moves through a **basis-DV01**, plus optional carry, minus transaction costs
  charged on turnover (in bps of spread, valued through the same DV01).
  Positions earn the *next-day* move to avoid look-ahead.
- Reports Sharpe, annualised P&L/vol, max drawdown and hit rate.

### `risk/` — what actually hurts a basis book
- Historical / parametric **VaR** and **Expected Shortfall**.
- **Basis-DV01** book exposure.
- **Jump-to-default**: the dominant tail. The package is hedged against the
  common credit move but *not* against a default, because the bond's realised
  recovery and the CDS settlement recovery differ and the bond may be held at a
  premium to recovery. `jump_to_default_pnl` makes this residual explicit.

## A note on the backtest results

The synthetic data is mean-reverting *by construction*, so in-sample Sharpe is
high. That is a demonstration that the plumbing is correct, **not** a
performance claim. On real data you should expect microstructure noise,
funding/repo costs, CTD effects and execution slippage to erode most of the
in-sample edge — which is exactly why the risk layer and realistic transaction
costs are first-class parts of the pipeline rather than afterthoughts.

## Plugging in real data

Build a DataFrame with columns:

```
date, name, cds_spread, bond_price, coupon, maturity, recovery
```

then:

```python
from cds_basis import run_pipeline
result = run_pipeline(panel=my_dataframe, recompute_basis=True)
```

`recompute_basis=True` re-derives the basis through the full pricing stack
(asset-swap spread vs CDS par spread on a supplied discount curve).

## Testing

```bash
pytest          # 17 tests, ~96% coverage
ruff check src tests examples
```

CI runs lint + tests on Python 3.10–3.12 (see `.github/workflows/ci.yml`).

## Disclaimer

This is research and educational software. It is not investment advice, makes
no warranty of correctness for any production use, and the included data is
synthetic.

## License

MIT — see [LICENSE](LICENSE).
