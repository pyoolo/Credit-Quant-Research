# Credit Quant Research

A working portfolio of quantitative-finance research projects, with a focus on **credit and fixed income**. Each project is self-contained — its own code, tests, and (where relevant) a short write-up — and is meant to be read, run, and built on.

## Projects

| Project | What it does | Stack |
|---|---|---|
| [bayesian-credit-updating](./bayesian-credit-updating) | Frames conditional credit pricing as Bayesian default updating: a spread-implied prior revised by signal likelihood ratios, with explicit handling of the correlation between signals (where naïve independence overstates risk). Includes a hierarchical Bayesian model and an out-of-sample calibration backtest. | Python, PyMC, scikit-learn, LaTeX |
| [convertible-bond-pricer](./convertible-bond-pricer) | Prices convertible bonds under the Tsiveriotis–Fernandes (1998) model and a Black–Scholes benchmark. Solves two coupled PDEs on a trinomial tree; handles coupons, call/put schedules. Characterises the three pricing regimes (equity / balanced / distressed) where TF and BS agree or diverge, using 200 synthetic scenarios. | Python, NumPy, SciPy, Matplotlib, LaTeX |
| [political-option-sovereign-debt](./political-option-sovereign-debt) | Prices defaulted sovereign bonds as political contingent claims: a distressed floor plus an embedded digital option on a sovereign-level political normalization event. Inverts observed prices into implied normalization probabilities, bootstraps a piecewise-constant hazard term structure across maturities, tests the cross-maturity monotonicity restriction V(t), runs event studies around political shocks, and falsifies the macro-financial channel with HAC regressions. Companion code to a working paper on Venezuela's post-2017 default; ships a synthetic DGP with known ground truth so every estimator is validated against simulated truth. | Python, NumPy, pandas, statsmodels, Matplotlib, pytest |
| [at1-coco-pricer](./at1-coco-pricer) | Prices perpetual AT1 contingent convertibles (CoCos) as a contingent claim on the issuer's CET1 ratio. A single simulated capital path drives, jointly, mechanical loss absorption (contractual trigger), regulatory absorption (PONV hazard), endogenous coupon cancellation through the CRD-IV Maximum Distributable Amount buffer, and rational call/extension (refinancing cost widens as capital erodes). Monte-Carlo valuation with common-random-number Greeks, an absorption term structure, and a discounted-payoff loss distribution (VaR/ES). | Python, NumPy, pandas, Matplotlib, pytest |

## Roadmap

Planned additions (same self-contained format):

- **cross-sectional-credit-rv** — fair-value model for corporate spreads; residual-as-signal ranking within rating peers, with purged/embargoed time-series cross-validation.
- **hazard-rate-toolkit** — bootstrap survival/default curves from CDS quotes (QuantLib / FinancePy); map posterior PDs back to fair spreads.
- **rating-migration-markov** — estimate and simulate corporate rating-transition matrices; link migration to default.
- **prob-bayes-notes** — the theoretical framework (probability, Bayes, and beyond) that underpins the applied work.
- **political-option extensions** — multi-event catalogue (2017–2026) for aggregated CARs, and bond-specific recovery estimation (sovereign vs. PDVSA seniority) within the contingent-claim framework.
- **at1-coco extensions** — a fully coupled equity process for genuine conversion (rather than a recovery-fraction proxy), and a multi-issuer calibration that disciplines the CET1 dynamics against a cross-section of each bank's AT1 spreads.

## Layout

```
quant-research/
├── README.md                          ← this index
├── LICENSE                            ← MIT (covers all projects)
├── .gitignore
├── bayesian-credit-updating/          ← project 1 (self-contained)
│   ├── README.md
│   ├── src/ tests/ scripts/ paper/
│   └── pyproject.toml
├── convertible-bond-pricer/           ← project 2 (self-contained)
│   ├── README.md
│   ├── src/ tests/ scripts/ paper/
│   └── pyproject.toml
├── political-option-sovereign-debt/   ← project 3 (self-contained)
│   ├── README.md
│   ├── src/ tests/ scripts/ data/ results/
│   └── pyproject.toml
└── at1-coco-pricer/                   ← project 4 (self-contained)
    ├── README.md
    ├── src/ tests/ scripts/ paper/ results/
    └── pyproject.toml
```

Each project installs and runs independently; see its own `README.md`.

## License

MIT — see `LICENSE`.
