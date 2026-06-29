# Credit Quant Research

A working portfolio of quantitative-finance research projects, with a focus on **credit and fixed income**. Each project is self-contained — its own code, tests, and (where relevant) a short write-up — and is meant to be read, run, and built on.

## Projects

| Project | What it does | Stack |
|---------|--------------|-------|
| [bayesian-credit-updating](./bayesian-credit-updating) | Frames conditional credit pricing as Bayesian default updating: a spread-implied prior revised by signal likelihood ratios, with explicit handling of the correlation between signals (where naïve independence overstates risk). Includes a hierarchical Bayesian model and an out-of-sample calibration backtest. | Python, PyMC, scikit-learn, LaTeX |
| [convertible-bond-pricer](./convertible-bond-pricer) | Prices convertible bonds under the Tsiveriotis-Fernandes (1998) model and a Black-Scholes benchmark. Solves two coupled PDEs on a trinomial tree; handles coupons, call/put schedules. Characterises the three pricing regimes (equity / balanced / distressed) where TF and BS agree or diverge, using 200 synthetic scenarios. | Python, NumPy, SciPy, Matplotlib, LaTeX |

## Roadmap

Planned additions (same self-contained format):

- **cross-sectional-credit-rv** — fair-value model for corporate spreads; residual-as-signal ranking within rating peers, with purged/embargoed time-series cross-validation.
- **hazard-rate-toolkit** — bootstrap survival/default curves from CDS quotes (QuantLib / FinancePy); map posterior PDs back to fair spreads.
- **rating-migration-markov** — estimate and simulate corporate rating-transition matrices; link migration to default.
- **prob-bayes-notes** — the theoretical framework (probability, Bayes, and beyond) that underpins the applied work.

## Layout

```
quant-research/
├── README.md                        ← this index
├── LICENSE                          ← MIT (covers all projects)
├── .gitignore
├── bayesian-credit-updating/        ← project 1 (self-contained)
│   ├── README.md
│   ├── src/ tests/ scripts/ paper/
│   └── pyproject.toml
└── convertible-bond-pricer/         ← project 2 (self-contained)
    ├── README.md
    ├── src/ tests/ scripts/ paper/
    └── pyproject.toml
```

Each project installs and runs independently; see its own `README.md`.

## License

MIT — see [LICENSE](./LICENSE).
