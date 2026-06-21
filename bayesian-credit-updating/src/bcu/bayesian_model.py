"""Hierarchical Bayesian logistic model of default (PyMC).

    logit p_i = a_{rating(i)} + sum_k beta_k * signal_{ik}
    a_r ~ Normal(mu_a, sigma_a)          (partial pooling across ratings)
    beta_k ~ Normal(0, 1.5)
    default_i ~ Bernoulli(p_i)

Because the signals enter jointly, each beta_k is the effect of signal k
*holding the others fixed*; exp(beta_k) is therefore a correlation-adjusted
likelihood ratio, typically smaller than the raw marginal LR. This is the
Bayesian analogue of the joint update.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["fit", "beta_summary", "posterior_pd"]

DEFAULT_SIGNALS = ("downgrade", "earnings_miss", "spread_widening")


def fit(data: pd.DataFrame, signals=DEFAULT_SIGNALS, rating_col: str = "rating",
        default_col: str = "default", draws: int = 500, tune: int = 500,
        chains: int = 2, seed: int = 0, progressbar: bool = False):
    """Fit the hierarchical model. Returns (model, idata, ratings, signals)."""
    import pymc as pm

    ratings = sorted(data[rating_col].unique())
    r_idx = {r: i for i, r in enumerate(ratings)}
    ridx = data[rating_col].map(r_idx).to_numpy()
    X = data[list(signals)].astype(float).to_numpy()
    y = data[default_col].astype(int).to_numpy()

    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=-3.5, sigma=2.0)
        sigma_a = pm.HalfNormal("sigma_a", sigma=2.0)
        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=len(ratings))
        beta = pm.Normal("beta", mu=0.0, sigma=1.5, shape=X.shape[1])
        eta = a[ridx] + pm.math.dot(X, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.Bernoulli("obs", p=p, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=0.9, random_seed=seed,
                          progressbar=progressbar)
    return model, idata, ratings, list(signals)


def beta_summary(idata, signals) -> pd.DataFrame:
    """Posterior summary of the signal coefficients and implied LRs."""
    post = idata.posterior["beta"].stack(s=("chain", "draw")).values  # (k, S)
    rows = []
    for i, name in enumerate(signals):
        b = post[i]
        rows.append({
            "signal": name,
            "beta_mean": float(b.mean()),
            "implied_LR": float(np.exp(b.mean())),
            "hdi2.5": float(np.percentile(b, 2.5)),
            "hdi97.5": float(np.percentile(b, 97.5)),
        })
    return pd.DataFrame(rows)


def posterior_pd(idata, ratings, signals, data, rating_col="rating") -> np.ndarray:
    """Posterior-mean default probability for each row of `data`."""
    a = idata.posterior["a"].stack(s=("chain", "draw")).values        # (R, S)
    beta = idata.posterior["beta"].stack(s=("chain", "draw")).values  # (k, S)
    r_idx = {r: i for i, r in enumerate(ratings)}
    ridx = data[rating_col].map(r_idx).to_numpy()
    X = data[list(signals)].astype(float).to_numpy()                  # (N, k)
    eta = a[ridx, :] + X @ beta                                       # (N, S)
    p = 1.0 / (1.0 + np.exp(-eta))
    return p.mean(axis=1)
