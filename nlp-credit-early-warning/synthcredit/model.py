"""Discrete-time hazard model, leakage-safe validation, honest metrics.

Model
-----
A discrete-time hazard (Shumway 2001) is a logistic regression on the
issuer-quarter panel; the trailing aggregate event rate (lagged, observable)
serves as the baseline-hazard proxy. Gradient boosting is available as a
robustness check (``model="hgb"``) but the headline model stays linear:
on a desk, an early-warning score you cannot decompose is a score you
cannot defend.

Validation
----------
* expanding-window walk-forward, refit per fold;
* purging: training rows must have their label window fully resolved before
  the training cutoff (t + h <= train_end);
* embargo of ``h`` quarters between the training cutoff and the test window,
  so overlapping label windows cannot bridge the split;
* out-of-cycle holdout: train strictly before the largest stress episode,
  test inside it. With ~3 stress episodes per sample, this is the honest
  estimate of how the model meets a recession it has never seen.

Metrics
-------
* pooled AUC over all test rows;
* incremental AUC of feature set A over B with a block bootstrap that
  resamples *quarters* (events arrive in waves; resampling rows would
  pretend they are independent and shrink the intervals dishonestly);
* precision@k per test quarter, averaged: the watchlist metric. k names on
  a desk get read; ranking quality beyond k is decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .features import FEATURE_SETS


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
                 model: str = "logit") -> np.ndarray:
    Xtr, Xte = train[feats].values, test[feats].values
    ytr = train.y.values
    if model == "logit":
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=4000, C=0.7)
        clf.fit(sc.transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.06,
            l2_regularization=1.0, random_state=0)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    raise ValueError(model)


def walk_forward(mm: pd.DataFrame, feature_set: str, horizon: int = 4,
                 first_train: int = 28, step: int = 4,
                 model: str = "logit") -> pd.DataFrame:
    """Expanding-window walk-forward. Returns test rows with predictions.

    For a fold with training cutoff T0:
      train: rows with quarter + horizon <= T0   (purged: labels resolved)
      test : rows with quarter in [T0 + horizon, T0 + horizon + step)
             (embargo of ``horizon`` quarters)
    """
    feats = FEATURE_SETS[feature_set]
    T = int(mm.quarter.max())
    out = []
    t0 = first_train
    while t0 + horizon <= T:
        train = mm[mm.quarter + horizon <= t0]
        test = mm[(mm.quarter >= t0 + horizon)
                  & (mm.quarter < t0 + horizon + step)]
        if len(test) and train.y.sum() >= 10:
            p = _fit_predict(train, test, feats, model)
            fold = test[["issuer", "quarter", "y"]].copy()
            fold["p"] = p
            fold["fold_cutoff"] = t0
            out.append(fold)
        t0 += step
    if not out:
        raise RuntimeError("no folds produced — sample too short")
    return pd.concat(out, ignore_index=True)


def pooled_auc(preds: pd.DataFrame) -> float:
    return roc_auc_score(preds.y, preds.p)


def precision_at_k(preds: pd.DataFrame, k: int = 15) -> float:
    vals = []
    for _, g in preds.groupby("quarter"):
        if len(g) < k:
            continue
        top = g.nlargest(k, "p")
        vals.append(top.y.mean())
    return float(np.mean(vals)) if vals else np.nan


def bootstrap_delta_auc(preds_a: pd.DataFrame, preds_b: pd.DataFrame,
                        n_boot: int = 400, seed: int = 0) -> dict:
    """AUC(A) - AUC(B), CI by resampling *quarters* with replacement.

    Rows within a quarter are correlated (events cluster); the quarter is
    the exchangeable unit, not the row.
    """
    rng = np.random.default_rng(seed)
    a = preds_a.set_index(["issuer", "quarter"]).p
    b = preds_b.set_index(["issuer", "quarter"]).p
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    y = preds_a.set_index(["issuer", "quarter"]).y.loc[common]
    df = pd.DataFrame({"pa": a, "pb": b, "y": y}).reset_index()

    quarters = df.quarter.unique()
    by_q = {q: g for q, g in df.groupby("quarter")}
    point = roc_auc_score(df.y, df.pa) - roc_auc_score(df.y, df.pb)
    draws = []
    for _ in range(n_boot):
        qs = rng.choice(quarters, size=len(quarters), replace=True)
        s = pd.concat([by_q[q] for q in qs], ignore_index=True)
        if s.y.nunique() < 2:
            continue
        draws.append(roc_auc_score(s.y, s.pa) - roc_auc_score(s.y, s.pb))
    draws = np.array(draws)
    return {"delta": point,
            "lo": float(np.percentile(draws, 5)),
            "hi": float(np.percentile(draws, 95))}


def find_stress_episodes(cycle: pd.DataFrame) -> list[tuple[int, int]]:
    eps, start = [], None
    for t, s in zip(cycle.quarter, cycle.stress):
        if s and start is None:
            start = t
        elif not s and start is not None:
            eps.append((start, t - 1)); start = None
    if start is not None:
        eps.append((start, int(cycle.quarter.max())))
    return sorted(eps, key=lambda e: e[0] - e[1])   # longest first


def cycle_holdout(mm: pd.DataFrame, cycle: pd.DataFrame, feature_set: str,
                  horizon: int = 4, model: str = "logit"):
    """Train strictly before the largest stress episode, test inside it.
    Returns (preds, episode) or (None, None) if the episode is unusable."""
    eps = [e for e in find_stress_episodes(cycle) if e[0] >= 20]
    if not eps:
        return None, None
    start, end = eps[0]
    train = mm[mm.quarter + horizon <= start - 1]
    test = mm[(mm.quarter >= start) & (mm.quarter <= end)]
    if train.y.sum() < 10 or test.y.nunique() < 2:
        return None, None
    p = _fit_predict(train, test, FEATURE_SETS[feature_set], model)
    out = test[["issuer", "quarter", "y"]].copy()
    out["p"] = p
    return out, (start, end)
