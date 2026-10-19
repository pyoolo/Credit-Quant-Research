"""Track-1 fair-spread Lasso: feature winsorising, alpha selection, final fit.

Alpha is selected on a 30% *issuer* hold-out rather than the internal
GroupKFold CV, because throughout development the CV optimum kept preferring a
higher alpha than the point that generalises to unseen issuers. Selecting on a
held-out issuer split removes that search-vs-holdout divergence at its root; the
final model is then refit on all training rows at the chosen alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from crv.config import settings
from crv.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FittedModel:
    """The fitted Lasso and everything needed to score a new cross-section."""
    model: Lasso
    scaler: StandardScaler
    feats: list[str]            # numeric features, in scaler order
    xcols: pd.Index             # full design-matrix columns (num + sector + week)
    alpha: float
    r2_oof: float
    r2_holdout: float
    winsor_bounds: dict         # feature -> (lo, hi), train-only


def _winsorise_train(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clip coupon/age to train-only winsor bounds (applied to the whole panel)."""
    df = df.copy()
    train_mask = df["in_train"].fillna(False)
    bounds: dict = {}
    for f in ["coupon", "age"]:
        s = pd.to_numeric(df.loc[train_mask, f], errors="coerce")
        lo, hi = float(s.quantile(settings.WINSOR)), float(s.quantile(1 - settings.WINSOR))
        bounds[f] = (lo, hi)
        df[f] = pd.to_numeric(df[f], errors="coerce").clip(lo, hi)
    return df, bounds


def _build_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_zs"] = np.log(df["z_spread"].clip(lower=1))
    train = df[df["in_train"]].dropna(subset=["log_zs", "duration", "rating_notch"]).copy()
    train["wk"] = train["date"].astype(str)
    sec_n = train.groupby("industry")["isin"].transform("nunique")
    train["sector"] = np.where(sec_n >= settings.MIN_SEC, train["industry"], "OTHER")
    return train


def _design_matrix(train: pd.DataFrame, feats: list[str]):
    scaler = StandardScaler()
    Xnum = train[feats].copy().fillna(train[feats].median())
    Xnum_s = pd.DataFrame(scaler.fit_transform(Xnum), columns=feats, index=Xnum.index)
    Xcat = pd.get_dummies(train[["sector", "wk"]], drop_first=True)
    X = pd.concat([Xnum_s, Xcat], axis=1)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    return X, scaler


def lasso_path(df: pd.DataFrame) -> pd.DataFrame:
    """Document feature selection: which features survive as alpha rises.

    Returns a frame of ``(alpha, n_active, r2_oof, active_features)`` rows.
    """
    df, _ = _winsorise_train(df)
    train = _build_train(df)
    y = train["log_zs"].values
    groups = train["issuer_key"].values
    gkf = GroupKFold(n_splits=settings.N_SPLITS)
    X, _ = _design_matrix(train, settings.SET_FEATS)

    fset = set(settings.SET_FEATS)
    rows = []
    for a in np.logspace(-3.5, -0.5, 18):
        oof = np.full(len(train), np.nan)
        for tr, te in gkf.split(X, y, groups):
            m = Lasso(alpha=a, max_iter=8000).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
        coef = pd.Series(Lasso(alpha=a, max_iter=8000).fit(X, y).coef_, index=X.columns)
        active = [f for f in fset if abs(coef.get(f, 0)) > 1e-6]
        rows.append({"alpha": a, "n_active": len(active),
                     "r2_oof": r2_score(y, oof),
                     "features": sorted(active, key=lambda f: -abs(coef[f]))})
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame) -> tuple[FittedModel, pd.DataFrame]:
    """Fit the final model. Returns ``(FittedModel, train_frame)``."""
    df, winsor_bounds = _winsorise_train(df)
    train = _build_train(df)
    y = train["log_zs"].values
    groups = train["issuer_key"].values
    gkf = GroupKFold(n_splits=settings.N_SPLITS)
    X, scaler = _design_matrix(train, settings.SEL_FEATS)

    # alpha selected on a 30% held-out issuer split
    alphas = np.logspace(-3.5, -2.0, 30)
    rng = np.random.RandomState(0)
    issuers = np.array(sorted(train["issuer_key"].dropna().unique()), dtype=object)
    perm = rng.permutation(len(issuers))
    n_hold = int(len(issuers) * 0.30)
    hold = set(issuers[perm[:n_hold]])
    search = set(issuers[perm[n_hold:]])
    msk_s = train["issuer_key"].isin(search).values
    msk_h = train["issuer_key"].isin(hold).values

    best_a, best_r2 = None, -9.0
    for a in alphas:
        m = Lasso(alpha=a, max_iter=8000).fit(X[msk_s], y[msk_s])
        r = r2_score(y[msk_h], m.predict(X[msk_h]))
        if r > best_r2:
            best_r2, best_a = r, a
    log.info("alpha selected on held-out issuers: %.4f (held-out R2 %.3f)", best_a, best_r2)

    # issuer-grouped OOF for comparison
    oof = np.full(len(train), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = Lasso(alpha=best_a, max_iter=8000).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    r2_oof = r2_score(y, oof)
    log.info("final model — R2 OOF: %.3f | held-out: %.3f", r2_oof, best_r2)

    final = Lasso(alpha=best_a, max_iter=8000).fit(X, y)
    fitted = FittedModel(
        model=final, scaler=scaler, feats=list(settings.SEL_FEATS),
        xcols=X.columns, alpha=float(best_a), r2_oof=float(r2_oof),
        r2_holdout=float(best_r2), winsor_bounds=winsor_bounds,
    )
    return fitted, train


def contribution_table(fitted: FittedModel) -> pd.DataFrame:
    """Standardised coefficients and contribution share (|coef| share -> 100)."""
    coef = pd.Series(fitted.model.coef_, index=fitted.xcols).loc[fitted.feats]
    coef = coef[coef.abs() > 1e-9]
    share = 100 * coef.abs() / coef.abs().sum()
    return (pd.DataFrame({"coef": coef.round(4), "share_%": share.round(1)})
            .sort_values("share_%", ascending=False))
