"""Calibration and discrimination metrics for predicted default probabilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

__all__ = ["metrics", "calibration_table"]


def metrics(pd_pred, y) -> dict:
    pd_pred = np.clip(np.asarray(pd_pred, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, int)
    return {
        "AUC": float(roc_auc_score(y, pd_pred)),
        "Brier": float(brier_score_loss(y, pd_pred)),
        "LogLoss": float(log_loss(y, pd_pred, labels=[0, 1])),
    }


def calibration_table(pd_pred, y, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": np.asarray(pd_pred, float), "y": np.asarray(y, int)})
    df["bucket"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = (df.groupby("bucket", observed=True)
           .agg(predicted=("p", "mean"), realized=("y", "mean"), n=("y", "size"))
           .reset_index(drop=True))
    return g
