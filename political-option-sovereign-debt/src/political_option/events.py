"""
Event-study machinery for discrete political repricing (paper, Sec. 5.3).

Abnormal returns are computed against a constant-mean benchmark estimated
on a pre-event window — appropriate here because the falsification tests
(Sec. 5.4 / App. B) show no systematic macro-factor exposure to net out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def abnormal_returns(
    series: pd.Series,
    event_date: pd.Timestamp,
    estimation_window: int = 120,
    gap: int = 10,
) -> pd.Series:
    """Return series minus the mean return over the estimation window.

    The estimation window ends ``gap`` trading days before the event to
    avoid contamination from pre-event drift.
    """
    rets = series.pct_change().dropna()
    idx = rets.index.get_indexer([pd.Timestamp(event_date)], method="nearest")[0]
    est_end = max(0, idx - gap)
    est_start = max(0, est_end - estimation_window)
    if est_end - est_start < 20:
        raise ValueError("Estimation window too short (< 20 obs).")
    mu = rets.iloc[est_start:est_end].mean()
    return rets - mu


def cumulative_abnormal_returns(
    series: pd.Series,
    event_date: pd.Timestamp,
    window: tuple[int, int] = (-10, 4),
    estimation_window: int = 120,
    gap: int = 10,
) -> pd.Series:
    """CAR over trading-day event window [window[0], window[1]] (Figs. 5-6)."""
    ar = abnormal_returns(series, event_date, estimation_window, gap)
    idx = ar.index.get_indexer([pd.Timestamp(event_date)], method="nearest")[0]
    lo, hi = window
    seg = ar.iloc[idx + lo : idx + hi + 1]
    car = seg.cumsum()
    car.index = pd.RangeIndex(lo, lo + len(car), name="event_time")
    return car.rename("CAR")


@dataclass(frozen=True)
class EventStudyResult:
    car: pd.Series
    jump_on_impact: float      # AR at event time 0..+1
    pre_event_drift: float     # CAR from window start to t = -1
    t_stat_impact: float       # impact AR / sd of estimation-window ARs


def event_study(
    series: pd.Series,
    event_date: pd.Timestamp,
    window: tuple[int, int] = (-10, 4),
    estimation_window: int = 120,
    gap: int = 10,
) -> EventStudyResult:
    """Full diagnostic: sharp on-impact repricing with no pre-event drift is
    the signature of *discrete belief updating* (Sec. 5.3)."""
    ar = abnormal_returns(series, event_date, estimation_window, gap)
    idx = ar.index.get_indexer([pd.Timestamp(event_date)], method="nearest")[0]
    est_end = max(0, idx - gap)
    est_start = max(0, est_end - estimation_window)
    sigma = ar.iloc[est_start:est_end].std()
    car = cumulative_abnormal_returns(series, event_date, window, estimation_window, gap)
    impact = float(ar.iloc[idx : idx + 2].sum())
    pre = float(car.loc[:-1].iloc[-1]) if (car.index < 0).any() else 0.0
    t_stat = impact / (sigma * np.sqrt(2)) if sigma > 0 else np.nan
    return EventStudyResult(car=car, jump_on_impact=impact, pre_event_drift=pre, t_stat_impact=float(t_stat))
