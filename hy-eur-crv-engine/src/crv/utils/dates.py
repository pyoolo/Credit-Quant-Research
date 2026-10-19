"""Weekly date-grid helpers used by the fetch and the tracks."""

from __future__ import annotations

from datetime import datetime, timedelta


def last_friday(d: datetime) -> datetime:
    """Most recent Friday on or before ``d``."""
    return d - timedelta(days=int((d.weekday() - 4) % 7))


def iso_week(date_str: str) -> str:
    """ISO year-week key (``%G-%V``) for a ``YYYY-MM-DD`` string."""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%G-%V")


def weekly_grid(n_weeks: int, anchor: datetime | None = None) -> list[str]:
    """``n_weeks`` Friday-anchored dates, most recent first.

    Returns ``["YYYY-MM-DD", ...]`` going back ``n_weeks`` from the last Friday.
    """
    a = last_friday(anchor or datetime.today())
    return [(a - timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(n_weeks)]
