"""Bloomberg BQL access: session, response parsing, retry/chunk primitives.

This module is the *only* place that imports :mod:`bql`. Everything downstream
takes already-parsed :class:`pandas.DataFrame` objects, which keeps the tracks
testable without a Bloomberg terminal.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from crv.config import settings
from crv.utils.logging import get_logger

log = get_logger(__name__)

_SERVICE = None


def service():
    """Return a singleton BQL service (opens the Bloomberg session on first use)."""
    global _SERVICE
    if _SERVICE is None:
        import bql  # imported lazily so the package imports without a terminal
        _SERVICE = bql.Service()
    return _SERVICE


def parse_response(res) -> pd.DataFrame:
    """Flatten a BQL response (list of per-field frames) into one tidy frame."""
    frames = []
    for item in res:
        df = item.df()
        drop = [c for c in df.columns
                if c in ("MULTIPLIER", "CURRENCY_OF_ISSUE") or "Partial" in c]
        frames.append(df.drop(columns=drop))
    out = pd.concat(frames, axis=1)
    out.columns = [c.split("(")[0].strip() for c in out.columns]  # strip BQL arg suffixes
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    out.index.name = "isin_full"
    return out.reset_index()


def fetch_dated(universe, fields, date_str: str) -> pd.DataFrame:
    """Run a request as of a single historical date (inject a ``dates=range`` clause)."""
    import bql
    bq = service()
    req = bql.Request(universe, fields)
    q = req.to_string().replace(
        "get(", f"with(dates=range('{date_str}','{date_str}')) get(", 1)
    return parse_response(bq.execute(q))


def is_transient(exc: Exception) -> bool:
    """Heuristic: timeouts and ``#N/A`` are worth retrying; other errors are not."""
    m = str(exc).lower()
    return "timed out" in m or "timeout" in m or "#n/a" in m


def as_corp_universe(ids):
    """Append the ``Corp`` yellow-key where missing."""
    return [f"{x} Corp" if not str(x).endswith("Corp") else x for x in ids]


def fetch_retry(fetch_fn, universe, date_str: str, tag: str,
                n_retry: int = settings.N_RETRY, base_wait: int = settings.BASE_WAIT):
    """Call ``fetch_fn(universe, date_str)`` with back-off on transient failures."""
    for attempt in range(1, n_retry + 1):
        try:
            return fetch_fn(universe, date_str)
        except Exception as exc:  # noqa: BLE001 - re-raised below if not transient
            if attempt < n_retry and is_transient(exc):
                wait = base_wait * attempt
                log.warning("[%s] retry %d in %ds", tag, attempt, wait)
                time.sleep(wait)
                continue
            raise


def fetch_chunked(fetch_fn, ids, date_str: str, currency: str = settings.CURRENCY,
                  chunk_size: int | None = settings.CHUNK_SIZE) -> pd.DataFrame | None:
    """Fetch one date in batches to stay under terminal limits."""
    if chunk_size is None or len(ids) <= chunk_size:
        return fetch_retry(fetch_fn, as_corp_universe(ids), date_str, currency)
    parts: list[pd.DataFrame] = []
    n = int(np.ceil(len(ids) / chunk_size))
    for i, sub in enumerate(np.array_split(ids, n), 1):
        d = fetch_retry(fetch_fn, as_corp_universe(sub), date_str, f"{currency}#{i}/{n}")
        if d is not None and len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None
