"""Bloomberg universe definition and field request builders.

All BQL field factories live here so the fetch layer reads as a sequence of
named requests rather than inline lambdas.
"""

from __future__ import annotations

import pandas as pd

from crv.config import settings
from crv.utils import bqlio


def hy_eur_universe():
    """Active bonds, EUR, rated HY by Moody's OR S&P."""
    bq = bqlio.service()
    base = bq.univ.bondsuniv(types="ACTIVE", consolidateduplicates=True)
    is_eur = bq.data.crncy().in_([settings.CURRENCY])
    is_hy = bq.func.or_(
        bq.data.rtg_moody().in_(settings.HY_RATING_CODES),
        bq.data.rtg_sp().in_(settings.HY_RATING_CODES),
    )
    return bq.univ.filter(base, bq.func.and_(is_eur, is_hy))


def fetch_static(universe) -> pd.DataFrame:
    """One-per-bond descriptive fields (rating, sector, maturity, seniority, ...)."""
    import bql
    bq = bqlio.service()
    fields = [
        bq.data.security_des(), bq.data.maturity(), bq.data.cpn(), bq.data.cpn_typ(),
        bq.data.amt_outstanding(), bq.data.callable(), bq.data.industry_group(),
        bq.data.rtg_moody(), bq.data.rtg_sp(), bq.data.rtg_fitch(),
        bq.data.mifid_seniority_of_bond(), bq.data.crncy(), bq.data.issue_dt(),
        bq.data.issuer(), bq.data.id_bb_company(), bq.data.id_isin(),
    ]
    return bqlio.parse_response(bq.execute(bql.Request(universe, fields)))


def _dynamic_field_specs():
    """Per-week market fields as ``(name, factory)`` pairs."""
    bq = bqlio.service()
    return [
        ("duration", bq.data.mty_dur_mid), ("z_spread", bq.data.spread),
        ("ytm", bq.data.yield_), ("px_bid", bq.data.px_bid), ("px_ask", bq.data.px_ask),
        ("next_call_dt", bq.data.nxt_call_dt), ("next_call_px", bq.data.nxt_call_px),
        ("oas", bq.data.risk_oas_mid), ("workout_dt", bq.data.workout_date_mid_to_worst),
    ]


def fetch_dynamic(universe, date_str: str) -> pd.DataFrame:
    """Per-week market fields as of ``date_str``. Skips fields the terminal rejects."""
    from crv.utils.logging import get_logger
    log = get_logger(__name__)
    fields = []
    for name, factory in _dynamic_field_specs():
        try:
            fields.append(factory())
        except Exception as exc:  # noqa: BLE001 - terminal may reject a field
            log.warning("skip %s: %s", name, str(exc)[:30])
    df = bqlio.fetch_dated(universe, fields, date_str)
    df["date"] = date_str
    return df


def dynamic_fetcher(universe):
    """Return a ``(corp_universe, date_str) -> DataFrame`` closure for chunked fetch.

    The chunker passes a *sub-universe* (the ``Corp`` ids it is batching) as the
    first argument, so we fetch against that rather than the full filtered set.
    """
    def _fetch(sub_universe, date_str):
        return fetch_dynamic(sub_universe, date_str)
    return _fetch
